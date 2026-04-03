import os
import pathlib

from tqdm import tqdm

from models.mrhp_1.base_model import ColBERT
from models.mrhp_1.candidate_generation import CandidateGeneration
from models.mrhp_1.doc_tokenization import DocTokenizer
from models.mrhp_1.new_hdr_search import HDRSearch
from models.mrhp_1.index_loader import IndexLoader
from models.mrhp_1.query_tokenization import QueryTokenizer
from models.mrhp_1.residual_embeddings_strided import ResidualEmbeddingsStrided
from models.mrhp_1.util import pool_embeddings_hierarchical, _stack_3D_tensors, MixedPrecisionManager
import torch
from math import ceil
from torch.utils.cpp_extension import load

class PLAID_HDRC(ColBERT):
    def __init__(self, config, verbose: int = 3):
        super().__init__(config)
        assert self.training is False

        self.verbose = verbose

        self.query_tokenizer = QueryTokenizer(config, verbose=self.verbose)
        self.doc_tokenizer = DocTokenizer(config)
        #
        self.amp_manager = MixedPrecisionManager(True)

    def query(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                Q = super().query(*args, **kw_args)
                return Q.cpu() if to_cpu else Q

    def doc(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                D = super().doc(*args, **kw_args)
                if to_cpu:
                    return (D[0].cpu(), *D[1:]) if isinstance(D, tuple) else D.cpu()

                return D

    def queryFromText(
        self, queries, context=None, full_length_search=False
    ):

        input_ids, attention_mask = self.query_tokenizer.tensorize(
            queries, context=context, full_length_search=full_length_search
        )
        Q = self.query(input_ids, attention_mask)
        re = Q.cpu()
        return re

    def docFromText(
        self,
        docs,
        bsize=None,
        keep_dims=True,
        to_cpu=False,
        showprogress=True,
        return_tokens=False,
        pool_factor=1,
        protected_tokens=0,
        clustering_mode: str = "hierarchical",
    ):
        assert keep_dims in [True, False, "flatten"]
        assert clustering_mode in ["hierarchical"]

        if bsize:
            text_batches, reverse_indices = self.doc_tokenizer.tensorize(
                docs, bsize=bsize
            )

            returned_text = []
            if return_tokens:
                returned_text = [text for batch in text_batches for text in batch[0]]
                returned_text = [returned_text[idx] for idx in reverse_indices.tolist()]
                returned_text = [returned_text]

            keep_dims_ = "return_mask" if keep_dims == "flatten" else keep_dims
            batches = [
                self.doc(input_ids, attention_mask, keep_dims=keep_dims_, to_cpu=to_cpu)
                for input_ids, attention_mask in
                    text_batches
            ]

            if keep_dims is True:
                D = _stack_3D_tensors(batches)
                return (D[reverse_indices], *returned_text)

            elif keep_dims == "flatten":
                D, mask = [], []

                for D_, mask_ in batches:
                    D.append(D_)
                    mask.append(mask_)


                D, mask = (
                    torch.cat(D)[reverse_indices],
                    torch.cat(mask)[reverse_indices],
                )
                # D = D[:,2:,:]  # 或者 D = D[2:, :, :]
                # mask = mask[:,2:,:]

                doclens = mask.squeeze(-1).sum(-1).tolist()

                D = D.reshape(-1, self.config["model_config"]["compression_dim"])
                D = D[mask.bool().flatten()].cpu()

                if pool_factor > 1:
                    print(f"Clustering tokens with a pool factor of {pool_factor}")
                    D, doclens = pool_embeddings_hierarchical(
                        D,
                        doclens,
                        pool_factor=pool_factor,
                        protected_tokens=protected_tokens,
                        showprogress=showprogress,
                    )
                return (D, doclens, *returned_text)

            assert keep_dims is False

            D = [d for batch in batches for d in batch]
            return ([D[idx] for idx in reverse_indices.tolist()], *returned_text)

        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        return self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)


class IndexScorer(IndexLoader, CandidateGeneration):
    def __init__(self, config, index_path, use_gpu=False, load_index_with_mmap=False):
        super().__init__(
            config= config,
            index_path=index_path,
            use_gpu=use_gpu,
            load_index_with_mmap=load_index_with_mmap
        )

        IndexScorer.try_load_torch_extensions(use_gpu)

        self.set_embeddings_strided()

        # 创建一个聚类器，并且读入相关的文件
        self.hdr_cluster = HDRSearch(
            beta= self.config["run_config"]["beta"],
            l_max=self.config["run_config"]["l_max"],
            update_threshold_radius=self.config["run_config"]["update_threshold_radius"],
            update_radius_start_layer=self.config["run_config"]["update_radius_start_layer"]
        )
        self.hdr_cluster.load(index_path)
        print(1)


    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, "loaded_extensions") or use_gpu:
            return

        print(f"Loading filter_pids_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)...")
        filter_pids_cpp = load(
            name="filter_pids_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "filter_pids.cpp"
                ),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.filter_pids = filter_pids_cpp.filter_pids_cpp

        print(f"Loading decompress_residuals_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)...")
        decompress_residuals_cpp = load(
            name="decompress_residuals_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "decompress_residuals.cpp"
                ),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.decompress_residuals = decompress_residuals_cpp.decompress_residuals_cpp

        print(
            f"Loading segmented_maxsim_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)...")
        segmented_maxsim_cpp = load(
            name="segmented_maxsim_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "segmented_maxsim.cpp"
                ),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.segmented_maxsim = segmented_maxsim_cpp.segmented_maxsim_cpp

        cls.loaded_extensions = True

    def set_embeddings_strided(self):
        if self.load_index_with_mmap:
            assert self.num_chunks == 1
            self.offsets = torch.cumsum(self.doclens, dim=0)
            self.offsets = torch.cat( (torch.zeros(1, dtype=torch.int64), self.offsets) )
        else:
            self.embeddings_strided = ResidualEmbeddingsStrided(self.codec, self.embeddings, self.doclens)
            self.offsets = self.embeddings_strided.codes_strided.offsets

    def lookup_pids(self, passage_ids, out_device='cuda', return_mask=False):
        return self.embeddings_strided.lookup_pids(passage_ids, out_device)

    def retrieve(self, config, Q):
        Q = Q[:, :config["run_config"]["max_query_len"]]   # NOTE: Candidate generation uses only the query tokens
        pids, centroid_scores = self.generate_candidates(config, Q, self.hdr_cluster)

        return pids, centroid_scores

    def embedding_ids_to_pids(self, embedding_ids):
        all_pids = torch.unique(self.emb2pid[embedding_ids.long()].cuda(), sorted=False)
        return all_pids

    def rank(self, config, Q, pids=None):
        with torch.inference_mode():
            if pids is None:
                pids, centroid_scores = self.retrieve(config, Q)
            else:
                pids = torch.tensor(pids, dtype=torch.int32, device=Q.device)
                centroid_scores = None

            scores, pids = self.score_pids(config, Q, pids, centroid_scores)

            scores_sorter = scores.sort(descending=True)
            pids, scores = pids[scores_sorter.indices].tolist(), scores_sorter.values.tolist()

            return pids, scores

    def score_pids(self, config, Q, pids, centroid_scores):
        """
            Always supply a flat list or tensor for `pids`.

            Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
            If Q.size(0) is 1, the matrix will be compared with all passages.
            Otherwise, each query matrix will be compared against the *aligned* passage.
        """

        if centroid_scores is not None:

            idx = centroid_scores.max(-1).values >= config["run_config"][r"centroid_score_threshold"]


            pids = IndexScorer.filter_pids(
                    pids, centroid_scores, self.embeddings.codes, self.doclens,
                    self.offsets, idx, config["run_config"][r"ndocs"]
                )


        D_packed = IndexScorer.decompress_residuals(
                pids,
                self.doclens,
                self.offsets,
                self.codec.bucket_weights,
                self.codec.reversed_bit_map,
                self.codec.decompression_lookup_table,
                self.embeddings.residuals,
                self.embeddings.codes,
                self.codec.centroids,
                self.codec.dim,
                self.codec.nbits
            )
        D_packed = torch.nn.functional.normalize(D_packed.to(torch.float32), p=2, dim=-1)
        D_mask = self.doclens[pids.long()]

        return self.colbert_score_packed(Q, D_packed, D_mask), pids



    def colbert_score_packed(self, Q, D_packed, D_lengths):
        """
            Works with a single query only.
        """

        Q = Q.squeeze(0)

        assert Q.dim() == 2, Q.size()
        assert D_packed.dim() == 2, D_packed.size()

        scores = D_packed @ Q.to(dtype=D_packed.dtype).T

        return self.segmented_maxsim(scores, D_lengths)
