import gc
import gzip
import json
import os
import pickle
import time
from collections import defaultdict

import faiss
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import ir_measures
from ir_measures import nDCG, RR, Success

from loader.pressure_dataloader import CorpusLoader, QueryLoader, StreamLoader
from models.model_factory import ModelsFactory
from models.mrhp_g.hdr_search import HDRSearch
from models.mrhp_g.index_saver import IndexSaver
from models.mrhp_g.main_model import IndexScorer
from models.mrhp_g.residual import ResidualCodec
from models.mrhp_g.util import optimize_ivf, build_ivf_dict, compile_dict_to_ivf
from module.dynamic_process import dynamic_process_factory, dynamic_process_doc_id_factory
from module.retriever import Retriever
import torch
import os

from utils.util_effectiveness import save_effectiveness_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from utils.util_timer import timer


def batch(group, bsize, provide_offset=False):
    offset = 0
    while offset < len(group):
        L = group[offset: offset + bsize]
        yield ((offset, L) if provide_offset else L)
        offset += len(L)
    return

class MRHPPressureRetrieverUp:
    def __init__(self, config):
        self.model = ModelsFactory.build_model(config)
        self.saver = IndexSaver(config)
        self.config = config
        self.global_token_vecs_array = None
        self.global_token_doc_ids_array = None
        self.hdrc_cluster = None
        self.num_partitions = None
        self._load_data()
        self.use_reindex = [0]


    def _load_data(self):
        """
        Loads the corpus, queries, and labels from the specified configuration.
        :return: None
        """
        # Load corpus
        corpus_loader = CorpusLoader()
        self.corpus_dict = corpus_loader.load(self.config)

        # Load queries
        query_loader = QueryLoader()
        self.queries_dict = query_loader.load(self.config)

        # Load streaming data
        stream_loader = StreamLoader()
        self.stream = stream_loader.load(self.config)


    def load_embedding_vectors(self, save_file=None):
        if os.path.exists(save_file):
            with h5py.File(save_file, 'r') as f:
                all_token_vecs_array = f['token_vecs'][:]  # 二维浮点数组（压缩维度向量）
                all_doc_lens_array = f['doc_lens'][:]  # 文档ID索引
                print(r"Load Embedding")
                return all_token_vecs_array, all_doc_lens_array


    def embedding_corpus(self, corpus_dict, streaming_step=0):
        corpus = list(corpus_dict.values())
        save_file = self.set_up_embedding_save_file(streaming_step)
        if os.path.exists(save_file):
            print(f"Embedding vectors already exist at {save_file}. Loading existing vectors.")
            all_token_vecs_array, all_token_doc_ids = self.load_embedding_vectors(save_file)
            return all_token_vecs_array, all_token_doc_ids
        else:
            print(f"Num of Corpus : {len(corpus)}")
            all_token_vecs_array = []
            all_doc_lens_array =[]
            with h5py.File(save_file, 'a') as f:
                all_token_vecs = f.create_dataset('token_vecs',
                                                  shape=(0, self.config["model_config"]['compression_dim']),
                                                  maxshape=(None, self.config["model_config"]['compression_dim']),
                                                  chunks=(1000, self.config["model_config"]['compression_dim']),
                                                  # 按需设置分块
                                                  compression='gzip',
                                                  dtype='float32')

                all_doc_lens = f.create_dataset('doc_lens',
                                                     shape=(0,),
                                                     maxshape=(None,),
                                                     chunks=(1000,),
                                                     compression='gzip',
                                                     dtype='int32')

                with torch.inference_mode():
                    for batch_idx in tqdm(range(0, len(corpus), self.config["run_config"]["batch_size"]*10)):
                        batch_text = corpus[batch_idx:batch_idx + self.config["run_config"]["batch_size"]*10]

                        embs_, doclens_ = self.model.docFromText(
                            batch_text,
                            bsize=self.config["run_config"]["batch_size"],
                            keep_dims="flatten",
                            pool_factor=self.config["run_config"]["pool_factor"],
                            clustering_mode=self.config["run_config"]["clustering_mode"],
                            protected_tokens=self.config["run_config"]["protected_tokens"],
                        )

                        embs_ = embs_.detach().cpu().numpy()
                        all_token_vecs_array.append(embs_)
                        all_token_vecs_current_size = all_token_vecs.shape[0]
                        all_token_vecs.resize(all_token_vecs_current_size + embs_.shape[0], axis=0)
                        all_token_vecs[all_token_vecs_current_size:, :] = embs_
                        del embs_

                        doclens_ = np.array(doclens_, dtype=np.int32)
                        all_doc_lens_array.append(doclens_)
                        all_doc_lens_current_size = all_doc_lens.shape[0]
                        all_doc_lens.resize(all_doc_lens_current_size + doclens_.shape[0], axis=0)
                        all_doc_lens[all_doc_lens_current_size:] = doclens_

            all_token_vecs_array = np.vstack(all_token_vecs_array)
            all_doc_lens_array = np.concatenate(all_doc_lens_array)
        return all_token_vecs_array, all_doc_lens_array

    def embedding_add_corpus(self, corpus_dict):
        corpus = list(corpus_dict.values())
        all_token_vecs_array = []
        all_doc_lens_array = []

        for batch_idx in tqdm(range(0, len(corpus), self.config["run_config"]["batch_size"] * 10)):
            batch_text = corpus[batch_idx:batch_idx + self.config["run_config"]["batch_size"] * 10]

            embs_, doclens_ = self.model.docFromText(
                batch_text,
                bsize=self.config["run_config"]["batch_size"],
                keep_dims="flatten",
                pool_factor=self.config["run_config"]["pool_factor"],
                clustering_mode=self.config["run_config"]["clustering_mode"],
                protected_tokens=self.config["run_config"]["protected_tokens"],
            )
            embs_ = embs_.detach().cpu().numpy()
            all_token_vecs_array.append(embs_)
            del embs_

            doclens_ = np.array(doclens_, dtype=np.int32)
            all_doc_lens_array.append(doclens_)
            del doclens_

        all_token_vecs_array = np.vstack(all_token_vecs_array)
        all_doc_lens_array = np.concatenate(all_doc_lens_array)

        return all_token_vecs_array, all_doc_lens_array

    def init_experiment(self):
        init_cycle = self.stream[0]
        init_corpus_id = init_cycle["steps"][0]["doc_ids"]
        init_queries_id = init_cycle["related_queries"]
        init_labels = init_cycle["cycle_labels"]
        init_corpus = {d_id: self.corpus_dict[d_id] for d_id in init_corpus_id}
        init_queries = {q_id: self.queries_dict[q_id] for q_id in init_queries_id}
        labels_df = pd.DataFrame(init_labels)
        index_time_folder = self.set_up_index_time_folder()
        with timer(0, f"index_embedding", index_time_folder):
            init_token_vecs_array, init_doc_lens_array = self.embedding_corpus(init_corpus, streaming_step=0)
        gc.collect()
        torch.cuda.empty_cache()
        self.indexing_corpus_all_from_zero(init_token_vecs_array,
                                           init_doc_lens_array,
                                           task=0)

        analysis_save_path = self.run_retrieve_this_step(init_queries,
                                                         init_queries_id,
                                                         init_corpus_id,
                                                         task=0)
        self.evaluate(analysis_save_path, labels_df, task=0)

    def run_retrieve_this_step(self, queries_dict, query_ids, corpus_id, task=0):
        analysis_save_folder, run_time_save_folder = self.set_up_retrieve_result_path()
        analysis_save_path = os.path.join(analysis_save_folder, f'task_{task}_performance.run.gz')
        index_folder_path = self.set_up_index_save_path(task=task)
        self.ranker = IndexScorer(self.config, index_folder_path)
        top_k_doc = self.config["run_config"]["top_k_doc"]
        results_list = []
        idx = 0
        for q_id, query in tqdm(queries_dict.items(), desc=f"Retrieving Queries at Task {task}"):
            with timer(idx, f"task_{task}_query_embedding_time", run_time_save_folder):
                query_vecs_array = self.model.queryFromText([query])
            with timer(idx, f"task_{task}_performance_query_time", run_time_save_folder):
                result = self.dense_search(query_vecs_array, k=top_k_doc)
            results_list.append(result)
            idx += 1
        with gzip.open(analysis_save_path, 'wt') as fout:
            q_idx = 0
            for results in results_list:
                q_id = query_ids[q_idx]
                d_idx_list, rank_list, score_list = results
                for d_idx, rank, score in zip(d_idx_list, rank_list, score_list):
                    did = corpus_id[d_idx]
                    fout.write(f'{q_id} 0 {did} {rank} {score} run\n')
                q_idx += 1
        return analysis_save_path

    def run_retrieve_other_step(self, queries_dict, query_ids, corpus_id, streaming_step=0, task=0):
        analysis_save_folder, run_time_save_folder = self.set_up_retrieve_result_path(task=task)
        analysis_save_path = os.path.join(analysis_save_folder, f'task_{task}_performance_in_step_{streaming_step}.run.gz')
        index_folder_path = self.set_up_index_save_path(streaming_step=streaming_step)
        self.ranker = IndexScorer(self.config, index_folder_path)
        top_k_doc = self.config["run_config"]["top_k_doc"]
        results_list = []
        idx = 0
        for q_id, query in tqdm(queries_dict.items(), desc=f"Retrieving Queries at Step {streaming_step}"):
            with timer(idx, f"task_{task}_performance_in_step_{streaming_step}_query_embedding_time", run_time_save_folder):
                query_vecs_array = self.model.queryFromText([query])
            with timer(idx, f"task_{task}_performance_in_step_{streaming_step}_query_time", run_time_save_folder):
                result = self.dense_search(query_vecs_array, k=top_k_doc)
            results_list.append(result)
            idx += 1
        with gzip.open(analysis_save_path, 'wt') as fout:
            q_idx = 0
            for results in results_list:
                q_id = query_ids[q_idx]
                d_idx_list, rank_list, score_list = results
                for d_idx, rank, score in zip(d_idx_list, rank_list, score_list):
                    did = corpus_id[d_idx]
                    fout.write(f'{q_id} 0 {did} {rank} {score} run\n')
                q_idx += 1

        return analysis_save_path


    def evaluate(self, analysis_save_path, labels_df, task=0):
        effectiveness_file = self.set_up_effectiveness()
        measure = [nDCG @ 10, RR @ 10, Success @ 10]
        faiss_results_pd = pd.DataFrame(list(ir_measures.read_trec_run(analysis_save_path)))
        eval_results = ir_measures.calc_aggregate(measure, labels_df, faiss_results_pd)
        save_results = {}
        for k, v in eval_results.items():
            save_results[k.NAME] = v
        save_effectiveness_metrics(effectiveness_file, task, save_results)

    def dense_search(self, Q, k=10):
        pids, scores = self.ranker.rank(self.config, Q)
        return pids[:k], list(range(1, k + 1)), scores[:k]

    def streaming_embedding(self, streaming_step):
        add_corpus = {d_id: self.corpus_dict[d_id] for d_id in self.streaming_data[streaming_step]["corpus"]}
        add_token_vecs_array, add_doc_lens_array = self.embedding_corpus(add_corpus, streaming_step=streaming_step)
        return add_token_vecs_array, add_doc_lens_array

    def streaming_experiment(self):
        #================= 1.读入 init 的 index ======================
        dynamic_doc_id_processor = dynamic_process_doc_id_factory.build_process(self.config)
        init_index_folder_path = self.set_up_index_save_path(task=0)
        init_vector_file = self.set_up_embedding_save_file(0)
        if os.path.exists(init_vector_file):
            print(f"Embedding vectors already exist at {init_vector_file}. Loading existing vectors.")
            all_token_vecs_array, all_doc_lens_array = self.load_embedding_vectors(init_vector_file)
        else:
            print(f"Embedding vectors do not exist at {init_vector_file}. Please run init experiment.")
            return


        self.hdrc_cluster = HDRSearch(
                beta=self.config["run_config"]["beta"],
                l_max=self.config["run_config"]["l_max"],
                update_threshold_radius=self.config["run_config"]["update_threshold_radius"],
                update_radius_start_layer=self.config["run_config"]["update_radius_start_layer"],
                G_th= self.config["run_config"]["G_th"]
            )

        self.hdrc_cluster.load(init_index_folder_path)
        self.hdrc_cluster.data = all_token_vecs_array


        self.num_embeddings = all_token_vecs_array.shape[0]
        codec = ResidualCodec.load(self.config, init_index_folder_path)
        compressed_embs = ResidualCodec.Embeddings.load_chunks(
            init_index_folder_path,
            range(1),
            self.num_embeddings
        )

        MAX_CAPACITY = self.config["run_config"]["max_capacity"]
        # --- 挖坑 1: 残差 (Residuals) 缓冲区 ---
        dim = compressed_embs.residuals.shape[1]
        dtype = compressed_embs.residuals.dtype
        device = compressed_embs.residuals.device
        self.global_residuals_buffer = torch.empty((MAX_CAPACITY, dim), dtype=dtype, device=device)
        current_size = compressed_embs.residuals.shape[0]
        self.global_residuals_buffer[:current_size] = compressed_embs.residuals

        # --- 挖坑 2: 聚类分配 ID (Codes) 缓冲区 ---
        codes_dtype = compressed_embs.codes.dtype
        codes_device = compressed_embs.codes.device
        self.global_codes_buffer = torch.empty(MAX_CAPACITY, dtype=codes_dtype, device=codes_device)
        self.global_codes_buffer[:self.num_embeddings] = compressed_embs.codes

        with open(os.path.join(init_index_folder_path, "doclens.json"), 'r', encoding='utf-8') as file:
            doc_lens = json.load(file)

        optimized_ivf_path = os.path.join(init_index_folder_path, 'ivf_dict.pkl')
        # 注意：这里必须用 'rb' (read binary) 二进制读取模式
        with open(optimized_ivf_path, 'rb') as f:
            pre_ivf_dict = pickle.load(f)
        init_cycle = self.stream[0]
        all_corpus_id = init_cycle["steps"][0]["doc_ids"]
        #====================== 2.设置流数据 ===========================
        add_tokens_num_list = []
        total_step = 1
        while total_step < 1000:
            for task_step in range(1, len(self.stream)):
                all_docs = self.stream[task_step]["steps"]
                index_folder_path = self.set_up_index_save_path(task=task_step)
                this_queries_id = [self.stream[task_step]["target_query_id"]]
                this_labels = self.stream[task_step]["cycle_labels"]
                this_queries = {q_id: self.queries_dict[q_id] for q_id in this_queries_id}
                labels_df = pd.DataFrame(this_labels)
                for this_docs in all_docs:
                    doc_ids = this_docs["doc_ids"]
                    add_corpus = {d_id: self.corpus_dict[d_id] for d_id in doc_ids}
                    all_corpus_id = dynamic_doc_id_processor(*[all_corpus_id, list(add_corpus.keys())])
        #===================== 3.嵌入新增向量 ===========================
                    add_token_vecs_array, add_doc_lens_array = self.embedding_add_corpus(add_corpus)
                    all_token_vecs_array = np.concatenate([all_token_vecs_array, add_token_vecs_array], axis=0)
                    all_doc_lens_array = np.concatenate([all_doc_lens_array, add_doc_lens_array], axis=0)
                    arg = {
                        "add_token_vecs_array": add_token_vecs_array,
                        "add_doc_lens_array": add_doc_lens_array,
                        "all_token_vecs_array": all_token_vecs_array,
                        "doc_lens": doc_lens,
                        "codec": codec,
                        "compressed_embs": compressed_embs,
                        "pre_ivf_dict": pre_ivf_dict,
                        "step" : total_step
                    }
                    self.set_faiss_index()
                    this_ivf_dict = self.indexing_corpus_by_hdrc(arg)
                    total_step += 1
                    print(f"num of ivf dict :{len(this_ivf_dict)}, num of partition: {self.num_partitions}")
                    pre_ivf_dict = this_ivf_dict
                    add_tokens_num = np.sum(add_doc_lens_array)
                    add_tokens_num_list.append(int(add_tokens_num))

                metadata = {
                    "num_embeddings": self.num_embeddings,
                    "num_partitions": self.num_partitions,
                    "dim": add_token_vecs_array.shape[1],
                    "nbits": self.config[r"run_config"]["nbits"],
                }
                self.saver.save_codec(codec, index_folder_path)
                self.saver.save_chunk(index_folder_path, compressed_embs, doc_lens)
                self.saver.save_ivf(index_folder_path, pre_ivf_dict)
                self.saver.save_metadata(index_folder_path, metadata)
                self.hdrc_cluster.save(index_folder_path)

        #======================== 4.当前这个 task 来测评检索效果 ======================
                analysis_save_path = self.run_retrieve_this_step(this_queries, this_queries_id, all_corpus_id, task=task_step)
                self.evaluate(analysis_save_path, labels_df, task=task_step)

        results_folder = self.set_up_results_folder()
        add_tokens_num_save_path = os.path.join(results_folder, "add_tokens_num_list.json")
        use_reindex_list_path = os.path.join(results_folder, "use_reindex_list.json")
        with open(add_tokens_num_save_path, 'w', encoding='utf-8') as f:
            json.dump(add_tokens_num_list, f)

        with open(use_reindex_list_path, 'w', encoding='utf-8') as f:
            json.dump(self.use_reindex, f)

    def set_faiss_index(self):
        faiss_centroids_list = []
        max_layer_id = len(self.hdrc_cluster.layers) - 1
        for layer_id, layer in enumerate(self.hdrc_cluster.layers):
            if max_layer_id != layer_id:
                centroids = layer.centroids
                dim = centroids.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(centroids)
                faiss_centroids_list.append(index)
            else:
                centroids = layer.centroids
                dim = centroids.shape[1]
                non_zero_mask = np.any(centroids != 0, axis=1)
                valid_centroids = centroids[non_zero_mask]
                valid_ids = np.arange(len(centroids))[non_zero_mask].astype(np.int64)

                # 构建带映射的索引
                base_index = faiss.IndexFlatIP(dim)
                index = faiss.IndexIDMap(base_index)
                index.add_with_ids(valid_centroids, valid_ids)
                faiss_centroids_list.append(index)
        self.hdrc_cluster.faiss_centroids_list = faiss_centroids_list


    def indexing_add_corpus(self, add_token_vecs_array, add_doc_lens_array, streaming_step=1):
        add_token_vecs_array = torch.tensor(add_token_vecs_array, dtype=torch.float32)
        index_time_folder = self.set_up_index_time_folder()
        index_folder_path = self.set_up_index_save_path(streaming_step)
        last_index_folder_path = self.set_up_index_save_path(streaming_step=streaming_step-1)
        #读入初始化的关键数据
        if os.path.exists(os.path.join(index_folder_path, "metadata.json")):
            print(f"Index already exists at {index_folder_path}. Loading existing index.")
            return

        with open(os.path.join(last_index_folder_path, "metadata.json"), 'r', encoding='utf-8') as file:
            metadata = json.load(file)
        self.num_partitions = metadata["num_partitions"]
        self.num_embeddings = metadata["num_embeddings"]
        codec = ResidualCodec.load(self.config, last_index_folder_path)
        compressed_embs = ResidualCodec.Embeddings.load_chunks(
            last_index_folder_path,
            range(1),
            self.num_embeddings
        )

        with open(os.path.join(last_index_folder_path, "doclens.json"), 'r', encoding='utf-8') as file:
            doc_lens = json.load(file)

        with timer(streaming_step, f"indexing", index_time_folder):
            add_doc_lens = add_doc_lens_array.tolist()
            doc_lens += add_doc_lens
            add_compressed_embs = codec.compress(add_token_vecs_array)
            compressed_embs.add_compress_embs(add_compressed_embs)
            ivf, ivf_lengths = self._build_ivf(doc_lens, compressed_embs)
            metadata = {
                "num_embeddings": self.num_embeddings + add_token_vecs_array.size(0),
                "num_partitions": self.num_partitions,
                "dim": add_token_vecs_array.size(1),
                "nbits": self.config[r"run_config"]["nbits"],
            }
        self.saver.save_codec(codec, index_folder_path)
        self.saver.save_chunk(index_folder_path, compressed_embs, doc_lens)
        self.saver.save_ivf(index_folder_path, ivf, ivf_lengths)
        self.saver.save_metadata(index_folder_path, metadata)

    def indexing_corpus_by_hdrc(self, arg):
        add_token_vecs_array = arg["add_token_vecs_array"]
        add_doc_lens_array = arg["add_doc_lens_array"]
        all_token_vecs_array = arg["all_token_vecs_array"]
        doc_lens = arg["doc_lens"]
        codec = arg["codec"]
        compressed_embs = arg["compressed_embs"]
        step = arg["step"]
        pre_ivf_dict = arg["pre_ivf_dict"]

        # doc_lens_arr = np.asarray(doc_lens)
        #
        # # 生成文档 ID 并按对应长度重复
        # doc_ids = np.arange(len(doc_lens_arr))
        # token_to_doc_id_arr = np.repeat(doc_ids, doc_lens_arr)
        # token_to_doc_id_tensor = torch.tensor(token_to_doc_id_arr, dtype=torch.int32, device="cpu")
        old_centroids = codec.centroids
        index_time_folder = self.set_up_index_time_folder()
        self.hdrc_cluster.data = all_token_vecs_array
        start_index_time = time.time()
        with timer(step, f"indexing", index_time_folder):
            start_pid = len(doc_lens)
            add_doc_lens = add_doc_lens_array.tolist() # add_doc_lens
            doc_lens += add_doc_lens

            doc_lens_arr = np.asarray(doc_lens)

            # 生成文档 ID 并按对应长度重复
            doc_ids = np.arange(len(doc_lens_arr))
            token_to_doc_id_arr = np.repeat(doc_ids, doc_lens_arr)
            temporary_cells, to_update_clusters = self.hdrc_cluster.add(add_token_vecs_array)
            # 这里要先全部更新一遍
            add_compressed_embs = codec.compress(add_token_vecs_array, old_centroids, temporary_cells)
            add_residuals = add_compressed_embs.residuals
            num_new = add_residuals.shape[0]
            start_idx = self.num_embeddings  # 当前已有的向量总数
            self.global_residuals_buffer[start_idx: start_idx + num_new] = add_residuals
            new_residuals = self.global_residuals_buffer[: start_idx + num_new]
            del add_residuals, add_compressed_embs
            if len(to_update_clusters) == 0:
                self.use_reindex.append(0)
                temp_cells_tensor = torch.as_tensor(
                    temporary_cells,
                    dtype=self.global_codes_buffer.dtype,
                    device=self.global_codes_buffer.device
                )
                self.global_codes_buffer[start_idx: start_idx + num_new] = temp_cells_tensor
                new_codes = self.global_codes_buffer[: start_idx + num_new]
                new_centroids = self.hdrc_cluster.get_all_centroids()
                compressed_embs.residuals = new_residuals
                compressed_embs.codes = new_codes
                num_partition = new_centroids.shape[0]
                ivf_dict = self._build_ivf_dict(add_doc_lens, temporary_cells, num_partition, start_pid =  start_pid)
                this_ivf_dict = self._merge_ivf_dict(pre_ivf_dict, ivf_dict)
            else:
                # 重新聚类
                self.use_reindex.append(1)
                update_vecs_ids = self.hdrc_cluster.get_vector_ids_from_requests(to_update_clusters)
                temp_cells_tensor = torch.as_tensor(
                    temporary_cells,
                    dtype=self.global_codes_buffer.dtype,
                    device=self.global_codes_buffer.device
                )
                self.global_codes_buffer[start_idx: start_idx + num_new] = temp_cells_tensor
                update_vecs_2_cent_dict, global_cancel_cells_ids = self.hdrc_cluster.update_clusters_with_dict_and_cell_ids(to_update_clusters)
                update_vecs_codes = [int(update_vecs_2_cent_dict[int(vec_id)]) for vec_id in update_vecs_ids]
                update_vecs_ids_tensor = torch.tensor(update_vecs_ids, dtype=torch.long,
                                                      device=self.global_codes_buffer.device)
                update_vecs_cells_tensor = torch.tensor(update_vecs_codes,
                                                        dtype=self.global_codes_buffer.dtype,
                                                        device=self.global_codes_buffer.device)


                self.global_codes_buffer[update_vecs_ids_tensor] = update_vecs_cells_tensor
                new_codes_2 = self.global_codes_buffer[: start_idx + num_new]
                # 更新所有聚类中心
                new_centroids = self.hdrc_cluster.get_all_centroids()
                new_centroids_tensor = torch.tensor(new_centroids, dtype=torch.float32)
                codec.centroids = new_centroids_tensor
                # 更新残差
                # 1.先拿到所有需要更新的向量
                update_vecs = all_token_vecs_array[update_vecs_ids]

                # 2. 计算被调整过的向量的ivf_dict
                update_vec_pid = token_to_doc_id_arr[update_vecs_ids]
                update_vec_cell = new_codes_2.cpu().numpy()[update_vecs_ids]
                update_ivf_dict = self.cal_update_ivf_dict(update_vec_pid, update_vec_cell)

                # 3. 把新加进来的向量也加到ivf_dict里
                num_partition = new_centroids.shape[0]
                ivf_dict = self._build_ivf_dict(add_doc_lens, temporary_cells, num_partition, start_pid=start_pid)
                new_ivf_dict = self._merge_ivf_dict(pre_ivf_dict, ivf_dict)

               # 4. 要把丢弃的中心给去掉，再把 update_ivf_dict 加入进去

                for cancel_cell_id in global_cancel_cells_ids:
                    new_ivf_dict[cancel_cell_id]  = []

                this_ivf_dict = self._merge_ivf_dict(new_ivf_dict, update_ivf_dict)

                new_cluster_compressed_embs = codec.compress(update_vecs, new_centroids_tensor, update_vecs_cells_tensor)
                new_residuals_tensor = new_cluster_compressed_embs.residuals
                new_residuals[update_vecs_ids] = new_residuals_tensor
                compressed_embs.residuals = new_residuals
                compressed_embs.codes = new_codes_2

                del update_vecs, new_centroids_tensor, update_vecs_cells_tensor, new_residuals, new_residuals_tensor

            self.num_partitions = new_centroids.shape[0]
            self.num_embeddings = self.num_embeddings + add_token_vecs_array.shape[0]

        end_index_time = time.time()
        print(f"Indexing time for mrhp in step {step}: {end_index_time - start_index_time} seconds")

        return this_ivf_dict


    def cal_update_ivf_dict(self, update_vec_pid, update_vec_cell):
        update_ivf_dict = defaultdict(set)

        # 2. 将两个数组打包，同步遍历
        for pid, cell_id in zip(update_vec_pid, update_vec_cell):
            # 将 numpy 的数据类型转成 python 原生的 int，避免后续 JSON 序列化等报错
            update_ivf_dict[int(cell_id)].add(int(pid))

        return update_ivf_dict


    # def indexing_corpus_by_hdrc(self, arg):
    #     add_token_vecs_array = arg["add_token_vecs_array"]
    #     add_doc_lens_array = arg["add_doc_lens_array"]
    #     all_token_vecs_array = arg["all_token_vecs_array"]
    #     doc_lens = arg["doc_lens"]
    #     codec = arg["codec"]
    #     compressed_embs = arg["compressed_embs"]
    #     step = arg["step"]
    #     pre_ivf_dict = arg["pre_ivf_dict"]
    #
    #     old_centroids = codec.centroids
    #     old_residuals = compressed_embs.residuals
    #
    #     index_time_folder = self.set_up_index_time_folder()
    #
    #     # 拿到上一个阶段的 codec 和 compressed_embs
    #
    #     self.hdrc_cluster.data = all_token_vecs_array
    #     with timer(step, f"indexing", index_time_folder):
    #         start_time =time.time()
    #         add_doc_lens = add_doc_lens_array.tolist()
    #         doc_lens += add_doc_lens
    #
    #         this_start_time = time.time()
    #         temporary_cells, to_update_clusters = self.hdrc_cluster.add(add_token_vecs_array)
    #         this_end_time = time.time()
    #         print(f"add time: {this_end_time - this_start_time} seconds")
    #         # 这里要先全部更新一遍
    #         add_compressed_embs = codec.compress(add_token_vecs_array, old_centroids, temporary_cells)
    #         add_residuals = add_compressed_embs.residuals
    #         new_residuals =  torch.cat((old_residuals, add_residuals), dim=0)
    #         del old_residuals, add_residuals, add_compressed_embs
    #         if len(to_update_clusters) == 0:
    #             new_cells = self.hdrc_cluster.get_point_to_cluster_map()
    #             new_centroids = self.hdrc_cluster.get_all_centroids()
    #             compressed_embs.residuals = new_residuals
    #             compressed_embs.codes = torch.tensor(new_cells)
    #         else:
    #         # 重新聚类
    #             update_vecs_ids = self.hdrc_cluster.get_vector_ids_from_requests(to_update_clusters)
    #             self.hdrc_cluster.update_clusters(to_update_clusters)
    #
    #             # 更新所有聚类中心
    #             new_centroids = self.hdrc_cluster.get_all_centroids()
    #             new_centroids_tensor = torch.tensor(new_centroids, dtype=torch.float32)
    #             codec.centroids = new_centroids_tensor
    #             # 更新残差
    #             # 1.先拿到所有需要更新的向量
    #             update_vecs =  all_token_vecs_array[update_vecs_ids]
    #             # 2. 重新计算残差
    #             new_cells = self.hdrc_cluster.get_point_to_cluster_map()
    #             vecs_cents_ids = new_cells[update_vecs_ids]
    #             new_cluster_compressed_embs = codec.compress(update_vecs, new_centroids_tensor, vecs_cents_ids)
    #             new_residuals_tensor = new_cluster_compressed_embs.residuals
    #             new_residuals[update_vecs_ids] = new_residuals_tensor
    #             compressed_embs.residuals = new_residuals
    #             compressed_embs.codes = torch.tensor(new_cells)
    #             del update_vecs, new_centroids_tensor, vecs_cents_ids, new_residuals, new_residuals_tensor
    #
    #
    #         self.num_partitions = new_centroids.shape[0]
    #         self.num_embeddings = self.num_embeddings + add_token_vecs_array.shape[0]
    #         end_time = time.time()
    #         print(f"Indexing time for mrhp: {end_time - start_time} seconds")
    #         start_time = time.time()
    #         if len(pre_ivf_dict) == 0:
    #             arg["pre_ivf_dict"] = self._build_ivf_dict(doc_lens, compressed_embs)
    #
    #         else:
    #             ivf_dict = self._build_ivf_dict(doc_lens, compressed_embs)
    #             arg["pre_ivf_dict"] = self._merge_ivf_dict(pre_ivf_dict, ivf_dict)
    #         end_time = time.time()
    #         print(f"IVF building time for mrhp: {end_time - start_time} seconds")
    #
    #
    #
    #     # self.saver.save_codec(codec, index_folder_path)
    #     # self.saver.save_chunk(index_folder_path, compressed_embs, doc_lens)
    #     # self.saver.save_ivf(index_folder_path, ivf, ivf_lengths)
    #     # self.saver.save_metadata(index_folder_path, metadata)
    #     # self.hdrc_cluster.save(index_folder_path)

    def separate_old_and_new_vectors(self, all_global_points_ids):
        ids = np.asarray(all_global_points_ids)  # 保证是数组
        mask_new = ids >= self.num_embeddings  # True 代表新
        mask_old = ~mask_new  # True 代表旧
        new_global_ids = ids[mask_new]
        old_global_ids = ids[mask_old]
        return old_global_ids, new_global_ids, mask_old, mask_new


    def indexing_corpus_all_from_zero(self, token_vecs_array, doc_lens_array, task=0, streaming_step = 0):
            index_time_folder = self.set_up_index_time_folder()
            index_folder_path = self.set_up_index_save_path(task)

            if os.path.exists(os.path.join(index_folder_path, "metadata.json")):
                print(f"Index already exists at {index_folder_path}. Loading existing index.")
                return

            with timer(streaming_step, f"indexing", index_time_folder):
                self.set_up_index()  # 计算聚类中心的数量
                cells_tensor, points_centroids_tensor , all_centroids_tensor = self._train_hdrc(token_vecs_array, index_folder_path)
                token_vecs_array = torch.tensor(token_vecs_array, dtype=torch.float32)
                doc_lens = doc_lens_array.tolist()
                heldout_fraction = 0.05
                heldout_size = int(min(heldout_fraction * token_vecs_array.size(0), 50_000))
                sample, heldout = token_vecs_array.split([token_vecs_array.size(0) - heldout_size, heldout_size], dim=0)
                bucket_cutoffs, bucket_weights, avg_residual = self._compute_avg_residual(all_centroids_tensor, heldout)
                codec = ResidualCodec(config=self.config, centroids=all_centroids_tensor, avg_residual=avg_residual,
                                      bucket_cutoffs=bucket_cutoffs, bucket_weights=bucket_weights, use_gpu=False)
                compressed_embs = codec.compress(token_vecs_array, all_centroids_tensor, cells_tensor)
                self.num_partitions = all_centroids_tensor.size(0)
                # ivf, ivf_lengths = self._build_ivf(doc_lens, compressed_embs)

                ivf_dict = self._build_ivf_dict(doc_lens, compressed_embs.codes, self.num_partitions, start_pid=0)
                metadata = {
                    "num_embeddings": token_vecs_array.size(0),
                    "num_partitions": self.num_partitions,
                    "dim": token_vecs_array.size(1),
                    "nbits": self.config[r"run_config"]["nbits"],
                }

            self.saver.save_codec(codec, index_folder_path)
            self.saver.save_chunk(index_folder_path, compressed_embs, doc_lens)
            self.saver.save_ivf(index_folder_path, ivf_dict)
            self.saver.save_metadata(index_folder_path, metadata)
            self.hdrc_cluster.save(index_folder_path)

    def set_up_index_time_folder(self):
        model_name = self.config["model_config"].get('model_name', 'xtr')
        root = self.config["data_config"].get('result_root', './result')
        stream_rate = self.config["data_config"].get('stream_rate', 100)
        init_domain = self.config["data_config"].get('init_domain', 'lifestyle')
        pressure_type = self.config["run_config"].get('pressure_type', 'all')
        index_time_folder = r"{}/{}/{}/stream_{}_{}_up/index_time".format(root,
                                                                       model_name,
                                                                       pressure_type,
                                                                       init_domain,
                                                                       str(stream_rate))
        if not os.path.exists(index_time_folder):
            os.makedirs(index_time_folder, exist_ok=True)
        return index_time_folder

    def set_up_index_save_path(self, task):
        model_name = self.config["model_config"].get('model_name', 'xtr')
        root = self.config["data_config"].get('index_root', './result')
        stream_rate = self.config["data_config"].get('stream_rate', 100)
        init_domain = self.config["data_config"].get('init_domain', 'lifestyle')
        pressure_type = self.config["run_config"].get('pressure_type', 'all')
        save_folder = r"{}/{}/{}/stream_{}_{}_up/task_{}".format(root,
                                                              model_name,
                                                              pressure_type,
                                                              init_domain,
                                                              str(stream_rate),
                                                              str(task))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        return save_folder

    def set_up_query_vecs_path(self):
        model_name = self.config["model_config"].get('model_name', 'xtr')
        root = self.config["data_config"].get('index_root', './index')
        benchmark_type = self.config["data_config"].get('benchmark_type', 'beir')
        dataset = self.config["data_config"].get('dataset', 'nq')
        if benchmark_type == "lotte":
            dataset = r"{}_{}".format(dataset, self.config["data_config"].get('queries_style', 'search'))
        centroids_file = r"{}/{}/{}/{}/{}".format(root, model_name, benchmark_type, dataset, f'queries_vec.npy')
        return centroids_file

    def set_up_embedding_save_file(self, streaming_step):
        model_name = self.config["model_config"].get('model_name', 'xtr')
        root = self.config["data_config"].get('embedding_root', './result')
        stream_rate = self.config["data_config"].get('stream_rate', 100)
        init_domain = self.config["data_config"].get('init_domain', 'lifestyle')
        pressure_type = self.config["run_config"].get('pressure_type', 'all')
        save_folder = r"{}/{}/{}/stream_{}_{}".format(root, model_name, pressure_type, stream_rate, init_domain)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        save_file = os.path.join(save_folder, f"embedding_vectors.h5")
        return save_file

    def set_up_effectiveness(self):
        model_name = self.config["model_config"].get('model_name', 'xtr')
        root = self.config["data_config"].get('result_root', './result')
        stream_rate = self.config["data_config"].get('stream_rate', 100)
        init_domain = self.config["data_config"].get('init_domain', 'lifestyle')
        pressure_type = self.config["run_config"].get('pressure_type', 'all')
        effectiveness_file = r"{}/{}/{}/stream_{}_{}_up/effectiveness_metrics.csv".format(root, model_name,
                                                                                       pressure_type,
                                                                                       init_domain,
                                                                                       str(stream_rate))
        return effectiveness_file

    def set_up_results_folder(self):
        model_name = self.config["model_config"].get('model_name', 'xtr')
        root = self.config["data_config"].get('result_root', './result')
        stream_rate = self.config["data_config"].get('stream_rate', 100)
        init_domain = self.config["data_config"].get('init_domain', 'lifestyle')
        pressure_type = self.config["run_config"].get('pressure_type', 'all')
        results_folder = r"{}/{}/{}/stream_{}_{}_up".format(root, model_name, pressure_type, init_domain, str(stream_rate))

        return results_folder

    def set_up_retrieve_result_path(self):
        model_name = self.config["model_config"].get('model_name', 'xtr')
        root = self.config["data_config"].get('result_root', './result')
        stream_rate = self.config["data_config"].get('stream_rate', 100)
        init_domain = self.config["data_config"].get('init_domain', 'lifestyle')
        pressure_type = self.config["run_config"].get('pressure_type', 'all')
        analysis_save_folder = r"{}/{}/{}/stream_{}_{}_up/analysis".format(root,
                                                                        model_name,
                                                                        pressure_type,
                                                                        init_domain,
                                                                        str(stream_rate))
        run_time_save_folder = r"{}/{}/{}/stream_{}_{}_up/query_time".format(root,
                                                                          model_name,
                                                                          pressure_type,
                                                                          init_domain,
                                                                          str(stream_rate))

        if not os.path.exists(analysis_save_folder):
            os.makedirs(analysis_save_folder, exist_ok=True)
        if not os.path.exists(run_time_save_folder):
            os.makedirs(run_time_save_folder, exist_ok=True)

        return analysis_save_folder, run_time_save_folder

    def _build_ivf(self, doc_lens, compressed_embs):

        codes = compressed_embs.codes
        codes = codes.sort()
        ivf, values = codes.indices, codes.values
        ivf_lengths = torch.bincount(values, minlength=self.num_partitions)
        assert ivf_lengths.size(0) == self.num_partitions

        # Transforms centroid->embedding ivf to centroid->passage ivf
        # ivf_dict = build_ivf_dict(ivf, ivf_lengths, doc_lens)
        # ivf_d, ivf_lengths_d = compile_dict_to_ivf(ivf_dict)


        ivf, ivf_lengths= optimize_ivf(ivf, ivf_lengths, doc_lens)
        # print(ivf_d==ivf)

        return ivf, ivf_lengths

    def _build_ivf_dict(self, doc_lens, codes, num_partitions, start_pid=0):
        if not isinstance(codes, torch.Tensor):
            codes = torch.as_tensor(codes)
        codes = codes.sort()
        ivf, values = codes.indices, codes.values
        ivf_lengths = torch.bincount(values, minlength=num_partitions)
        assert ivf_lengths.size(0) == num_partitions

        ivf_dict = build_ivf_dict(ivf, ivf_lengths, doc_lens, start_pid=start_pid)

        return ivf_dict

    # def _merge_ivf_dict(self, pre_ivf_dict, new_ivf_dict):
    #     for centroid_id, passage_ids in new_ivf_dict.items():
    #         if not passage_ids:  # 更 Pythonic 的判空方式，等同于 len == 0
    #             continue
    #
    #         if isinstance(passage_ids, set):
    #             passage_ids = list(passage_ids)
    #         if centroid_id in pre_ivf_dict:
    #             # 直接将两个列表相加生成新列表，然后转 set 去重，再转回 list
    #             merged_list = pre_ivf_dict[centroid_id] + passage_ids
    #             pre_ivf_dict[centroid_id] = list(set(merged_list))
    #         else:
    #             pre_ivf_dict[centroid_id] = passage_ids
    #
    #     return pre_ivf_dict

    def _merge_ivf_dict(self, pre_ivf_dict, new_ivf_dict):
        for centroid_id, passage_ids in new_ivf_dict.items():
            if len(passage_ids) == 0:
                continue
            if isinstance(passage_ids, set):
                passage_ids = list(passage_ids)

            if centroid_id in pre_ivf_dict:
                pre_ivf_dict[centroid_id].extend(passage_ids)
            else:
                pre_ivf_dict[centroid_id] = passage_ids

        return pre_ivf_dict

    def _train_hdrc(self, token_vecs_array, index_folder_path):
        cluster_file = r"{}/index.pkl".format(index_folder_path)
        if os.path.exists(cluster_file):
            print(f"Loading existing HDRC clusters from {index_folder_path}.")
            # =============== Load existing HDRC ================
            self.hdrc_cluster.load_cluster(index_folder_path)
            return self._get_cluster_results()
        # ============ Training HDRC ============
        self.hdrc_cluster.fit(token_vecs_array)
        return self._get_cluster_results()

    def _get_cluster_results(self):
        # ============ Get clustering results ============
        cells = self.hdrc_cluster.get_point_to_cluster_map()
        points_centroids_vecs = self.hdrc_cluster.get_point_to_centroid_vectors()
        all_centroids_vecs = self.hdrc_cluster.get_all_centroids_by_global_id()
        all_centroids_tensor = torch.tensor(all_centroids_vecs, dtype=torch.float32)
        cells_tensor = torch.tensor(cells, dtype=torch.int64)
        points_centroids_tensor = torch.tensor(points_centroids_vecs, dtype=torch.float32)
        return cells_tensor, points_centroids_tensor , all_centroids_tensor

    def _compute_avg_residual(self, centroids, heldout):
        compressor = ResidualCodec(config=self.config, centroids=centroids, avg_residual=None)
        heldout_reconstruct = compressor.compress_into_codes(heldout, self.hdrc_cluster)
        heldout_reconstruct = compressor.lookup_centroids(heldout_reconstruct, self.hdrc_cluster)
        heldout_avg_residual = heldout - heldout_reconstruct

        avg_residual = torch.abs(heldout_avg_residual).mean(dim=0).cpu()
        print([round(x, 3) for x in avg_residual.squeeze().tolist()])

        num_options = 2 ** self.config[r"run_config"]["nbits"]
        quantiles = torch.arange(0, num_options, device=heldout_avg_residual.device) * (1 / num_options)
        bucket_cutoffs_quantiles, bucket_weights_quantiles = quantiles[1:], quantiles + (0.5 / num_options)
        bucket_cutoffs = heldout_avg_residual.float().quantile(bucket_cutoffs_quantiles)
        bucket_weights = heldout_avg_residual.float().quantile(bucket_weights_quantiles)
        print(
            f"#> Got bucket_cutoffs_quantiles = {bucket_cutoffs_quantiles} and bucket_weights_quantiles = {bucket_weights_quantiles}")
        print(f"#> Got bucket_cutoffs = {bucket_cutoffs} and bucket_weights = {bucket_weights}")

        return bucket_cutoffs, bucket_weights, avg_residual.mean()

    def set_up_index(self):
        # 初始化 HDRC 聚类器
        self.hdrc_cluster = HDRSearch(
            beta=self.config["run_config"]["beta"],
            l_max=self.config["run_config"]["l_max"],
            update_threshold_radius=self.config["run_config"]["update_threshold_radius"],
            update_radius_start_layer=self.config["run_config"]["update_radius_start_layer"],
            G_th=self.config["run_config"]["G_th"]
        )

def compute_faiss_kmeans(dim, num_partitions, kmeans_niters, shared_lists,):
    kmeans = faiss.Kmeans(dim, num_partitions, niter=kmeans_niters, gpu=True, verbose=True, seed=123)
    sample = shared_lists[0][0]
    kmeans.train(sample)
    centroids = torch.from_numpy(kmeans.centroids)
    return centroids

if __name__ == '__main__':
    config = {
        "run_config": {
            "device": "cuda:0",
            "max_query_len": 32,
            "max_doc_len": 300,
            "batch_size": 32,
            "top_k_token": 256,
            "top_k_doc": 30,
            "pool_factor": 1,
            "kmeans_niters":5,
            "nbits": 4,
            "clustering_mode":"hierarchical",
            "protected_tokens": 0,
            "rank":0,
            "ncells":1,
            "centroid_score_threshold":0.5,
            "update_threshold_radius":0.85,
            "update_radius_start_layer":4,
            "ndocs":1000,
            "beta":0.83,
            "l_max":3,
            "radius_step":0.2,
            "is_from_zero":True,
            "stream_type": "only_corpus_add",
            "max_capacity": 50000000,
            "G_th" : 0.006
        },
        "model_config": {
            "model_name": "plaid_hdrc_g",
            "model_path": r"../checkpoints/colbertv2.0",  # 改成你的 checkpoint 路径
            "doc_token_id": "[unused1]",
            "query_token_id": "[unused0]",
            "doc_token": "[D]",
            "query_token": "[Q]",
            "compression_dim": 128,
            "attend_to_mask_tokens": False
        },
        "data_config": {
            "data_root": r"/media/xianpe/17fee0f7-8905-40ec-881c-2477771df149/chunming/project_streaming_mvdr/baselines/data",
            "embedding_root": r"/media/xianpe/17fee0f7-8905-40ec-881c-2477771df149/chunming/project_streaming_mvdr/mrhp/embedding",
            "index_root": r"/media/xianpe/17fee0f7-8905-40ec-881c-2477771df149/chunming/project_streaming_mvdr/mrhp/index",
            "result_root": r"/media/xianpe/17fee0f7-8905-40ec-881c-2477771df149/chunming/project_streaming_mvdr/mrhp/result",
            "benchmark_type": "pressure_data_lotte",
            "init_domain": "science",
            "stream_rate": 10
        }
    }
    retriever = MRHPPressureRetrieverUp(config)
    retriever.init_experiment()
    retriever.streaming_experiment()
