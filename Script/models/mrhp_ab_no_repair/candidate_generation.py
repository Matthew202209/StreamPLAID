from models.mrhp_ab_no_repair.strided_tensor import StridedTensor
import torch

class CandidateGeneration:

    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu

    def get_cells_and_scores_by_cluster(self, Q, hdr_cluster):
        # 这个默认ncells=1
        cells,all_scores = hdr_cluster.cents_query(Q)
        # all_scores = hdr_cluster.cal_scores_with_all_cents(Q)
        cells = torch.tensor(cells, dtype=torch.float32)
        all_scores = torch.tensor(all_scores, dtype=torch.float32)
        cells = cells.unique(sorted=False)
        return cells, all_scores

    def get_cells(self, Q, ncells):

        scores = (self.codec.centroids @ Q.T)
        if ncells == 1:
            cells = scores.argmax(dim=0, keepdim=True).permute(1, 0)
        else:
            cells = scores.topk(ncells, dim=0, sorted=False).indices.permute(1, 0)  # (32, ncells)
        cells = cells.flatten().contiguous()  # (32 * ncells,)
        cells = cells.unique(sorted=False)
        return cells, scores

    def generate_candidate_eids(self, Q, ncells):
        cells, scores = self.get_cells(Q, ncells)

        eids, cell_lengths = self.ivf.lookup(cells)  # eids = (packedlen,)  lengths = (32 * ncells,)
        eids = eids.long()
        return eids, scores

    def generate_candidate_pids(self, Q, hdr_cluster):
        cells, scores = self.get_cells_and_scores_by_cluster(Q, hdr_cluster)
        pids, cell_lengths = self.ivf.lookup(cells)
        return pids, scores

    def generate_candidate_scores(self, Q, eids):
        E = self.lookup_eids(eids)
        return (Q.unsqueeze(0) @ E.unsqueeze(2)).squeeze(-1).T

    def generate_candidates(self, config, Q, hdr_cluster):
        ncells = config["run_config"][r"ncells"]

        assert isinstance(self.ivf, StridedTensor)

        Q = Q.squeeze(0)
        assert Q.dim() == 2
        # Q = Q.cuda().half()
        pids, centroid_scores = self.generate_candidate_pids(Q, hdr_cluster)

        sorter = pids.sort()
        pids = sorter.values

        pids, pids_counts = torch.unique_consecutive(pids, return_counts=True)

        return pids, centroid_scores
