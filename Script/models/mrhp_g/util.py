import os
import re
import os
import ujson

import torch
from scipy.cluster.hierarchy import linkage, fcluster
from tqdm import tqdm

def flatten(L):
    # return [x for y in L for x in y]

    result = []
    for _list in L:
        result += _list

    return result


class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass


class MixedPrecisionManager():
    def __init__(self, activated):
        self.activated = activated

        if self.activated:
            self.scaler = torch.cuda.amp.GradScaler()

    def context(self):
        return torch.cuda.amp.autocast() if self.activated else NullContextManager()

    def backward(self, loss):
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, colbert, optimizer, scheduler=None):
        if self.activated:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0, error_if_nonfinite=False)

            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()


def pool_embeddings_hierarchical(
    p_embeddings,
    token_lengths,
    pool_factor,
    protected_tokens: int = 0,
    showprogress: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_embeddings = p_embeddings.to(device)
    pooled_embeddings = []
    pooled_token_lengths = []
    start_idx = 0

    T = tqdm(token_lengths, desc="Pooling tokens") if showprogress else token_lengths
    for token_length in T:
        # Get the embeddings for the current passage
        passage_embeddings = p_embeddings[start_idx : start_idx + token_length]

        # Remove the tokens at protected_tokens indices
        protected_embeddings = passage_embeddings[:protected_tokens]
        passage_embeddings = passage_embeddings[protected_tokens:]

        # Cosine similarity computation (vector are already normalized)
        similarities = torch.mm(passage_embeddings, passage_embeddings.t())

        # Convert similarities to a distance for better ward compatibility
        similarities = 1 - similarities.cpu().numpy()

        # Create hierarchical clusters using ward's method
        Z = linkage(similarities, metric="euclidean", method="ward")
        # Determine the number of clusters we want in the end based on the pool factor
        max_clusters = (
            token_length // pool_factor if token_length // pool_factor > 0 else 1
        )
        cluster_labels = fcluster(Z, t=max_clusters, criterion="maxclust")

        # Pool embeddings within each cluster
        for cluster_id in range(1, max_clusters + 1):
            cluster_indices = torch.where(
                torch.tensor(cluster_labels == cluster_id, device=device)
            )[0]
            if cluster_indices.numel() > 0:
                pooled_embedding = passage_embeddings[cluster_indices].mean(dim=0)
                pooled_embeddings.append(pooled_embedding)

        # Re-add the protected tokens to pooled_embeddings
        pooled_embeddings.extend(protected_embeddings)

        # Store the length of the pooled tokens (number of total tokens - number of tokens from previous passages)
        pooled_token_lengths.append(len(pooled_embeddings) - sum(pooled_token_lengths))
        start_idx += token_length

    pooled_embeddings = torch.stack(pooled_embeddings)
    return pooled_embeddings, pooled_token_lengths


def _stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)

    output = torch.zeros(
        bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype
    )

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, : x.size(1)] = x
        offset = endpos

    return output

def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize]))

    return batches



def _insert_prefix_token(tensor: torch.Tensor, prefix_id: int):
    prefix_tensor = torch.full(
    (tensor.size(0), 1),
    prefix_id,
    dtype=tensor.dtype,
    device=tensor.device,
    )
    return torch.cat([tensor[:, :1], prefix_tensor, tensor[:, 1:]], dim=1)


def build_ivf_dict(orig_ivf, orig_ivf_lengths, all_doclens, start_pid: int = 0, verbose: int = 3):
    """
    步骤 1：将底层的 Embedding ID 映射为全局 Document ID，去重后生成 Python 字典。

    Args:
        orig_ivf: 排序后的 local embedding 索引 (1D Tensor)
        orig_ivf_lengths: 每个中心包含的 embedding 数量 (1D Tensor)
        all_doclens: 当前批次每篇文档的长度列表
        start_pid: 增量更新时的起始文档编号 (全局 PID)。例如已存 1000 篇，传 1000。
        verbose: 打印日志等级

    Returns:
        ivf_dict: 格式为 {centroid_id: [global_pid_1, global_pid_2, ...]}
    """
    if verbose > 1:
        print(f"#> Building the emb2pid mapping and dictionary (starting from PID {start_pid})...")

    total_num_embeddings = sum(all_doclens)

    # 注意：这里 emb2pid 的大小依然是“当前这批新增的词向量总数”
    # 它的作用是把当前批次的 Local ID 映射为 Global PID
    emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)
    # 1. 构建 Embedding ID 到 全局 PID (文档编号) 的映射表
    offset_doclens = 0
    for i, dlength in enumerate(all_doclens):
        global_pid = start_pid + i  # <--- 核心修改：加上全局起始偏移量
        emb2pid[offset_doclens: offset_doclens + dlength] = global_pid
        offset_doclens += dlength
    # 2. 将 orig_ivf 里的“本地词向量编号”全部替换为“全局文档编号”
    mapped_pids = emb2pid[orig_ivf]
    ivf_dict = {}
    offset = 0
    # 3. 按聚类中心切分，去重，存入字典
    for cid, length in enumerate(tqdm(orig_ivf_lengths.tolist(), disable=verbose < 2)):
        if length > 0:
            # 提取该中心的 全局 PIDs 并去重
            pids = torch.unique(mapped_pids[offset:offset + length])
            # 存为纯 Python list
            ivf_dict[cid] = pids.tolist()
        else:
            # 处理空中心
            ivf_dict[cid] = []
        offset += length
    return ivf_dict


# def build_ivf_dict(orig_ivf, orig_ivf_lengths, all_doclens, verbose: int = 3):
#     """
#     步骤 1：将底层的 Embedding ID 映射为 Document ID，去重后生成 Python 字典。
#
#     Returns:
#         ivf_dict: 格式为 {centroid_id: [pid_1, pid_2, ...]}
#     """
#     if verbose > 1:
#         print("#> Building the emb2pid mapping and dictionary...")
#
#     total_num_embeddings = sum(all_doclens)
#     emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)
#
#     # 1. 构建 Embedding ID 到 PID (文档编号) 的映射表
#     offset_doclens = 0
#     for pid, dlength in enumerate(all_doclens):
#         emb2pid[offset_doclens: offset_doclens + dlength] = pid
#         offset_doclens += dlength
#
#     # 2. 将 orig_ivf 里的词向量编号全部替换为文档编号
#     mapped_pids = emb2pid[orig_ivf]
#
#     ivf_dict = {}
#     offset = 0
#
#     # 3. 按聚类中心切分，去重，存入字典
#     for cid, length in enumerate(tqdm(orig_ivf_lengths.tolist(), disable=verbose < 2)):
#         if length > 0:
#             # 提取该中心的 PIDs 并去重
#             pids = torch.unique(mapped_pids[offset:offset + length])
#             # 存为纯 Python list，方便后续自由 append 和 clear
#             ivf_dict[cid] = pids.tolist()
#         else:
#             # 处理空中心
#             ivf_dict[cid] = []
#         offset += length
#
#     return ivf_dict


def compile_dict_to_ivf(ivf_dict, device=torch.device('cpu')):
    """
    步骤 2：将字典按顺序展平，并添加底层 CUDA 所需的内存对齐 Padding。

    Args:
        ivf_dict: {centroid_id: [pid_1, pid_2, ...]}
        device: 最终 Tensor 存放的设备 ('cpu' 或 'cuda')

    Returns:
        ivf: 展平且带 Padding 的 1D Tensor
        ivf_lengths: 记录每个中心长度的 1D Tensor
    """
    unique_pids_per_centroid = []
    ivf_lengths_list = []

    # 必须保证从 0 到 max_centroid_id 严格按顺序遍历
    max_cid = max(ivf_dict.keys()) if ivf_dict else -1

    for cid in range(max_cid + 1):
        pids_list = ivf_dict.get(cid, [])
        if len(pids_list) > 0:
            # 保底操作：再次去重（因为你在增量更新字典时，可能不小心加入了重复的 PID）
            pids_tensor = torch.unique(torch.tensor(pids_list, dtype=torch.int32, device=device))
            unique_pids_per_centroid.append(pids_tensor)
            ivf_lengths_list.append(pids_tensor.shape[0])
        else:
            ivf_lengths_list.append(0)

    # 将所有短数组拼接成一根长面条
    if unique_pids_per_centroid:
        ivf = torch.cat(unique_pids_per_centroid)
    else:
        ivf = torch.empty(0, dtype=torch.int32, device=device)

    ivf_lengths = torch.tensor(ivf_lengths_list, dtype=torch.int32, device=device)

    # ==========================================
    # 核心：复原原版代码的 Padding 逻辑 (防越界)
    # ==========================================
    if ivf_lengths.numel() > 0:
        max_stride = ivf_lengths.max().item()

        if max_stride > 0:
            zero = torch.zeros(1, dtype=torch.int32, device=device)
            offsets = torch.cat((zero, torch.cumsum(ivf_lengths, dim=0)))
            inner_dims = ivf.size()[1:]

            # 检查最后一个 block 是否有越界风险
            if offsets[-2] + max_stride > ivf.size(0):
                padding = torch.zeros(max_stride, *inner_dims, dtype=ivf.dtype, device=device)
                ivf = torch.cat((ivf, padding))

    return ivf, ivf_lengths



def optimize_ivf(orig_ivf, orig_ivf_lengths, all_doclens, verbose: int = 3):
    if verbose > 1:
        print("#> Optimizing IVF to store map from centroids to list of pids..")

        print("#> Building the emb2pid mapping..")

    # assert self.num_embeddings == sum(flatten(all_doclens))

    total_num_embeddings = sum(all_doclens)

    emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)

    """
    EVENTUALLY: Use two tensors. emb2pid_offsets will have every 256th element.
    emb2pid_delta will have the delta from the corresponding offset,
    """

    offset_doclens = 0
    for pid, dlength in enumerate(all_doclens):
        emb2pid[offset_doclens: offset_doclens + dlength] = pid
        offset_doclens += dlength

    if verbose > 1:
        print("len(emb2pid) =", len(emb2pid))

    ivf = emb2pid[orig_ivf]
    unique_pids_per_centroid = []
    ivf_lengths = []

    offset = 0
    for length in tqdm(orig_ivf_lengths.tolist()):
        pids = torch.unique(ivf[offset:offset + length])
        unique_pids_per_centroid.append(pids)
        ivf_lengths.append(pids.shape[0])
        offset += length
    ivf = torch.cat(unique_pids_per_centroid)
    ivf_lengths = torch.tensor(ivf_lengths)

    max_stride = ivf_lengths.max().item()
    zero = torch.zeros(1, dtype=torch.long, device=ivf_lengths.device)
    offsets = torch.cat((zero, torch.cumsum(ivf_lengths, dim=0)))
    inner_dims = ivf.size()[1:]

    if offsets[-2] + max_stride > ivf.size(0):
        padding = torch.zeros(max_stride, *inner_dims, dtype=ivf.dtype, device=ivf.device)
        ivf = torch.cat((ivf, padding))
    return ivf, ivf_lengths


def load_doclens(directory, flatten=True):
    doclens_filenames = {}

    for filename in os.listdir(directory):
        match = re.match(r"doclens.(\d+).json", filename)

        if match is not None:
            doclens_filenames[int(match.group(1))] = filename

    doclens_filenames = [os.path.join(directory, doclens_filenames[i]) for i in sorted(doclens_filenames.keys())]

    all_doclens = [ujson.load(open(filename)) for filename in doclens_filenames]

    if flatten:
        all_doclens = [x for sub_doclens in all_doclens for x in sub_doclens]

    if len(all_doclens) == 0:
        raise ValueError("Could not load doclens")

    return all_doclens



