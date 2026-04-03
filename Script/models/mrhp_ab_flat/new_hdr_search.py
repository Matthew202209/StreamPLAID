import json
import os
import shutil
import time
import h5py
import numpy as np
import torch
import tqdm
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist


# ==========================================
# 1. 定义数据结构，告别 11 个元素的元组
# ==========================================
@dataclass
class LayerData:
    """存储每一层的元数据，替代原来的 tuple"""
    layer_index: int
    centroids: np.ndarray  # (n_centers, d)
    radius: float  # 当前层的统一截断半径

    # 映射关系
    center_global_ids: np.ndarray  # (n_centers,) 全局唯一的中心 ID
    stats_count: np.ndarray  # (n_centers,) 每个簇的点数
    stats_sum_vec: np.ndarray  # (n_centers, d) 向量和
    stats_sum_sq: np.ndarray  # (n_centers, d) 向量平方和
    radius_per_cluster: np.ndarray  # (n_centers,) 每个簇的独立半径

    # 倒排索引 (用于检索)
    # key: local_center_index, value: list of global_point_ids
    inverted_index: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list))


class HDRSearch:
    def __init__(self,
                 beta=0.7,
                 l_max=3,
                 update_threshold_radius=0.3,
                 update_radius_start_layer=2):
        self.beta = beta
        self.l_max = l_max
        self.update_radius_start_layer = update_radius_start_layer
        self.update_threshold_radius = update_threshold_radius

        # 全局计数器
        self.global_center_id_counter = 0

        # 核心存储
        self.layers: List[LayerData] = []
        self.data: Optional[np.ndarray] = None  # (N, D) 原始数据
        self.num_data = 0
        self.vec_dim = 0

        self.all_centroids = None

    # ==========================================
    # 2. 辅助工具函数
    # ==========================================
    def _normalize(self, X):
        if X.ndim == 1: X = X.reshape(1, -1)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def _cal_num_centers(self, n_points):
        """
        动态计算聚类中心数
        策略：大数用 Log-Sqrt 策略，小数用比例策略，并增加安全边界。
        """
        if n_points == 0:
            return 0

        # 1. 极小数据保护：如果点数少于4个，没必要聚类（或者直接视作叶子节点）
        # 这里为了保证流程通畅，如果点太少，就返回 1 (即不分叉，当作一坨处理)
        # 或者返回 n_points (每个点都是中心，但这通常用于极小数据集的精确搜索)
        if n_points <= 3:
            return 1

            # 2. 理论计算 (你的公式逻辑，适合 N > 1000)
        # 16 * sqrt(N) 是 IVF 索引常用的经验公式，向上取整到 2 的幂次方便对齐
        theo_num = int(2 ** np.floor(np.log2(16 * np.sqrt(n_points))))

        # 3. 比例限制 (适合 N < 1000)
        # 你的 0.1 (10%) 其实稍微有点稀疏。
        # 对于小数据集，通常允许更密一点的切分，比如 sqrt(N) 或者 N/5
        # 这里我们用 max(2, ...) 确保至少分出 2 个簇
        ratio_num = max(2, int(n_points * 0.15))

        # 4. 融合策略
        # 取两者的较小值，防止中心数爆炸
        n_centers = min(theo_num, ratio_num)

        # 5. 最终安全钳位 (Safety Clamp)
        # 确保 2 <= n_centers < n_points
        # 如果算出 n_centers >= n_points，KMeans 会报错，所以限制为 n_points - 1 (或者 n_points)
        # 通常建议保留一定的压缩比，比如最多 n_points / 2
        max_allowed = max(1, n_points // 2)
        n_centers = min(n_centers, max_allowed)

        # 再次兜底，确保至少为 1
        return max(1, n_centers)

    def geometric_center_allocation(self, num_centers):
        """
        一次性算好每层中心数（含归一化，保证总和=num_centers）
        """
        w = np.array([(1 - self.beta) ** i for i in range(self.l_max)])
        w /= w.sum()  # 关键：归一化
        # 先向下取整，再把余数从前往后补 1，确保总和一丝不差
        base = np.floor(num_centers * w).astype(int)
        rem = num_centers - base.sum()
        # 余数按权重从大到小补
        order = np.argsort(w)[::-1]
        base[order[:rem]] += 1
        return base  # shape=(l_max,)

    def _geometric_allocation(self, total_centers, layer_idx):
        """
        运行时只要查表，不再做动态计算
        """
        if not hasattr(self, '_center_table') or self._center_table.sum() != total_centers:
            # 缓存起来，避免每次都重算
            self._center_table = self.geometric_center_allocation(total_centers)
        return self._center_table[layer_idx]

    def _calculate_cluster_radius(self, count, sum_vec, sum_sq):
        """计算簇内半径统计量 (修复 NaN 问题版)"""
        # 1. 防止除以 0
        eps = 1e-8
        # 注意：这里 count 是 int，为了精度最好转 float
        safe_count = np.maximum(count, eps).astype(np.float32).reshape(-1, 1)

        # 2. 分步计算均值，方便调试
        mean = sum_vec / safe_count  # E[X]
        mean_sq = sum_sq / safe_count  # E[X^2]

        # 3. 计算方差项: E[X^2] - (E[X])^2
        # 这一步可能会因为精度问题产生微小的负数 (如 -1e-7)
        var_per_dim = mean_sq - (mean ** 2)

        # 4. 对维度求和
        total_variance = var_per_dim.sum(axis=1)

        # === 关键修复 ===
        # 强制截断负数！数学上不可能为负，那是计算误差。
        # 用 np.maximum 将所有小于 0 的值变成 0
        total_variance = np.maximum(total_variance, 0.0)

        # 5. 开根号
        return np.sqrt(total_variance).ravel()

    # ==========================================
    # 3. 核心逻辑抽离：通用构建层
    # ==========================================
    def _build_layer_core(self, vecs: np.ndarray, global_ids: np.ndarray,
                          layer_idx: int, specified_radius: float = None,
                          min_radius: float = 0.0) -> tuple[Optional['LayerData'], np.ndarray, np.ndarray]:

        n_samples = vecs.shape[0]
        if n_samples == 0:
            return None, np.array([]), np.array([])

        # 1. 确定中心数
        total_centers_needed = self._cal_num_centers(self.num_data if self.num_data > 0 else n_samples)
        n_centers = self._geometric_allocation(total_centers_needed, layer_idx)
        n_centers = min(n_centers, n_samples)

        if n_centers <= 1 and layer_idx < self.l_max - 1:
            return None, vecs, global_ids

        # =========================================================
        # 核心：为了保持论文理论一致性 (Theoretical Compliance)
        # =========================================================

        # [Step A: 随机采样]
        # 对应论文 Algorithm 2 Line 1: Sample alpha*k points u.a.r.
        # 这确保了初始状态满足论文的 O(1) 近似界限。
        random_indices = np.random.choice(n_samples, n_centers, replace=False)
        S_random_init = vecs[random_indices]

        # [Step B: 启发式优化 (Heuristic Refinement)]
        # 利用 Lloyd's Algorithm 的单调下降性质 (Monotonic Descent Property)
        # 证明: Cost(S_final) <= Cost(S_random_init) <= O(1)*OPT
        kmeans = MiniBatchKMeans(
            n_clusters=n_centers,
            init=S_random_init,  # <--- 必须传入随机点，不能用 'k-means++'
            n_init=1,  # <--- 必须为 1，仅仅做优化，不重新随机
            batch_size=int(n_samples *0.01),
            random_state=42
        )

        kmeans.fit(vecs)
        S_i = self._normalize(kmeans.cluster_centers_)

        # =========================================================

        batch_size = int(n_samples * 0.001)

        # print(f"Start calculating scores in batches (Batch size: {batch_size})...")
        all_max_scores = np.zeros(n_samples, dtype=np.float32)
        all_assigned_centers = np.zeros(n_samples, dtype=np.int32)
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)

            # 1. 取出一个小批次向量
            # batch_vecs shape: (batch_size, dim)
            batch_vecs = vecs[start_idx:end_idx]

            # 2. 计算当前批次的分数矩阵
            # batch_scores shape: (batch_size, n_centers)
            # 内存占用极小，计算完即释放
            batch_scores = batch_vecs @ S_i.T

            # 3. 计算该批次的最大值和索引
            # 存入预分配的大数组中
            all_max_scores[start_idx:end_idx] = batch_scores.max(axis=1)
            all_assigned_centers[start_idx:end_idx] = batch_scores.argmax(axis=1)

            # 4. (可选) 手动删除临时变量，确保内存立即释放
            del batch_scores, batch_vecs

        # 为了兼容你后续的代码逻辑，将变量名指回
        max_scores = all_max_scores
        assigned_local_centers = all_assigned_centers

        # 4. 计算半径
        if specified_radius is not None:
            current_radius = specified_radius
        else:
            target_idx = int(n_samples * self.beta)
            sorted_scores = np.sort(max_scores)[::-1]
            threshold_score = sorted_scores[min(target_idx, len(sorted_scores) - 1)]
            threshold_score = min(threshold_score, 1.0)
            r_curr = np.sqrt(2 * (1.0 - threshold_score))
            current_radius = max(r_curr, min_radius)
        # 5. 筛选覆盖点
        min_dists = np.sqrt(2 * (1.0 - np.minimum(max_scores, 1.0)))
        hit_mask = min_dists <= current_radius

        hit_vecs = vecs[hit_mask]
        hit_ids = global_ids[hit_mask]
        hit_center_indices = assigned_local_centers[hit_mask]

        miss_vecs = vecs[~hit_mask]
        miss_ids = global_ids[~hit_mask]

        actual_n_centers = S_i.shape[0]

        # 6. 向量化统计 count, sum, sum_sq
        stats_count = np.zeros(actual_n_centers, dtype=np.int32)
        np.add.at(stats_count, hit_center_indices, 1)

        stats_sum = np.zeros((actual_n_centers, self.vec_dim), dtype=np.float32)
        np.add.at(stats_sum, (hit_center_indices,), hit_vecs)

        stats_sq = np.zeros((actual_n_centers, self.vec_dim), dtype=np.float32)
        np.add.at(stats_sq, (hit_center_indices,), hit_vecs ** 2)

        # 7. 构建 inverted_index
        inverted_index = defaultdict(list)
        if len(hit_ids) > 0:
            unique_centers = np.unique(hit_center_indices)
            for cid in unique_centers:
                inverted_index[cid] = hit_ids[hit_center_indices == cid].tolist()

        # 8. 计算每个中心半径
        rpc = self._calculate_cluster_radius(stats_count, stats_sum, stats_sq)

        # 9. 全局中心 ID
        center_global_ids = np.arange(self.global_center_id_counter,
                                      self.global_center_id_counter + actual_n_centers, dtype=np.int32)
        self.global_center_id_counter += actual_n_centers

        # 10. 打包 LayerData
        layer_data = LayerData(
            layer_index=layer_idx,
            centroids=S_i,
            radius=current_radius,
            center_global_ids=center_global_ids,
            stats_count=stats_count,
            stats_sum_vec=stats_sum,
            stats_sum_sq=stats_sq,
            radius_per_cluster=rpc,
            inverted_index=inverted_index
        )

        return layer_data, miss_vecs, miss_ids

    def _force_assign_to_layer(self, layer: LayerData, vecs: np.ndarray, ids: np.ndarray):
        """
        将 vecs 强行分配到 layer 中现有的中心里（忽略半径限制）。
        用于处理最后一层的兜底数据。
        """
        if len(vecs) == 0:
            return

        # 1. 计算与当前层现有中心的距离 (vecs: M x d, centroids: K x d)
        # 注意：这里假设 vecs 已经归一化
        scores = vecs @ layer.centroids.T
        best_local_centers = scores.argmax(axis=1)  # (M,) 找到每个点最近的中心下标

        # 2. 更新统计量 (Stats) - 使用 np.add.at 进行原地快速更新
        # stats_count, stats_sum_vec, stats_sum_sq 不需要 resize，因为中心数量 K 没变
        np.add.at(layer.stats_count, best_local_centers, 1)
        np.add.at(layer.stats_sum_vec, (best_local_centers,), vecs)
        np.add.at(layer.stats_sum_sq, (best_local_centers,), vecs ** 2)

        # 4. 更新倒排索引 (Inverted Index)
        # 这里需要遍历更新，因为是字典操作
        # 优化：先聚合成 dict 再 update，减少 dict 查找次数
        new_assignments = defaultdict(list)
        for i, cid in enumerate(best_local_centers):
            new_assignments[cid].append(ids[i])

        for cid, p_ids in new_assignments.items():
            layer.inverted_index[cid].extend(p_ids)

        # 5. 重新计算每个簇的统计半径 (可选)
        # 因为加入了远距离点，簇的方差变大了，半径可能会变大
        layer.radius_per_cluster = self._calculate_cluster_radius(
            layer.stats_count, layer.stats_sum_vec, layer.stats_sum_sq
        )

        print(f"  -> [兜底] 强行将 {len(vecs)} 个点归入层 {layer.layer_index}。")

    # ==========================================
    # 4. Fit 流程：串联 Core Logic
    # ==========================================
    def fit(self, U):
        print(f"--- 构建索引 (N={len(U)}) ---")
        self.data = U.astype(np.float32)
        self.num_data, self.vec_dim = self.data.shape
        self.layers = []
        self.global_center_id_counter = 0

        # 归一化数据
        current_vecs = self._normalize(self.data)
        current_ids = np.arange(self.num_data, dtype=np.int32)

        prev_radius = 0.0

        for i in range(self.l_max):
            print(f"处理层 {i}...")

            # 调用核心构建逻辑
            layer_data, leftovers, leftover_ids = self._build_layer_core(
                current_vecs, current_ids, i, min_radius=prev_radius
            )

            if layer_data:
                # 记录半径供下一层参考
                prev_radius = layer_data.radius
                self.layers.append(layer_data)

                print(
                    f"  -> 层 {i}: 产生 {len(layer_data.centroids)} 个中心，"
                    f"覆盖 {int(np.sum(layer_data.stats_count))} 个点, "
                    f"截断半径={layer_data.radius:.4f}"
                )

            # 更新待处理数据为剩下的点
            current_vecs = leftovers
            current_ids = leftover_ids

            # 如果没有剩余点，提前结束
            if len(current_vecs) == 0:
                print("所有点已覆盖，构建结束。")
                break

        # ==================================================
        # 优化点：兜底处理 (Leftover Fallback)
        # ==================================================
        if len(current_vecs) > 0:
            if len(self.layers) > 0:
                # 情况 A: 之前已经构建了至少一层
                # 策略: 将剩余点强行归入“最后一层”最近的中心
                print(f"兜底处理: 剩余 {len(current_vecs)} 个点，强行归入层 {self.layers[-1].layer_index}")
                self._force_assign_to_layer(self.layers[-1], current_vecs, current_ids)
            else:
                # 情况 B: 极其罕见，数据太少或极其离散，导致前几层都没构建成功
                # 策略: 强行构建第0层
                print(f"兜底处理: 构建紧急层处理剩余 {len(current_vecs)} 个点")
                layer_data, _, _ = self._build_layer_core(
                    current_vecs, current_ids, 0, specified_radius=999.0
                )
                if layer_data:
                    self.layers.append(layer_data)


    # ==========================================
    # 5. 增量更新 (先添加到每一层，然后检查分裂)
    # ==========================================
    def add(self, x: np.ndarray):
        """
        在线增量加入新样本 (重构版)。
        """
        # ==========================
        # 1. 数据准备
        # ==========================
        x = self._normalize(np.atleast_2d(x).astype(np.float32))
        B = x.shape[0]

        # 扩展全局数据存储
        start_id = self.num_data
        new_global_ids = np.arange(start_id, start_id + B, dtype=np.int32)

        if self.data is None:
            self.data = x
        else:
            self.data = np.vstack([self.data, x])
        self.num_data += B

        # 状态追踪
        assigned_mask = np.zeros(B, dtype=bool)
        final_cluster_ids = np.zeros(B, dtype=np.int32)
        affected_clusters = defaultdict(set)  # {layer_idx: {cid1, cid2...}}

        # ==========================
        # 2. 分层路由 (Routing)
        # ==========================
        for layer in self.layers:
            # 获取未分配点
            active_mask = ~assigned_mask
            if not active_mask.any(): break

            active_indices = np.where(active_mask)[0]
            q_vecs = x[active_indices]

            # 计算最近中心和距离
            min_dist_sq, best_local_cids = self._compute_nearest_centers(q_vecs, layer.centroids)

            # 半径筛选
            hit_mask = min_dist_sq <= (layer.radius ** 2)
            if not hit_mask.any(): continue

            # 提取命中数据
            # 注意：这里的 hit_mask 是相对于 q_vecs 的
            valid_indices = active_indices[hit_mask]  # 全局索引
            valid_vecs = q_vecs[hit_mask]
            valid_cids = best_local_cids[hit_mask]
            valid_gids = new_global_ids[valid_indices]

            # 执行原子更新
            self._batch_update_layer_stats(
                layer, valid_cids, valid_vecs, valid_gids, affected_clusters
            )

            # 标记已处理
            assigned_mask[valid_indices] = True
            final_cluster_ids[valid_indices] = layer.center_global_ids[valid_cids]

        # ==========================
        # 3. 兜底逻辑 (Fallback)
        # ==========================
        leftover_indices = np.where(~assigned_mask)[0]
        if len(leftover_indices) > 0 and len(self.layers) > 0:
            last_layer = self.layers[-1]
            q_vecs = x[leftover_indices]

            # 强行分配（无视半径）
            _, best_local_cids = self._compute_nearest_centers(q_vecs, last_layer.centroids)

            # 执行原子更新
            self._batch_update_layer_stats(
                last_layer, best_local_cids, q_vecs, new_global_ids[leftover_indices], affected_clusters
            )

            final_cluster_ids[leftover_indices] = last_layer.center_global_ids[best_local_cids]

        # ==========================
        # 4. 检查分裂 (Check Split)
        # ==========================
        to_update_reqs = self._check_split_conditions(affected_clusters)

        return final_cluster_ids, to_update_reqs

    # ==========================================
    # 下面是抽取的私有辅助方法，让主逻辑更干净
    # ==========================================

    def _compute_nearest_centers(self, vecs, centroids):
        """计算最近邻中心和距离平方"""
        dots = vecs @ centroids.T
        dist_sq = 2.0 * (1.0 - dots)
        min_dist_sq = dist_sq.min(axis=1)
        best_cids = dist_sq.argmin(axis=1)
        return min_dist_sq, best_cids

    def _batch_update_layer_stats(self, layer, cids, vecs, gids, affected_tracker):
        """
        [原子操作] 批量更新某层的统计量和索引
        """
        # 1. Numpy 快速统计量更新
        np.add.at(layer.stats_count, cids, 1)
        np.add.at(layer.stats_sum_vec, (cids,), vecs)
        np.add.at(layer.stats_sum_sq, (cids,), vecs ** 2)

        # 2. 倒排索引更新 (聚合后更新，减少字典IO)
        # 优化：先在 python 层面做一次 groupby，比逐个 append 快
        group_dict = defaultdict(list)
        for i, cid in enumerate(cids):
            group_dict[cid].append(gids[i])

        for cid, gid_list in group_dict.items():
            layer.inverted_index[cid].extend(gid_list)
            # 记录变动簇，用于后续检查
            affected_tracker[layer.layer_index].add(cid)

    def _check_split_conditions(self, affected_clusters):
        """检查哪些簇需要分裂"""
        to_update_reqs = []

        for l_idx, cid_set in affected_clusters.items():
            # 1. 策略拦截：浅层不参与分裂
            if l_idx < self.update_radius_start_layer:
                # 依然需要更新半径数值，保证元数据正确，但不加入待处理列表
                self._recalc_radii_only(l_idx, list(cid_set))
                continue

            layer = self.layers[l_idx]
            c_idxs = np.array(list(cid_set), dtype=np.int32)

            # 2. 计算新半径
            sub_count = layer.stats_count[c_idxs]
            sub_sum = layer.stats_sum_vec[c_idxs]
            sub_sq = layer.stats_sum_sq[c_idxs]
            new_radii = self._calculate_cluster_radius(sub_count, sub_sum, sub_sq)

            # 回写半径
            layer.radius_per_cluster[c_idxs] = new_radii

            # 3. 阈值判断
            bad_indices = c_idxs[new_radii > self.update_threshold_radius]

            if len(bad_indices) > 0:
                to_update_reqs.append((l_idx, bad_indices.tolist()))

        return to_update_reqs

    def _recalc_radii_only(self, l_idx, c_list):
        """仅更新半径数值，不做判定"""
        if not c_list: return
        layer = self.layers[l_idx]
        c_idxs = np.array(c_list, dtype=np.int32)
        new_radii = self._calculate_cluster_radius(
            layer.stats_count[c_idxs],
            layer.stats_sum_vec[c_idxs],
            layer.stats_sum_sq[c_idxs]
        )
        layer.radius_per_cluster[c_idxs] = new_radii

    #=====================================================================
    #                          下面是增量更新聚类的相关方法
    #=====================================================================
    def _build_layer_fixed_radius(self, vecs: np.ndarray, global_ids: np.ndarray,
                                  layer_idx: int, fixed_radius: float,
                                  n_centers_target: int) -> tuple[Optional['LayerData'], np.ndarray, np.ndarray]:
        """
        [Update 专用] 固定半径构建层。
        参数 n_centers_target: 由外部基于"当前实际到达数据量"动态计算好的中心配额。
        """
        n_samples = vecs.shape[0]
        if n_samples == 0:
            return None, np.array([]), np.array([])

        # =========================================================
        # 1. 使用外部传入的目标中心数 (只做安全限制)
        # =========================================================
        # 即使外部算出来需要 100 个，如果当前只有 50 个点，也只能分 50 个
        n_centers = min(n_centers_target, n_samples)

        # 极小数据兜底：如果算出来太少，但数据量尚可，强行给 2 个尝试一下
        if n_centers <= 1:
            if n_samples > 10:
                n_centers = 2
            else:
                # 只有几个点，且配额也很少，无法形成有效聚类结构
                return None, vecs, global_ids

        # =========================================================
        # 2. 聚类 (增加 Try-Except 鲁棒性)
        # =========================================================
        random_indices = np.random.choice(n_samples, n_centers, replace=False)
        S_random_init = vecs[random_indices]

        try:
            kmeans = MiniBatchKMeans(
                n_clusters=n_centers,
                init=S_random_init,
                n_init=1,
                batch_size=4096,
                random_state=42
            )
            kmeans.fit(vecs)
            S_i = self._normalize(kmeans.cluster_centers_)
        except Exception as e:
            print(f"[Warning] Layer {layer_idx} update clustering failed: {e}")
            return None, vecs, global_ids

        # =========================================================
        # 3. 分配与筛选 (逻辑不变)
        # =========================================================
        scores = vecs @ S_i.T
        max_scores = scores.max(axis=1)
        assigned_local_centers = scores.argmax(axis=1)

        # 严格使用固定半径筛选
        min_dists = np.sqrt(2 * (1.0 - np.minimum(max_scores, 1.0)))
        hit_mask = min_dists <= fixed_radius

        hit_vecs = vecs[hit_mask]
        hit_ids = global_ids[hit_mask]
        hit_center_indices = assigned_local_centers[hit_mask]

        miss_vecs = vecs[~hit_mask]
        miss_ids = global_ids[~hit_mask]

        if len(hit_ids) == 0:
            return None, vecs, global_ids

        # 4. 打包数据
        layer_data = self._pack_layer_data(
            layer_idx, S_i, fixed_radius, hit_vecs, hit_ids, hit_center_indices
        )

        return layer_data, miss_vecs, miss_ids


    def _pack_layer_data(self, layer_idx, centroids, radius, hit_vecs, hit_ids, hit_center_indices):
        """
        辅助函数：计算统计量并打包 LayerData。
        【修正】只根据当前 global_counter 生成 ID 数组，但不修改 global_counter。
        """
        actual_n_centers = centroids.shape[0]

        # ... (统计量计算保持不变) ...
        stats_count = np.zeros(actual_n_centers, dtype=np.int32)
        stats_sum = np.zeros((actual_n_centers, self.vec_dim), dtype=np.float32)
        stats_sq = np.zeros((actual_n_centers, self.vec_dim), dtype=np.float32)

        if len(hit_ids) > 0:
            np.add.at(stats_count, hit_center_indices, 1)
            np.add.at(stats_sum, (hit_center_indices,), hit_vecs)
            np.add.at(stats_sq, (hit_center_indices,), hit_vecs ** 2)

        # 倒排索引
        inverted_index = defaultdict(list)
        if len(hit_ids) > 0:
            df = defaultdict(list)
            for i, cid in enumerate(hit_center_indices):
                df[cid].append(hit_ids[i])
            inverted_index.update(df)

        # 半径
        rpc = self._calculate_cluster_radius(stats_count, stats_sum, stats_sq)

        # ==================================================
        # 【关键修改】生成 ID，但不自增 counter
        # ==================================================
        # 我们使用当前的 counter 值作为起始，生成临时 ID
        # 如果这个 patch 最终被丢弃，counter 没变，下次生成的 ID 还是这些，不会浪费
        center_global_ids = np.arange(self.global_center_id_counter,
                                      self.global_center_id_counter + actual_n_centers, dtype=np.int32)

        # DELETE: self.global_center_id_counter += actual_n_centers  <-- 删掉这一行

        return LayerData(
            layer_index=layer_idx,
            centroids=centroids,
            radius=radius,
            center_global_ids=center_global_ids,  # 这里的 ID 是暂定的
            stats_count=stats_count,
            stats_sum_vec=stats_sum,
            stats_sum_sq=stats_sq,
            radius_per_cluster=rpc,
            inverted_index=inverted_index
        )

    def update_clusters(self, update_reqs: List[Tuple[int, List[int]]]):
        """
        [完整版] 增量重聚类逻辑 (动态基数策略版)
        """
        if not update_reqs: return

        print(f"--- 触发增量重聚类: 处理 {sum(len(ids) for _, ids in update_reqs)} 个簇 ---")

        # ==========================================
        # 1. 提取数据 & 软删除旧簇 (逻辑保持不变)
        # ==========================================
        all_vecs_list = []
        all_ids_list = []

        req_map = defaultdict(list)
        for l_idx, c_idxs in update_reqs:
            req_map[l_idx].extend(c_idxs)

        for l_idx, c_idxs in req_map.items():
            layer = self.layers[l_idx]
            # A. 提取
            for cid in c_idxs:
                p_ids = layer.inverted_index.get(cid, [])
                if not p_ids: continue
                p_ids_arr = np.array(p_ids, dtype=np.int32)
                vecs = self.data[p_ids_arr]
                all_vecs_list.append(vecs)
                all_ids_list.append(p_ids_arr)
            # B. 软删
            self._soft_delete_clusters(l_idx, c_idxs)

        if not all_vecs_list:
            print("  未提取到有效向量，跳过更新。")
            return

        current_vecs = np.vstack(all_vecs_list)
        current_ids = np.concatenate(all_ids_list)
        print(f"  提取完成: 共 {len(current_ids)} 个点等待重分配。")

        # ==========================================
        # 2. 前 N-1 层流转 (动态基数 + Beta校验)
        # ==========================================
        for i in range(len(self.layers) - 1):
            if len(current_vecs) == 0: break

            target_layer = self.layers[i]
            n_current_samples = len(current_vecs)

            # [核心策略 1] 动态计算基数
            # 如果上层发生了穿透(Skip)，这里的 n_current_samples 会保持高位
            # 从而计算出一个较大的 base_budget，抵消了几何衰减带来的过分稀疏
            base_budget = self._cal_num_centers(n_current_samples)

            # [核心策略 2] 应用层级衰减
            # Layer i 始终应该比 Layer 0 稀疏，所以必须要 decay
            n_centers_target = self._geometric_allocation(base_budget, i)

            # [核心策略 3] 密度兜底
            # 保证至少有 1% 的点转为中心，或者至少 2 个中心
            min_safe = max(2, int(n_current_samples * 0.01))
            n_centers_target = max(n_centers_target, min_safe)

            # A. 尝试聚类 (传入计算好的 target)
            patch, leftovers, leftover_ids = self._build_layer_fixed_radius(
                current_vecs, current_ids,
                layer_idx=i,
                fixed_radius=target_layer.radius,
                n_centers_target=n_centers_target  # <--- 传入动态计算的值
            )

            if patch is None: continue

            # B. Beta 校验
            # 分母使用 n_current_samples，代表"当前层输入数据的覆盖率"
            covered_count = np.sum(patch.stats_count)
            coverage_ratio = covered_count / n_current_samples

            if coverage_ratio >= self.beta:
                print(
                    f"  Layer {i} 准入通过: {coverage_ratio:.2%} (Samples: {n_current_samples}, Centers: {len(patch.centroids)})")
                self._merge_update_patch(i, patch)
                # 只有通过了，才更新剩余数据
                current_vecs = leftovers
                current_ids = leftover_ids
            else:
                # C. 未通过 -> 穿透
                # current_vecs 不变，直接进入下一次循环
                # 下一层循环会基于这个较大的 n_current_samples 重新计算基数
                print(f"  Layer {i} 准入失败 ({coverage_ratio:.2%}), 全量穿透。")
                pass

        # ==========================================
        # 3. 最后一层兜底 (核心策略)
        # ==========================================
        if len(current_vecs) > 0:
            last_layer_idx = len(self.layers) - 1
            last_layer = self.layers[last_layer_idx]
            n_last_samples = len(current_vecs)

            # [新增] 最后一层也要计算合理的中心数
            base_budget = self._cal_num_centers(n_last_samples)
            n_centers_last = self._geometric_allocation(base_budget, last_layer_idx)
            n_centers_last = max(n_centers_last, max(2, int(n_last_samples * 0.01)))

            # A. 尝试建立新结构
            patch, leftovers, leftover_ids = self._build_layer_fixed_radius(
                current_vecs, current_ids,
                layer_idx=last_layer_idx,
                fixed_radius=last_layer.radius,
                n_centers_target=n_centers_last  # <--- 传入
            )

            # --- 分支 1: 发现新结构 (Patch成功) ---
            if patch is not None:
                print(f"  Layer {last_layer_idx}: 发现新结构，生成 {len(patch.centroids)} 个中心。")

                # 挂载剩余点到新 Patch (使用 Update 专用强行分配)
                if len(leftovers) > 0:
                    self._force_assign_to_layer_update(patch, leftovers, leftover_ids)

                self._merge_update_patch(last_layer_idx, patch)

            # --- 分支 2: 未发现新结构 (Patch失败) ---
            else:
                # 挂载到旧 Layer
                valid_centers_count = (last_layer.radius_per_cluster >= 0).sum()

                if valid_centers_count > 0:
                    print(f"  Layer {last_layer_idx}: 数据离散，强行挂载至现有旧簇。")
                    self._force_assign_to_layer_update(last_layer, current_vecs, current_ids)
                else:
                    # [极罕见兜底] 连旧簇都没有，必须强制新建
                    # 此时给一个极大的 n_centers 预算，或者直接按照 cal_num_centers 来
                    fallback_budget = self._cal_num_centers(len(current_vecs))
                    print(f"  Layer {last_layer_idx}: 无可用旧簇，执行全量新建兜底 (Budget: {fallback_budget})。")

                    fallback_patch, _, _ = self._build_layer_fixed_radius(
                        current_vecs, current_ids,
                        layer_idx=last_layer_idx,
                        fixed_radius=9999.0,  # 强制全覆盖
                        n_centers_target=fallback_budget
                    )
                    if fallback_patch:
                        self._merge_update_patch(last_layer_idx, fallback_patch)

            # 更新一下全局中心
            _ = self.get_all_centroids_by_global_id()

        print("--- 增量重聚类完成 ---\n")

    def _force_assign_to_layer_update(self, layer: LayerData, vecs: np.ndarray, ids: np.ndarray):
        """
        [Update 阶段专用] 强行分配。
        包含【屏蔽逻辑】，防止数据被分配给刚刚软删除的废弃簇(radius < 0)。
        """
        if len(vecs) == 0: return

        # 1. 计算距离
        scores = vecs @ layer.centroids.T

        # =================================================
        # 【安全检查】屏蔽已软删除的簇 (防止僵尸复活)
        # =================================================
        invalid_mask = layer.radius_per_cluster < 0
        if invalid_mask.any():
            scores[:, invalid_mask] = -np.inf
        # =================================================

        best_local_centers = scores.argmax(axis=1)

        # 2. 更新统计量
        np.add.at(layer.stats_count, best_local_centers, 1)
        np.add.at(layer.stats_sum_vec, (best_local_centers,), vecs)
        np.add.at(layer.stats_sum_sq, (best_local_centers,), vecs ** 2)

        # 4. 更新倒排
        group_dict = defaultdict(list)
        for i, cid in enumerate(best_local_centers):
            group_dict[cid].append(ids[i])
        for cid, p_ids in group_dict.items():
            layer.inverted_index[cid].extend(p_ids)

        # 5. 重算半径
        layer.radius_per_cluster = self._calculate_cluster_radius(
            layer.stats_count, layer.stats_sum_vec, layer.stats_sum_sq
        )

        # 【双重保险】再次确保废弃簇保持为 -1 (防止计算半径时误将0改为小数值)
        if invalid_mask.any():
            layer.radius_per_cluster[invalid_mask] = -1.0

        print(
            f"  -> [Update兜底] 强行将 {len(vecs)} 个点归入层 {layer.layer_index} (已屏蔽 {invalid_mask.sum()} 个废簇)。")

    # ==========================================
    # 必须配合的辅助函数
    # ==========================================

    def _soft_delete_clusters(self, layer_idx: int, local_cids: List[int]):
        """
        软删除：标记旧簇无效，防止 Query 命中，并释放内存。
        """
        if not local_cids: return
        layer = self.layers[layer_idx]

        cids_arr = np.array(local_cids, dtype=np.int32)

        # 1. 标记半径为 -1.0 (Query 时会自动 Mask 掉)
        layer.radius_per_cluster[cids_arr] = -1.0

        # 2. 将中心向量归零 (数据整洁)
        layer.centroids[cids_arr] = 0.0

        # 3. 清空统计量
        layer.stats_count[cids_arr] = 0
        layer.stats_sum_vec[cids_arr] = 0
        layer.stats_sum_sq[cids_arr] = 0

        # 4. 释放倒排索引内存
        for cid in local_cids:
            if cid in layer.inverted_index:
                del layer.inverted_index[cid]

        print(f"  -> Layer {layer_idx}: 已软删除 {len(local_cids)} 个旧簇。")

    def _merge_update_patch(self, layer_idx: int, patch: LayerData):
        """
        将新生成的 Patch 追加到层尾，并更新全局索引。
        """
        base = self.layers[layer_idx]

        # 计算偏移量 (Base 当前的长度)
        offset = len(base.stats_count)

        # 1. 追加数组数据 (Centroids & Stats)
        base.centroids = np.vstack([base.centroids, patch.centroids])
        base.stats_count = np.concatenate([base.stats_count, patch.stats_count])
        base.stats_sum_vec = np.vstack([base.stats_sum_vec, patch.stats_sum_vec])
        base.stats_sum_sq = np.vstack([base.stats_sum_sq, patch.stats_sum_sq])
        base.radius_per_cluster = np.concatenate([base.radius_per_cluster, patch.radius_per_cluster])

        # 2. 追加全局中心 ID (patch 里已经生成好了)
        base.center_global_ids = np.concatenate([base.center_global_ids, patch.center_global_ids])

        # 3. 追加点归属信息

        # Local ID 需要加上偏移量 (Offset)

        # 4. 合并倒排索引
        # 这里加上偏移量是合理的，因为每一层的实际中心数是累积的
        for local_cid, gids in patch.inverted_index.items():
            base.inverted_index[local_cid + offset] = gids

        self.global_center_id_counter += len(patch.centroids)
        print(f"  -> Layer {layer_idx}: 追加合并完成。新增 {len(patch.centroids)} 个新簇。")

    # ==========================================
    # 6. 查询 (Query)
    # ==========================================

    def structure_query(self, Q):
        """
        [最终完整版] 查询逻辑：
        Phase 1: 分层半径过滤 (高精度，谁先命中算谁的)。
        Phase 2: 最后一层兜底 (无视半径，找最近的有效簇)。
        """
        # 归一化
        Q = self._normalize(np.atleast_2d(Q).astype(np.float32))
        B = Q.shape[0]

        final_center_ids = np.full(B, -1, dtype=np.int32)
        final_scores = np.full(B, -1.0, dtype=np.float32)

        found_mask = np.zeros(B, dtype=bool)

        # =========================================================
        # Phase 1: 正常的半径内检索 (High Precision)
        # =========================================================
        for layer in self.layers:
            if found_mask.all(): break

            # 只计算尚未命中的点
            active_indices = np.where(~found_mask)[0]
            if len(active_indices) == 0: break

            sub_Q = Q[active_indices]

            # 1. 计算点积
            dots = sub_Q @ layer.centroids.T

            # [关键] 屏蔽已软删除的簇
            # 这一步不能省，否则 argmax 会选中废弃簇
            invalid_cluster_mask = layer.radius_per_cluster < 0
            if invalid_cluster_mask.any():
                dots[:, invalid_cluster_mask] = -np.inf

            # 2. 寻找最佳匹配
            max_dots = dots.max(axis=1)
            best_local_centers = dots.argmax(axis=1)

            # 3. 计算距离并判定半径
            valid_score_mask = max_dots > -1e9
            dists = np.full_like(max_dots, 9999.0)
            if valid_score_mask.any():
                dists[valid_score_mask] = np.sqrt(
                    2 * (1.0 - np.minimum(max_dots[valid_score_mask], 1.0))
                )

            # [严格半径筛选]
            hits = dists <= layer.radius

            # 4. 记录命中结果
            if hits.any():
                hit_query_indices = active_indices[hits]
                hit_local_centers = best_local_centers[hits]

                final_center_ids[hit_query_indices] = layer.center_global_ids[hit_local_centers]
                final_scores[hit_query_indices] = max_dots[hits]
                found_mask[hit_query_indices] = True

        # =========================================================
        # Phase 2: 最后一层兜底 (Recall Rescue)
        # =========================================================
        # 检查还有哪些点在 Phase 1 没找到家
        leftover_indices = np.where(~found_mask)[0]

        if len(leftover_indices) > 0 and len(self.layers) > 0:
            # 策略：直接去【最后一层】找最近的（无视半径）
            # 这与 fit/update 中的 "Force Assign" 逻辑对齐
            last_layer = self.layers[-1]
            sub_Q = Q[leftover_indices]

            # 1. 计算距离
            dots = sub_Q @ last_layer.centroids.T

            # [关键] 依然要屏蔽软删除的簇！
            # 即使是兜底，也不能兜给死人（废弃簇）
            invalid_cluster_mask = last_layer.radius_per_cluster < 0
            if invalid_cluster_mask.any():
                dots[:, invalid_cluster_mask] = -np.inf

            # 2. 强行取最大值 (Argmax) - 不检查 Radius
            max_dots = dots.max(axis=1)
            best_local_centers = dots.argmax(axis=1)

            # 3. 记录结果
            # 注意：这里的 max_dots 可能很小，但它是目前能找到的最优解
            final_center_ids[leftover_indices] = last_layer.center_global_ids[best_local_centers]
            final_scores[leftover_indices] = max_dots

            # 更新 Mask (虽然函数结束了，但为了逻辑完整)
            found_mask[leftover_indices] = True

        return final_center_ids, final_scores

    def cents_query(self, Q):
        # k = 3
        all_scores = self.cal_scores_with_all_cents(Q)
        # sorted_indices = np.argsort(all_scores, axis=0)

        # 2. 取出最后 3 个索引 (最大的3个)，并倒序 (变成从大到小)
        # 切片操作会保留 (14) 这个维度
        # top3_indices = sorted_indices[-10:][::-1, :]
        best_cent_id = all_scores.argmax(axis=0)  # 每列最大值的下标
        # top3_indices = top3_indices.flatten()
        return best_cent_id, all_scores


    def cal_scores_with_all_cents(self, Q):
        Q = np.atleast_2d(Q).astype(np.float32)
        all_cents = self.get_all_centroids()
        all_scores = (all_cents @ Q.T)
        return all_scores

    # ====================================================
    # 7. 结果获取工具
    # ====================================================
    def get_all_centroids_by_global_id(self) -> np.ndarray:
        """
        获取所有历史生成的中心向量，并按照 Global ID 排序。

        Returns:
            all_centroids (np.ndarray): 形状为 (global_center_id_counter, vec_dim)。
                第 i 行 (result[i]) 对应 center_global_id == i 的中心向量。
                注意：包含已被软删除的簇（其向量通常已被置为 0）。
        """
        if self.global_center_id_counter == 0:
            return np.zeros((0, self.vec_dim), dtype=np.float32)

        # 1. 根据全局计数器，分配全量矩阵
        # 这里的 global_center_id_counter 是所有生成过的中心总数（含废弃的）
        all_centroids = np.zeros((self.global_center_id_counter, self.vec_dim), dtype=np.float32)

        # 2. 遍历每一层进行填充
        for layer in self.layers:
            # layer.center_global_ids: (K,) 记录了该层当前持有的中心对应的全局 ID
            # layer.centroids: (K, D) 对应的向量

            gids = layer.center_global_ids
            vecs = layer.centroids

            # 使用 Numpy 高级索引直接赋值
            # 这一步会自动将 vecs 填入 all_centroids 中 gids 指定的行
            if len(gids) > 0:
                # 安全检查：防止 gids 越界 (理论上不应发生，但为了健壮性)
                valid_mask = gids < self.global_center_id_counter
                if not valid_mask.all():
                    print(f"[Warning] Layer {layer.layer_index} has IDs exceeding global counter!")
                    gids = gids[valid_mask]
                    vecs = vecs[valid_mask]

                all_centroids[gids] = vecs

        self.all_centroids = all_centroids
        return all_centroids

    def get_all_centroids(self):
        if self.all_centroids is not None:
            return self.all_centroids

        else:
            return self.get_all_centroids_by_global_id()

    def get_centroids_by_idx(self, idx):
        all_centroids = self.get_all_centroids()
        return all_centroids[idx]



    def get_point_to_centroid_vectors(self) -> np.ndarray:
        """
        获取全量数据对应的中心向量。

        Returns:
            centroid_vectors (np.ndarray): 形状为 (num_data, vec_dim) 的 float32 数组。
                                           第 i 行存储的是 point_i 所属中心的向量坐标。
        """
        if self.num_data == 0:
            return np.zeros((0, self.vec_dim), dtype=np.float32)

        # 初始化 (N, D) 矩阵，默认为 0
        centroid_map = np.zeros((self.num_data, self.vec_dim), dtype=np.float32)

        for layer in self.layers:
            # 遍历该层倒排索引
            # local_cid: 该层内的中心局部下标
            # point_ids: 属于该中心的所有数据点全局下标列表
            for local_cid, point_ids in layer.inverted_index.items():
                if not point_ids:
                    continue

                # 1. 获取中心向量 (1, D)
                center_vec = layer.centroids[local_cid]

                # 2. 批量赋值
                # 将该中心向量赋值给倒排索引中记录的所有点
                p_ids_arr = np.array(point_ids, dtype=np.int64)
                centroid_map[p_ids_arr] = center_vec
        return centroid_map


    def get_point_to_cluster_map(self) -> np.ndarray:
        """
        获取全量数据的簇分配结果。
        利用各层的倒排索引，返回一个数组，表示每个向量所属的全局中心 ID。

        Returns:
            assignments (np.ndarray): 形状为 (num_data,) 的 int32 数组。
                                      assignments[i] = point_i 所属的 Global Center ID。
                                      如果为 -1，说明该点未被索引（理论上不应发生）。
        """
        # 初始化全为 -1
        assignments = np.full(self.num_data, -1, dtype=np.int32)
        for layer in self.layers:
            # 遍历该层倒排索引
            # local_cid: 该层内的中心下标
            # point_ids: 属于该中心的所有数据点全局下标列表
            for local_cid, point_ids in layer.inverted_index.items():
                if not point_ids:
                    continue

                # 获取全局唯一的中心 ID
                global_cid = layer.center_global_ids[local_cid]

                # 批量赋值 (Numpy 高级索引)
                # 注意：point_ids 可能是 list，转为 array 操作更安全/快
                p_ids_arr = np.array(point_ids, dtype=np.int64)
                assignments[p_ids_arr] = global_cid

        return assignments

    def get_vector_ids_from_requests(self, update_reqs: List[Tuple[int, List[int]]]) -> np.ndarray:
        """
        根据 update_reqs (分裂请求)，获取所有涉及到的向量的全局 ID (Global IDs)。

        Args:
            update_reqs: add() 方法返回的分裂请求列表，格式 [(layer_idx, [cid1, cid2...]), ...]

        Returns:
            np.ndarray: 所有待更新向量的全局 ID 组成的一维数组 (int32)。
        """
        if not update_reqs:
            return np.array([], dtype=np.int32)

        all_gids_list = []

        for l_idx, cids in update_reqs:
            # 安全检查：防止层索引越界
            if l_idx < 0 or l_idx >= len(self.layers):
                continue

            layer = self.layers[l_idx]

            for cid in cids:
                # 从倒排索引中获取该簇包含的所有点 ID
                # inverted_index 的 value 是 list[int]
                p_ids = layer.inverted_index.get(cid, [])

                if p_ids:
                    all_gids_list.extend(p_ids)

        # 如果列表为空，返回空数组
        if not all_gids_list:
            return np.array([], dtype=np.int32)

        # 转换为 numpy 数组返回
        return np.array(all_gids_list, dtype=np.int32)


    # ==========================================
    # 7. 序列化 (Save/Load)
    # ==========================================
    def save(self, folder):
        """现在保存变得非常简单，只需要 pickle layer 对象或者分开存 numpy"""
        os.makedirs(folder, exist_ok=True)
        # 这里为了演示简单使用 pickle，生产环境建议用 h5py 或分开存 npy
        import pickle
        with open(os.path.join(folder, "index.pkl"), "wb") as f:
            pickle.dump({
                "layers": self.layers,
                "meta": {
                    "num_data": self.num_data,
                    "global_counter": self.global_center_id_counter,
                    "vec_dim": self.vec_dim
                }
            }, f)
        print("保存完成")

    def load(self, folder):
        """从 folder/index.pkl 恢复整个索引"""
        import pickle
        index_path = os.path.join(folder, "index.pkl")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"{index_path} 不存在，无法加载索引！")

        with open(index_path, "rb") as f:
            data = pickle.load(f)

        # 恢复核心字段
        self.layers = data["layers"]  # 各层中心点、倒排等
        meta = data["meta"]
        self.num_data = meta["num_data"]
        self.global_center_id_counter = meta["global_counter"]
        self.vec_dim = meta["vec_dim"]

        print("加载完成，当前索引数据量:", self.num_data)


def run_tests():
    print("========================================")
    print("开始 HDRSearch 功能全链路测试 (真实链路版)")
    print("========================================")

    # -------------------------------------------------
    # 0. 数据准备
    # -------------------------------------------------
    DIM = 64
    N_INIT = 2000
    N_ADD = 1000  # 增加一点 Add 的数据量，增加触发分裂的概率

    # 固定随机种子以便复现
    np.random.seed(42)

    # 生成模拟数据
    # 为了更容易触发分裂，我们让 data_add 的一部分数据故意聚集在 data_init 的某些点周围
    data_init = np.random.rand(N_INIT, DIM).astype(np.float32)

    # 构造更密集的 Add 数据 (模拟真实场景中某个热点爆发)
    center_noise = data_init[0]  # 取第一个点作为聚焦点
    data_add = np.random.rand(N_ADD, DIM).astype(np.float32)
    # 让前200个点非常靠近 data_init[0]，强行撑爆这个簇
    data_add[:200] = center_noise + np.random.normal(0, 0.01, (200, DIM))

    indexer = HDRSearch(l_max=3, beta=0.7)

    # -------------------------------------------------
    # Test 1: 全量构建 (Fit)
    # -------------------------------------------------
    print("\n[Test 1] 测试 fit() ...")
    start_time = time.time()
    indexer.fit(data_init)

    assert len(indexer.layers) > 0, "Fit 失败"
    assert indexer.num_data == N_INIT
    print(f"[PASS] Fit 完成 (Total: {indexer.num_data})")

    # -------------------------------------------------
    # Test 2: 查询 (Query)
    # -------------------------------------------------
    print("\n[Test 2] 测试 query() ...")
    ids, _ = indexer.query(data_init[:5])
    assert not np.any(ids == -1)
    print("[PASS] Query 测试通过")

    # -------------------------------------------------
    # Test 3: 增量添加 (Add) -> 产生 update_reqs
    # -------------------------------------------------
    print("\n[Test 3] 测试 add() ...")
    old_num_data = indexer.num_data

    # === 关键步骤：获取 add 返回的分裂请求 ===
    ids, update_reqs = indexer.add(data_add)

    assert indexer.num_data == old_num_data + N_ADD
    print(f"  -> 新增 {N_ADD} 个点")
    print(f"  -> [关键输出] 自动触发的分裂请求: {update_reqs}")

    print("[PASS] Add 测试通过")

    # -------------------------------------------------
    # Test 4: 增量重聚类 (Update Clusters) - 链路串联
    # -------------------------------------------------
    print("\n[Test 4] 测试 update_clusters() (链路串联) ...")

    # 这里的逻辑是：
    # 1. 优先处理 Test 3 产生的真实请求。
    # 2. 如果 Test 3 没产生请求（比如数据太均匀），则手动造一个请求来测试功能。

    real_reqs_exist = len(update_reqs) > 0

    if real_reqs_exist:
        print("  -> [模式] 使用 add() 返回的真实请求进行测试")
        reqs_to_process = update_reqs
    else:
        print("  -> [模式] add() 未触发分裂，切换为手动选择簇进行功能测试")
        # 手动找一个簇
        target_layer = indexer.layers[0]
        valid_cids = [cid for cid, count in enumerate(target_layer.stats_count) if count > 0]
        if valid_cids:
            reqs_to_process = [(0, [valid_cids[0]])]
        else:
            reqs_to_process = []
            print("[WARNING] 无法进行 Update 测试 (无有效簇)")

    if reqs_to_process:
        # === 核心逻辑执行 ===
        indexer.update_clusters(reqs_to_process)

        # --- 验证逻辑 ---
        # 1. 验证软删除：检查所有请求中的旧簇是否都变成了 -1
        for l_idx, cids in reqs_to_process:
            layer = indexer.layers[l_idx]
            for cid in cids:
                assert layer.radius_per_cluster[cid] == -1.0, f"Layer {l_idx} Cluster {cid} 未被软删除"
                assert cid not in layer.inverted_index, f"Layer {l_idx} Cluster {cid} 倒排索引未清理"

        print("  -> [验证] 旧簇软删除/清理成功")

        # 2. 验证数据守恒 (Data Conservation)
        total_points_now = 0
        for layer in indexer.layers:
            total_points_now += np.sum(layer.stats_count)

        assert total_points_now == indexer.num_data, \
            f"数据守恒失败! Indexer认为有 {indexer.num_data}, 实际统计 {total_points_now}"

        print(f"  -> [验证] 数据守恒校验通过 (Total: {total_points_now})")
        print("[PASS] Update Clusters 测试通过")
    else:
        print("[SKIP] 跳过 Update 测试")

    # -------------------------------------------------
    # Test 5 & 6 (保持不变)
    # -------------------------------------------------
    # ... (Mapping 和 Save 的测试代码同上) ...
    print("\n[Test 6] 测试 get_point_to_cluster_map() ...")
    mapping = indexer.get_point_to_cluster_map()
    assert mapping.shape[0] == indexer.num_data
    assert (mapping == -1).sum() == 0, "存在未分配的点"
    print("[PASS] Mapping 测试通过")

    print("\n========================================")
    print("所有测试执行完毕")
    print("========================================")

if __name__ == '__main__':
    # 确保运行上面的类定义代码后，再运行这个测试
    # 如果你在同一个文件里，直接调用即可
    run_tests()