import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class PressureResultsVisualizer:
    def __init__(self, root_dir):
        """
                初始化类并指定数据根目录
                :param root_dir: 数据所在的root文件夹路径
                """
        self.root_dir = root_dir
        self.results = {}
        # 用于存储最终结果的嵌套字典: dict[method][frequency] = {'performance': df, 'index_update_time': df, 'QPS': df}
        self._load_data()

    def _load_data(self):
        """
        遍历根目录，读取不同方法和不同更新频率下的实验结果
        """
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"找不到指定的根目录: {self.root_dir}")

        # 1. 遍历方法文件夹 (一层)
        for method_name in os.listdir(self.root_dir):
            method_path = os.path.join(self.root_dir, method_name)
            if not os.path.isdir(method_path):
                continue

            self.results[method_name] = {}

            # 2. 遍历参数文件夹 (二层，例如 stream_lifestyle_100)
            for param_folder in os.listdir(method_path):
                param_path = os.path.join(method_path, param_folder)
                if not os.path.isdir(param_path):
                    continue

                # 从文件夹名字提取更新频率 (假设总是用 '_' 分隔且最后一个是数字)
                try:
                    frequency = int(param_folder.split('_')[-1])
                except ValueError:
                    print(f"警告: 无法从文件夹 {param_folder} 解析出频率数字，跳过该文件夹。")
                    continue

                metrics_dict = {}

                # ==========================================
                # (1) 读取 effectiveness_metrics.csv
                # ==========================================
                eff_path = os.path.join(param_path, 'effectiveness_metrics.csv')
                if os.path.exists(eff_path):
                    df_eff = pd.read_csv(eff_path)
                    # 只提取 task 和 nDCG
                    if 'task' in df_eff.columns and 'nDCG' in df_eff.columns:
                        metrics_dict['performance'] = df_eff[['task', 'nDCG']].copy()
                    else:
                        print(f"警告: {eff_path} 缺失 task 或 nDCG 列。")

                # ==========================================
                # (2) 读取 index_time/indexing.csv
                # ==========================================
                idx_path = os.path.join(param_path, 'index_time', 'indexing.csv')
                if os.path.exists(idx_path):
                    df_idx = pd.read_csv(idx_path)
                    # 去掉第0步，只保留增量更新部分
                    df_idx = df_idx[df_idx['Index'] != 0].copy()
                    metrics_dict['index_update_time'] = df_idx

                # ==========================================
                # (3) 读取 query_time 下的 task_X_performance_query_time.csv
                # ==========================================
                query_dir = os.path.join(param_path, 'query_time')
                if os.path.exists(query_dir):
                    # 获取所有满足模式的文件
                    query_files = glob.glob(os.path.join(query_dir, 'task_*_performance_query_time.csv'))

                    # 定义排序函数，根据文件名中的 task id 排序
                    def extract_task_id(filepath):
                        filename = os.path.basename(filepath)
                        try:
                            # 根据 task_X_... 提取数字 X
                            return int(filename.split('_')[1])
                        except:
                            return -1

                    query_files.sort(key=extract_task_id)

                    qps_records = []
                    for qf in query_files:
                        task_id = extract_task_id(qf)
                        if task_id == -1:
                            continue

                        df_query = pd.read_csv(qf)
                        # 【注意】：由于你未提供query_time里的具体列名，这里假设列名包含'time'字样
                        time_cols = [col for col in df_query.columns if 'time' in col.lower()]

                        if time_cols:
                            time_col = time_cols[0]
                            # 假设我们要取平均查询时间来计算QPS，或者如果里面只有一个时间那就直接取
                            avg_time = df_query[time_col].mean()
                            # 计算QPS (取时间的倒数)
                            qps = 1.0 / avg_time if avg_time > 0 else np.nan
                            qps_records.append({'task': task_id, 'QPS': qps})

                    if qps_records:
                        metrics_dict['QPS'] = pd.DataFrame(qps_records)

                    if method_name == "StreamPLAID":
                        use_reindex_list_path = os.path.join(param_path, 'use_reindex_list.json')
                        with open(use_reindex_list_path, 'r') as f:
                            use_reindex_list = json.load(f)
                        metrics_dict["use_reindex"] = np.array(use_reindex_list).astype(bool)

                # 将三个dataframe打包存入字典
                self.results[method_name][frequency] = metrics_dict





    def get_results_dict(self):
        """返回要求的嵌套字典"""
        if not self.results:
            self._load_data()
        return self.results

    def plot_index_update_time(self, target_frequency):
        """
        绘制同一频率下，不同方法的 index 增量更新时间曲线
        :param target_frequency: 输入频率 (例如 100)
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))

        # 标记是否找到了该频率下的数据
        data_found = False

        # 遍历 self.results 中的每一个方法
        for method, freqs in self.results.items():
            # 如果该方法下包含目标频率，并且成功读取了 index_update_time 数据
            if target_frequency in freqs and 'index_update_time' in freqs[target_frequency]:
                df_idx = freqs[target_frequency]['index_update_time']

                # 为了防止列名大小写有出入，这里做一下安全获取
                x_col = 'Index' if 'Index' in df_idx.columns else df_idx.columns[0]
                y_col = 'run_time' if 'run_time' in df_idx.columns else df_idx.columns[-1]

                # 画出该方法的更新时间曲线
                plt.plot(df_idx[x_col], df_idx[y_col], marker='o', linestyle='-', label=method)
                data_found = True

        if not data_found:
            print(f"提示: 未找到频率为 {target_frequency} 的索引更新时间数据。")
            return

        plt.title(f'Incremental Index Update Time Comparison (Frequency: {target_frequency})')
        plt.xlabel('Update Step (Index)')
        plt.ylabel('Run Time (seconds)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_index_update_time_improved(self, target_frequency, use_log_scale=True, do_smoothing=True, window_size=50):
        """
        改进版的绘制索引增量更新时间曲线函数。
        解决数据点密集和尺度差异大的问题。
        :param target_frequency: 输入频率 (例如 100)
        :param use_log_scale: 是否使用对数Y轴 (默认True，解决跨度大问题)
        :param do_smoothing: 是否添加移动平均平滑曲线 (默认True，解决点密集看不清趋势问题)
        :param window_size: 移动平均的窗口大小，越大曲线越平滑 (默认50)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns  # 使用seaborn获取更好看的颜色

        # 设置绘图风格
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(14, 7))

        data_found = False
        # 获取一个颜色调色板
        palette = sns.color_palette("deep", len(self.results))
        method_colors = {}

        for i, (method, freqs) in enumerate(self.results.items()):
            if target_frequency in freqs and 'index_update_time' in freqs[target_frequency]:
                df_idx = freqs[target_frequency]['index_update_time'].copy()
                # 确保按 Index 排序，否则画出来的线是乱的
                df_idx.sort_values(by='Index', inplace=True)

                x_col = 'Index' if 'Index' in df_idx.columns else df_idx.columns[0]
                y_col = 'run_time' if 'run_time' in df_idx.columns else df_idx.columns[-1]

                # 分配固定颜色
                color = palette[i]
                method_colors[method] = color

                # --- 改进点: 只画线，不画点，且线调细，设置半透明 ---
                # alpha=0.3 让原始数据的线变为背景，不喧宾夺主
                ax.plot(df_idx[x_col], df_idx[y_col], linestyle='-', linewidth=0.8, color=color, alpha=0.3,
                        label=f'{method} (Raw Data)')

                # --- 改进点 (进阶): 添加移动平均平滑曲线 ---
                if do_smoothing and len(df_idx) > window_size:
                    # 计算移动平均
                    smoothed_time = df_idx[y_col].rolling(window=window_size, center=True, min_periods=1).mean()
                    # 绘制平滑曲线，颜色加深，线宽加粗，作为视觉主体
                    ax.plot(df_idx[x_col], smoothed_time, linestyle='-', linewidth=2.5, color=color,
                            label=f'{method} (Trend, MA-{window_size})')

                data_found = True

        if not data_found:
            print(f"提示: 未找到频率为 {target_frequency} 的索引更新时间数据。")
            plt.close(fig)
            return

        ax.set_title(f'Incremental Index Update Time Comparison (Frequency: {target_frequency})', fontsize=14)
        ax.set_xlabel('Update Step (Index)', fontsize=12)
        ax.set_ylabel('Run Time (seconds)', fontsize=12)

        # --- 改进点: 使用对数坐标轴 ---
        if use_log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('Run Time (seconds) - Log Scale', fontsize=12)
            # 手动设置对数刻度的显示的格式，避免显示成 10^0 这种科学计数法
            from matplotlib.ticker import ScalarFormatter
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(axis='y', style='plain')

        # 优化图例显示
        ax.legend(loc='best', frameon=True, shadow=True)
        # 显示更细致的网格
        ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_index_update_time_advanced(self, target_frequency, y_max=None, clip_percentile=None, use_log_scale=True,
                                        do_smoothing=True, window_size=50):
        """
        修复版：完美兼容【对数坐标】与极小值/零值的进阶绘制函数。
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        from matplotlib.ticker import FuncFormatter  # 【修复点2：引入更智能的格式化器】

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(14, 7))

        data_found = False
        palette = sns.color_palette("deep", len(self.results))

        all_y_values = []

        for i, (method, freqs) in enumerate(self.results.items()):
            if target_frequency in freqs and 'index_update_time' in freqs[target_frequency]:
                df_idx = freqs[target_frequency]['index_update_time'].copy()
                df_idx.sort_values(by='Index', inplace=True)

                x_col = 'Index' if 'Index' in df_idx.columns else df_idx.columns[0]
                y_col = 'run_time' if 'run_time' in df_idx.columns else df_idx.columns[-1]

                all_y_values.extend(df_idx[y_col].tolist())
                color = palette[i]

                # 原始数据线
                ax.plot(df_idx[x_col], df_idx[y_col], linestyle='-', linewidth=0.8, color=color, alpha=0.3,
                        label=f'{method} (Raw)')

                # 平滑趋势线
                if do_smoothing and len(df_idx) > window_size:
                    smoothed_time = df_idx[y_col].rolling(window=window_size, center=True, min_periods=1).mean()
                    ax.plot(df_idx[x_col], smoothed_time, linestyle='-', linewidth=2.5, color=color,
                            label=f'{method} (Trend, MA-{window_size})')

                data_found = True

        if not data_found:
            print(f"提示: 未找到频率为 {target_frequency} 的索引更新时间数据。")
            plt.close(fig)
            return

        title_suffix = ""

        # ----------------------------------------------------
        # 1. 设置 Y 轴上限 (天花板)
        # ----------------------------------------------------
        if y_max is not None:
            ax.set_ylim(top=y_max)
            title_suffix = f" (Y max={y_max}s)"
        elif clip_percentile is not None and len(all_y_values) > 0:
            s_all = pd.Series(all_y_values)
            auto_y_max = s_all.quantile(clip_percentile)
            ax.set_ylim(top=auto_y_max * 1.2)
            title_suffix = f" (Zoomed to {clip_percentile * 100}% data)"

        # ----------------------------------------------------
        # 2. 设置对数轴及 Y 轴下限 (地板) - 核心修复区域
        # ----------------------------------------------------
        if use_log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('Run Time (seconds) - Log Scale', fontsize=12)

            # 【修复点1：寻找大于0的最小正数值，防止对数轴因为0值无限拉伸】
            positive_y_values = [y for y in all_y_values if y > 0]
            if positive_y_values:
                min_pos_y = min(positive_y_values)
                # 把地板设在最小正数值的一半位置，留出视觉空间
                ax.set_ylim(bottom=min_pos_y * 0.5)

            # 【修复点2：使用 g 格式化，智能保留有效数字，不盲目补零】
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        else:
            ax.set_ylabel('Run Time (seconds)', fontsize=12)
            ax.set_ylim(bottom=0)  # 线性坐标下，地板放心设为 0

        # 设置标题和基础Label
        ax.set_title(f'Incremental Index Update Time Comparison (Frequency: {target_frequency}){title_suffix}',
                     fontsize=14)
        ax.set_xlabel('Update Step (Index)', fontsize=12)

        # 优化图例和网格
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_index_update_time_stream_special(self, target_frequency, y_max=None, clip_percentile=None,
                                              use_log_scale=True, do_smoothing=True, window_size=50):
        """
        StreamPLAID 专属特化版绘制函数：
        - 针对 StreamPLAID：将其耗时严格拆分为 '增量更新(连续折线+圆点)' 和 '局部重聚类(虚线+X标记)'。
        - 针对 其他对比方法：保持 '淡色原始数据 + 深色平滑趋势线' 的高级画法。
        兼容对数坐标轴与极小值修正。
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        from matplotlib.ticker import FuncFormatter

        # 设置绘图风格
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(14, 7))

        data_found = False
        palette = sns.color_palette("deep", len(self.results))
        all_y_values = []

        for i, (method, freqs) in enumerate(self.results.items()):
            if target_frequency in freqs and 'index_update_time' in freqs[target_frequency]:
                df_idx = freqs[target_frequency]['index_update_time'].copy()
                df_idx.sort_values(by='Index', inplace=True)

                x_col = 'Index' if 'Index' in df_idx.columns else df_idx.columns[0]
                y_col = 'run_time' if 'run_time' in df_idx.columns else df_idx.columns[-1]

                all_y_values.extend(df_idx[y_col].tolist())
                color = palette[i]

                # ==========================================
                # 核心逻辑 1：专门针对 StreamPLAID 的拆分画法
                # ==========================================
                if method == "StreamPLAID":
                    if 'use_reindex' in freqs[target_frequency]:
                        is_reidx_raw = freqs[target_frequency]['use_reindex']

                        # 【安全对齐逻辑】
                        # 因为在 _load_data 中 df_idx 剔除了 Index == 0 的行
                        # 如果 use_reindex 列表包含第 0 步，长度会多 1，需要截掉第一个元素
                        if len(is_reidx_raw) == len(df_idx) + 1:
                            is_reidx = is_reidx_raw[1:]
                        else:
                            # 兜底对齐，取最小公共长度
                            min_len = min(len(is_reidx_raw), len(df_idx))
                            is_reidx = is_reidx_raw[:min_len]
                            df_idx = df_idx.iloc[:min_len]

                        # 提取 numpy 数组，方便用布尔掩码过滤
                        x_vals = df_idx[x_col].values
                        y_vals = df_idx[y_col].values

                        # 拆分掩码
                        is_inc = ~is_reidx

                        x_inc, y_inc = x_vals[is_inc], y_vals[is_inc]
                        x_reidx, y_reidx = x_vals[is_reidx], y_vals[is_reidx]

                        # 绘制: 常规增量更新 (蓝色, 实线 + 圆点)
                        ax.plot(x_inc, y_inc, label='StreamPLAID (Incremental Update)',
                                color='#c44e52', linestyle='-', marker='o', markersize=3, linewidth=1.5, alpha=0.85)

                        # 绘制: 局部重聚类 (红色, 线 + X标记)
                        ax.plot(x_reidx, y_reidx, label='StreamPLAID (Local Reindex Triggered)',
                                color='#c44e52', linestyle='-', marker='x', markersize=5, linewidth=1.5, alpha=0.85)
                    else:
                        print("警告: StreamPLAID 缺失 use_reindex 掩码数据，将采用普通画法。")
                        ax.plot(df_idx[x_col], df_idx[y_col], label=f'{method} (Raw)', color=color, linewidth=1.5)

                # ==========================================
                # 核心逻辑 2：针对其他 Baseline 保持原有平滑画法
                # ==========================================
                else:
                    # 原始数据线（背景半透明）
                    ax.plot(df_idx[x_col], df_idx[y_col], linestyle='-', linewidth=0.8, color=color, alpha=0.3,
                            label=f'{method} (Raw)')

                    # 平滑趋势线（视觉主体）
                    if do_smoothing and len(df_idx) > window_size:
                        smoothed_time = df_idx[y_col].rolling(window=window_size, center=True, min_periods=1).mean()
                        ax.plot(df_idx[x_col], smoothed_time, linestyle='-', linewidth=2.5, color=color,
                                label=f'{method} (Trend, MA-{window_size})')

                data_found = True

        if not data_found:
            print(f"提示: 未找到频率为 {target_frequency} 的索引更新时间数据。")
            plt.close(fig)
            return

        # ==========================================
        # 坐标轴与范围设置 (复用 advanced 版本逻辑)
        # ==========================================
        title_suffix = ""

        # 1. 设置 Y 轴上限 (天花板)
        if y_max is not None:
            ax.set_ylim(top=y_max)
            title_suffix = f" (Y max={y_max}s)"
        elif clip_percentile is not None and len(all_y_values) > 0:
            s_all = pd.Series(all_y_values)
            auto_y_max = s_all.quantile(clip_percentile)
            ax.set_ylim(top=auto_y_max * 1.2)
            title_suffix = f" (Zoomed to {clip_percentile * 100}% data)"

        # 2. 设置对数轴及 Y 轴下限 (地板)
        if use_log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('Run Time (seconds) - Log Scale', fontsize=12)

            positive_y_values = [y for y in all_y_values if y > 0]
            if positive_y_values:
                min_pos_y = min(positive_y_values)
                ax.set_ylim(bottom=min_pos_y * 0.5)

            # 使用智能保留有效数字的格式化器
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        else:
            ax.set_ylabel('Run Time (seconds)', fontsize=12)
            ax.set_ylim(bottom=0)

        # 设置标题和基础Label
        ax.set_title(f'Incremental Index Update Time Comparison (Frequency: {target_frequency}){title_suffix}',
                     fontsize=14)
        ax.set_xlabel('Update Step (Index)', fontsize=12)

        # 优化图例 (如果图例太多遮挡，可以将 loc 改为 'upper right' 或 'upper left')
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)
        ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_performance_cdf(self, target_frequency):
        """
        绘制同一频率下，不同方法 Performance (nDCG) 的 CDF (累积分布函数) 图
        :param target_frequency: 输入频率 (例如 100)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 设置绘图风格
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))

        data_found = False
        # 获取颜色调色板
        palette = sns.color_palette("deep", len(self.results))

        for i, (method, freqs) in enumerate(self.results.items()):
            if target_frequency in freqs and 'performance' in freqs[target_frequency]:
                df_perf = freqs[target_frequency]['performance'].copy()

                # 动态获取 nDCG 列名，防止大小写不一致
                score_col = 'nDCG' if 'nDCG' in df_perf.columns else df_perf.columns[-1]

                # 过滤掉可能的 NaN 值，防止绘图报错
                df_perf = df_perf.dropna(subset=[score_col])

                if not df_perf.empty:
                    # 使用 seaborn 的 ecdfplot 直接绘制经验累积分布函数
                    sns.ecdfplot(data=df_perf, x=score_col, label=method, color=palette[i], linewidth=2.5)
                    data_found = True

        if not data_found:
            print(f"提示: 未找到频率为 {target_frequency} 的 performance 数据。")
            plt.close(fig)
            return

        # 设置标题和标签
        ax.set_title(f'Performance (nDCG) CDF Comparison (Frequency: {target_frequency})', fontsize=14)
        ax.set_xlabel('nDCG Score', fontsize=12)
        ax.set_ylabel('Cumulative Probability (Proportion of Tasks)', fontsize=12)

        # 强制设置坐标轴范围（nDCG 和概率通常都在 0~1 之间）
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        # 优化图例和网格显示
        ax.legend(loc='upper left', frameon=True, shadow=True)
        ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_performance_cdf_multi_freq(self, target_frequencies=[10, 20, 30]):
        """
        绘制多频率在一张大图上的 Performance (nDCG) CDF 对比。
        - 三个子图横向排布。
        - 去除主标题。
        - 统一提取全局图例放置在底部，排成一行，字号 16。
        - 坐标轴标题字体放大到 21，刻度字号放大到 14。
        - 每个子图右下角添加英文指示箭头 (极简极小版 + 文字下移防遮挡)。
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        from matplotlib.lines import Line2D

        # 设置绘图风格
        sns.set_theme(style="whitegrid")

        # 动态创建子图，保持扁平比例 (6.0)
        num_freqs = len(target_frequencies)
        fig, axes = plt.subplots(1, num_freqs, figsize=(8 * num_freqs, 6.0), dpi=300)

        if num_freqs == 1:
            axes = [axes]

        palette = sns.color_palette("deep", len(self.results))
        legend_dict = {}

        for ax_idx, target_frequency in enumerate(target_frequencies):
            ax = axes[ax_idx]
            data_found = False

            for i, (method, freqs) in enumerate(self.results.items()):
                if target_frequency in freqs and 'performance' in freqs[target_frequency]:
                    df_perf = freqs[target_frequency]['performance'].copy()

                    # 动态获取 nDCG 列名，防止大小写不一致
                    score_col = 'nDCG' if 'nDCG' in df_perf.columns else df_perf.columns[-1]

                    # 过滤掉可能的 NaN 值，防止绘图报错
                    df_perf = df_perf.dropna(subset=[score_col])

                    if not df_perf.empty:
                        color = palette[i]
                        # 绘制 CDF 折线 (seaborn 默认 zorder>=2)
                        sns.ecdfplot(data=df_perf, x=score_col, ax=ax, color=color, linewidth=2.5)

                        # 手动收集图例句柄
                        if method not in legend_dict:
                            legend_dict[method] = Line2D([0], [0], color=color, lw=2.5)

                        data_found = True

            if not data_found:
                ax.set_title(f'Frequency: {target_frequency} Docs/Step', fontsize=21)
                continue

            # ==========================================
            # 子图坐标轴与范围设置
            # ==========================================
            ax.set_title(f'Frequency: {target_frequency} Docs/Step', fontsize=21)

            # 强制设置坐标轴范围
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)

            # 横纵坐标注释大小放大到 21，刻度放大到 14
            ax.set_xlabel('nDCG@10', fontsize=21)
            ax.set_ylabel('Cumulative Probability', fontsize=21)
            ax.tick_params(axis='both', which='major', labelsize=14)

            ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)

            # ==========================================
            # 右下角的引导箭头与注释 (文字下移 + 箭头缩小版)
            # ==========================================
            ax.annotate('Better Performance',
                        xy=(0.96, 0.04),  # 箭头尖端保持在极右下角
                        xytext=(0.68, 0.10),  # 文本位置
                        xycoords='axes fraction',
                        textcoords='axes fraction',
                        # 【核心修改】将箭头颜色 facecolor 和 edgecolor 改为红色 (red)
                        arrowprops=dict(facecolor='red', edgecolor='red', shrink=0.05, width=0.5, headwidth=4,
                                        headlength=5, alpha=0.5),
                        # 【核心修改】将文字颜色 color 改为红色 (red)
                        fontsize=12, color='red', ha='center', va='center', alpha=0.6, fontweight='bold', zorder=1)

        # ==========================================
        # 生成全局统一图例 (单行排列)
        # ==========================================
        filtered_labels = list(legend_dict.keys())
        filtered_handles = list(legend_dict.values())

        fig.legend(filtered_handles, filtered_labels, loc='lower center',
                   bbox_to_anchor=(0.5, 0.02), ncol=len(filtered_labels),
                   frameon=True, shadow=True, fontsize=16)

        # ==========================================
        # 布局微调
        # ==========================================
        plt.tight_layout(pad=0.2, w_pad=3.0, rect=[0, 0.12, 1, 0.95])
        plt.show()

    def plot_qps_line_chart(self, target_frequency, use_log_scale=False):
        """
        绘制同一频率下，不同方法的 QPS 折线图。
        自动过滤掉 QPS 为 0 以及 Task 为 0 的数据。

        :param target_frequency: 输入频率 (例如 100)
        :param use_log_scale: 是否使用对数Y轴 (如果不同方法的QPS差距极大，可以设为True)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 设置绘图风格
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(14, 6))

        data_found = False
        # 获取颜色调色板
        palette = sns.color_palette("deep", len(self.results))

        for i, (method, freqs) in enumerate(self.results.items()):
            if target_frequency in freqs and 'QPS' in freqs[target_frequency]:
                df_qps = freqs[target_frequency]['QPS'].copy()

                # ==========================================
                # 核心逻辑：去掉 0
                # 1. 过滤掉 QPS <= 0 或 NaN 的异常数据
                # 2. 过滤掉 Task == 0 的预热/初始状态
                # ==========================================
                df_qps = df_qps[(df_qps['QPS'] > 0) & (df_qps['task'] != 0)].copy()

                if df_qps.empty:
                    continue

                # 确保按 task 排序，保证折线图连线正确，不会乱穿插
                df_qps.sort_values(by='task', inplace=True)

                # 绘制折线图：带有小圆点标记，方便看清每一个 task 的 QPS
                ax.plot(df_qps['task'], df_qps['QPS'], marker='o', markersize=4, linestyle='-',
                        linewidth=1.5, color=palette[i], label=method, alpha=0.8)

                data_found = True

        if not data_found:
            print(f"提示: 未找到频率为 {target_frequency} 的有效 QPS 数据。")
            plt.close(fig)
            return

        # 设置标题和基础Label
        ax.set_title(f'QPS (Queries Per Second) Comparison (Frequency: {target_frequency})', fontsize=14)
        ax.set_xlabel('Task ID', fontsize=12)

        # ==========================================
        # 根据是否开启对数轴，调整 Y 轴的展示
        # ==========================================
        if use_log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('QPS - Log Scale', fontsize=12)
            # 使用智能格式化器，防止显示科学计数法
            from matplotlib.ticker import FuncFormatter
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        else:
            ax.set_ylabel('QPS (Queries Per Second)', fontsize=12)
            ax.set_ylim(bottom=0)  # 线性坐标下，下限固定为 0 比较好看

        # 优化图例和网格显示
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_qps_line_chart(self, target_frequency, use_log_scale=False):
        """
        绘制同一频率下，不同方法的 QPS 折线图。
        自动过滤掉 QPS 为 0 以及 Task 为 0 的数据。

        :param target_frequency: 输入频率 (例如 100)
        :param use_log_scale: 是否使用对数Y轴 (如果不同方法的QPS差距极大，可以设为True)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 设置绘图风格
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(14, 6))

        data_found = False
        # 获取颜色调色板
        palette = sns.color_palette("deep", len(self.results))

        for i, (method, freqs) in enumerate(self.results.items()):
            if target_frequency in freqs and 'QPS' in freqs[target_frequency]:
                df_qps = freqs[target_frequency]['QPS'].copy()

                # ==========================================
                # 核心逻辑：去掉 0
                # 1. 过滤掉 QPS <= 0 或 NaN 的异常数据
                # 2. 过滤掉 Task == 0 的预热/初始状态
                # ==========================================
                df_qps = df_qps[(df_qps['QPS'] > 0) & (df_qps['task'] != 0)].copy()

                if df_qps.empty:
                    continue

                # 确保按 task 排序，保证折线图连线正确，不会乱穿插
                df_qps.sort_values(by='task', inplace=True)

                # 绘制折线图：带有小圆点标记，方便看清每一个 task 的 QPS
                ax.plot(df_qps['task'], df_qps['QPS'], marker='o', markersize=4, linestyle='-',
                        linewidth=1.5, color=palette[i], label=method, alpha=0.8)

                data_found = True

        if not data_found:
            print(f"提示: 未找到频率为 {target_frequency} 的有效 QPS 数据。")
            plt.close(fig)
            return

        # 设置标题和基础Label
        ax.set_title(f'QPS (Queries Per Second) Comparison (Frequency: {target_frequency})', fontsize=14)
        ax.set_xlabel('Task ID', fontsize=12)

        # ==========================================
        # 根据是否开启对数轴，调整 Y 轴的展示
        # ==========================================
        if use_log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('QPS - Log Scale', fontsize=12)
            # 使用智能格式化器，防止显示科学计数法
            from matplotlib.ticker import FuncFormatter
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        else:
            ax.set_ylabel('QPS (Queries Per Second)', fontsize=12)
            ax.set_ylim(bottom=0)  # 线性坐标下，下限固定为 0 比较好看

        # 优化图例和网格显示
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()

    def print_qps_average_table(self, target_frequencies=[10, 20, 30]):
        """
        计算并打印不同频率下，各个方法的平均 QPS ± 标准差表格。
        自动过滤掉 QPS <= 0 以及 Task == 0 的数据。

        :param target_frequencies: 输入频率列表 (例如 [10, 20, 30])
        """
        import pandas as pd

        # 用于存储格式化后的字符串结果
        table_data = {}

        for freq in target_frequencies:
            table_data[freq] = {}
            for method, freqs in self.results.items():
                if freq in freqs and 'QPS' in freqs[freq]:
                    df_qps = freqs[freq]['QPS'].copy()

                    # 核心逻辑：过滤异常数据和预热数据
                    df_qps = df_qps[(df_qps['QPS'] > 0) & (df_qps['task'] != 0)]

                    if not df_qps.empty:
                        avg_qps = df_qps['QPS'].mean()
                        std_qps = df_qps['QPS'].std()

                        # 处理只有一个数据点时 std 为 NaN 的情况
                        std_val = std_qps if pd.notnull(std_qps) else 0.0

                        # 格式化为 "均值 ± 标准差"
                        table_data[freq][method] = f"{avg_qps:.2f} ± {std_val:.2f}"
                    else:
                        table_data[freq][method] = "N/A"
                else:
                    table_data[freq][method] = "N/A"

        # 构建 DataFrame
        df_result = pd.DataFrame.from_dict(table_data, orient='index')

        # 按照你要求的顺序排列列名
        # 注意：请确保这些名字与 self.results 中的 key 完全一致
        expected_cols = ['FrozenPLAID', 'ReindexPLAID', 'StreamPLAID']
        available_cols = [c for c in expected_cols if c in df_result.columns]
        if available_cols:
            df_result = df_result[available_cols]

        df_result.index.name = 'Frequency'
        df_result.columns.name = 'Method'

        # 打印结果
        print("\n" + "=" * 60)
        print(" Average QPS (Mean ± Std Dev) Table")
        print("=" * 60)

        # 因为现在是字符串，round(2) 不再起作用，我们在上面格式化时已经处理了
        print(df_result)
        print("=" * 60 + "\n")

    def plot_index_update_time_multi_freq(self, target_frequencies=[10, 20, 30], y_max=None, clip_percentile=None,
                                          use_log_scale=True, do_smoothing=True, window_size=50):
        """
        绘制多频率在一张大图上的对比。
        - 隐藏所有方法的原始数据。
        - StreamPLAID 分离出两条趋势线：Incremental Update 和 Local Reindex。
        - 其他 Baseline 只显示各自的趋势线。
        - 去除所有标题和注释文字。
        - 图例字体大小 16，且排成一行。
        - 横纵坐标标题字体放大1.5倍 (fontsize=21)。
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        from matplotlib.ticker import FuncFormatter

        # 设置绘图风格
        sns.set_theme(style="whitegrid")

        # 创建子图，保持扁平比例 (6.0)
        num_freqs = len(target_frequencies)
        fig, axes = plt.subplots(1, num_freqs, figsize=(8 * num_freqs, 6.0), dpi=300)

        if num_freqs == 1:
            axes = [axes]

        palette = sns.color_palette("deep", len(self.results))
        legend_dict = {}

        for ax_idx, target_frequency in enumerate(target_frequencies):
            ax = axes[ax_idx]
            all_y_values = []
            data_found = False

            for i, (method, freqs) in enumerate(self.results.items()):
                if target_frequency in freqs and 'index_update_time' in freqs[target_frequency]:
                    df_idx = freqs[target_frequency]['index_update_time'].copy()
                    df_idx.sort_values(by='Index', inplace=True)

                    x_col = 'Index' if 'Index' in df_idx.columns else df_idx.columns[0]
                    y_col = 'run_time' if 'run_time' in df_idx.columns else df_idx.columns[-1]

                    all_y_values.extend(df_idx[y_col].tolist())
                    color = palette[i]

                    # StreamPLAID 逻辑：拆分两条趋势线
                    if method == "StreamPLAID":
                        if 'use_reindex' in freqs[target_frequency]:
                            is_reidx_raw = freqs[target_frequency]['use_reindex']
                            if len(is_reidx_raw) == len(df_idx) + 1:
                                is_reidx = is_reidx_raw[1:]
                            else:
                                min_len = min(len(is_reidx_raw), len(df_idx))
                                is_reidx = is_reidx_raw[:min_len]
                                df_idx = df_idx.iloc[:min_len]

                            is_reidx = np.array(is_reidx, dtype=bool)
                            is_inc = ~is_reidx
                            df_inc = df_idx[is_inc].copy()
                            df_reidx = df_idx[is_reidx].copy()

                            if not df_inc.empty:
                                win_size = min(window_size, len(df_inc)) if do_smoothing else 1
                                smoothed_inc = df_inc[y_col].rolling(window=win_size, center=True,
                                                                     min_periods=1).mean() if do_smoothing else df_inc[
                                    y_col]
                                line_inc, = ax.plot(df_inc[x_col], smoothed_inc, linestyle='-', linewidth=2.5,
                                                    color=color)
                                legend_dict['StreamPLAID (Incremental Update)'] = line_inc

                            if not df_reidx.empty:
                                win_size = min(window_size, len(df_reidx)) if do_smoothing else 1
                                smoothed_reidx = df_reidx[y_col].rolling(window=win_size, center=True,
                                                                         min_periods=1).mean() if do_smoothing else \
                                df_reidx[y_col]
                                line_reidx, = ax.plot(df_reidx[x_col], smoothed_reidx, linestyle='--', linewidth=2.5,
                                                      color='#c44e52')
                                legend_dict['StreamPLAID (Local Repair)'] = line_reidx
                        else:
                            # 缺失掩码时的兜底
                            smoothed = df_idx[y_col].rolling(window=window_size, center=True,
                                                             min_periods=1).mean() if do_smoothing else df_idx[y_col]
                            line, = ax.plot(df_idx[x_col], smoothed, linestyle='-', linewidth=2.5, color=color)
                            legend_dict[method] = line
                    else:
                        # 其他 Baseline 逻辑：只显示趋势
                        smoothed = df_idx[y_col].rolling(window=window_size, center=True,
                                                         min_periods=1).mean() if do_smoothing else df_idx[y_col]
                        line, = ax.plot(df_idx[x_col], smoothed, linestyle='-', linewidth=2.5, color=color)
                        legend_dict[method] = line

                    data_found = True

            if not data_found:
                continue

            # 坐标轴设置
            if y_max:
                ax.set_ylim(top=y_max)
            elif clip_percentile and all_y_values:
                ax.set_ylim(top=pd.Series(all_y_values).quantile(clip_percentile) * 1.2)

            if use_log_scale:
                ax.set_yscale('log')
                pos_y = [y for y in all_y_values if y > 0]
                if pos_y: ax.set_ylim(bottom=min(pos_y) * 0.5)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
            else:
                ax.set_ylim(bottom=0)

            ax.set_title(f'Frequency: {target_frequency} Docs/Step', fontsize=21)

            # --- 核心修改区：将14放大1.5倍到21，并稍微调大刻度数字防止比例失调 ---
            ax.set_xlabel('Index update step', fontsize=21)
            ax.set_ylabel('Runtime (s) - LogScale' if use_log_scale else 'Runtime (s)', fontsize=21)
            ax.tick_params(axis='both', which='major', labelsize=14)
            # -------------------------------------------------------------

            ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)

        # ==========================================
        # 生成全局统一图例 (单行排列，去除注释)
        # ==========================================
        filtered_labels = list(legend_dict.keys())
        filtered_handles = list(legend_dict.values())

        # ncol=len(filtered_labels) 确保排成一行
        fig.legend(filtered_handles, filtered_labels, loc='lower center',
                   bbox_to_anchor=(0.5, 0.02), ncol=len(filtered_labels),
                   frameon=True, shadow=True, fontsize=16)

        # ==========================================
        # 布局微调 (更加紧凑)
        # ==========================================
        # bottom=0.12 适合单行图例，top=0.95 留出子图标题空间
        plt.tight_layout(pad=0.2, w_pad=3.0, rect=[0, 0.12, 1, 0.95])
        plt.show()

    def plot_index_update_time_multi_freq_2_0(self, target_frequencies=[10, 20, 30], y_max=None, clip_percentile=None,
                                          use_log_scale=True, do_smoothing=True, window_size=50):
        """
        绘制多频率在一张大图上的对比。
        - 隐藏所有方法的原始数据。
        - StreamPLAID 分离出两条趋势线：Incremental Update 和 Local Reindex。
        - 其他 Baseline 只显示各自的趋势线。
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        from matplotlib.ticker import FuncFormatter

        # 设置绘图风格
        sns.set_theme(style="whitegrid")

        # 动态创建子图，加大 figsize 并设置高分辨率 (dpi=300)
        num_freqs = len(target_frequencies)
        fig, axes = plt.subplots(1, num_freqs, figsize=(8 * num_freqs, 7), dpi=300)

        fig.suptitle('Incremental index update time comparison', fontsize=18, fontweight='bold')

        if num_freqs == 1:
            axes = [axes]

        palette = sns.color_palette("deep", len(self.results))
        legend_dict = {}

        for ax_idx, target_frequency in enumerate(target_frequencies):
            ax = axes[ax_idx]
            all_y_values = []
            data_found = False

            for i, (method, freqs) in enumerate(self.results.items()):
                if target_frequency in freqs and 'index_update_time' in freqs[target_frequency]:
                    df_idx = freqs[target_frequency]['index_update_time'].copy()
                    df_idx.sort_values(by='Index', inplace=True)

                    x_col = 'Index' if 'Index' in df_idx.columns else df_idx.columns[0]
                    y_col = 'run_time' if 'run_time' in df_idx.columns else df_idx.columns[-1]

                    all_y_values.extend(df_idx[y_col].tolist())
                    color = palette[i]

                    # ==========================================
                    # 针对 StreamPLAID：两条独立的趋势线
                    # ==========================================
                    if method == "StreamPLAID":
                        if 'use_reindex' in freqs[target_frequency]:
                            is_reidx_raw = freqs[target_frequency]['use_reindex']

                            # 对齐数组长度
                            if len(is_reidx_raw) == len(df_idx) + 1:
                                is_reidx = is_reidx_raw[1:]
                            else:
                                min_len = min(len(is_reidx_raw), len(df_idx))
                                is_reidx = is_reidx_raw[:min_len]
                                df_idx = df_idx.iloc[:min_len]

                            # 转化为布尔数组
                            is_reidx = np.array(is_reidx, dtype=bool)
                            is_inc = ~is_reidx

                            # 将数据拆分为两个独立的 DataFrame
                            df_inc = df_idx[is_inc].copy()
                            df_reidx = df_idx[is_reidx].copy()

                            # --- 画 Incremental Update 趋势线 (实线，分配的颜色) ---
                            if not df_inc.empty:
                                if do_smoothing:
                                    # 动态调整窗口大小，防止数据过少
                                    win_size = min(window_size, len(df_inc)) if len(df_inc) > 0 else 1
                                    smoothed_inc = df_inc[y_col].rolling(window=win_size, center=True,
                                                                         min_periods=1).mean()
                                    line_inc, = ax.plot(df_inc[x_col], smoothed_inc, linestyle='-', linewidth=2.5,
                                                        color=color)
                                else:
                                    line_inc, = ax.plot(df_inc[x_col], df_inc[y_col], linestyle='-', linewidth=2.5,
                                                        color=color)
                                legend_dict['StreamPLAID (Incremental Update)'] = line_inc

                            # --- 画 Local Reindex 趋势线 (红色，虚线，显著区分) ---
                            if not df_reidx.empty:
                                if do_smoothing:
                                    win_size = min(window_size, len(df_reidx)) if len(df_reidx) > 0 else 1
                                    smoothed_reidx = df_reidx[y_col].rolling(window=win_size, center=True,
                                                                             min_periods=1).mean()
                                    # 颜色固定为显眼的偏红色，使用虚线
                                    line_reidx, = ax.plot(df_reidx[x_col], smoothed_reidx, linestyle='--',
                                                          linewidth=2.5, color='#c44e52')
                                else:
                                    line_reidx, = ax.plot(df_reidx[x_col], df_reidx[y_col], linestyle='--',
                                                          linewidth=2.5, color='#c44e52')
                                legend_dict['StreamPLAID (Local Reindex)'] = line_reidx

                        else:
                            # 缺失掩码数据的兜底处理（仅画一条平滑线）
                            if do_smoothing and len(df_idx) > window_size:
                                smoothed_time = df_idx[y_col].rolling(window=window_size, center=True,
                                                                      min_periods=1).mean()
                                line_raw, = ax.plot(df_idx[x_col], smoothed_time, linestyle='-', linewidth=2.5,
                                                    color=color)
                            else:
                                line_raw, = ax.plot(df_idx[x_col], df_idx[y_col], linestyle='-', linewidth=2.5,
                                                    color=color)
                            legend_dict[method] = line_raw

                    # ==========================================
                    # 针对其他 Baseline：【只】显示趋势线
                    # ==========================================
                    else:
                        if do_smoothing and len(df_idx) > window_size:
                            smoothed_time = df_idx[y_col].rolling(window=window_size, center=True, min_periods=1).mean()
                            # 粗实线代表趋势，图例直接使用方法名
                            line_smooth, = ax.plot(df_idx[x_col], smoothed_time, linestyle='-', linewidth=2.5,
                                                   color=color)
                            legend_dict[method] = line_smooth
                        else:
                            # 兜底：如果数据量极少不够平滑，就直接画出连线
                            line_raw, = ax.plot(df_idx[x_col], df_idx[y_col], linestyle='-', linewidth=2.5, color=color)
                            legend_dict[method] = line_raw

                    data_found = True

            # ==========================================
            # 坐标轴与范围设置
            # ==========================================
            if not data_found:
                ax.set_title(f'Frequency: {target_frequency} (No Data)', fontsize=14)
                continue

            if y_max is not None:
                ax.set_ylim(top=y_max)
            elif clip_percentile is not None and len(all_y_values) > 0:
                s_all = pd.Series(all_y_values)
                auto_y_max = s_all.quantile(clip_percentile)
                ax.set_ylim(top=auto_y_max * 1.2)

            if use_log_scale:
                ax.set_yscale('log')
                positive_y_values = [y for y in all_y_values if y > 0]
                if positive_y_values:
                    min_pos_y = min(positive_y_values)
                    ax.set_ylim(bottom=min_pos_y * 0.5)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
            else:
                ax.set_ylim(bottom=0)

            ax.set_title(f'Frequency: {target_frequency}', fontsize=14)
            ax.set_xlabel('Index update step', fontsize=12)
            if use_log_scale:
                ax.set_ylabel('Runtime(Secondes)-LogScale', fontsize=12)
            else:
                ax.set_ylabel('Runtime (Seconds)', fontsize=12)

            ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)

        # ==========================================
        # 生成全局统一图例
        # ==========================================
        filtered_labels = list(legend_dict.keys())
        filtered_handles = list(legend_dict.values())

        # 将图例放在底部居中
        fig.legend(filtered_handles, filtered_labels, loc='lower center',
                   bbox_to_anchor=(0.5, 0), ncol=4, frameon=True, shadow=True, fontsize=11)

        # bottom=0.15 留出足够空间放置全局图例
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])
        plt.show()


    def plot_index_update_time_multi_freq_1_0(self, target_frequencies=[10, 20, 30], y_max=None, clip_percentile=None,
                                          use_log_scale=True, do_smoothing=True, window_size=50):
        """
        绘制多频率在一张大图上的对比（针对 StreamPLAID 特化），并共享统一图例。
        :param target_frequencies: 频率列表，默认 [10, 20, 30]
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        from matplotlib.ticker import FuncFormatter

        # 设置绘图风格
        sns.set_theme(style="whitegrid")

        # 动态创建子图 (1行 N列)
        num_freqs = len(target_frequencies)
        # 将单个图的宽度从 6 增加到 8（或更宽，如 10），并加上 dpi=300 提高清晰度
        fig, axes = plt.subplots(1, num_freqs, figsize=(8 * num_freqs, 7), dpi=300)

        # 统一大标题
        fig.suptitle('Incremental index update time comparison', fontsize=18, fontweight='bold')

        # 如果只有一个频率，axes不是数组，将其转为列表方便遍历
        if num_freqs == 1:
            axes = [axes]

        palette = sns.color_palette("deep", len(self.results))

        # 用于收集全局图例的字典，确保同一个 label 只出现一次
        legend_dict = {}

        for ax_idx, target_frequency in enumerate(target_frequencies):
            ax = axes[ax_idx]
            all_y_values = []
            data_found = False

            for i, (method, freqs) in enumerate(self.results.items()):
                if target_frequency in freqs and 'index_update_time' in freqs[target_frequency]:
                    df_idx = freqs[target_frequency]['index_update_time'].copy()
                    df_idx.sort_values(by='Index', inplace=True)

                    x_col = 'Index' if 'Index' in df_idx.columns else df_idx.columns[0]
                    y_col = 'run_time' if 'run_time' in df_idx.columns else df_idx.columns[-1]

                    all_y_values.extend(df_idx[y_col].tolist())
                    color = palette[i]

                    # ==========================================
                    # 针对 StreamPLAID 的拆分画法
                    # ==========================================
                    if method == "StreamPLAID":
                        if 'use_reindex' in freqs[target_frequency]:
                            is_reidx_raw = freqs[target_frequency]['use_reindex']

                            # 安全对齐逻辑
                            if len(is_reidx_raw) == len(df_idx) + 1:
                                is_reidx = is_reidx_raw[1:]
                            else:
                                min_len = min(len(is_reidx_raw), len(df_idx))
                                is_reidx = is_reidx_raw[:min_len]
                                df_idx = df_idx.iloc[:min_len]

                            x_vals = df_idx[x_col].values
                            y_vals = df_idx[y_col].values
                            is_inc = ~is_reidx

                            x_inc, y_inc = x_vals[is_inc], y_vals[is_inc]
                            x_reidx, y_reidx = x_vals[is_reidx], y_vals[is_reidx]

                            # 绘制并保存到图例字典
                            line_inc, = ax.plot(x_inc, y_inc, color='#c44e52', linestyle='-', marker='o',
                                                markersize=3, linewidth=1.5, alpha=0.85)
                            legend_dict['StreamPLAID (Incremental Update)'] = line_inc

                            line_reidx, = ax.plot(x_reidx, y_reidx, color='#c44e52', linestyle='-', marker='x',
                                                  markersize=5, linewidth=1.5, alpha=0.85)
                            legend_dict['StreamPLAID (Local Reindex Triggered)'] = line_reidx
                        else:
                            line_raw, = ax.plot(df_idx[x_col], df_idx[y_col], color=color, linewidth=1.5)
                            legend_dict[f'{method} (Raw)'] = line_raw

                    # ==========================================
                    # 针对其他 Baseline 的平滑画法
                    # ==========================================
                    else:
                        line_raw, = ax.plot(df_idx[x_col], df_idx[y_col], linestyle='-', linewidth=0.8,
                                            color=color, alpha=0.3)
                        legend_dict[f'{method} (Raw)'] = line_raw

                        if do_smoothing and len(df_idx) > window_size:
                            smoothed_time = df_idx[y_col].rolling(window=window_size, center=True, min_periods=1).mean()
                            line_smooth, = ax.plot(df_idx[x_col], smoothed_time, linestyle='-', linewidth=2.5,
                                                   color=color)
                            legend_dict[f'{method} (Trend, MA-{window_size})'] = line_smooth

                    data_found = True

            # ==========================================
            # 坐标轴与范围设置 (应用到当前子图)
            # ==========================================
            if not data_found:
                ax.set_title(f'Frequency: {target_frequency} (No Data)', fontsize=14)
                continue

            # 1. 设置 Y 轴上限
            if y_max is not None:
                ax.set_ylim(top=y_max)
            elif clip_percentile is not None and len(all_y_values) > 0:
                s_all = pd.Series(all_y_values)
                auto_y_max = s_all.quantile(clip_percentile)
                ax.set_ylim(top=auto_y_max * 1.2)

            # 2. 设置对数轴及 Y 轴下限
            if use_log_scale:
                ax.set_yscale('log')
                positive_y_values = [y for y in all_y_values if y > 0]
                if positive_y_values:
                    min_pos_y = min(positive_y_values)
                    ax.set_ylim(bottom=min_pos_y * 0.5)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
            else:
                ax.set_ylim(bottom=0)

            # 3. 设置子图标题和 XY 标签
            ax.set_title(f'Frequency: {target_frequency}', fontsize=14)
            ax.set_xlabel('Index update step', fontsize=12)
            if use_log_scale:
                ax.set_ylabel('Runtime(Secondes)-LogScale', fontsize=12)
            else:
                ax.set_ylabel('Runtime (Seconds)', fontsize=12)

            ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)

        # ==========================================
        # 生成全局过滤后的统一图例
        # ==========================================
        filtered_handles = []
        filtered_labels = []

        # 定义需要屏蔽的关键词（兼容带有空格或无空格的写法）
        exclude_keywords = ["ReindexPLAID (Raw)", "ReindexPLAID(Raw)",
                            "FronzenPLAID (Raw)", "FronzenPLAID(Raw)",
                            "FrozenPLAID (Raw)", "FrozenPLAID(Raw)"]

        for label, handle in legend_dict.items():
            if any(keyword in label for keyword in exclude_keywords):
                continue
            filtered_labels.append(label)
            filtered_handles.append(handle)

        # 将图例放在整个画布的底部正中间
        fig.legend(filtered_handles, filtered_labels, loc='lower center',
                   bbox_to_anchor=(0.5, 0), ncol=4, frameon=True, shadow=True, fontsize=11)

        # 调整布局，给底部的全局图例和顶部的大标题留出空间
        # rect=[left, bottom, right, top]
        plt.tight_layout(rect=[0, 0.12, 1, 0.95])
        plt.show()


if __name__ == '__main__':
    root = r"../p_science_lotte"
    pr_vis = PressureResultsVisualizer(root)