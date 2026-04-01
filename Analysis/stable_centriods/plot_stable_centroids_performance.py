
if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # ================= 1. 配置区域 =================

    # 请修改为你的实际文件路径
    file_path = "./results/stable_centroids_arg_performance.csv"

    # --- A. 方法名称映射 ---
    method_name_map = {
        'IGP(Full)': 'IGP (Full)',
        'IGP(wo_update_centriods)': 'IGP (w/o Update)',
        'PLAID(Full)': 'PLAID (Full)',
        'PLAID(wo_update_centriods)': 'PLAID (w/o Update)',
        'StreamPLAID': 'StreamPLAID (Ours)'
    }

    # --- B. 数据集名称映射 ---
    dataset_name_map = {
        'beir-ood': 'BEIR-OOD',
        'ood_dataset': 'BEIR-OOD',
        'ood': 'BEIR-OOD',
        'BEIR-OOD': 'BEIR-OOD'
    }

    # --- C. 排序逻辑 ---
    method_order = [
        'IGP (Full)',
        'IGP (w/o Update)',
        'PLAID (Full)',
        'PLAID (w/o Update)',
        'StreamPLAID (Ours)'
    ]

    # --- D. 颜色配置 ---
    palette_config = {
        'IGP (Full)': '#1f77b4',  # 深蓝
        'IGP (w/o Update)': '#aec7e8',  # 浅蓝
        'PLAID (Full)': '#2ca02c',  # 深绿
        'PLAID (w/o Update)': '#98df8a',  # 浅绿
        'StreamPLAID (Ours)': '#d62728'  # 红色
    }


    # ================= 2. 数据处理 =================

    def load_and_process(filepath):
        try:
            df = pd.read_csv(filepath)
            print(f">> 成功读取文件: {filepath}")
        except FileNotFoundError:
            print("!! 找不到文件，使用模拟数据演示...")
            # 模拟 BEIR-OOD 数据
            methods_raw = ['IGP(Full)', 'IGP(wo_update_centriods)',
                           'PLAID(Full)', 'PLAID(wo_update_centriods)',
                           'StreamPLAID']
            data = []
            for m in methods_raw:
                base_score = 0.4 if 'PLAID' in m else 0.35
                if 'wo' in m: base_score -= 0.05
                if 'Stream' in m: base_score = 0.55

                data.append({
                    'Method': m,
                    'Dataset': 'beir-ood',
                    'Task0_Mean': base_score + 0.05, 'Task0_Std': 0.012,
                    'Task1_Mean': base_score + 0.02, 'Task1_Std': 0.023,
                    'Task2_Mean': base_score - 0.01, 'Task2_Std': 0.015
                })
            df = pd.DataFrame(data)

        # 映射与转换
        df['Dataset'] = df['Dataset'].map(lambda x: dataset_name_map.get(x, x))
        dataset_name = df['Dataset'].iloc[0] if not df.empty else "Unknown"

        df['Method_Display'] = df['Method'].map(lambda x: method_name_map.get(x, x))

        task_mean_cols = [c for c in df.columns if "Task" in c and "_Mean" in c]
        if not task_mean_cols:
            return pd.DataFrame(), [], ""

        task_ids = sorted([int(c.replace("Task", "").replace("_Mean", "")) for c in task_mean_cols])

        long_data = []
        for _, row in df.iterrows():
            method_disp = row['Method_Display']
            if method_disp not in method_order: continue

            for t in task_ids:
                mean_col = f"Task{t}_Mean"
                std_col = f"Task{t}_Std"

                if mean_col in df.columns:
                    mean_val = row[mean_col]
                    std_val = row.get(std_col, 0.0)
                    if pd.isna(std_val): std_val = 0.0

                    long_data.append({
                        'Method': method_disp,
                        'Task': t,
                        'Mean': mean_val,
                        'Std': std_val
                    })

        return pd.DataFrame(long_data), task_ids, dataset_name


    plot_df, task_ids, dataset_name = load_and_process(file_path)

    # ================= 3. 绘图 =================

    if not plot_df.empty:
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(14, 8))

        # --- A. 绘制柱状图 ---
        ax = sns.barplot(
            data=plot_df,
            x='Task',
            y='Mean',
            hue='Method',
            hue_order=method_order,
            palette=palette_config,
            edgecolor='black',
            linewidth=1,
            errorbar=None,
            zorder=2
        )

        # --- B. 手动添加 Error Bar 和 数值 (错位显示逻辑) ---
        y_max_limit = 0

        # enumerate(ax.containers) 中的 i 对应的是 Method 的索引
        # 0: IGP Full, 1: IGP wo, 2: PLAID Full, 3: PLAID wo, 4: StreamPLAID
        for i, container in enumerate(ax.containers):
            if i >= len(method_order): break
            current_method_name = method_order[i]

            for j, bar in enumerate(container):
                if j >= len(task_ids): break
                current_task = task_ids[j]

                record = plot_df[
                    (plot_df['Method'] == current_method_name) &
                    (plot_df['Task'] == current_task)
                    ]
                if record.empty: continue

                mean_val = record['Mean'].values[0]
                std_val = record['Std'].values[0]

                x_pos = bar.get_x() + bar.get_width() / 2
                y_pos = bar.get_height()
                top_val = mean_val + std_val

                if top_val > y_max_limit: y_max_limit = top_val

                # 1. 画误差线
                if std_val > 0:
                    ax.errorbar(x_pos, y_pos, yerr=std_val, fmt='none', c='black', capsize=3, elinewidth=1.5)

                # -----------------------------------------------------------
                # 2. 标数值 (方案二：高低错位)
                # -----------------------------------------------------------
                # 定义基础偏移量
                base_offset = 0.01
                # 定义额外偏移量 (用于把字顶上去)
                stagger_height = -0.1

                # 逻辑：如果 i 是奇数 (1, 3)，则抬高显示；如果是偶数 (0, 2, 4)，则正常显示
                # 这样相邻的柱子文字高度就会错开
                if i % 2 != 0:
                    text_y = top_val + base_offset + stagger_height
                else:
                    text_y = top_val + base_offset

                label_text = f"{mean_val:.3f}\n(±{std_val:.3f})"

                ax.text(x_pos, text_y,
                        label_text,
                        ha='center',
                        va='bottom',
                        fontsize=7,  # 字体稍微调小
                        color='black',
                        fontweight='bold')
                # -----------------------------------------------------------

        # --- C. 装饰图表 ---
        plt.title(f"Impact of Centroids Update on {dataset_name}", fontsize=18, fontweight='bold', pad=25)
        plt.ylabel("nDCG Score", fontsize=14, fontweight='bold')
        plt.xlabel("Streaming Tasks", fontsize=14, fontweight='bold')

        plt.xticks(ticks=range(len(task_ids)), labels=[f"Task {t}" for t in task_ids], fontsize=12)
        plt.yticks(fontsize=12)

        if y_max_limit == 0: y_max_limit = 1.0

        # 增加 Y 轴高度，因为错位显示需要更多顶部空间
        plt.ylim(0, y_max_limit * 1.45)

        plt.legend(
            title="Comparison Variants",
            title_fontsize=12,
            fontsize=11,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.98),
            ncol=5,
            frameon=True,
            shadow=True
        )

        plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        plt.tight_layout()

        save_name = "centroids_ablation_staggered.png"
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f">> 图表已保存至: {save_name}")
        plt.show()
    else:
        print("没有数据用于绘图")