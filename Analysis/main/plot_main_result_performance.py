import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_iid_ood_bar_separate_no_title(data_root="."):
    """
    分别生成 In-Domain 和 Out-Of-Domain 的独立柱状图。
    - 去掉了图表内部的标题 (Title)。
    - 图例内置于图表内部。
    - 极致去白边。
    """
    iid_file = os.path.join(data_root, "iid_merge_arg.csv")
    ood_file = os.path.join(data_root, "ood_merge_arg.csv")

    # 1. 数据读取逻辑
    def get_data(file):
        if not os.path.exists(file): return None
        df = pd.read_csv(file)
        df['Method'] = df['Method'].replace({'StreamPLAID': 'StreamPLAID (Our)'})
        df_melt = df.melt(id_vars='Method', var_name='Task', value_name='nDCG@10')
        df_melt['Task'] = df_melt['Task'].str.replace('_Mean', '').str.replace('Task', 'Task ')
        return df_melt

    df_iid_melt = get_data(iid_file)
    df_ood_melt = get_data(ood_file)

    # 2. 样式配置
    method_order = ['Dessert', 'PLAID', 'IGP', 'MUVERA', 'XTR', 'StreamPLAID (Our)']
    palette = sns.color_palette("deep", len(method_order))
    sns.set_theme(style="whitegrid")

    # 3. 绘图函数
    def create_single_bar_plot(data, filename):
        if data is None: return

        # 稍微调低画布高度，因为没有了标题，(10, 5) 会显得更紧凑
        plt.figure(figsize=(10, 5), dpi=300)

        ax = sns.barplot(
            data=data, x='Task', y='nDCG@10', hue='Method',
            hue_order=method_order, palette=palette,
            edgecolor='black', linewidth=1.0
        )

        # 【核心修改】去掉了 plt.title(...) 逻辑

        plt.xlabel("", fontsize=14)
        plt.ylabel("nDCG@10", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # 图例内置
        plt.legend(loc='best',
                   bbox_to_anchor=(0.5, 0.98),
                   ncol=3,
                   fontsize=10,
                   frameon=True,
                   framealpha=0.6,
                   edgecolor='gray')

        # 极致去白边：保存时 bbox_inches='tight' 会自动把没有标题后的顶部空白切掉
        plt.tight_layout()
        save_path = os.path.join(data_root, filename)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.01)
        print(f">> 已保存 (无标题版): {save_path}")
        plt.show()

    # 4. 执行
    create_single_bar_plot(df_iid_melt, "iid_bar.png")
    create_single_bar_plot(df_ood_melt, "ood_bar.png")


def plot_iid_ood_step_line_separate(data_root="."):
    """
    分别生成 IID (Step 0-5) 和 OOD (Step 0-4) 两张独立的折线图。
    - 图例内置于图表内部。
    - 极致去白边。
    """
    iid_file = os.path.join(data_root, "iid_merge_arg_task0.csv")
    ood_file = os.path.join(data_root, "ood_merge_arg_task0.csv")

    # 1. 读取数据逻辑
    def get_data(file):
        if not os.path.exists(file):
            return None
        df = pd.read_csv(file)
        df['Method'] = df['Method'].replace({'StreamPLAID': 'StreamPLAID (Our)'})
        df_melt = df.melt(id_vars='Method', var_name='Step', value_name='nDCG@10')
        df_melt['Step_Idx'] = df_melt['Step'].str.extract('(\d+)').astype(int)
        return df_melt.dropna(subset=['nDCG@10']).sort_values(['Method', 'Step_Idx'])

    df_iid_melt = get_data(iid_file)
    df_ood_melt = get_data(ood_file)

    # 2. 绘图样式配置
    method_order = ['Dessert', 'PLAID', 'IGP', 'MUVERA', 'XTR', 'StreamPLAID (Our)']
    palette = sns.color_palette("deep", len(method_order))
    markers = ["o", "s", "D", "^", "v", "p"]
    sns.set_theme(style="whitegrid")

    # 3. 核心绘图函数
    def create_single_plot(data, title, filename):
        if data is None:
            print(f"跳过: {title} 数据不存在")
            return

        # 创建独立的画布，单张图建议比例 (10, 6)
        plt.figure(figsize=(10, 6), dpi=300)
        ax = plt.gca()

        actual_steps = sorted(data['Step_Idx'].unique())

        for i, method in enumerate(method_order):
            m_data = data[data['Method'] == method]
            if m_data.empty: continue

            plt.plot(m_data['Step_Idx'], m_data['nDCG@10'],
                     label=method, color=palette[i], marker=markers[i],
                     linewidth=2.5, markersize=10, markeredgecolor='white', markeredgewidth=0.6)

        # 设置标题和标签

        plt.ylabel("nDCG@10", fontsize=12)

        # 设置刻度
        plt.xticks(actual_steps, [f"Step {s}" for s in actual_steps], fontsize=12)
        plt.yticks(fontsize=12)

        # 【核心修改】图例内置
        # loc='lower left' 通常比较适合 Step 趋势向下的图，不会遮挡线条
        # 如果遮挡了，可以改为 loc='best'
        plt.legend(loc='best', fontsize=12, frameon=True, shadow=True, ncol=2)

        # 极致去白边
        plt.tight_layout()
        save_path = os.path.join(data_root, filename)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.01)
        print(f">> 已保存: {save_path}")
        plt.show()

    # 4. 执行生成两张图
    create_single_plot(df_iid_melt, "In-Domain Datasets", "iid_step_line.png")
    create_single_plot(df_ood_melt, "Out-Of-Domain Datasets", "ood_step_line.png")

if __name__ == '__main__':
    plot_iid_ood_step_line_separate(data_root="./results")
    plot_iid_ood_bar_separate_no_title(data_root="./results")