import os
from collections import defaultdict

import numpy as np
import pandas as pd


def calculate_wide_format_stats(data_root, methods, datasets,
                                save_filename="final_wide_stats.csv", max_step=4):
    """
    功能：
    生成宽表格式。每一行是一个 Method + Dataset。
    列包含每个 Task (0 到 max_step) 在对角线 (Task==Step) 处的 Mean 和 Std。

    聚合逻辑：
    1. 找到 Task i 在 Step i 的 nDCG。
    2. 对 3 个 Seed (Iteration) 的该值进行聚合 (Mean, Std)。
    3. 特殊处理 lotte-pooled (单 Seed)。
    """

    final_rows = []

    print(f"开始处理... (目标: 宽表格式, Max Task = {max_step})")

    for method in methods:
        for dataset in datasets:

            # --- 1. 准备容器 ---
            # 结构: { step_id: [score_seed0, score_seed1, score_seed2] }
            # 用于收集该 Method+Dataset 下，每一个 Task 在不同种子里的表现
            step_data_collector = defaultdict(list)

            # --- 2. 确定种子列表 ---

            target_iterations = ["0", "1", "2"]

            # --- 3. 遍历种子读取数据 ---
            for iteration in target_iterations:
                file_path = os.path.join(data_root,
                                         method, dataset, iteration,
                                         "effectiveness_metrics.csv")

                if not os.path.exists(file_path):
                    # 如果某个种子缺失，就跳过该种子
                    continue

                try:
                    df = pd.read_csv(file_path)

                    # 清洗列名 (防止 ' StreamingStep' 这种带空格的情况)
                    df.columns = df.columns.str.strip()

                    # 提取对角线 (Task == StreamingStep)
                    # 这里假设 CSV 列名为 'Task', 'StreamingStep', 'nDCG'
                    diagonal_df = df[df['Task'] == df['StreamingStep']]

                    # 遍历对角线数据，存入收集器
                    for _, row in diagonal_df.iterrows():
                        step = int(row['StreamingStep'])
                        score = row['nDCG']

                        # 只收集我们关心的范围 (0 到 max_step)
                        if 0 <= step <= max_step:
                            step_data_collector[step].append(score)

                except Exception as e:
                    print(f"读取出错 {file_path}: {e}")

            # --- 4. 聚合计算 (行生成) ---
            # 如果收集器是空的 (说明所有种子文件都缺失)，生成全 NaN 行

            row_dict = {
                "Method": method,
                "Dataset": dataset
            }

            # 遍历每一个 Task (从 0 到 max_step) 生成列
            for step in range(max_step + 1):
                scores = step_data_collector.get(step, [])

                col_mean = f"Task{step}_Mean"
                col_std = f"Task{step}_Std"

                if not scores:
                    # 无数据
                    row_dict[col_mean] = np.nan
                    row_dict[col_std] = np.nan
                elif dataset == "lotte-pooled":
                    # 单种子
                    row_dict[col_mean] = scores[0]
                    row_dict[col_std] = np.nan  # 单种子无方差
                else:
                    # 多种子
                    row_dict[col_mean] = np.mean(scores)
                    row_dict[col_std] = np.std(scores)

            final_rows.append(row_dict)

    # --- 5. 保存结果 ---
    if final_rows:
        df_res = pd.DataFrame(final_rows)

        # 格式化数字，保留4位小数
        # 找出所有包含 Mean 或 Std 的列进行 round
        numeric_cols = [c for c in df_res.columns if "Mean" in c or "Std" in c]
        df_res[numeric_cols] = df_res[numeric_cols].round(4)

        output_path = os.path.join(data_root, save_filename)
        df_res.to_csv(output_path, index=False)

        print(f"\n统计完成！宽表已保存至: {output_path}")
        # 打印前几行预览
        print(df_res.head())
        return df_res
    else:
        print("未生成任何结果。")


def calculate_task0_retention_stats(data_root, methods, datasets,
                                    save_filename="task0_retention_stats.csv", max_step=5):
    """
    功能：
    统计 Task 0 在整个流式过程 (Step 0 到 max_step) 中的表现。
    用于分析模型对初始任务的遗忘情况。

    逻辑：
    1. 锁定 Task == 0 的行。
    2. 提取 Step 0 到 max_step 的 nDCG。
    3. 对 3 个 Iteration 进行聚合 (Mean, Std)。
    """

    final_rows = []

    print(f"开始处理... (目标: Task 0 Retention, Step 0-{max_step})")

    for method in methods:
        for dataset in datasets:

            # --- 1. 准备容器 ---
            # 结构: { step_id: [score_seed0, score_seed1, score_seed2] }
            step_data_collector = defaultdict(list)

            # --- 2. 确定种子列表 ---
            target_iterations = ["0", "1", "2"]

            # --- 3. 遍历种子读取数据 ---
            for iteration in target_iterations:
                file_path = os.path.join(data_root,method, dataset, iteration,
                                         "effectiveness_metrics.csv")

                if not os.path.exists(file_path):
                    continue

                try:
                    df = pd.read_csv(file_path)

                    # 清洗列名
                    df.columns = df.columns.str.strip()

                    # --- 核心修改：只筛选 Task == 0 的数据 ---
                    task0_df = df[df['Task'] == 0]

                    # 遍历 Task 0 在各个 Step 的表现
                    for _, row in task0_df.iterrows():
                        step = int(row['StreamingStep'])
                        score = row['nDCG']

                        # 只收集 Step 0 到 max_step 的数据
                        if 0 <= step <= max_step:
                            step_data_collector[step].append(score)

                except Exception as e:
                    print(f"读取出错 {file_path}: {e}")

            # --- 4. 聚合计算 (行生成) ---
            row_dict = {
                "Method": method,
                "Dataset": dataset
            }

            # 遍历 Step 0 到 max_step 生成列
            for step in range(max_step + 1):
                scores = step_data_collector.get(step, [])

                # 列名改为 StepX_Mean (表示在 Step X 时 Task 0 的分数)
                col_mean = f"Step{step}_Mean"
                col_std = f"Step{step}_Std"

                if not scores:
                    row_dict[col_mean] = np.nan
                    row_dict[col_std] = np.nan
                elif dataset == "lotte-pooled":
                    row_dict[col_mean] = scores[0]
                    row_dict[col_std] = np.nan
                else:
                    row_dict[col_mean] = np.mean(scores)
                    row_dict[col_std] = np.std(scores)

            final_rows.append(row_dict)

    # --- 5. 保存结果 ---
    if final_rows:
        df_res = pd.DataFrame(final_rows)

        # 格式化数字
        numeric_cols = [c for c in df_res.columns if "Mean" in c or "Std" in c]
        df_res[numeric_cols] = df_res[numeric_cols].round(4)

        output_path = os.path.join(data_root, save_filename)
        df_res.to_csv(output_path, index=False)

        print(f"\n统计完成！结果已保存至: {output_path}")
        print(df_res.head())
        return df_res
    else:
        print("未生成任何结果。")

# 示例调用
# datasets_list = ["lotte-pooled", "product", "clue"]
# methods_list = ["Finetune", "MUVERA", "Replay"]
# calculate_final_stats_special("./data", methods_list, datasets_list)


if __name__ == '__main__':
    data_root = r"./results"
    methods = ['w_MRHP&Repairing', "wo_MRHP", "wo_Repairing"]
    # methods = ["StreamPLAID"]
    # datasets = ["scifact", "arguana", "fiqa", "touche2020", "trec-covid", "lotte-pooled"]
    datasets = ["beir-ood", "lotte-ood"]
    # distribution_type = "iid"
    save_filename = "ab_arg_performance.csv"
    calculate_wide_format_stats(data_root, methods, datasets, save_filename,max_step=4)
    # merged_df = merge_index_time_stats_int(data_root, methods, datasets, distribution_type, save_filename)