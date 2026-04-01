import os

import pandas as pd


def merge_index_time_stats(data_root, methods, datasets, distribution_type="iid", save_filename="index_time_stats.csv",
                           max_step=10):
    """
    读取每个方法、数据集、3个种子的文件。
    计算每个 step 的运行时间均值和方差。
    最后保存为横向宽表。
    """

    # 1. 收集所有原始数据
    raw_data = []

    for method in methods:
        for dataset in datasets:
            # 遍历 3 个种子 (0, 1, 2)
            for iteration in ["0", "1", "2"]:

                # --- 构建文件路径 (保持原有逻辑) ---
                if method == "xtr":
                    file_path = os.path.join(data_root, distribution_type,
                                             method, dataset, iteration,
                                             "index_time", "index_updata.csv")
                else:
                    file_path = os.path.join(data_root, distribution_type,
                                             method, dataset, iteration,
                                             "index_time", "indexing.csv")

                if not os.path.exists(file_path):
                    # print(f"[Warning] 文件不存在: {file_path}") # 可选：减少刷屏
                    continue

                try:
                    df = pd.read_csv(file_path)
                    df.columns = df.columns.str.strip()  # 去空格

                    # 兼容列名
                    if "run_time" in df.columns:
                        time_col = "run_time"
                    elif "time" in df.columns:
                        time_col = "time"
                    else:
                        print(f"[Skip] {file_path} 缺少 time 列")
                        continue

                    # 确保按 Index 排序
                    if "Index" in df.columns:
                        df = df.sort_values("Index")

                    # --- 提取每个 Step 的数据 ---
                    # 循环提取 step 1 到 max_step
                    for step in range(1, max_step + 1):
                        current_time = None

                        # 处理 Index 偏移逻辑
                        if method == "xtr":
                            # xtr: Index 1 对应 step 1
                            target_row = df[df["Index"] == step]
                        else:
                            # plaid/others: Index 0 对应 step 1
                            target_row = df[df["Index"] == step - 1]

                        if not target_row.empty:
                            current_time = target_row[time_col].values[0]

                        # 将原始数据加入列表
                        # 只要有数据就加，后续由 pandas 处理 NaN
                        if current_time is not None:
                            raw_data.append({
                                "method": method,
                                "dataset": dataset,
                                "iteration": iteration,  # 记录种子，虽然聚合时会折叠掉
                                "step": step,
                                "time": current_time
                            })

                except Exception as e:
                    print(f"读取错误 {file_path}: {e}")

    if not raw_data:
        print("未找到任何有效数据。")
        return pd.DataFrame()

    # 2. 转换为 DataFrame 并计算统计量
    df_raw = pd.DataFrame(raw_data)

    # 按 method, dataset, step 分组，计算 time 的 mean 和 var
    # reset_index 后结构: method | dataset | step | mean | var
    df_stats = df_raw.groupby(["method", "dataset", "step"])["time"].agg(["mean", "std"]).reset_index()

    # 3. 转换格式 (Pivot)
    # 将 step 这一列的值，变成宽表的列头
    df_pivot = df_stats.pivot(index=["method", "dataset"], columns="step", values=["mean", "std"])

    # 4. 扁平化列名
    # df_pivot 的列现在是多级索引: ('mean', 1), ('mean', 2)... ('var', 1)...
    # 我们需要将其变成: step1_mean, step2_mean ... step1_var, step2_var

    new_columns = []
    for stat_type, step_num in df_pivot.columns:
        new_columns.append(f"step{step_num}_{stat_type}")

    df_pivot.columns = new_columns
    df_pivot = df_pivot.reset_index()

    # 5. 可选：为了美观，调整列顺序
    # 现在的顺序可能是 step1_mean, step2_mean... step1_var...
    # 如果你想把 mean 和 var 放一起 (step1_mean, step1_var, step2_mean...)，可以做额外的排序
    # 这里做简单的字母排序通常就够了，或者按 Step 数值重排

    # 构建理想的列顺序
    sorted_cols = ["method", "dataset"]
    for i in range(1, max_step + 1):
        if f"step{i}_mean" in df_pivot.columns:
            sorted_cols.append(f"step{i}_mean")
        if f"step{i}_var" in df_pivot.columns:
            sorted_cols.append(f"step{i}_std")

    # 只保留存在的列
    final_cols = [c for c in sorted_cols if c in df_pivot.columns]
    df_final = df_pivot[final_cols]

    # 保存
    # save_file = os.path.join(data_root, save_filename)
    # df_final.to_csv(save_file, index=False)
    # print(f"统计完成！文件已保存至: {save_file}")
    print(f"包含列示例: {final_cols[:6]} ...")

    return df_final


def merge_index_time_stats_std(data_root, methods, datasets, distribution_type="iid",
                               save_filename="index_time_std.csv", max_step=10):
    """
    读取每个方法、数据集、3个种子的文件。
    计算每个 step 的运行时间【均值】和【标准差】。
    最后保存为横向宽表。
    """

    # 1. 收集所有原始数据
    raw_data = []

    for method in methods:
        for dataset in datasets:
            # 遍历 3 个种子
            for iteration in ["0", "1", "2"]:

                # --- 构建文件路径 ---
                if method == "xtr":
                    file_path = os.path.join(data_root, distribution_type,
                                             method, dataset, iteration,
                                             "index_time", "index_updata.csv")
                else:
                    file_path = os.path.join(data_root, distribution_type,
                                             method, dataset, iteration,
                                             "index_time", "indexing.csv")

                if not os.path.exists(file_path):
                    continue

                try:
                    df = pd.read_csv(file_path)
                    df.columns = df.columns.str.strip()  # 去空格

                    # 兼容列名
                    if "run_time" in df.columns:
                        time_col = "run_time"
                    elif "time" in df.columns:
                        time_col = "time"
                    else:
                        continue

                    # 确保按 Index 排序
                    if "Index" in df.columns:
                        df = df.sort_values("Index")

                    # --- 提取每个 Step 的数据 ---
                    for step in range(1, max_step + 1):
                        current_time = None

                        # 处理 Index 偏移逻辑
                        if method == "XTR" or method== r"StreamPLAID":
                            # xtr: Index 1 对应 step 1
                            target_row = df[df["Index"] == step]
                        else:
                            # plaid/others: Index 0 对应 step 1
                            target_row = df[df["Index"] == step - 1]

                        if not target_row.empty:
                            current_time = target_row[time_col].values[0]

                        if current_time is not None:
                            raw_data.append({
                                "method": method,
                                "dataset": dataset,
                                "step": step,
                                "time": current_time
                                # 注意：这里不需要存 iteration，因为我们要把它聚合掉
                            })

                except Exception as e:
                    print(f"读取错误 {file_path}: {e}")

    if not raw_data:
        print("未找到任何有效数据。")
        return pd.DataFrame()

    # 2. 转换为 DataFrame 并计算统计量
    df_raw = pd.DataFrame(raw_data)

    # ---------------------------------------------------------
    # [核心修改]：将 "var" 改为 "std"
    # reset_index 后结构: method | dataset | step | mean | std
    # ---------------------------------------------------------
    df_stats = df_raw.groupby(["method", "dataset", "step"])["time"].agg(["mean", "std"]).reset_index()

    # 3. 转换格式 (Pivot)
    # 将 step 转为列，values 包含 mean 和 std
    df_pivot = df_stats.pivot(index=["method", "dataset"], columns="step", values=["mean", "std"])

    # 4. 扁平化列名
    # 列名会变成: step1_mean, step1_std, step2_mean, step2_std ...
    new_columns = []
    # df_pivot.columns 是 MultiIndex: [('mean', 1), ('std', 1), ...]
    for stat_type, step_num in df_pivot.columns:
        new_columns.append(f"step{step_num}_{stat_type}")

    df_pivot.columns = new_columns
    df_pivot = df_pivot.reset_index()

    # 5. 调整列顺序 (交替排列 Mean 和 Std)
    sorted_cols = ["method", "dataset"]
    for i in range(1, 5):
        col_mean = f"step{i}_mean"
        col_std = f"step{i}_std"  # [修改]: 这里的后缀对应上面的 stat_type

        if col_mean in df_pivot.columns:
            sorted_cols.append(col_mean)
        if col_std in df_pivot.columns:
            sorted_cols.append(col_std)

    # 只保留存在的列
    final_cols = [c for c in sorted_cols if c in df_pivot.columns]
    df_final = df_pivot[final_cols]

    # 保存
    # save_file = os.path.join(data_root, save_filename)
    # df_final.to_csv(save_file, index=False)
    # print(f"统计完成 (均值+标准差)！文件已保存至: {save_file}")

    return df_final


import pandas as pd
import os


def merge_index_time_stats_int(data_root, methods, datasets, distribution_type="iid",
                               save_filename="index_time_stats_int.csv", max_step=5):
    """
    读取每个方法、数据集、3个种子的文件。
    计算每个 step 的运行时间【均值】和【标准差】。
    最终结果【保留整数】（四舍五入）。
    """

    # 1. 收集所有原始数据
    raw_data = []

    for method in methods:
        for dataset in datasets:
            if dataset==r"lotte-pooled" and method =="MUVERA":
                continue
            if dataset==r"lotte-pooled":
                iteration_list = ["0"]

            else:
                iteration_list = ["0", "1", "2"]
            for iteration in iteration_list:

                # --- 构建文件路径 ---
                if method == "XTR":
                    file_path = os.path.join(data_root, distribution_type,
                                             method, dataset, iteration,
                                             "index_time", "index_updata.csv")
                else:
                    file_path = os.path.join(data_root, distribution_type,
                                             method, dataset, iteration,
                                             "index_time", "indexing.csv")

                if not os.path.exists(file_path):
                    continue

                try:
                    df = pd.read_csv(file_path)
                    df.columns = df.columns.str.strip()  # 去空格

                    # 兼容列名
                    if "run_time" in df.columns:
                        time_col = "run_time"
                    elif "time" in df.columns:
                        time_col = "time"
                    else:
                        continue

                    # 确保按 Index 排序
                    if "Index" in df.columns:
                        df = df.sort_values("Index")

                    # --- 提取每个 Step 的数据 ---
                    for step in range(1, max_step + 1):
                        current_time = None

                        # 处理 Index 偏移逻辑

                        target_row = df[df["Index"] == step]

                        if not target_row.empty:
                            current_time = target_row[time_col].values[0]

                        if current_time is not None:
                            raw_data.append({
                                "method": method,
                                "dataset": dataset,
                                "step": step,
                                "time": current_time
                            })

                except Exception as e:
                    print(f"读取错误 {file_path}: {e}")

    if not raw_data:
        print("未找到任何有效数据。")
        return pd.DataFrame()

    # 2. 转换为 DataFrame 并计算统计量
    df_raw = pd.DataFrame(raw_data)

    # 计算均值和标准差
    df_stats = df_raw.groupby(["method", "dataset", "step"])["time"].agg(["mean", "std"]).reset_index()

    # 3. 透视表 (Pivot)
    df_pivot = df_stats.pivot(index=["method", "dataset"], columns="step", values=["mean", "std"])

    # 4. 扁平化列名
    new_columns = []
    # df_pivot.columns 是 MultiIndex: [('mean', 1), ('std', 1), ...]
    for stat_type, step_num in df_pivot.columns:
        # 重命名为 step1_mean, step1_std
        new_columns.append(f"step{step_num}_{stat_type}")

    df_pivot.columns = new_columns
    df_pivot = df_pivot.reset_index()

    # ---------------------------------------------------------
    # [核心修改]：四舍五入并转为整数
    # ---------------------------------------------------------
    # 找出所有包含数据的列（除了 method 和 dataset 之外的列）
    numeric_cols = [c for c in df_pivot.columns if c not in ["method", "dataset"]]

    # 1. round(0): 四舍五入到整数位 (此时还是 float, 如 12.0)
    df_pivot[numeric_cols] = df_pivot[numeric_cols].round(2)



    # 5. 调整列顺序 (交替排列 Mean 和 Std)
    sorted_cols = ["method", "dataset"]
    for i in range(1, max_step + 1):
        col_mean = f"step{i}_mean"
        col_std = f"step{i}_std"

        if col_mean in df_pivot.columns:
            sorted_cols.append(col_mean)
        if col_std in df_pivot.columns:
            sorted_cols.append(col_std)

    df_final = df_pivot[sorted_cols]

    # 保存
    save_file = os.path.join(data_root, save_filename)
    df_final.to_csv(save_file, index=False)
    print(f"统计完成 (整数格式)！文件已保存至: {save_file}")

    return df_final


def calculate_experiment_stats(data_root, methods, datasets, distribution_type="iid",
                               save_filename="final_index_stats.csv", max_step=5):
    """
    计算逻辑：
    1. 读取每个 Method + Dataset + Iteration 的文件。
    2. 提取 Step 1 到 Step {max_step} 的运行时间。
    3. [一级聚合]：计算单次 Iteration 内的平均时间 (例如 step1-5 的均值)。
    4. [二级聚合]：计算 3 个 Iteration 结果的均值(Mean)和标准差(Std)。

    结果保留两位小数。
    """

    all_step_data = []

    print(f"开始处理... (Max Step = {max_step})")

    for method in methods:
        for dataset in datasets:
            # 遍历 3 个种子 (Iteration 0, 1, 2)
            for iteration in ["0", "1", "2"]:

                # --- 1. 构建文件路径 ---
                # 根据 method 不同，文件名和路径略有不同
                if method == "XTR":
                    file_path = os.path.join(data_root, distribution_type,
                                             method, dataset, iteration,
                                             "index_time", "index_updata.csv")
                else:
                    file_path = os.path.join(data_root, distribution_type,
                                             method, dataset, iteration,
                                             "index_time", "indexing.csv")

                # 如果文件不存在，跳过
                if not os.path.exists(file_path):
                    # print(f"[Skip] 文件缺失: {file_path}")
                    continue

                try:
                    # --- 2. 读取并清洗数据 ---
                    df = pd.read_csv(file_path)
                    df.columns = df.columns.str.strip()  # 去除列名空格

                    # 统一时间列名
                    if "run_time" in df.columns:
                        time_col = "run_time"
                    elif "time" in df.columns:
                        time_col = "time"
                    else:
                        print(f"[Skip] {file_path} 缺少时间列")
                        continue

                    # 确保按 Index 排序
                    if "Index" in df.columns:
                        df = df.sort_values("Index")

                    # --- 3. 提取指定 Step 的数据 ---
                    # 我们需要收集 step 1 到 max_step 的所有数据
                    valid_times = []

                    for step in range(1, max_step + 1):
                        # 处理 Index 偏移逻辑
                        target_row = df[df["Index"] == step]

                        if not target_row.empty:
                            t = target_row[time_col].values[0]
                            valid_times.append(float(t))

                    # 如果这一轮实验提取到了数据，存入列表
                    # 我们记录每一行数据属于哪个 iteration，方便后续分组
                    for t in valid_times:
                        all_step_data.append({
                            "method": method,
                            "dataset": dataset,
                            "iteration": iteration,
                            "time": t
                        })

                except Exception as e:
                    print(f"[Error] 读取错误 {file_path}: {e}")

    # --- 4. 数据聚合计算 ---
    if not all_step_data:
        print("未找到任何有效数据，请检查路径或文件名。")
        return pd.DataFrame()

    df_raw = pd.DataFrame(all_step_data)

    # 【Step A: 一级聚合】
    # 计算每个 Iteration 内部的平均时间
    # 结果：每个 method+dataset+iteration 只有一个值 (iter_mean_time)
    df_iter_avg = df_raw.groupby(["method", "dataset", "iteration"])["time"].mean().reset_index()
    df_iter_avg.rename(columns={"time": "iter_mean_time"}, inplace=True)

    # 【Step B: 二级聚合】
    # 计算 3 个 Iteration 之间的 均值 和 标准差
    # 结果：每个 method+dataset 只有一行
    df_final = df_iter_avg.groupby(["method", "dataset"])["iter_mean_time"].agg(["mean", "std"]).reset_index()

    # --- 5. 格式化输出 ---
    # 重命名列
    df_final.rename(columns={"mean": "avg_update_time", "std": "std_dev"}, inplace=True)

    # 填充 NaN (如果某个实验只跑成功了1次，std会是NaN，这里填0或者保留NaN看你需求，通常保留NaN)
    # df_final = df_final.fillna(0)

    # 保留两位小数
    cols_to_round = ["avg_update_time", "std_dev"]
    df_final[cols_to_round] = df_final[cols_to_round].round(2)

    # 保存文件
    save_path = os.path.join(data_root, save_filename)
    df_final.to_csv(save_path, index=False, float_format='%.2f')

    print("-" * 30)
    print(f"统计完成！")
    print(f"结果已保存至: {save_path}")
    print("-" * 30)
    print("数据预览:")
    print(df_final.head())

    return df_final
if __name__ == '__main__':
    data_root = r"./results"
    methods = ['PLAID', "IGP", "MUVERA", "Dessert", "XTR", "StreamPLAID"]
    # methods = ["StreamPLAID"]
    datasets = ["scifact", "arguana", "fiqa", "touche2020", "lotte-pooled"]
    # datasets = ["beir-ood", "lotte-ood"]
    # distribution_type = "iid"
    distribution_type = "iid"
    save_filename = "iid_arg_index_time.csv"
    calculate_experiment_stats(data_root, methods, datasets, distribution_type, save_filename)
    # merged_df = merge_index_time_stats_int(data_root, methods, datasets, distribution_type, save_filename)