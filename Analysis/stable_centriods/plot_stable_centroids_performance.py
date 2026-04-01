
if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # ================= 1. 配置与映射 =================

    # 更新映射：w/o Update -> w/o Re-cluster
    method_name_map = {
        'IGP(Full)': 'IGP (Full)',
        'IGP(wo_update_centriods)': 'IGP (w/o Re-cluster)',
        'PLAID(Full)': 'PLAID (Full)',
        'PLAID(wo_update_centriods)': 'PLAID (w/o Re-cluster)',
        'StreamPLAID': 'StreamPLAID (Ours)'
    }

    # 数据集标准名映射
    dataset_name_map = {
        'beir-ood': 'BEIR-OOD',
        'ood_dataset': 'BEIR-OOD',
        'ood': 'BEIR-OOD',
        'lotte-ood': 'LoTTE-OOD'
    }

    file_path = "./results/stable_centroids_arg_performance.csv"


    # ================= 2. 数据处理与逻辑计算 =================

    def process_ablation_tables(filepath):
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            # 如果没有文件，这里提供一个包含两个数据集的模拟数据结构
            print("!! 找不到文件，使用模拟多数据集数据...")
            methods_raw = list(method_name_map.keys())
            datasets = ['beir-ood', 'lotte-ood']
            data = []
            for ds in datasets:
                for m in methods_raw:
                    base = 0.45 if 'PLAID' in m else 0.40
                    if 'wo' in m: base -= 0.04
                    if 'Stream' in m: base = 0.52
                    data.append({
                        'Method': m, 'Dataset': ds,
                        'Task0_Mean': base + 0.01, 'Task1_Mean': base + 0.02,
                        'Task2_Mean': base, 'Task3_Mean': base - 0.01, 'Task4_Mean': base - 0.02
                    })
            df = pd.DataFrame(data)

        # 1. 基础转换
        df['Dataset'] = df['Dataset'].map(lambda x: dataset_name_map.get(x.lower(), x))
        df['Method'] = df['Method'].map(method_name_map)

        task_cols = [c for c in df.columns if "_Mean" in c]
        task_labels = [c.replace("_Mean", "") for c in task_cols]

        # 2. 按数据集拆分处理
        for ds_name, ds_df in df.groupby('Dataset'):
            print(f"\n" + "=" * 80)
            print(f" Dataset: {ds_name}")
            print("=" * 80)

            # 将 Method 设为索引方便查询
            ds_df = ds_df.set_index('Method')

            final_rows = []

            # 定义对比组逻辑
            # 结构: (组名, Full方法名, Ablation方法名, Ours名)
            groups = [
                ("PLAID Variants", "PLAID (Full)", "PLAID (w/o Re-cluster)", "StreamPLAID (Ours)"),
                ("IGP Variants", "IGP (Full)", "IGP (w/o Re-cluster)", "StreamPLAID (Ours)")
            ]

            for group_label, full_m, wo_m, ours_m in groups:
                # 提取各行数据
                try:
                    row_full = ds_df.loc[full_m, task_cols]
                    row_wo = ds_df.loc[wo_m, task_cols]
                    row_ours = ds_df.loc[ours_m, task_cols]
                except KeyError as e:
                    print(f"缺失方法数据: {e}")
                    continue

                # 构建展示行
                # 1. Full 行 (只显示均值)
                res_full = {"Method": full_m}
                for col, label in zip(task_cols, task_labels):
                    res_full[label] = f"{row_full[col]:.4f}"
                final_rows.append(res_full)

                # 2. w/o 行 (显示均值及相比 Full 的变动)
                res_wo = {"Method": wo_m}
                for col, label in zip(task_cols, task_labels):
                    diff = row_wo[col] - row_full[col]
                    res_wo[label] = f"{row_wo[col]:.4f} ({diff:+.4f})"
                final_rows.append(res_wo)

                # 3. StreamPLAID 行 (显示均值及相比 Full 的变动)
                res_ours = {"Method": ours_m}
                for col, label in zip(task_cols, task_labels):
                    diff = row_ours[col] - row_full[col]
                    res_ours[label] = f"{row_ours[col]:.4f} ({diff:+.4f})"
                final_rows.append(res_ours)

                # 插入一个空行或分割标记
                final_rows.append({"Method": "-" * 20})

            # 打印表格
            output_table = pd.DataFrame(final_rows)
            # 移除最后一行多余的分隔符
            if output_table.iloc[-1]["Method"].startswith("-"):
                output_table = output_table.iloc[:-1]

            print(output_table.to_string(index=False))


    # 执行
    process_ablation_tables(file_path)