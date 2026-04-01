if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # ================= 1. 配置区域 =================

    file_path = "./results/ab_arg_performance.csv"

    # --- A. 核心映射 (已更新原名与展示名) ---
    method_name_map = {
        'w_MRHP&Repairing': 'StreamPLAID (Full)',
        'wo_Repairing': 'w/o Local Repairing',
        'wo_MRHP': 'w/o MRHP'
    }

    # --- B. 数据集名称映射 (确保完美识别 beir-ood 和 lotte-ood) ---
    dataset_name_map = {
        'beir-ood': 'BEIR-OOD',
        'ood_dataset': 'BEIR-OOD',
        'ood': 'BEIR-OOD',
        'BEIR-OOD': 'BEIR-OOD',
        'lotte-ood': 'LoTTE-OOD',
        'lotte': 'LoTTE-OOD',
        'LoTTE-OOD': 'LoTTE-OOD'
    }

    # --- C. 排序逻辑：Full 放在最上面作为基准 ---
    method_order = [
        'StreamPLAID (Full)',
        'w/o Local Repairing',
        'w/o MRHP'
    ]


    # ================= 2. 数据处理与表格生成 =================

    def process_ablation_comparison_table(filepath):
        try:
            df = pd.read_csv(filepath)
            print(f">> 成功读取文件: {filepath}")
        except FileNotFoundError:
            print("!! 找不到文件，生成模拟数据进行演示...")
            # 模拟数据：包含 BEIR-OOD 和 LoTTE-OOD 两个数据集，使用新的原名
            data = []
            for ds in ['beir-ood', 'lotte-ood']:
                for m_raw in method_name_map.keys():
                    base = 0.55 if 'beir' in ds else 0.48
                    if 'wo_Repairing' in m_raw: base -= 0.08
                    if 'wo_MRHP' in m_raw: base -= 0.12

                    data.append({
                        'Method': m_raw,
                        'Dataset': ds,
                        'Task0_Mean': base + 0.02, 'Task1_Mean': base + 0.05,
                        'Task2_Mean': base + 0.01, 'Task3_Mean': base - 0.01, 'Task4_Mean': base - 0.03
                    })
            df = pd.DataFrame(data)

        # 1. 规范化数据集名称并映射方法名
        if 'Dataset' in df.columns:
            df['Dataset'] = df['Dataset'].map(
                lambda x: dataset_name_map.get(str(x).lower(), dataset_name_map.get(x, x)))
        else:
            df['Dataset'] = 'Unknown Dataset'

        df['Method_Display'] = df['Method'].map(lambda x: method_name_map.get(x, x))

        # 2. 提取所有 Task 列
        task_mean_cols = [c for c in df.columns if "_Mean" in c]
        task_labels = [c.replace("_Mean", "") for c in task_mean_cols]

        # 3. 按数据集分组处理
        datasets = df['Dataset'].unique()

        for ds in datasets:
            print(f"\n" + "=" * 95)
            print(f" Ablation Study Comparison: {ds}")
            print("=" * 95)

            ds_df = df[df['Dataset'] == ds].set_index('Method_Display')

            # 提取基准值 (Full 版)
            try:
                base_row = ds_df.loc['StreamPLAID (Full)', task_mean_cols]
            except KeyError:
                print(f"警告: 数据集 {ds} 中缺失 StreamPLAID (Full) 基准数据，跳过。\n")
                continue

            final_table_rows = []

            for m_name in method_order:
                if m_name not in ds_df.index: continue

                current_row = ds_df.loc[m_name, task_mean_cols]
                row_data = {"Method": m_name}

                for col, label in zip(task_mean_cols, task_labels):
                    mean_val = current_row[col]

                    if m_name == 'StreamPLAID (Full)':
                        # 基准行只显示数值
                        row_data[label] = f"{mean_val:.4f}"
                    else:
                        # 消融行显示数值和变动 (Delta)
                        delta = mean_val - base_row[col]
                        row_data[label] = f"{mean_val:.4f} ({delta:+.4f})"

                final_table_rows.append(row_data)

            # 打印结果表格
            if final_table_rows:
                res_df = pd.DataFrame(final_table_rows)
                print(res_df.to_string(index=False))
            print("-" * 95)


    # 执行
    process_ablation_comparison_table(file_path)