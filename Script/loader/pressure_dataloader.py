import glob
import json
import os
import subprocess
from abc import ABC, abstractmethod

import pandas as pd
from tqdm import tqdm
import ujson


class DataLoader(ABC):
    @abstractmethod
    def load(self, config):
        pass

    @abstractmethod
    def _preprocess(self, data):
        pass

class CorpusLoader(DataLoader):
    def load(self, config):
        root = config["data_config"].get('data_root', '/raw_data')
        benchmark_type = config["data_config"].get('benchmark_type', 'pressure_data')

        corpus_folder = r"{}/{}/corpus".format(root, benchmark_type)

        # 1. 获取该目录下所有的 .jsonl 文件路径
        jsonl_files = glob.glob(os.path.join(corpus_folder, "*.jsonl"))

        if not jsonl_files:
            print(f"Warning: No .jsonl files found in {corpus_folder}")
            return {}

        corpus_dict = {}
        total_lines = 0
        # 2. 预先统计所有文件的总行数，以便 tqdm 进度条准确（保留了你原本的 wc -l 逻辑）
        print("Calculating total lines across all files...")
        for file_path in jsonl_files:
            try:
                output_linecnt = subprocess.check_output(["wc", "-l", file_path]).decode("utf-8")
                total_lines += int(output_linecnt.split()[0])
            except Exception as e:
                # 如果 wc 命令失败（例如在 Windows 上），可以选择跳过统计或用 Python 读取
                print(f"Could not count lines for {file_path}: {e}")

        print(f"Total lines to process: {total_lines}")

        # 3. 初始化进度条
        pbar = tqdm(total=total_lines, desc="Loading all corpus files")

        # 4. 遍历所有文件并读取
        for data_file in jsonl_files:
            with open(data_file, encoding='utf8') as fIn:
                for line in fIn:
                    try:
                        line = ujson.loads(line)
                        doc_id, text = self._preprocess(line)

                        if doc_id is None:
                            continue

                        # 合并入大字典
                        corpus_dict[doc_id] = text
                    except ValueError:
                        # 防止空行或格式错误导致崩溃
                        continue
                    finally:
                        # 无论成功与否，都更新进度条（或者只在成功时更新，看你需求）
                        pbar.update(1)

        pbar.close()
        return corpus_dict

    def _preprocess(self, data):
        # Implement logic to preprocess corpus data
        if 'doc_id' in data.keys():
            if len(data.get("text")) < 3:
                print(data.get("_id"))
                return None, None
            doc_id = data.get("doc_id")
            text = data.get("text")
            return doc_id, text


class QueryLoader(DataLoader):
    def load(self, config):
        root = config["data_config"].get('data_root', 'Benchmark/data')
        benchmark_type = config["data_config"].get('benchmark_type', 'pressure_data')

        query_folder = r"{}/{}/queries".format(root, benchmark_type)

        # 2. 获取该目录下所有的 .jsonl 文件
        # 原本针对 lotte 的文件名拼接逻辑被移除，改为直接读取文件夹内所有文件
        query_files = glob.glob(os.path.join(query_folder, "*.jsonl"))

        if not query_files:
            print(f"Warning: No .jsonl files found in {query_folder}")
            return {}

        # 3. 统计所有文件的总行数 (为了进度条)
        total_lines = 0
        print(f"Found {len(query_files)} query files. Calculating total lines...")

        for q_file in query_files:
            try:
                # 使用系统命令 wc -l 快速统计行数
                output_linecnt = subprocess.check_output(["wc", "-l", q_file]).decode("utf-8")
                total_lines += int(output_linecnt.split()[0])
            except Exception as e:
                print(f"Error counting lines for {q_file}: {e}")

        print(f"Total queries to load: {total_lines}")

        # 4. 初始化进度条和大字典
        pbar = tqdm(total=total_lines, desc="Loading all queries")
        queries_dict = {}

        # 5. 遍历每个文件并读取内容
        for q_file in query_files:
            with open(q_file, 'r', encoding="utf-8") as f:
                for line in f:
                    try:
                        query = ujson.loads(line)
                        q_id, text = self._preprocess(query)

                        if q_id is not None:
                            queries_dict[q_id] = text
                    except ValueError:
                        continue
                    finally:
                        pbar.update(1)

        pbar.close()
        return queries_dict

    def _preprocess(self, data):
        # Implement logic to preprocess query data
        if 'query_id' in data.keys():
            if len(data.get("text")) < 3:
                print(data.get("query_id"))
                return None
            query_id = data.get("query_id")
            text = data.get("text")
            return query_id, text

        return data  # Assuming data is already in the desired format

class StreamLoader(DataLoader):
    def load(self, config):
        root = config["data_config"].get('data_root', '/raw_data')
        benchmark_type = config["data_config"].get('benchmark_type', 'pressure_data')
        stream_rate = config["data_config"].get('stream_rate', 100)
        init_domain = config["data_config"].get('init_domain', 'lifestyle')

        stream_folder = r"{}/{}/stream".format(root, benchmark_type)
        # 注意后缀改成了 .json
        stream_file = r"{}/stream_{}_{}.json".format(stream_folder, init_domain, str(stream_rate))

        if not os.path.exists(stream_file):
            print(f"Warning: File not found: {stream_file}")
            return {}

        print(f"Loading stream data from: {stream_file} ...")

        # 1. 一次性读取整个 JSON 文件
        try:
            with open(stream_file, 'r', encoding='utf-8') as f:
                data = ujson.load(f)

        except ValueError as e:
            print(f"Error parsing JSON: {e}")
            return {}
        return data

    def _preprocess(self, data):
        pass




if __name__ == '__main__':
    config = {
        "data_config": {
            "data_root": "/media/xianpe/17fee0f7-8905-40ec-881c-2477771df149/chunming/project_streaming_mvdr/baselines/data",
            "benchmark_type": "pressure_data",
            "init_domain": "lifestyle",
            "stream_rate": 100
        }
    }
    corpus_loader = CorpusLoader()
    queries_loader = QueryLoader()
    stream_loader = StreamLoader()

    corpus = corpus_loader.load(config)
    queries = queries_loader.load(config)
    stream = stream_loader.load(config)
    print(1)

    print(f"Loaded {len(corpus)} documents and {len(queries)} queries.")