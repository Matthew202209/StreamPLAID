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
        benchmark_type = config["data_config"].get('benchmark_type', 'beir')
        dataset = config["data_config"].get('dataset', 'nq')
        interation = config["data_config"].get('iteration', None)
        if config["data_config"].get('is_ood', False):
            corpus_folder = r"{}/{}/corpus/{}".format(root, benchmark_type, str(interation))
        else:
            corpus_folder = r"{}/{}/corpus".format(root, benchmark_type)
        data_file = r"{}/{}.jsonl".format(corpus_folder, dataset)
        output_linecnt = subprocess.check_output(["wc", "-l", data_file]).decode("utf-8")
        print("line cnt", output_linecnt)
        all_linecnt = int(output_linecnt.split()[0])
        pbar = tqdm(total=all_linecnt, desc=f"load data from {data_file}")
        corpus_dict = {}
        with open(data_file, encoding='utf8') as fIn:
            for line in fIn:
                line = ujson.loads(line)
                doc_id, text = self._preprocess(line)
                if doc_id is None:
                    continue
                corpus_dict[doc_id] = text
                pbar.update(1)
        return corpus_dict

    def _preprocess(self, data):
        # Implement logic to preprocess corpus data
        if '_id' in data.keys():
            if len(data.get("text")) < 3:
                print(data.get("_id"))
                return None, None
            doc_id = data.get("doc_id")
            text = data.get("text")
            return doc_id, text
            # Process the corpus data

        elif "doc_id" in data.keys():
            if len(data.get("text")) < 3:
                print(data.get("doc_id"))
                return None, None
            # Process the corpus data
            doc_id = data.get("doc_id")
            text = data.get("text")
            return doc_id, text
        else:
            print("Unknown corpus format")


class QueryLoader(DataLoader):
    def load(self, config):
        root = config["data_config"].get('data_root', 'Benchmark/data')
        benchmark_type = config["data_config"].get('benchmark_type', 'beir')
        dataset = config["data_config"].get('dataset', 'nq')
        interation = config["data_config"].get('iteration', None)
        if config["data_config"].get('is_ood', False):
            query_folder = r"{}/{}/queries/{}".format(root, benchmark_type, str(interation))
        else:
            query_folder = r"{}/{}/queries".format(root, benchmark_type)
        if benchmark_type == "lotte":
            queries_style = config["data_config"].get('queries_style', 'forum')
            query_file = r"{}/{}_{}.jsonl".format(query_folder, dataset, queries_style)
        else:
            query_file = r"{}/{}.jsonl".format(query_folder, dataset)


        output_linecnt = subprocess.check_output(["wc", "-l", query_file]).decode("utf-8")
        print("line cnt", output_linecnt)
        all_linecnt = int(output_linecnt.split()[0])
        pbar = tqdm(total=all_linecnt, desc=f"load data from {query_file}")
        queries_dict = {}

        for line in open(query_file, 'r', encoding="utf-8"):
            query = ujson.loads(line)
            q_id, text = self._preprocess(query)
            queries_dict[q_id] = text
            pbar.update(1)
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

class LabelsLoader(DataLoader):
    def load(self, config):
        root = config["data_config"].get('data_root', 'Benchmark/data')
        benchmark_type = config["data_config"].get('benchmark_type', 'beir')
        dataset = config["data_config"].get('dataset', 'nq')
        if benchmark_type == "lotte":
            queries_style = config["data_config"].get('queries_style', 'forum')
            labels_file = r"{}/{}/labels/{}_{}.csv".format(root, benchmark_type, dataset, queries_style)
        else:
            labels_file = r"{}/{}/labels/{}.csv".format(root, benchmark_type, dataset)
        labels = pd.read_csv(labels_file)
        labels = self._preprocess(labels)
        return labels

    def _preprocess(self, data):
        # Implement logic to preprocess labels data
        data["query_id"] = data["query_id"].astype(str)
        data["doc_id"] = data["doc_id"].astype(str)
        return data

class StreamingDataLoader:
    def load_only_corpus_add(self, config):
        root = config["data_config"].get('data_root', 'Benchmark/data')
        benchmark_type = config["data_config"].get('benchmark_type', 'beir')
        dataset = config["data_config"].get('dataset', 'nq')
        iteration = config["data_config"].get('iteration', None)


        if benchmark_type == "lotte":
            dataset = r"{}_{}".format(dataset, config["data_config"].get('queries_style', 'forum'))
        init_file = r"{}/{}/stream/{}/{}/0.json".format(root, benchmark_type, dataset, iteration)
        with open(init_file, "r", encoding="utf-8") as f:
            init_data = json.load(f)
        stream_folder_path = r"{}/{}/stream/{}/{}".format(root, benchmark_type, dataset, iteration)
        merged_data = {}
        for file_name in sorted(os.listdir(stream_folder_path)):
            if file_name.endswith(".json"):
                file_path = os.path.join(stream_folder_path, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    streaming_step = int(file_name.split('.')[0])  # 提取文件名中的数字作为步骤
                    merged_data[streaming_step] = data  # 用文件名作为 key
        return init_data, merged_data

    def load_only_corpus_add_remove(self, config):
        root = config["data_config"].get('data_root', 'Benchmark/data')
        benchmark_type = config["data_config"].get('benchmark_type', 'beir')
        dataset = config["data_config"].get('dataset', 'nq')

        init_file = r"{}/{}/stream/{}/0.json".format(root, benchmark_type, dataset)
        with open(init_file, "r", encoding="utf-8") as f:
            init_data = json.load(f)

        stream_folder_path = r"{}/{}/stream/{}/only_corpus_add_remove".format(root, benchmark_type, dataset)
        merged_data = {}
        for file_name in sorted(os.listdir(stream_folder_path)):
            if file_name.endswith(".json"):
                file_path = os.path.join(stream_folder_path, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    streaming_step = int(file_name.split('.')[0])  # 提取文件名中的数字作为步骤
                    merged_data[streaming_step] = data  # 用文件名作为 key
        return init_data, merged_data


if __name__ == '__main__':
    config = {
        "data_config": {
            "data_root": "../data",
            "benchmark_type": "beir",
            "dataset": "nq"
        }
    }

    corpus_loader = CorpusLoader()
    corpus, corpus_id = corpus_loader.load(config)
    print("Corpus:", corpus[:5])
    print("Corpus IDs:", corpus_id[:5])

    query_loader = QueryLoader()
    queries, queries_id = query_loader.load(config)
    print("Queries:", queries[:5])
    print("Query IDs:", queries_id[:5])

    labels_loader = LabelsLoader()
    labels = labels_loader.load(config)
    print("Labels:", labels.head())