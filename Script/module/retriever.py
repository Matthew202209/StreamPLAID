import gc
import gzip
import json
import os


import h5py
import numpy as np
import pandas as pd
import scann
from tqdm import tqdm
import torch
from Script.loader.dataloader import CorpusLoader, QueryLoader, StreamingDataLoader
from Script.models.model_factory import ModelsFactory
from Script.module.dynamic_process import dynamic_process_factory
from Script.utils.util_effectiveness import save_effectiveness_metrics
from Script.utils.util_timer import timer
import ir_measures
from ir_measures import nDCG, RR, Success


class Retriever:
    def __init__(self, config):
        self.model = ModelsFactory.build_model(config)
        self.config = config
        self.global_token_vecs_array = None
        self.global_token_doc_ids_array = None
        self._load_data()

    def _load_data(self):
        """
        Loads the corpus, queries, and labels from the specified configuration.
        :return: None
        """
        # Load corpus
        corpus_loader = CorpusLoader()
        self.corpus_dict = corpus_loader.load(self.config)

        # Load queries
        query_loader = QueryLoader()
        self.queries_dict = query_loader.load(self.config)
        self.init_data, self.streaming_data = StreamingDataLoader().load_only_corpus_add(self.config)

    def set_up_embedding_save_file(self, streaming_step):
        model_name = self.config["model_config"].get('model_name', 'xtr')
        root = self.config["data_config"].get('embedding_root', './embedding')
        benchmark_type = self.config["data_config"].get('benchmark_type', 'beir')
        dataset = self.config["data_config"].get('dataset', 'nq')
        save_folder = r"{}/{}/{}/{}/step_{}".format(root, model_name, benchmark_type, dataset, str(streaming_step))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        save_file = os.path.join(save_folder, f"embedding_vectors.h5")
        return save_file

    def set_up_index_save_path(self, streaming_step):
        model_name = self.config["model_config"].get('model_name', 'xtr')
        root = self.config["data_config"].get('index_root', './index')
        benchmark_type = self.config["data_config"].get('benchmark_type', 'beir')
        dataset = self.config["data_config"].get('dataset', 'nq')
        stream_type = self.config["run_config"].get('stream_type', 'only_corpus_add')

        save_folder = r"{}/{}/{}/{}/{}/step_{}".format(root, model_name, benchmark_type, dataset, stream_type, str(streaming_step))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        return save_folder

    def set_up_effectiveness(self):
        model_name = self.config["model_config"].get('model_name', 'xtr')
        root = self.config["data_config"].get('result_root', './index')
        benchmark_type = self.config["data_config"].get('benchmark_type', 'beir')
        dataset = self.config["data_config"].get('dataset', 'nq')
        stream_type = self.config["run_config"].get('stream_type', 'only_corpus_add')
        effectiveness_file = r"{}/{}/{}/{}/{}/effectiveness_metrics.csv".format(root, model_name, benchmark_type, dataset, stream_type)
        return effectiveness_file

    def set_up_retrieve_result_path(self):
        model_name = self.config["model_config"].get('model_name', 'xtr')
        root = self.config["data_config"].get('result_root', './index')
        benchmark_type = self.config["data_config"].get('benchmark_type', 'beir')
        dataset = self.config["data_config"].get('dataset', 'nq')
        stream_type = self.config["run_config"].get('stream_type', 'only_corpus_add')
        analysis_save_folder = r"{}/{}/{}/{}/{}/analysis".format(root, model_name,benchmark_type, dataset, stream_type)
        run_time_save_folder = r"{}/{}/{}/{}/{}/run_time".format(root, model_name,benchmark_type, dataset, stream_type)

        if not os.path.exists(analysis_save_folder):
            os.makedirs(analysis_save_folder, exist_ok=True)
        if not os.path.exists(run_time_save_folder):
            os.makedirs(run_time_save_folder, exist_ok=True)

        return analysis_save_folder, run_time_save_folder


    def load_embedding_vectors(self, save_file=None):
        if os.path.exists(save_file):
            with h5py.File(save_file, 'r') as f:
                all_token_vecs_array = f['token_vecs'][:]  # 二维浮点数组（压缩维度向量）
                all_token_doc_ids = json.loads(f.attrs['all_token_doc_ids'])  # 文档ID索引
                print(r"Load Embedding")
                return all_token_vecs_array, all_token_doc_ids


    def embedding_corpus(self, corpus_dict, streaming_step=0):
        corpus = list(corpus_dict.values())
        corpus_id = list(corpus_dict.keys())
        save_file = self.set_up_embedding_save_file(streaming_step)
        if os.path.exists(save_file):
            print(f"Embedding vectors already exist at {save_file}. Loading existing vectors.")
            all_token_vecs_array, all_token_doc_ids = self.load_embedding_vectors(save_file)
            return all_token_vecs_array, all_token_doc_ids
        else:
            print(f"Running Embedding: {self.config['model_config']['model_name']}")
            print(f"Dataset: {self.config['data_config']['dataset']}")
            print(f"Num of Corpus : {len(corpus)}")
            all_token_vecs_array = []
            with h5py.File(save_file, 'a') as f:
                all_token_vecs = f.create_dataset('token_vecs',
                                                  shape=(0, self.config["model_config"]['compression_dim']),
                                                  maxshape=(None, self.config["model_config"]['compression_dim']),
                                                  chunks=(1000, self.config["model_config"]['compression_dim']),
                                                  # 按需设置分块
                                                  compression='gzip',
                                                  dtype='float32')

                all_token_doc_ids = []
                for batch_idx in tqdm(range(0, len(corpus), self.config["run_config"]["batch_size"])):
                    batch_text = corpus[batch_idx:batch_idx + self.config["run_config"]["batch_size"]]
                    doc_ids_list = [id for id in range(batch_idx, batch_idx + self.config["run_config"]["batch_size"])]
                    token_vecs, token_ids = self.model.encode_doc(batch_text, doc_ids_list)
                    all_token_doc_ids += [corpus_id[idx] for idx in token_ids]
                    all_token_vecs_array.append(token_vecs)
                    all_token_vecs_current_size = all_token_vecs.shape[0]
                    all_token_vecs.resize(all_token_vecs_current_size + token_vecs.shape[0], axis=0)
                    all_token_vecs[all_token_vecs_current_size:, :] = token_vecs
                all_token_vecs_array = np.vstack(all_token_vecs_array)
                f.attrs['all_token_doc_ids'] = json.dumps(all_token_doc_ids)
        return all_token_vecs_array, all_token_doc_ids

    def indexing_corpus(self, token_vecs_array, streaming_step=0):
        # import shutil
        # embedding_file = self.set_up_embedding_save_file(streaming_step)
        index_folder = self.set_up_index_save_path(streaming_step)
        if os.path.exists(os.path.join(index_folder, "dataset.npy")):
            print(f"Index already exists at {index_folder}. Loading existing index.")
            index_folder = os.path.abspath(index_folder)  # 转换为绝对路径
            index = scann.scann_ops_pybind.load_searcher(index_folder)
            return index
        else:
            os.makedirs(index_folder, exist_ok=True)
        index = self.model.indexing(token_vecs_array)
        save_dir = os.path.abspath(index_folder)  # 转换为绝对路径
        index.serialize(save_dir)
        # shutil.copy2(embedding_file, index_folder)
        return index

    def init_experiment(self):
        init_corpus = {d_id: self.corpus_dict[d_id] for d_id in self.init_data["corpus"]}
        init_queries = {q_id: self.queries_dict[q_id] for q_id in self.init_data["queries"]}
        labels_df = pd.DataFrame(self.init_data["labels"])
        init_token_vecs_array, init_token_doc_ids = self.embedding_corpus(init_corpus, streaming_step=0)
        gc.collect()
        torch.cuda.empty_cache()
        index = self.indexing_corpus(init_token_vecs_array, streaming_step=0)
        analysis_save_path = self.run_retrieve(index, init_queries, init_token_doc_ids, streaming_step=0)
        self.evaluate(analysis_save_path, labels_df)


    def evaluate(self, analysis_save_path, labels_df, streaming_step=0):
        effectiveness_file = self.set_up_effectiveness()
        measure = [nDCG @ 10, RR @ 10, Success @ 10]
        faiss_results_pd = pd.DataFrame(list(ir_measures.read_trec_run(analysis_save_path)))
        eval_results = ir_measures.calc_aggregate(measure, labels_df, faiss_results_pd)
        save_results = {}
        for k,v in eval_results.items():
            save_results[k.NAME] = v
        save_effectiveness_metrics(effectiveness_file, streaming_step, save_results)

    def run_retrieve(self, index, queries_dict, init_token_doc_ids, streaming_step=0):
        analysis_save_folder, run_time_save_folder = self.set_up_retrieve_result_path()
        analysis_save_path = os.path.join(analysis_save_folder, f'step_{streaming_step}.run.gz')
        top_k_token = self.config["run_config"]["top_k_token"]
        top_k_doc = self.config["run_config"]["top_k_doc"]
        results = []
        idx = 0
        for q_id, query in tqdm(queries_dict.items(), desc=f"Retrieving Queries at Step {streaming_step}"):
            with timer(idx, f"embedding_{streaming_step}", run_time_save_folder):
                query_vecs_array = self.model.encode_query([query])
            with timer(idx, f"retrieval_{streaming_step}", run_time_save_folder):
                batch_ranking = self.model.retrieve(query_vecs_array, index, init_token_doc_ids, top_k_token=top_k_token, top_k_doc=top_k_doc)
            results.append({
                "q_text": query,
                "query_id": q_id,
                "results": batch_ranking[0]
            })
            idx += 1

        with gzip.open(analysis_save_path, 'wt') as fout:
            for result in results:
                q_idx = result["query_id"]
                for did, rank, score in result["results"]:
                    fout.write(f'{q_idx} 0 {did} {rank} {score} run\n')
        return analysis_save_path

    def dynamic_indexing(self, add_token_vecs_array , add_token_doc_ids_array, streaming_step):
        vec_processor = dynamic_process_factory.build_process(self.config)
        pass

    def streaming_add_remove_experiment(self):
        index_time_folder = self.set_up_index_time_folder()
        dynamic_vector_processor = dynamic_process_factory.build_process(self.config)
        init_vector_file = self.set_up_embedding_save_file(0)
        if os.path.exists(init_vector_file):
            print(f"Embedding vectors already exist at {init_vector_file}. Loading existing vectors.")
            all_token_vecs_array, all_token_doc_ids = self.load_embedding_vectors(init_vector_file)
        else:
            print(f"Embedding vectors do not exist at {init_vector_file}. Please run init experiment.")
            return
        num_step = len(self.merge_data)
        for this_step in range(num_step):
            streaming_step = this_step + 1
            print(f"\n===== 正在进行第 {streaming_step}/{num_step} 组实验 =====")
            with timer(streaming_step, f"index_embedding", index_time_folder):
                add_token_vecs_array, add_token_doc_ids = self.streaming_add_embedding(streaming_step)
            with timer(streaming_step, f"index_updata", index_time_folder):
                filtered_vecs_array, filtered_token_doc_ids = self.get_remove_data(all_token_vecs_array, all_token_doc_ids, streaming_step)
                input_vector = [filtered_vecs_array, add_token_vecs_array, filtered_token_doc_ids, add_token_doc_ids]
                all_token_vecs_array, all_token_doc_ids = dynamic_vector_processor(*input_vector)
                index = self.indexing_corpus(all_token_vecs_array, streaming_step=streaming_step)
            gc.collect()
            torch.cuda.empty_cache()
            this_queries = {q_id: self.queries_dict[q_id] for q_id in self.merge_data[streaming_step]["queries"]}
            labels_df = pd.DataFrame(self.merge_data[streaming_step]["labels"])
            analysis_save_path = self.run_retrieve(index, this_queries, all_token_doc_ids,
                                                   streaming_step=streaming_step)
            self.evaluate(analysis_save_path, labels_df, streaming_step=streaming_step)


    def streaming_experiment(self):
        index_time_folder = self.set_up_index_time_folder()
        dynamic_vector_processor = dynamic_process_factory.build_process(self.config)
        init_vector_file = self.set_up_embedding_save_file(0)
        if os.path.exists(init_vector_file):
            print(f"Embedding vectors already exist at {init_vector_file}. Loading existing vectors.")
            all_token_vecs_array, all_token_doc_ids = self.load_embedding_vectors(init_vector_file)
        else:
            print(f"Embedding vectors do not exist at {init_vector_file}. Please run init experiment.")
            return
        num_step = len(self.merge_data)
        for this_step in range(num_step):
            streaming_step = this_step + 1
            print(f"\n===== 正在进行第 {streaming_step}/{num_step} 组实验 =====")
            with timer(streaming_step, f"index_embedding", index_time_folder):
                add_token_vecs_array, add_token_doc_ids = self.streaming_embedding(streaming_step)
                input_vector = [all_token_vecs_array, add_token_vecs_array, all_token_doc_ids, add_token_doc_ids]
            # Process the input vectors dynamically
            with timer(streaming_step, f"index_updata", index_time_folder):
                all_token_vecs_array, all_token_doc_ids = dynamic_vector_processor(*input_vector)
                index = self.indexing_corpus(all_token_vecs_array, streaming_step=streaming_step)
            gc.collect()
            torch.cuda.empty_cache()
            this_queries = {q_id: self.queries_dict[q_id] for q_id in self.merge_data[streaming_step]["queries"]}
            labels_df = pd.DataFrame(self.merge_data[streaming_step]["labels"])
            analysis_save_path = self.run_retrieve(index, this_queries, all_token_doc_ids, streaming_step=streaming_step)
            self.evaluate(analysis_save_path, labels_df, streaming_step=streaming_step)


    def streaming_embedding(self, streaming_step):
        add_corpus = {d_id: self.corpus_dict[d_id] for d_id in self.merge_data[streaming_step]["corpus"]}
        all_token_vecs_array, all_token_doc_ids = self.embedding_corpus(add_corpus, streaming_step=streaming_step)
        return all_token_vecs_array, all_token_doc_ids

    def streaming_add_embedding(self, streaming_step):
        add_corpus = {d_id: self.corpus_dict[d_id] for d_id in self.merge_data[streaming_step]["corpus_add"]}
        all_token_vecs_array, all_token_doc_ids = self.embedding_corpus(add_corpus, streaming_step=streaming_step)
        return all_token_vecs_array, all_token_doc_ids

    def get_remove_data(self, all_token_vecs_array, all_token_doc_ids, streaming_step):
        remove_corpus = [d_id for d_id in self.merge_data[streaming_step]["corpus_remove"]]
        remove_set = set(remove_corpus)
        remove_vector_indices = [index for index, doc_id in enumerate(all_token_doc_ids) if doc_id in remove_set]
        indices = sorted(set(remove_vector_indices))
        filtered_vecs_array = np.delete(all_token_vecs_array, indices, axis=0)
        filtered_token_doc_ids = [doc_id for doc_id in all_token_doc_ids if doc_id not in remove_set]
        return filtered_vecs_array, filtered_token_doc_ids


    def set_up_index_time_folder(self):
        model_name = self.config["model_config"].get('model_name', 'xtr')
        root = self.config["data_config"].get('result_root', './embedding')
        benchmark_type = self.config["data_config"].get('benchmark_type', 'beir')
        dataset = self.config["data_config"].get('dataset', 'nq')
        streaming_type = self.config["run_config"]["stream_type"]
        index_time_folder = r"{}/{}/{}/{}/index_time/{}".format(root, model_name, benchmark_type, dataset, streaming_type)
        if not os.path.exists(index_time_folder):
            os.makedirs(index_time_folder, exist_ok=True)
        return index_time_folder

    def set_streaming_data_folder(self):
        model_name = self.config["model_config"].get('model_name', 'xtr')
        root = self.config["data_config"].get('embedding_root', './embedding')
        benchmark_type = self.config["data_config"].get('benchmark_type', 'beir')
        dataset = self.config["data_config"].get('dataset', 'nq')
        streaming_type = self.config["run_config"]["stream_type"]
        streaming_folder = r"{}/{}/{}/{}/{}".format(root, model_name, benchmark_type, dataset, streaming_type)

        return streaming_folder



if __name__ == '__main__':
    config = {
        "run_config": {
            "device": "cuda:0",
            "max_query_len": 32,
            "max_doc_len": 300,
            "batch_size":32,
            "top_k_token":256,
            "top_k_doc": 30,
            "stream_type" :"only_corpus_add"
        },
        "model_config": {
            "model_name":"xtr",
            "model_path": r"/media/xianpe/17fee0f7-8905-40ec-881c-2477771df149/chunming/project/MVDR_bm/exp_stream/checkpoints/xtr-base-en",
            "compression_dim": 128
        },
        "data_config":{
            "data_root": r"/media/xianpe/17fee0f7-8905-40ec-881c-2477771df149/chunming/project/MVDR_bm/exp_stream/data",
            "embedding_root":r"/media/xianpe/17fee0f7-8905-40ec-881c-2477771df149/chunming/project/MVDR_bm/exp_stream/embedding",
            "index_root": r"/media/xianpe/17fee0f7-8905-40ec-881c-2477771df149/chunming/project/MVDR_bm/exp_stream/index",
            "result_root": r"/media/xianpe/17fee0f7-8905-40ec-881c-2477771df149/chunming/project/MVDR_bm/exp_stream/result",
            "benchmark_type": "beir",
            "dataset": 'scifact'
        }
    }
    retriever = Retriever(config)
    retriever.init_experiment()
    retriever.streaming_add_remove_experiment()


























    #
    #
    # def indexing(self):
    #     pass
    #
    # def dynamic_indexing(self):
    #     pass
    #









