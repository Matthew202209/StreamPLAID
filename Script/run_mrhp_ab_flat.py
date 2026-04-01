from module.mrhp_ab_flat_retriever import ABFlatRetriever

if __name__ == '__main__':
    config = {
        "run_config": {
            "device": "cuda:3",
            "max_query_len": 32,
            "max_doc_len": 300,
            "batch_size": 32,
            "top_k_token": 256,
            "top_k_doc": 30,
            "pool_factor": 1,
            "kmeans_niters":5,
            "nbits": 4,
            "clustering_mode":"hierarchical",
            "protected_tokens": 0,
            "rank":0,
            "ncells":1,
            "centroid_score_threshold":0.5,
            "update_threshold_radius":0.80,
            "update_radius_start_layer":2,
            "ndocs":1000,
            "beta":0.83,
            "l_max":3,
            "n_init":5,
            "radius_step":0.2,
            "is_from_zero":True,
            "stream_type": "only_corpus_add"
        },
        "model_config": {
            "model_name": "ab_flat",
            "model_path": r"./checkpoints/colbertv2.0",  # 改成你的 checkpoint 路径
            "doc_token_id": "[unused1]",
            "query_token_id": "[unused0]",
            "doc_token": "[D]",
            "query_token": "[Q]",
            "compression_dim": 128,
            "attend_to_mask_tokens": False
        },
        "data_config": {
            "data_root": r"/data/user/cma859/export/ChunmingMA/project_mvdr/baselines/data/ood_process_data",
            "embedding_root": r"./embedding",
            "index_root": r"./index",
            "result_root": r"./result",
            "benchmark_type": "new_label_lotte",
            "dataset": "scifact",
            "iteration": "0",
            "is_ood" : True
        }
    }

    for dataset in ["lotte-ood"]:
        config["data_config"]["dataset"] = dataset
        print(r"\n==== Running dataset: {} ====\n".format(config["data_config"]["dataset"]))
        for iteration in [0,1,2]:
            print(r"\n==== iteration: {} ====\n".format(str(iteration)))
            config["data_config"]["iteration"] = str(iteration)
            retriever = ABFlatRetriever(config)
            retriever.init_experiment()
            retriever.streaming_experiment()
