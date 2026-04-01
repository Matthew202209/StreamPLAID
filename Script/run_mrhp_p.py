from module.mrhp_g_retriever import PlaidHDRCRetriever
from module.mrhp_pressure_retriever import MRHPPressureRetriever
from module.mrhp_pressure_retriever_up import MRHPPressureRetrieverUp

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
            "update_threshold_radius":0.85,
            "update_radius_start_layer":4,
            "ndocs":1000,
            "beta":0.83,
            "l_max":3,
            "radius_step":0.2,
            "is_from_zero":True,
            "stream_type": "only_corpus_add",
            "max_capacity": 500000000,
            "G_th" : 0.008,
            "pressure_type": "stream_pressure"
        },
        "model_config": {
            "model_name": "plaid_hdrc_g",
            "model_path": r"./checkpoints/colbertv2.0",  # 改成你的 checkpoint 路径
            "doc_token_id": "[unused1]",
            "query_token_id": "[unused0]",
            "doc_token": "[D]",
            "query_token": "[Q]",
            "compression_dim": 128,
            "attend_to_mask_tokens": False,
        },
        "data_config": {
            "data_root": r"/data/user/cma859/export/ChunmingMA/project_mvdr/baselines/data",
            "embedding_root": r"/data/user/cma859/export/ChunmingMA/project_mvdr/mrhp/embedding",
            "index_root": r"/data/user/cma859/export/ChunmingMA/project_mvdr/mrhp/index",
            "result_root": r"/data/user/cma859/export/ChunmingMA/project_mvdr/mrhp/result",
            "benchmark_type": "pressure_data_lotte",
            "init_domain": "science",
            "stream_rate": 20
        }
    }
    for stream_rate in [10]:
        config["data_config"]["stream_rate"] = stream_rate
        print(r"\n==== Running stream_rate: {} ====\n".format(str(stream_rate)))
        retriever = MRHPPressureRetrieverUp(config)
        retriever.init_experiment()
        retriever.streaming_experiment()
