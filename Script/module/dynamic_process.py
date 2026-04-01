import json
import os

import h5py
import numpy as np


def only_corpus_add_vectors_process(*args):
    args = list(args)
    a = args[0]
    b = args[1]
    c = args[2]
    d = args[3]
    return np.vstack([a,b]), np.concatenate([c,d])

def only_corpus_add_doc_process(*args):
    args = list(args)
    return  args[0] + args[1]

class dynamic_process_factory:
    @staticmethod
    def build_process(config):
        process_name = config["run_config"]["stream_type"]
        if process_name == "only_corpus_add" or process_name == "only_corpus_add_remove":

            return only_corpus_add_vectors_process
        else:
            raise ValueError(f"Process {process_name} is not supported.")

class dynamic_process_doc_id_factory:
    @staticmethod
    def build_process(config):
        process_name = config["run_config"]["stream_type"]
        if process_name == "only_corpus_add" or process_name == "only_corpus_add_remove":
            return only_corpus_add_doc_process
        else:
            raise ValueError(f"Process {process_name} is not supported.")