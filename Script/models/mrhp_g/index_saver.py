import json
import os
import pickle
import queue
import ujson
import threading
import torch

from contextlib import contextmanager

from models.mrhp_g.residual import ResidualCodec


class IndexSaver():
    def __init__(self, config):
        self.config = config

    def save_codec(self, codec, index_path):
        print(f"#> Saving codec...")
        codec.save(index_path=index_path)

    def load_codec(self, index_path):
        return ResidualCodec.load(index_path=index_path)

    def try_load_codec(self, index_path):
        try:
            ResidualCodec.load(index_path=index_path)
            return True
        except Exception as e:
            return False

    def check_chunk_exists(self, chunk_idx):
        # TODO: Verify that the chunk has the right amount of data?

        doclens_path = os.path.join(self.config.index_path_, f'doclens.{chunk_idx}.json')
        if not os.path.exists(doclens_path):
            return False

        metadata_path = os.path.join(self.config.index_path_, f'{chunk_idx}.metadata.json')
        if not os.path.exists(metadata_path):
            return False

        path_prefix = os.path.join(self.config.index_path_, str(chunk_idx))
        codes_path = f'{path_prefix}.codes.pt'
        if not os.path.exists(codes_path):
            return False

        residuals_path = f'{path_prefix}.residuals.pt'  # f'{path_prefix}.residuals.bn'
        if not os.path.exists(residuals_path):
            return False

        return True

    @contextmanager
    def thread(self):
        self.codec = self.load_codec()

        self.saver_queue = queue.Queue(maxsize=3)
        thread = threading.Thread(target=self._saver_thread)
        thread.start()

        try:
            yield

        finally:
            self.saver_queue.put(None)
            thread.join()

            del self.saver_queue
            del self.codec

    def save_chunk(self, index_folder_path, compressed_embs, doclens):
        print(f"#>Saving chunk...")
        self._write_chunk_to_disk(index_folder_path, compressed_embs, doclens)

    def _saver_thread(self):
        for args in iter(self.saver_queue.get, None):
            self._write_chunk_to_disk(*args)

    def save_ivf(self, index_path, ivf_dict):
        optimized_ivf_path = os.path.join(index_path, 'ivf_dict.pkl')
        # 注意：这里必须用 'wb' (write binary) 二进制写入模式
        with open(optimized_ivf_path, 'wb') as f:
            pickle.dump(ivf_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"成功保存 IVF 数据至: {optimized_ivf_path}")


    def save_metadata(self, index_path, metadata):
        metadata_path = os.path.join(index_path, 'metadata.json')
        with open(metadata_path, 'w') as output_metadata:
            ujson.dump(metadata, output_metadata)
        print(f"#> Saved metadata to {metadata_path}")

    def _write_chunk_to_disk(self, index_folder_path, compressed_embs, doclens):
        path_prefix = os.path.join(index_folder_path, str(0))
        compressed_embs.save(path_prefix)

        doclens_path = os.path.join(index_folder_path, f'doclens.json')
        with open(doclens_path, 'w') as output_doclens:
            ujson.dump(doclens, output_doclens)


