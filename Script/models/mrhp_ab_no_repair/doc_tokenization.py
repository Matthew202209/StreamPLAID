from models.mrhp_ab_no_repair.base_model import HF_ColBERT
from models.mrhp_ab_no_repair.util import _insert_prefix_token, _split_into_batches, _sort_by_length


class DocTokenizer():
    def __init__(self, config):
        self.tok = HF_ColBERT.raw_tokenizer_from_pretrained(config["model_config"]["model_path"])

        self.config = config
        self.doc_maxlen = config["run_config"]["max_doc_len"]

        self.D_marker_token, self.D_marker_token_id = config["model_config"]["doc_token"], self.tok.convert_tokens_to_ids(
            config["model_config"]["doc_token_id"])
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.device = config["run_config"]["device"]

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        tokens = [self.tok.tokenize(x, add_special_tokens=False).to(self.device) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.D_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        ids = self.tok(batch_text, add_special_tokens=False).to(self.device)['input_ids']

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.D_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix for lst in ids]

        return ids

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        obj = self.tok(batch_text, padding='longest', truncation='longest_first',
                       return_tensors='pt', max_length=(self.doc_maxlen - 1)).to(self.device)

        ids = _insert_prefix_token(obj['input_ids'], self.D_marker_token_id)
        mask = _insert_prefix_token(obj['attention_mask'], 1)

        if bsize:
            ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
            batches = _split_into_batches(ids, mask, bsize)
            return batches, reverse_indices

        return ids, mask