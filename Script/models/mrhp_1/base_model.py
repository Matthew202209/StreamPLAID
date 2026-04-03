import string

from torch.nn.utils.rnn import pad_sequence
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer
from torch import nn
import torch


class HF_ColBERT(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"cls"]
    def __init__(self, config, compression_dim=128):
        super().__init__(config)
        self.config = config
        self.dim = compression_dim
        self.linear = nn.Linear(self.config.hidden_size, self.dim , bias=False)
        setattr(self, self.base_model_prefix, BertModel(self.config))
        self.init_weights()

    @property
    def LM(self):
        base_model_prefix = getattr(self, "base_model_prefix")
        return getattr(self, base_model_prefix)

    @classmethod
    def from_pretrained(cls, colbert_config):

        obj = super().from_pretrained(colbert_config["model_config"]["model_path"], compression_dim=colbert_config["model_config"]["compression_dim"])
        obj.base = colbert_config["model_config"]["model_path"]
        return obj

    @staticmethod
    def raw_tokenizer_from_pretrained(name_or_path):
        if name_or_path.endswith('.dnn'):
            dnn = torch.load(name_or_path, map_location='cpu')
            base = dnn.get('arguments', {}).get('model', 'bert-base-uncased')
            obj = AutoTokenizer.from_pretrained(base)
            obj.base = base
            return obj

        obj = AutoTokenizer.from_pretrained(name_or_path)
        obj.base = name_or_path

        return obj



class BaseColBERT(torch.nn.Module):
    """
    Shallow module that wraps the ColBERT parameters, custom configuration, and underlying tokenizer.
    This class provides direct instantiation and saving of the model/colbert_config/tokenizer package.
    Like HF, evaluation mode is the default.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["run_config"]["device"]
        self.model = HF_ColBERT.from_pretrained(config)
        self.model.to(config["run_config"]["device"])
        self.raw_tokenizer = AutoTokenizer.from_pretrained(self.config["model_config"]["model_path"])
        self.eval()


    @property
    def bert(self):
        return self.model.LM

    @property
    def linear(self):
        return self.model.linear

    @property
    def score_scaler(self):
        return self.model.score_scaler


class ColBERT(BaseColBERT):
    def __init__(self, config):
        super().__init__(config)
        self.pad_token = self.raw_tokenizer.pad_token_id
        self.skiplist = {w: True
                         for symbol in string.punctuation
                         for w in [symbol, self.raw_tokenizer.encode(symbol, add_special_tokens=False)[0]]}

    def mask(self, input_ids, skiplist):
        mask = [[(x not in skiplist) and (x != self.pad_token) for x in d] for d in input_ids.cpu().tolist()]
        return mask

    def forward(self, Q, D):
        Q = self.query(*Q)
        D, D_mask = self.doc(*D, keep_dims='return_mask')

        # Repeat each query encoding for every corresponding document.
        Q_duplicated = Q.repeat_interleave(self.colbert_config.nway, dim=0).contiguous()
        scores = self.score(Q_duplicated, D, D_mask)

        if self.colbert_config.use_ib_negatives:
            ib_loss = self.compute_ib_loss(Q, D, D_mask)
            return scores, ib_loss

        return scores

    # def query(self, input_ids, attention_mask):
    #     input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
    #     Q = self.bert(input_ids, attention_mask=attention_mask)[0]
    #     Q = self.linear(Q)
    #
    #     mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
    #     Q = Q * mask
    #
    #     return torch.nn.functional.normalize(Q, p=2, dim=2)

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

        # 1. 得到 Token-level 表示
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]  # [B, L, H]
        Q = self.linear(Q)  # [B, L, D]

        # 2. mask: True = 保留，False = 删除
        mask_bool = attention_mask.bool()  # attention_mask 本身就是 1/0

        compact_outputs = []
        for b in range(Q.size(0)):
            q_b = Q[b][mask_bool[b]]  # 只保留 mask=1 的 token，形状变为 [L_b, D]
            q_b = torch.nn.functional.normalize(q_b, p=2, dim=1)
            compact_outputs.append(q_b)

        padded = pad_sequence(compact_outputs, batch_first=True)

        return padded

    def doc(self, input_ids, attention_mask, keep_dims=True):
        assert keep_dims in [True, False, 'return_mask']

        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)
        mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)
        if self.device!="cpu":
            D = D.half()

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        elif keep_dims == 'return_mask':

            return D, mask.bool()

        return D