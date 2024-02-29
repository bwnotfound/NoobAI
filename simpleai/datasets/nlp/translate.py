import torch
from torch.utils.data import Dataset
import tiktoken
import pandas as pd


class TranslateDataset(Dataset):
    def __init__(self, encoder: tiktoken.Encoding, data_path="./datasets/fra.txt"):
        self.dataset = pd.read_csv(data_path, delimiter='\t', header=None)
        self.dataset = self.dataset.iloc[:, :2]
        self.encoder = encoder

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.iloc[idx, 0], self.dataset.iloc[idx, 1]

    def collate_fn(self, batchs, len_align=True):
        src = []
        tgt = []
        for src_, tgt_ in batchs:
            src.append(self.encoder.encode_ordinary(src_))
            tgt.append(self.encoder.encode_ordinary(tgt_))
        if len_align:
            max_len1 = max([len(s) for s in src])
            max_len2 = max([len(t) for t in tgt])
            max_len = max(max_len1, max_len2)
        src = [s + [self.encoder.eot_token] * (max_len - len(s)) for s in src]
        src = torch.tensor(src)
        if not len_align:
            max_len = max([len(t) for t in tgt])
        tgt = [t + [self.encoder.eot_token] * (max_len - len(t)) for t in tgt]
        tgt = torch.tensor(tgt)
        return src, tgt
