from functools import partial

import torch
from torch.utils.data import Dataset
import tiktoken
from datasets import load_dataset


class OpenOrcaDataset(Dataset):
    sys_prompt_start = "<|sys_prompt_start|>"
    question_start = "<|question_start|>"
    answer_start = "<|answer_start|>"
    answer_end = "<|answer_end|>"

    def __init__(
        self,
        enc: tiktoken.Encoding,
        cache_dir=None,
        is_val=False,
        val_rate=0.0005,
        max_len=512,
        seed=1437,
    ):
        super().__init__()
        if not hasattr(enc, "_noobai_is_init"):
            raise ValueError(
                "You should init encoder by OpenOrcaDataset.init_encoder manually."
            )
        self.max_len = max_len
        self.eot_token = enc.eot_token
        self.dataset = load_dataset("Open-Orca/OpenOrca", cache_dir=cache_dir)
        self.split_dataset = self.dataset["train"].train_test_split(
            test_size=val_rate, seed=seed, shuffle=True
        )
        if is_val:
            self.dataset = self.split_dataset["test"]
        else:
            self.dataset = self.split_dataset["train"]
        self.dataset = self.dataset.to_list()
        self.dataset = list(map(partial(self.handle, enc=enc), self.dataset))

    @staticmethod
    def init_encoder(enc: tiktoken.Encoding):
        enc = tiktoken.Encoding(
            name="gpt2",
            pat_str=enc._pat_str,
            mergeable_ranks=enc._mergeable_ranks,
            special_tokens={
                **enc._special_tokens,
                OpenOrcaDataset.sys_prompt_start: enc.max_token_value + 1,
                OpenOrcaDataset.question_start: enc.max_token_value + 2,
                OpenOrcaDataset.answer_start: enc.max_token_value + 3,
                OpenOrcaDataset.answer_end: enc.max_token_value + 4,
            },
        )
        setattr(enc, "_noobai_is_init", True)
        return enc

    def handle(self, x, enc: tiktoken.Encoding):
        ids = []
        ids.append(enc._special_tokens[self.sys_prompt_start])
        ids.extend(enc.encode_ordinary(x['system_prompt']))
        ids.append(enc._special_tokens[self.question_start])
        ids.extend(enc.encode_ordinary(x['question']))
        answer_start_idx = len(ids)
        ids.append(enc._special_tokens[self.answer_start])
        ids.extend(enc.encode_ordinary(x['response']))
        ids.append(enc._special_tokens[self.answer_end])
        ids = ids[: self.max_len]
        answer_start_idx = min(answer_start_idx, self.max_len)
        return ids, answer_start_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def collate_fn(self, batchs):
        ids_list = []
        len_list = []
        answer_start_idx_list = []
        for b in batchs:
            ids, answer_start_idx = b
            ids_list.append(ids)
            len_list.append(len(ids))
            answer_start_idx_list.append(answer_start_idx)

        max_len = max(len_list)
        ids_tensor = torch.zeros(len(batchs), max_len, dtype=torch.long)
        ids_tensor = torch.fill_(ids_tensor, self.eot_token)
        for i, ids in enumerate(ids_list):
            ids_tensor[i, : len(ids)] = torch.tensor(ids)

        len_tensor = torch.tensor(len_list)
        answer_start_idx_tensor = torch.tensor(answer_start_idx_list)

        _, sorted_indices = torch.sort(len_tensor, descending=True)
        ids_tensor = ids_tensor[sorted_indices]
        len_tensor = len_tensor[sorted_indices]
        answer_start_idx_tensor = answer_start_idx_tensor[sorted_indices]

        return ids_tensor, len_tensor, answer_start_idx_tensor
