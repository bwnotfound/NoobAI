import torch
from torch.utils.data import Dataset
import tiktoken
from datasets import load_dataset


class BookCorpus(Dataset):
    def __init__(
        self,
        enc: tiktoken.Encoding,
        is_val=False,
        max_len=None,
        split=False,
        val_rate=0.005,
        seed=1437,
        cache_dir="D:/AI/HuggingFace/datasets",
    ):
        super().__init__()
        self.eot_token = enc.eot_token
        self.max_len = max_len
        self.dataset = load_dataset("bookcorpus", cache_dir=cache_dir)
        self.split_dataset = self.dataset["train"].train_test_split(
            test_size=val_rate, seed=seed, shuffle=True
        )
        if is_val:
            self.dataset = self.split_dataset["test"]
        else:
            self.dataset = self.split_dataset["train"]
        self.dataset = self.dataset.to_list()
        self.dataset = list(map(lambda x: enc.encode_ordinary(x['text']), self.dataset))
        if max_len is not None:
            if not split:
                self.dataset = list(
                    filter(lambda x: len(x) <= max_len - 1, self.dataset)
                )
            else:
                new_dataset = []
                for d in self.dataset:
                    for i in range(0, len(d), max_len - 1):
                        new_dataset.append(d[i : i + max_len - 1])
                self.dataset = new_dataset
        self.dataset = [d for d in new_dataset if len(d) > 0]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def collate_fn(self, batchs):
        ids_list = []
        len_list = []
        for b in batchs:
            ids = b + [self.eot_token]
            ids_list.append(ids)
            len_list.append(len(ids))

        max_len = max(len_list)
        ids_tensor = torch.zeros(len(batchs), max_len, dtype=torch.long)
        ids_tensor = torch.fill_(ids_tensor, self.eot_token)
        for i, ids in enumerate(ids_list):
            ids_tensor[i, : len(ids)] = torch.tensor(ids)

        len_tensor = torch.tensor(len_list)

        return ids_tensor, len_tensor
