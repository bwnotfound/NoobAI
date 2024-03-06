'''https://github.com/thaihungle/SAM'''

import os
from torch.utils.data import Dataset
import torch

import numpy as np
import random
import pickle

elem_num = 26 + 10 + 1


def get_one_hot(c):
    if ord('a') <= ord(c) <= ord('z'):
        return ord(c) - ord('a') + 10
    elif ord('0') <= ord(c) <= ord('9'):
        return ord(c) - ord('0')

    return elem_num - 1


def generate_one(step_num):
    a = np.zeros([step_num * 2 + 3], dtype=np.int64)
    d = {}
    st = ''
    tem = list(range(26))
    np.random.shuffle(tem)
    tem = tem[:step_num]
    for i, c in enumerate(tem):
        b = random.randint(0, 9)
        d[c] = b
        s, t = chr(c + ord('a')), chr(b + ord('0'))
        st += s + t
        a[i * 2] = get_one_hot(s)
        a[i * 2 + 1] = get_one_hot(t)

    s = random.choice(list(d.keys()))
    t = chr(s + ord('a'))
    r = chr(d[s] + ord('0'))
    a[step_num * 2] = get_one_hot('?')
    a[step_num * 2 + 1] = get_one_hot('?')
    a[step_num * 2 + 2] = get_one_hot(t)
    st += '??' + t + r
    e = get_one_hot(r)
    return a, e


def generate_data(save_path=None, step_num=24, row_nums=10000):
    step_num = max(1, min(int(step_num), 24))
    if save_path is not None:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, 'associative-retrieval.pkl')
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
    x = np.zeros([row_nums, step_num * 2 + 3], dtype=np.int64)

    y = np.zeros([row_nums], dtype=np.int64)
    for i in range(row_nums):
        x[i], y[i] = generate_one(step_num)

    d = {'data': x, 'label': y}
    data = pickle.dumps(d, protocol=2)
    if save_path is not None:
        with open(save_path, 'wb') as f:
            f.write(data)
        return save_path
    return data


class NumberRecallDataset(Dataset):
    n_vocab = elem_num

    def __init__(self, path=None):
        if path is None:
            path = './data/associative-retrieval.pkl'
        if not os.path.exists(path):
            path = generate_data(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.data = data['data']
        self.label = data['label']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def collate_fn(self, batch):
        data = [x[0] for x in batch]
        label = [x[1] for x in batch]
        data = torch.tensor(np.array(data), dtype=torch.long).permute(1, 0)
        label = torch.tensor(np.array(label), dtype=torch.long)
        return data, label


if __name__ == '__main__':
    generate_data('./data/associative-retrieval.pkl')
