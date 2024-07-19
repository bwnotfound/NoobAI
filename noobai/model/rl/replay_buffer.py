from threading import Lock

import numpy as np


class BaseReplayBuffer:
    def __init__(self, need_unzip=True, to_tensor=True):
        self.op_lock = Lock()
        self.need_unzip = need_unzip
        self.to_tensor = to_tensor

    def sample(self, batch_size):
        """
        can be sampled if real_size < batch_size
        """
        raise NotImplementedError

    def can_sample(self, batch_size):
        raise NotImplementedError

    def add(self, example):
        """
        should be a tuple or list of one step result.
        """
        raise NotImplementedError


class FIFOOfflineReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity, need_unzip=True, to_tensor=True):
        super().__init__(need_unzip=need_unzip, to_tensor=to_tensor)
        self.capacity = capacity
        self.container = [None for _ in range(capacity)]
        self.ptr = 0
        self.is_full = False

    def size(self):
        if self.is_full:
            return self.capacity
        else:
            return self.ptr

    def sample(self, batch_size):
        assert batch_size > 0
        with self.op_lock:
            result = []
            if batch_size == self.capacity:
                result = self.container
            else:
                if self.is_full:
                    ind = np.random.randint(0, self.capacity, size=batch_size)
                    for i in ind:
                        result.append(self.container[i])
                else:
                    result = self.container[: min(self.capacity, batch_size)]
            if self.need_unzip:
                result = list(zip(*result))
                if self.to_tensor:
                    import torch

                    result = [
                        torch.FloatTensor(
                            data
                            if not isinstance(data[0], (list, tuple))
                            else [torch.FloatTensor(line) for line in data]
                        )
                        for data in result
                    ]
            result = list(result)
            return result

    def add(self, example):
        with self.op_lock:
            self.container[self.ptr] = example
            self.ptr += 1
            if self.ptr >= self.capacity:
                self.ptr %= self.capacity
                self.is_full = True
