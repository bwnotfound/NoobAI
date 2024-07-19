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
        self.size = 0
        self.new_in_after_sample = 0

    def sample(self, batch_size):
        assert batch_size > 0
        with self.op_lock:
            self.new_in_after_sample = 0
            result = []
            ind = np.random.randint(0, self.size, size=batch_size)
            for i in ind:
                result.append(self.container[i])
            if self.need_unzip:
                result = list(zip(*result))
                if self.to_tensor:
                    import torch

                    result = [
                        torch.FloatTensor(
                            np.array(data
                            if not isinstance(data[0], (list, tuple))
                            else [np.array(line) for line in data])
                        )
                        for data in result
                    ]
            result = list(result)
            return result

    def add(self, example):
        with self.op_lock:
            self.container[self.ptr] = example
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            self.new_in_after_sample += 1
