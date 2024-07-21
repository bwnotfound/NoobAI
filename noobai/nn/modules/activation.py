import torch.nn as nn
import torch.nn.functional as F


class SoftmaxT(nn.Module):
    r'''temperature higher: distribution more uniform'''

    def __init__(self, temperature=1, dim=-1):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)
        self.temperature = temperature

    def forward(self, x):
        x = x / self.temperature
        return self.softmax(x)

    @staticmethod
    def apply(module, temperature, dim):
        if not isinstance(module, nn.Softmax):
            return module
        new_module = SoftmaxT(temperature, dim)
        return new_module


def softmax_t(x, temperature=1, dim=-1):
    return F.softmax(x / temperature, dim=dim)


def set_softmax_t(module: nn.Module, temperature=1, dim=-1):
    for name, layer in module.named_children():
        module.add_module(name, set_softmax_t(layer))
    if isinstance(module, nn.Softmax):
        return SoftmaxT.apply(module, temperature, dim)
    else:
        return module
