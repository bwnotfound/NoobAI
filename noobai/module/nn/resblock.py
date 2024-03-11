import torch.nn as nn


class SimpleResWrapper(nn.Module):

    def __init__(self, model, alpha=None):
        super().__init__()
        self.model = model
        self.alpha = alpha

    def forward(self, x):
        if self.alpha is None:
            return x + self.model(x)
        return x + self.alpha * self.model(x)
