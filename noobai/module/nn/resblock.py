import torch.nn as nn


class SimpleResWrapper(nn.Module):

    def __init__(self, model, alpha=None, raw_add=True):
        super().__init__()
        if not isinstance(model, (list, tuple)):
            model = [model]
        if raw_add:
            model.append(nn.Identity())
        self.model = nn.Sequential(*model)
        self.alpha = alpha

    def forward(self, x):
        result = 0
        for i in range(len(self.model)):
            if i == len(self.model) - 1 and self.alpha is not None:
                result = result * self.alpha + self.model[i](x)
                break
            result = result + self.model[i](x)
        return result
