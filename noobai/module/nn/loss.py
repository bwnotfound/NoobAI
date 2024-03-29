import torch
import torch.nn as nn


class MaskedCrossEntropyLoss:
    def __init__(self, ignore_index=-100, reduction="mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction=reduction
        )

    def loss_func(self, x, y, dim):
        if dim == -1:
            dim = len(x.shape) - 1
        if dim != 1:
            shape_list = []
            for i in range(len(x.shape)):
                if i == dim:
                    continue
                if i == 1:
                    shape_list.append(dim)
                shape_list.append(i)
            x = x.permute(*shape_list)
        return self.cross_entropy(x, y)

    def __call__(self, logits, target, mask=None, dim=-1, batch_first=False):
        if mask is None:
            return self.loss_func(logits, target, dim)
        if len(mask.shape) == 1 and len(logits.shape) == 3 and len(target.shape) == 2:
            if batch_first:
                logits = logits.permute(1, 0, 2)
                target = target.transpose(1, 0)
            mask = torch.arange(logits.size(0), device=mask.device).unsqueeze(
                0
            ) < mask.unsqueeze(1)
            mask = mask.transpose(0, 1)
        target = torch.clone(target)
        target[~mask] = self.ignore_index
        return self.loss_func(logits, target, dim)
