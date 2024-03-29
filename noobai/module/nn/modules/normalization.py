import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    r'''Root Mean Square Normalization'''

    def __init__(self, feature_dim, dim=-1, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(feature_dim))

    def l2_norm(self, x) -> torch.Tensor:
        return x * torch.rsqrt(
            torch.norm(x, p=2, dim=self.dim, keepdim=True) + self.eps
        )

    def forward(self, x):
        target_dim = self.dim if self.dim >= 0 else len(x.shape) - 1
        view_shape = [1] * len(x.shape)
        view_shape[target_dim] = -1
        output = self.l2_norm(x.float()) * self.weight.view(*view_shape)
        return output.type_as(x)
