import torch.nn as nn
from einops import rearrange


class Rearrange(nn.Module):

    def __init__(self, ops):
        '''
        einops.rearrange wrapper
        ops: eg "h w c -> c h w"
        '''
        super().__init__()
        self.ops = ops

    def forward(self, x):
        return rearrange(x, self.ops)
