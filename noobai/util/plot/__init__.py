import torch

import matplotlib.pyplot as plt


def plot_mask(mask, save_path=None, show=True):
    r'''mask: [H, W] bool'''
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()
    plt.imshow(mask, cmap='gray')
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
