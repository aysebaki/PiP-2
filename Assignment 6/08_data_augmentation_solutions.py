# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer

################################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

################################################################################

Example solutions for tasks in the provided tasks file.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

torch.manual_seed(0)  # Set a known random seed for reproducibility


#
# Task 1
#

# To make dropout on image data more effective, we can also drop out blocks of
# pixels that are adjacent in the spatial and channel dimensions (see DropBlock
# https://arxiv.org/abs/1810.12890) or replace them with random values (see
# Random Erasing https://arxiv.org/abs/1708.04896). Your task is to create a
# DropBlock PyTorch module that processes an input tensor with an expected shape
# of (..., H, W), where "..." can be arbitrary many leading dimensions.


class DropBlock(torch.nn.Module):
    
    def __init__(self, n: int, p: float):
        """DropBlock module drops out a block of `n` by `n` pixels with
        probability `p`."""
        super().__init__()
        self.n = n
        # We will sample the block centers from the entire input dimension, so
        # we can skip the second scaling factor mentioned in the paper
        self.gamma = p / n ** 2
    
    def forward(self, x: torch.Tensor):
        # Do not apply dropout in evaluation mode or if there is nothing to drop
        if not self.training or self.gamma <= 0:
            return x
        
        # Expected shape of "x": (..., H, W)
        # "one_dims" is just to create dimensions of size 1 for every other dim
        one_dims = [1] * (len(x.shape) - 2)  # -2 because of "H" and "W"
        # Make sure to use the same device as the input tensor
        block_centers = (torch.rand(size=(*one_dims, *x.shape[-2:]), device=x.device) < self.gamma).float()
        
        # Create blocks from the block centers. Since a drop-out block center is
        # currently 1.0, we can simply use max-pooling
        blocks = F.max_pool2d(input=block_centers, kernel_size=self.n, stride=1, padding=self.n // 2)
        # If "n" is an even number, max-pooling yields 1 pixel too much (height
        # and width)
        if self.n % 2 == 0:
            blocks = blocks[..., :-1, :-1]
        # Invert the blocks, so 1 indicates "keep" and 0 indicates "drop"
        blocks = 1 - blocks
        
        # Multiply "blocks" mask and normalize result
        return x * blocks * (blocks.numel() / blocks.sum())


n_vals = [2, 8, 32]
p_vals = [0.1, 0.2, 0.5]
image = Image.open("08_example_image.jpg")
fig, axes = plt.subplots(nrows=len(n_vals), ncols=len(p_vals) + 1, sharex=True, sharey=True)
for row, n in enumerate(n_vals):
    axes[row, 0].imshow(image)
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])
    axes[row, 0].set_title("Original image")
    for col, p in enumerate(p_vals):
        dropblock = DropBlock(n=n, p=p)
        image_dropout = dropblock(TF.to_tensor(image)).clamp(min=0, max=1)
        axes[row, col + 1].imshow(TF.to_pil_image(image_dropout))
        axes[row, col + 1].set_xticks([])
        axes[row, col + 1].set_yticks([])
        axes[row, col + 1].set_title(f"DropBlock\n(p: {p}, n: {n})")
fig.tight_layout()
plt.show()
