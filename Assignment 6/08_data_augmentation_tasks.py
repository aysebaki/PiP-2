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

Tasks for self-study. Try to solve these tasks on your own and compare your
solutions to the provided solutions file.
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
    pass
    # Your code here #


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
