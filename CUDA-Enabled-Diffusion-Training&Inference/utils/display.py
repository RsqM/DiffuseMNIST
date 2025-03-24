# -*- coding: utf-8 -*-
"""
Created on Mon Feb 02 16:57:32 2025

@author: Rohan
"""

import matplotlib.pyplot as plt
from einops import rearrange
from typing import List

def display_reverse(images: List):
    fig, axes = plt.subplots(1, 10, figsize=(10, 1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x, cmap='gray')
        ax.axis('off')
    plt.show()
