# -*- coding: utf-8 -*-
"""
Created on Mon Feb 03 17:46:09 2025

@author: Rohan
"""

import torch
import torch.nn as nn

class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int = 1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    def forward(self, t):
        return self.beta[t], self.alpha[t]
