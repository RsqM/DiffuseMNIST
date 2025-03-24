# -*- coding: utf-8 -*-
"""
Created on Mon Feb 02 16:57:52 2025

@author: Rohan
"""

import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
