# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 17:00:07 2025

@author: Rohan
"""

import torch

def save_checkpoint(model, optimizer, ema, filename="checkpoints/ddpm_checkpoint.pth"):
    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, ema, filename="checkpoints/ddpm_checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['weights'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    ema.load_state_dict(checkpoint['ema'])
