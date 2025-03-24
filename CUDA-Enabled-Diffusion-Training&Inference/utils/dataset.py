# -*- coding: utf-8 -*-
"""
Created on Mon Feb 02 16:56:12 2025

@author: Rohan
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_data(batch_size: int = 64, num_workers: int = 4):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    
    return train_loader
