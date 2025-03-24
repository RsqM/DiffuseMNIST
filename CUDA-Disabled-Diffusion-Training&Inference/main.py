# -*- coding: utf-8 -*-
"""
Created on Mon Feb 02 16:54:12 2025

@author: Rohan
"""

from train import train
from inference import inference

def main():
    train(checkpoint_path=None, lr=2e-5, num_epochs=75)
    inference('checkpoints/ddpm_checkpoint')

if __name__ == '__main__':
    main()
