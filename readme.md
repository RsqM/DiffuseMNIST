## U-Net Based Denoising Diffusion Model for Image Generation
This repository implements a Denoising Diffusion Probabilistic Model (DDPM) using a U-Net architecture, designed for image generation. The model is trained on the MNIST dataset and is implemented with PyTorch. It includes modular components for easier maintenance, scalability, and future enhancements.

### Features

* Implements DDPM for image generation.
* Uses a U-Net-based denoising model for high-quality image reconstruction.
* Supports training and inference workflows.
* Modularized for scalability and customizability.
* CUDA-enabled for efficient training.

### Requirements

Ensure the following dependencies are installed before running the project:

```bash
pip install torch torchvision numpy einops tqdm timm matplotlib
```

For CUDA support, install the appropriate PyTorch version following the official PyTorch Installation Guide.

### Usage

1. Training the Model
To train the model from scratch, run:

```bash
python train.py
```
To resume training from a checkpoint, specify the checkpoint path:

```bash
python train.py --checkpoint_path checkpoints/ddpm_checkpoint
```

2. Running Inference
To generate images using a pre-trained model, run:

```bash
python inference.py --checkpoint_path checkpoints/ddpm_checkpoint
```

This will generate denoised images from random noise and display the intermediate steps.

## Implementation Details
### 1. Model Architecture
* U-Net: Used as the backbone for denoising.
* Sinusoidal Time Embeddings: Encodes time steps for better model understanding.
* Residual Blocks & Attention Layers: Enhance feature extraction at multiple scales.
* Diffusion Scheduler: Handles noise addition and denoising step calculations.

### 2. Training Workflow
* Sample an image from MNIST.
* Add noise using the diffusion process.
* Train the U-Net model to predict noise.
* Compute loss and update model parameters.

### 3. Inference Process
* Start with a random noise image.
* Perform denoising over T time steps.
* Generate a final clean image.

### CUDA Support

#### CUDA Enabled: Allows efficient training and real-time inference.

#### CUDA Disabled: Training is extremely slow (>4h per epoch) and is not recommended. Inference is still possible but will be slower.
