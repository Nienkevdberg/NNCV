# Semantic Segmentation with DeepLabv3

## Overview

This project focuses on semantic segmentation of urban scenes using the Cityscapes dataset. A provided U-Net baseline is improved by implementing a DeepLabv3-based model. The project explores both a high-performance configuration and an efficient lightweight variant in order to analyze the trade-off between accuracy and computational cost.

## Models

### Baseline (unet_model.py)
- U-Net architecture provided in the assignment
- Serves as reference model

### Peak-Performance Model (PP_model.py and PP_train.py)
- DeepLabv3 with ResNet-101 backbone
- Loss: Cross-Entropy + Dice Loss
- Test-Time Augmentation (multi-scale + horizontal flip)
- Optimized for highest segmentation accuracy

### Efficiency Model (efficiency_model.py and efficiency_train.py)
- DeepLabv3 with MobileNetV3 backbone
- Lower input resolution (256×512)
- No auxiliary loss
- Optimized for speed and low computational cost

In the predict.py, only the resize of the image should be adjusted to the specific model.

## Requirements
The project is implemented in Python using PyTorch.

### Required libraries:
- torch
- torchvision
- numpy
- pillow
- wandb
