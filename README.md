# Cardiac-Amyloid: Neural Network for Detection and Segmentation of Cardiac Amyloidosis

This repository contains the training scripts, pre-trained models, and resources used to develop and validate a neural network for detecting and segmenting amyloid deposits in cardiac tissue from brightfield Congo Red-stained virtual slides.

## Overview

Cardiac amyloidosis is characterized by the extracellular deposition of amyloid fibrils, which can severely impair cardiac function. This project leverages image analysis and deep learning techniques to enhance the detection and segmentation of amyloid deposits in digital pathology. Using brightfield slides and a fluorescence-guided annotation pipeline, we trained AI models to classify and segment amyloid-positive regions with high accuracy.

### Key Features:
- **Tile Classification**: A ResNet18-based CNN for binary classification of tiles as amyloid-positive or amyloid-negative.
- **Segmentation**: A U-Net model for pixel-wise segmentation of amyloid deposits.
- **Explainability**: Grad-CAM overlays to visualize regions influencing model predictions.

### Results
Classification (Validation metrics)
  Accuracy: 0.93
  F1 Score: 0.93
  Precision: 0.93
  Recall: 0.94
Segmentation
  Intersection over Union (IoU):
    Training: 0.70
    Validation: 0.60

### Pre-Trained Models
Download the pre-trained models for direct use:

Tile Classification: resnet18.pth
Segmentation: unet.pth

### Prerequisites

- Python 3.10 or later
- NVIDIA GPU with CUDA support (optional but recommended)
- Install dependencies:
  ```bash
  pip install -r requirements.txt


