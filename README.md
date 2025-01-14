# Cardiac-Amyloid: AI for Detection and Segmentation of Cardiac Amyloidosis

This repository hosts the training scripts, pre-trained models, and resources for a neural network designed to detect and segment amyloid deposits in cardiac tissue from brightfield Congo Red-stained virtual slides.

## Overview

Cardiac amyloidosis is a condition marked by the extracellular deposition of amyloid fibrils, which can lead to significant cardiac dysfunction. This project employs state-of-the-art image analysis and deep learning techniques to improve detection and segmentation of amyloid deposits in digital pathology slides. Using fluorescence-guided annotations, our AI models deliver high accuracy for classification and segmentation tasks.

## Key Features

- **Tile Classification**:  
  A ResNet18-based Convolutional Neural Network (CNN) for binary classification of tiles as amyloid-positive or amyloid-negative.
  
- **Segmentation**:  
  A U-Net model for precise, pixel-wise segmentation of amyloid deposits.
  
- **Explainability**:  
  Grad-CAM overlays highlight regions most influential in model predictions, enhancing interpretability.

## Results

### Classification (Validation Metrics)
- **Accuracy**: `0.93`  
- **F1 Score**: `0.93`  
- **Precision**: `0.93`  
- **Recall**: `0.94`

### Segmentation
- **Intersection over Union (IoU)**:  
  - Training: `0.70`  
  - Validation: `0.60`  

## 📦 Pre-Trained Models

You can download our pre-trained models for immediate use:  
- **Tile Classification**: [resnet18.pth](#)  
- **Segmentation**: [unet_amyloid.h5](#)  

## Prerequisites

### Software Requirements
- Python `3.10` or later
- NVIDIA GPU with CUDA support (optional for faster training/inference)

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/Cardiac-Amyloid.git
cd Cardiac-Amyloid
pip install -r requirements.txt

