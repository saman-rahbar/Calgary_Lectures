# CNN Architecture Explanation

## Overview
This document provides a detailed explanation of the Convolutional Neural Network (CNN) architecture used in this project. The architecture is designed for MNIST digit classification and consists of several key components.

## Layer-by-Layer Explanation

### 1. Input Layer
- **Shape**: 28x28x1 (grayscale images)
- **Purpose**: Receives the raw MNIST digit images
- **Details**: Each pixel value is normalized to [0,1]

### 2. First Convolutional Block
#### 2.1 Convolutional Layer (Conv2D)
- **Input Channels**: 1 (grayscale)
- **Output Channels**: 16
- **Kernel Size**: 3x3
- **Padding**: 1 (maintains spatial dimensions)
- **Purpose**: Learns basic features like edges and textures
- **Output Shape**: 28x28x16

#### 2.2 ReLU Activation
- **Function**: max(0, x)
- **Purpose**: Introduces non-linearity
- **Benefits**: Helps network learn complex patterns

#### 2.3 MaxPooling Layer
- **Pool Size**: 2x2
- **Stride**: 2
- **Purpose**: Reduces spatial dimensions
- **Benefits**: 
  - Reduces computation
  - Provides translation invariance
- **Output Shape**: 14x14x16

### 3. Second Convolutional Block
#### 3.1 Convolutional Layer (Conv2D)
- **Input Channels**: 16
- **Output Channels**: 32
- **Kernel Size**: 3x3
- **Padding**: 1
- **Purpose**: Learns more complex features
- **Output Shape**: 14x14x32

#### 3.2 ReLU Activation
- **Function**: max(0, x)
- **Purpose**: Maintains non-linearity
- **Benefits**: Helps network learn hierarchical features

#### 3.3 MaxPooling Layer
- **Pool Size**: 2x2
- **Stride**: 2
- **Purpose**: Further reduces spatial dimensions
- **Output Shape**: 7x7x32

### 4. Flatten Layer
- **Input Shape**: 7x7x32
- **Output Shape**: 1568 (7 * 7 * 32)
- **Purpose**: Prepares data for fully connected layers
- **Operation**: Reshapes 3D tensor to 1D vector

### 5. Fully Connected Layers
#### 5.1 First Fully Connected Layer
- **Input Size**: 1568
- **Output Size**: 128
- **Purpose**: Learns high-level features
- **Operation**: Linear transformation with weights and bias

#### 5.2 ReLU Activation
- **Function**: max(0, x)
- **Purpose**: Maintains non-linearity
- **Benefits**: Helps network learn complex decision boundaries

#### 5.3 Second Fully Connected Layer
- **Input Size**: 128
- **Output Size**: 10 (number of classes)
- **Purpose**: Final classification layer
- **Operation**: Linear transformation to class scores

### 6. Output Layer
- **Shape**: 10 (one score per digit class)
- **Purpose**: Produces class probabilities
- **Operation**: Softmax activation (applied during inference)

## Training Process
1. Forward pass through all layers
2. Compute loss (Cross Entropy)
3. Backward pass to compute gradients
4. Update weights using Adam optimizer

## Key Features
- **Hierarchical Feature Learning**: Each convolutional block learns increasingly complex features
- **Spatial Invariance**: MaxPooling helps the network be robust to small translations
- **Non-linearity**: ReLU activations enable learning complex patterns
- **Efficient Computation**: Convolutional layers share parameters across spatial locations

## Performance
- **Training Accuracy**: ~99%
- **Test Accuracy**: ~98%
- **Training Time**: ~5 minutes on CPU
- **Model Size**: ~1.2MB
