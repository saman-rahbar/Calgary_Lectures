import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) implementation.
    
    Architecture:
    - Input: 28x28 grayscale images (MNIST)
    - Conv1: 32 filters, 3x3 kernel, ReLU activation
    - MaxPool1: 2x2 pooling
    - Conv2: 64 filters, 3x3 kernel, ReLU activation
    - MaxPool2: 2x2 pooling
    - Fully Connected: 128 neurons, ReLU activation
    - Output: 10 neurons (one for each digit)
    """
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # First convolutional layer
        # Input: 28x28x1 -> Output: 26x26x32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        
        # First pooling layer
        # Input: 26x26x32 -> Output: 13x13x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer
        # Input: 13x13x32 -> Output: 11x11x64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        
        # Second pooling layer
        # Input: 11x11x64 -> Output: 5x5x64
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # Input: 5x5x64 = 1600 -> Output: 128
        self.fc1 = nn.Linear(1600, 128)
        
        # Output layer
        # Input: 128 -> Output: 10 (number of classes)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 10)
        """
        # First conv block
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Second conv block
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten the tensor
        x = x.view(-1, 1600)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def get_model():
    """
    Creates and returns a new instance of the CNN model.
    
    Returns:
        SimpleCNN: A new instance of the CNN model
    """
    return SimpleCNN()

def count_parameters(model):
    """
    Counts the number of trainable parameters in the model.
    
    Args:
        model (nn.Module): The PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 