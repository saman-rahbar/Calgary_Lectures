import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.utils import make_grid

def plot_training_history(train_losses, test_losses, train_accs, test_accs):
    """
    Plots the training and testing loss/accuracy history.
    
    Args:
        train_losses (list): List of training losses
        test_losses (list): List of testing losses
        train_accs (list): List of training accuracies
        test_accs (list): List of testing accuracies
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(test_losses, label='Testing Loss')
    ax1.set_title('Loss History')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(test_accs, label='Testing Accuracy')
    ax2.set_title('Accuracy History')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, test_loader, num_samples=10):
    """
    Visualizes model predictions on test samples.
    
    Args:
        model (nn.Module): The trained model
        test_loader (DataLoader): The test data loader
        num_samples (int): Number of samples to visualize
    """
    model.eval()
    images, labels = next(iter(test_loader))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Create a grid of images
    img_grid = make_grid(images, nrow=5)
    
    plt.figure(figsize=(15, 3))
    plt.imshow(img_grid.permute(1, 2, 0).numpy(), cmap='gray')
    plt.title('Sample Predictions\n' + 
              'Predicted: ' + ' '.join(map(str, predicted.numpy())) + '\n' +
              'Actual: ' + ' '.join(map(str, labels.numpy())))
    plt.axis('off')
    plt.show()

def visualize_feature_maps(model, image):
    """
    Visualizes the feature maps from the first convolutional layer.
    
    Args:
        model (nn.Module): The trained model
        image (torch.Tensor): Input image tensor
    """
    model.eval()
    
    # Get the first convolutional layer
    conv1 = model.conv1
    
    # Forward pass until conv1
    with torch.no_grad():
        features = conv1(image.unsqueeze(0))
    
    # Create a grid of feature maps
    feature_grid = make_grid(features[0], nrow=8, normalize=True)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(feature_grid.permute(1, 2, 0).numpy(), cmap='viridis')
    plt.title('Feature Maps (Conv1)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show() 