import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64, num_workers=2):
    """
    Creates and returns DataLoader objects for MNIST training and test sets.
    
    Args:
        batch_size (int): Number of samples per batch
        num_workers (int): Number of subprocesses for data loading
        
    Returns:
        tuple: (train_loader, test_loader) DataLoader objects
    """
    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor and normalize to [0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize using MNIST mean and std
    ])
    
    # Load training data
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )
    
    # Load test data
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transform,
        download=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader

def get_sample_batch(train_loader):
    """
    Gets a single batch of data from the training loader.
    
    Args:
        train_loader (DataLoader): The training data loader
        
    Returns:
        tuple: (images, labels) from one batch
    """
    for images, labels in train_loader:
        return images, labels

def get_device():
    """
    Determines the best available device (CUDA GPU or CPU).
    
    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu") 