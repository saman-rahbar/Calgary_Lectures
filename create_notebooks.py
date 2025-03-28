import nbformat as nbf
import os

def create_main_notebook():
    nb = nbf.v4.new_notebook()
    
    # Title and Setup
    cells = []
    
    # Title
    cells.append(nbf.v4.new_markdown_cell("""# Introduction to Convolutional Neural Networks (CNNs)
    
This notebook provides a comprehensive introduction to Convolutional Neural Networks (CNNs), their architecture, and implementation in PyTorch. We'll use the MNIST dataset as an example."""))
    
    # Setup and Imports
    cells.append(nbf.v4.new_code_cell("""import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Create directories if they don't exist
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")"""))
    
    # Data Loading
    cells.append(nbf.v4.new_markdown_cell("""## Data Loading and Preprocessing
    
We'll use the MNIST dataset, which contains 28x28 grayscale images of handwritten digits (0-9)."""))
    
    cells.append(nbf.v4.new_code_cell("""# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Display some example images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('results/plots/mnist_examples.png')
plt.show()"""))
    
    # Model Definition
    cells.append(nbf.v4.new_markdown_cell("""## CNN Model Architecture
    
Let's define our CNN model with the following architecture:
1. Convolutional layer (1 → 16 channels)
2. ReLU activation and max pooling
3. Convolutional layer (16 → 32 channels)
4. ReLU activation and max pooling
5. Fully connected layers"""))
    
    cells.append(nbf.v4.new_code_cell("""class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
print(model)"""))
    
    # Training
    cells.append(nbf.v4.new_markdown_cell("""## Model Training
    
We'll train the model using:
- Cross Entropy Loss
- Adam optimizer
- 5 epochs"""))
    
    cells.append(nbf.v4.new_code_cell("""def train_model(model, train_loader, test_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Testing
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100 * correct / total
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    # Save the model
    torch.save(model.state_dict(), 'results/models/mnist_cnn.pth')
    return train_losses, test_losses, train_accs, test_accs

# Train the model
train_losses, test_losses, train_accs, test_accs = train_model(model, train_loader, test_loader)"""))
    
    # Visualization
    cells.append(nbf.v4.new_markdown_cell("""## Results Visualization
    
Let's visualize the training results and model predictions."""))
    
    cells.append(nbf.v4.new_code_cell("""# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Test Accuracy')

plt.tight_layout()
plt.savefig('results/plots/training_history.png')
plt.show()

# Generate confusion matrix
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        y_true.extend(target.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('results/plots/confusion_matrix.png')
plt.show()"""))
    
    # Model Inference
    cells.append(nbf.v4.new_markdown_cell("""## Model Inference
    
Let's test our model on some example images."""))
    
    cells.append(nbf.v4.new_code_cell("""def visualize_predictions(model, test_loader, num_samples=10):
    model.eval()
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            
            ax = axes.flat[i]
            ax.imshow(data[0].cpu().squeeze(), cmap='gray')
            ax.set_title(f'True: {target[0].item()}\\nPred: {predicted[0].item()}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/plots/model_predictions.png')
    plt.show()

# Visualize some predictions
visualize_predictions(model, test_loader)"""))
    
    # Add a new section for simple examples
    cells.append(nbf.v4.new_markdown_cell("""# Simple CNN Examples and Visualizations
Let's break down CNNs into simple, intuitive examples that you can run and understand.

## 1. Understanding Convolution with a Simple Example
Think of convolution like using a magnifying glass to look for specific patterns in an image. Let's create a simple example:"""))

    cells.append(nbf.v4.new_code_cell("""import numpy as np
import matplotlib.pyplot as plt

# Create a simple 5x5 image with a diagonal pattern
image = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
])

# Create a 3x3 filter to detect diagonal patterns
filter_3x3 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Perform convolution
def simple_convolution(image, filter):
    output = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            output[i, j] = np.sum(image[i:i+3, j:j+3] * filter)
    return output

# Get the output
output = simple_convolution(image, filter_3x3)

# Visualize
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Input Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(filter_3x3, cmap='gray')
plt.title('Filter (Pattern Detector)')
plt.axis('off')

plt.subplot(133)
plt.imshow(output, cmap='gray')
plt.title('Output (Detected Pattern)')
plt.axis('off')

plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("""### What's happening here?
1. **Input Image**: Think of it like a small photo (5x5 pixels)
2. **Filter**: Like a magnifying glass looking for diagonal patterns
3. **Output**: Shows where the pattern was found (brighter = stronger match)

## 2. Real MNIST Example with Visualization
Let's look at how a CNN processes a real digit image:"""))

    cells.append(nbf.v4.new_code_cell("""import torch
import torchvision
import matplotlib.pyplot as plt

# Load MNIST dataset
transform = torchvision.transforms.ToTensor()
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# Get one image
image, label = next(iter(test_loader))

# Create a simple CNN model
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# Create model and process image
model = SimpleCNN()
output = model(image)

# Visualize the process
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(image[0, 0].numpy(), cmap='gray')
plt.title(f'Input Digit: {label.item()}')
plt.axis('off')

plt.subplot(132)
plt.imshow(output[0, 0].detach().numpy(), cmap='gray')
plt.title('First Layer Output')
plt.axis('off')

plt.subplot(133)
plt.imshow(output[0, 15].detach().numpy(), cmap='gray')
plt.title('Another Feature Map')
plt.axis('off')

plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("""### Understanding the Process
1. **Input**: A handwritten digit (28x28 pixels)
2. **First Layer**: Creates 16 different "feature maps" (like looking for 16 different patterns)
3. **Output**: Shows how well each pattern matches the input

## 3. Real-World Analogy: Pattern Recognition
Think of CNNs like a detective looking for clues in a photo:

1. **Convolution Layer**: Like using different magnifying glasses to look for specific patterns
   - One might look for edges
   - Another for curves
   - Another for corners

2. **ReLU**: Like saying "if I see something interesting, mark it; if not, ignore it"

3. **MaxPooling**: Like taking a step back to see the bigger picture
   - Reduces the image size
   - Keeps the most important features

4. **Fully Connected Layer**: Like making a final decision based on all the clues found

Let's see this in action with a real image:"""))

    cells.append(nbf.v4.new_code_cell("""import cv2
import numpy as np

# Create a simple edge detection filter
edge_filter = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Create a simple corner detection filter
corner_filter = np.array([
    [1, -1, -1],
    [-1,  1, -1],
    [-1, -1,  1]
])

# Load and process a simple image
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

# Apply filters
edge_output = simple_convolution(image, edge_filter)
corner_output = simple_convolution(image, corner_filter)

# Visualize
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(edge_output, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

plt.subplot(133)
plt.imshow(corner_output, cmap='gray')
plt.title('Corner Detection')
plt.axis('off')

plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("""### What We Learned
1. **Different Filters Find Different Things**:
   - Edge filter finds edges (like looking for sudden changes)
   - Corner filter finds corners (like looking for intersections)

2. **CNN Layers Work Together**:
   - First layer finds basic patterns (edges, corners)
   - Later layers combine these to find more complex patterns

3. **Real-World Applications**:
   - Face recognition: Looks for eyes, nose, mouth patterns
   - Object detection: Looks for shapes and textures
   - Medical imaging: Looks for specific tissue patterns"""))

    # Add all cells to the notebook
    nb.cells = cells
    
    # Save notebook
    with open('notebooks/01_CNN_Introduction.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_slides_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Title Slide
    cells.append(nbf.v4.new_markdown_cell("""# Introduction to Convolutional Neural Networks
---
## University of Calgary
### Saman Rahbar, Senior Applied Scientist"""))
    
    # What are CNNs
    cells.append(nbf.v4.new_markdown_cell("""# What are CNNs?
---
- Specialized neural networks for processing grid-like data
- Particularly effective for image processing
- Inspired by the visual cortex of animals
- State-of-the-art performance in computer vision tasks"""))
    
    # Architecture Components
    cells.append(nbf.v4.new_markdown_cell("""# CNN Architecture Components
---
1. **Convolutional Layers**
   - Apply filters to detect features
   - Learn hierarchical representations

2. **Pooling Layers**
   - Reduce spatial dimensions
   - Provide translation invariance

3. **Activation Functions**
   - Introduce non-linearity
   - Common choice: ReLU

4. **Fully Connected Layers**
   - Combine features
   - Make final predictions"""))
    
    # How CNNs Work
    cells.append(nbf.v4.new_markdown_cell("""# How CNNs Work
---
![MNIST Examples](results/plots/mnist_examples.png)

- Convolution operation slides filters over input
- Each filter learns to detect specific features
- Pooling reduces spatial dimensions
- Final layers make predictions"""))
    
    # Implementation
    cells.append(nbf.v4.new_markdown_cell("""# Implementation in PyTorch
---
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
```"""))
    
    # Training Results
    cells.append(nbf.v4.new_markdown_cell("""# Training Results
---
![Training History](results/plots/training_history.png)

- Model converges quickly
- High accuracy on test set
- Good generalization"""))
    
    # Model Performance
    cells.append(nbf.v4.new_markdown_cell("""# Model Performance
---
![Confusion Matrix](results/plots/confusion_matrix.png)

- Strong performance across all classes
- Few misclassifications
- Robust predictions"""))
    
    # Predictions
    cells.append(nbf.v4.new_markdown_cell("""# Model Predictions
---
![Model Predictions](results/plots/model_predictions.png)

- Accurate predictions on test data
- Handles various writing styles
- Robust to noise"""))
    
    # Summary
    cells.append(nbf.v4.new_markdown_cell("""# Summary
---
- CNNs are powerful tools for image processing
- Key components work together effectively
- Easy to implement with modern frameworks
- Strong performance on real-world tasks

## Questions?"""))
    
    # Add all cells to the notebook
    nb.cells = cells
    
    # Save notebook
    with open('notebooks/CNN_Lecture.slides.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    # Create notebooks directory if it doesn't exist
    os.makedirs('notebooks', exist_ok=True)
    
    # Create both notebooks
    create_main_notebook()
    create_slides_notebook()
    print("Notebooks created successfully!") 