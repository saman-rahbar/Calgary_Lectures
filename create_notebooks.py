import nbformat as nbf
import os

def create_main_notebook():
    nb = nbf.v4.new_notebook()
    
    # Title
    title = nbf.v4.new_markdown_cell("""# Introduction to Convolutional Neural Networks (CNNs)
    
This notebook provides a comprehensive introduction to Convolutional Neural Networks (CNNs), their architecture, and implementation in PyTorch.""")
    nb.cells.append(title)
    
    # Introduction
    intro = nbf.v4.new_markdown_cell("""## What are CNNs?
    
Convolutional Neural Networks (CNNs) are a specialized type of neural network designed for processing grid-like data, such as images. They are particularly effective at capturing spatial relationships in data through the use of convolutional layers.""")
    nb.cells.append(intro)
    
    # CNN Components
    components = nbf.v4.new_markdown_cell("""## Key Components of CNNs
    
1. **Convolutional Layers**: Apply filters to input data to detect features
2. **Pooling Layers**: Reduce spatial dimensions of the data
3. **Activation Functions**: Introduce non-linearity
4. **Fully Connected Layers**: Make final predictions""")
    nb.cells.append(components)
    
    # Simple Convolution Example
    conv_example = nbf.v4.new_code_cell("""import numpy as np

def simple_convolution(image, kernel):
    # Get dimensions
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape
    
    # Calculate output dimensions
    o_height = i_height - k_height + 1
    o_width = i_width - k_width + 1
    
    # Initialize output
    output = np.zeros((o_height, o_width))
    
    # Perform convolution
    for i in range(o_height):
        for j in range(o_width):
            output[i, j] = np.sum(image[i:i+k_height, j:j+k_width] * kernel)
    
    return output

# Example usage
image = np.array([[1, 1, 1, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 1, 1, 1],
                  [0, 0, 1, 1, 0],
                  [0, 1, 1, 0, 0]])

kernel = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1]])

result = simple_convolution(image, kernel)
print("Convolution result:")
print(result)""")
    nb.cells.append(conv_example)
    
    # PyTorch Implementation
    pytorch_impl = nbf.v4.new_code_cell("""import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model instance
model = SimpleCNN()
print(model)""")
    nb.cells.append(pytorch_impl)
    
    # Training Process
    training = nbf.v4.new_markdown_cell("""## Training Process
    
1. **Forward Pass**: Input data flows through the network
2. **Loss Calculation**: Compare predictions with ground truth
3. **Backward Pass**: Compute gradients
4. **Parameter Update**: Adjust network weights""")
    nb.cells.append(training)
    
    # Training Example
    training_example = nbf.v4.new_code_cell("""import torch.optim as optim

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop example
def train(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')""")
    nb.cells.append(training_example)
    
    # Save notebook
    with open('notebooks/01_CNN_Introduction.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_slides_notebook():
    nb = nbf.v4.new_notebook()
    
    # Title Slide
    title = nbf.v4.new_markdown_cell("""# Introduction to Convolutional Neural Networks
---
## University of Calgary
### Dr. Saman Rahbar""")
    nb.cells.append(title)
    
    # Introduction Slide
    intro = nbf.v4.new_markdown_cell("""# What are CNNs?
---
- Specialized neural networks for processing grid-like data
- Particularly effective for image processing
- Inspired by the visual cortex of animals""")
    nb.cells.append(intro)
    
    # Architecture Components
    components = nbf.v4.new_markdown_cell("""# CNN Architecture Components
---
1. Convolutional Layers
2. Pooling Layers
3. Activation Functions
4. Fully Connected Layers""")
    nb.cells.append(components)
    
    # How CNNs Work
    how_it_works = nbf.v4.new_markdown_cell("""# How CNNs Work
---
- Convolution operation slides filters over input
- Each filter learns to detect specific features
- Pooling reduces spatial dimensions
- Final layers make predictions""")
    nb.cells.append(how_it_works)
    
    # Comparison with Traditional NNs
    comparison = nbf.v4.new_markdown_cell("""# CNNs vs Traditional Neural Networks
---
- CNNs preserve spatial relationships
- Parameter sharing reduces complexity
- Translation invariance
- Hierarchical feature learning""")
    nb.cells.append(comparison)
    
    # Implementation
    implementation = nbf.v4.new_markdown_cell("""# Implementation in PyTorch
---
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
```
""")
    nb.cells.append(implementation)
    
    # Training Process
    training = nbf.v4.new_markdown_cell("""# Training Process
---
1. Forward Pass
2. Loss Calculation
3. Backward Pass
4. Parameter Update""")
    nb.cells.append(training)
    
    # Results
    results = nbf.v4.new_markdown_cell("""# Results and Visualization
---
- Model performance metrics
- Feature map visualization
- Confusion matrix
- Learning curves""")
    nb.cells.append(results)
    
    # Summary
    summary = nbf.v4.new_markdown_cell("""# Summary
---
- Key concepts covered
- Applications of CNNs
- Future directions
- Questions?""")
    nb.cells.append(summary)
    
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