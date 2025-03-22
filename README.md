# Convolutional Neural Networks (CNNs) - Lecture Materials
## University of Calgary - Guest Lecture

This repository contains comprehensive materials for teaching Convolutional Neural Networks (CNNs) in Python. The materials include theoretical concepts, practical implementations, and hands-on exercises.

### Project Structure

```
.
├── README.md
├── requirements.txt
├── data/                    # Directory for storing datasets
├── notebooks/              # Jupyter notebooks
│   ├── 01_CNN_Introduction.ipynb    # Main lecture notebook
│   └── CNN_Lecture.slides.ipynb     # Jupyter slides
└── results/               # Directory for storing results
    ├── models/           # Trained model checkpoints
    └── plots/            # Generated visualizations
```

### Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter:
```bash
jupyter notebook
```

### Learning Objectives

1. Understand the fundamentals of CNNs
2. Learn about CNN architecture components
3. Implement a CNN from scratch
4. Train and evaluate CNN models
5. Visualize CNN features and results

### Topics Covered

- Introduction to CNNs
- CNN Architecture Components
  - Convolutional Layers
  - Pooling Layers
  - Fully Connected Layers
- CNN vs Traditional Neural Networks
- Implementation Details
- Training and Optimization
- Visualization and Interpretation

### Dataset

We'll be using the MNIST dataset for demonstration purposes. The dataset will be automatically downloaded when running the notebooks.

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Jupyter
- jupyter-slides

### License

This project is licensed under the MIT License - see the LICENSE file for details. 