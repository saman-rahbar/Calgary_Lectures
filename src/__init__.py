"""
CNN Lecture Materials Package
University of Calgary - Guest Lecture
"""

from .cnn_model import SimpleCNN, get_model, count_parameters
from .data_utils import get_data_loaders, get_sample_batch, get_device
from .visualization import plot_training_history, visualize_predictions, visualize_feature_maps

__all__ = [
    'SimpleCNN',
    'get_model',
    'count_parameters',
    'get_data_loaders',
    'get_sample_batch',
    'get_device',
    'plot_training_history',
    'visualize_predictions',
    'visualize_feature_maps'
] 