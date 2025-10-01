"""
BloomWatch: Plant Blooming Stage Detection and Tracking

A modular Python project for detecting and tracking plant blooming stages 
from image datasets using PyTorch.
"""

__version__ = "0.1.0"
__author__ = "BloomWatch Team"
__email__ = "contact@bloomwatch.com"

# Package imports for easy access
from . import data
from . import models
from . import utils
from . import visualization

__all__ = [
    "data",
    "models", 
    "utils",
    "visualization"
]