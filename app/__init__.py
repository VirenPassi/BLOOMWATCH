"""
FastAPI application for BloomWatch plant bloom detection.

This module provides a REST API for serving plant bloom detection models
with endpoints for prediction, health checks, and model information.
"""

from .main import app
from .endpoints import router
from .models import PredictionRequest, PredictionResponse
from .utils import load_model, preprocess_image

__all__ = [
    "app",
    "router",
    "PredictionRequest",
    "PredictionResponse", 
    "load_model",
    "preprocess_image"
]