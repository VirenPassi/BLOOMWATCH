"""
Utility functions for the FastAPI application.

This module provides model management, image preprocessing,
and other utility functions for the API.
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import asyncio
import time
import logging
import base64
import io
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import psutil

logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str = 'auto') -> nn.Module:
    """
    Load a model from file.
    
    Args:
        model_path: Path to the model file
        device: Device to load model on ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        nn.Module: Loaded model
    """
    # For now, return a dummy model since we don't have actual trained models
    # In production, this would load the actual saved model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, 5)
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc(x)
            return x
    
    model = DummyModel()
    
    # Set device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    model = model.to(torch.device(device))
    model.eval()
    
    return model


def preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> torch.Tensor:
    """
    Preprocess an image for model inference.
    
    Args:
        image: PIL Image to preprocess
        target_size: Target size (height, width)
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Resize image
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Convert to tensor and rearrange dimensions (H, W, C) -> (C, H, W)
    tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    
    # Apply ImageNet normalization if requested
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor


def postprocess_predictions(
    logits: torch.Tensor,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Postprocess model predictions.
    
    Args:
        logits: Raw model output logits
        class_names: Optional list of class names
        
    Returns:
        Dict with processed predictions
    """
    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=1)
    
    # Get predicted class
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0, predicted_class].item()
    
    # Prepare class confidences
    class_confidences = {}
    if class_names is None:
        class_names = ['bud', 'early_bloom', 'full_bloom', 'late_bloom', 'dormant']
    
    for i, name in enumerate(class_names):
        if i < probabilities.shape[1]:
            class_confidences[name] = probabilities[0, i].item()
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'class_confidences': class_confidences,
        'probabilities': probabilities.squeeze().tolist()
    }


def encode_image_to_base64(image: Image.Image, format: str = 'JPEG') -> str:
    """
    Encode PIL Image to base64 string.
    
    Args:
        image: PIL Image to encode
        format: Image format for encoding
        
    Returns:
        str: Base64 encoded image
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


def decode_base64_to_image(base64_str: str) -> Image.Image:
    """
    Decode base64 string to PIL Image.
    
    Args:
        base64_str: Base64 encoded image string
        
    Returns:
        PIL.Image: Decoded image
    """
    # Remove data URL prefix if present
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    
    # Decode base64
    img_data = base64.b64decode(base64_str)
    
    # Create PIL Image
    image = Image.open(io.BytesIO(img_data)).convert('RGB')
    
    return image


class ModelManager:
    """
    Manages multiple models for the API.
    
    Handles model loading, unloading, and inference with proper
    resource management and error handling.
    """
    
    def __init__(self, config=None):
        """
        Initialize model manager.
        
        Args:
            config: Configuration object
        """
        self.models: Dict[str, Dict[str, Any]] = {}
        self.config = config
        self.device = self._get_device()
        self.class_names = ['bud', 'early_bloom', 'full_bloom', 'late_bloom', 'dormant']
        
        logger.info(f"ModelManager initialized with device: {self.device}")
    
    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    async def load_model(self, model_name: str, model_path: str) -> bool:
        """
        Load a model from file.
        
        Args:
            model_name: Name to assign to the model
            model_path: Path to the model file
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Loading model '{model_name}' from {model_path}")
            
            # Check if model file exists
            if not Path(model_path).exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # For now, create a dummy model since we don't have actual trained models
            # In production, this would load the actual saved model
            model = self._create_dummy_model()
            model = model.to(self.device)
            model.eval()
            
            # Store model info
            self.models[model_name] = {
                'model': model,
                'path': model_path,
                'device': str(self.device),
                'num_classes': 5,
                'input_size': (224, 224),
                'architecture': 'SimpleCNN',
                'loaded_at': time.time(),
                'total_predictions': 0,
                'processing_times': [],
                'version': '1.0.0'
            }
            
            logger.info(f"Model '{model_name}' loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {str(e)}")
            return False
    
    def _create_dummy_model(self) -> nn.Module:
        """
        Create a dummy model for demonstration.
        
        In production, this would be replaced with actual model loading.
        """
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(64, 5)
                self.dropout = nn.Dropout(0.5)
            
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.dropout(x)
                x = self.fc(x)
                return x
        
        return DummyModel()
    
    async def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            bool: Success status
        """
        try:
            if model_name not in self.models:
                logger.warning(f"Model '{model_name}' not found for unloading")
                return False
            
            # Clean up model
            del self.models[model_name]['model']
            del self.models[model_name]
            
            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Model '{model_name}' unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model '{model_name}': {str(e)}")
            return False
    
    async def predict(
        self,
        model_name: str,
        image_tensor: torch.Tensor,
        include_confidence: bool = True,
        include_features: bool = False
    ) -> Dict[str, Any]:
        """
        Make a prediction using the specified model.
        
        Args:
            model_name: Name of the model to use
            image_tensor: Preprocessed image tensor
            include_confidence: Whether to include confidence scores
            include_features: Whether to include extracted features
            
        Returns:
            Dict with prediction results
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model_info = self.models[model_name]
        model = model_info['model']
        
        start_time = time.time()
        
        try:
            # Move tensor to model device
            image_tensor = image_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                logits = model(image_tensor)
            
            # Postprocess predictions
            results = postprocess_predictions(logits, self.class_names)
            
            # Add timing information
            processing_time = time.time() - start_time
            results['processing_time'] = processing_time
            
            # Update model statistics
            model_info['total_predictions'] += 1
            model_info['processing_times'].append(processing_time)
            
            # Keep only last 100 processing times for statistics
            if len(model_info['processing_times']) > 100:
                model_info['processing_times'] = model_info['processing_times'][-100:]
            
            # Add features if requested
            if include_features:
                # For demonstration, return random features
                # In production, this would extract actual features from the model
                results['features'] = torch.randn(512).tolist()
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed for model '{model_name}': {str(e)}")
            raise
    
    async def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get model information.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict with model information or None if not found
        """
        if model_name not in self.models:
            return None
        
        model_info = self.models[model_name].copy()
        
        # Remove the actual model object from the returned info
        if 'model' in model_info:
            del model_info['model']
        
        # Add computed statistics
        processing_times = self.models[model_name]['processing_times']
        if processing_times:
            model_info['avg_processing_time'] = np.mean(processing_times)
            model_info['min_processing_time'] = np.min(processing_times)
            model_info['max_processing_time'] = np.max(processing_times)
        
        return model_info
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List all loaded models.
        
        Returns:
            List of model information dictionaries
        """
        models_list = []
        
        for model_name in self.models:
            model_info = await self.get_model(model_name)
            if model_info:
                model_info['name'] = model_name
                models_list.append(model_info)
        
        return models_list
    
    def get_model_status(self) -> Dict[str, str]:
        """
        Get simple status of all models.
        
        Returns:
            Dict mapping model names to their status
        """
        return {name: 'loaded' for name in self.models.keys()}
    
    async def get_model_status_detailed(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status of a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict with detailed model status
        """
        if model_name not in self.models:
            return None
        
        model_info = self.models[model_name]
        processing_times = model_info['processing_times']
        
        # Get memory usage (approximate)
        memory_usage = 0
        if torch.cuda.is_available() and str(self.device) == 'cuda':
            memory_usage = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        status = {
            'name': model_name,
            'status': 'loaded',
            'device': model_info['device'],
            'memory_usage_mb': memory_usage,
            'total_predictions': model_info['total_predictions'],
            'load_time': model_info['loaded_at'],
            'health_status': 'healthy'
        }
        
        if processing_times:
            status['avg_processing_time'] = np.mean(processing_times)
            status['last_prediction_time'] = time.time()  # Approximate
        
        return status
    
    def cleanup(self):
        """Clean up all models and resources."""
        try:
            for model_name in list(self.models.keys()):
                asyncio.create_task(self.unload_model(model_name))
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("ModelManager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


class AppConfig:
    """
    Application configuration class.
    
    Manages configuration settings for the API.
    """
    
    def __init__(self):
        """Initialize configuration with default values."""
        self.default_model_path = None  # Path to default model
        self.max_batch_size = 50
        self.max_image_size = 10 * 1024 * 1024  # 10MB
        self.allowed_image_types = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff']
        self.model_timeout = 30  # seconds
        self.enable_caching = True
        self.log_level = 'INFO'


def get_system_stats() -> Dict[str, Any]:
    """
    Get current system statistics.
    
    Returns:
        Dict with system information
    """
    try:
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'gpu_memory_used': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        return {}


async def health_check_models(model_manager: ModelManager) -> Dict[str, str]:
    """
    Perform health check on all loaded models.
    
    Args:
        model_manager: ModelManager instance
        
    Returns:
        Dict with model health status
    """
    health_status = {}
    
    try:
        models = await model_manager.list_models()
        
        for model_info in models:
            model_name = model_info['name']
            
            try:
                # Try a test prediction with dummy data
                dummy_tensor = torch.randn(1, 3, 224, 224)
                await model_manager.predict(model_name, dummy_tensor)
                health_status[model_name] = 'healthy'
                
            except Exception as e:
                logger.error(f"Health check failed for model '{model_name}': {str(e)}")
                health_status[model_name] = 'unhealthy'
    
    except Exception as e:
        logger.error(f"Error during model health check: {str(e)}")
    
    return health_status