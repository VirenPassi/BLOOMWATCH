"""
Model utilities for training, evaluation, and inference.

This module provides utility functions and classes for model management,
training loops, evaluation metrics, and model deployment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
import time
import json
import os
from pathlib import Path
import logging

from .baseline import create_baseline_model, count_parameters
from .losses import create_loss_function


logger = logging.getLogger(__name__)


class ModelUtils:
    """
    Utility class for model operations including saving, loading, and evaluation.
    """
    
    @staticmethod
    def save_model(
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        loss: float,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save model checkpoint with metadata.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer state
            epoch: Current epoch
            loss: Current loss value
            filepath: Path to save checkpoint
            metadata: Additional metadata to save
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metadata': metadata or {}
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(
        model: nn.Module,
        filepath: str,
        optimizer: Optional[optim.Optimizer] = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            model: PyTorch model to load weights into
            filepath: Path to checkpoint file
            optimizer: Optional optimizer to load state into
            device: Device to load model on
            
        Returns:
            Dict with checkpoint metadata
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', 0.0),
            'metadata': checkpoint.get('metadata', {})
        }
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count trainable and total parameters in model."""
        return count_parameters(model)
    
    @staticmethod
    def get_model_size(model: nn.Module) -> Dict[str, float]:
        """
        Get model size in MB.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dict with size information
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        
        return {
            'parameters_mb': param_size / 1024 / 1024,
            'buffers_mb': buffer_size / 1024 / 1024,
            'total_mb': size_mb
        }
    
    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        dummy_input: torch.Tensor,
        filepath: str,
        input_names: List[str] = None,
        output_names: List[str] = None
    ):
        """
        Export model to ONNX format.
        
        Args:
            model: PyTorch model
            dummy_input: Example input tensor
            filepath: Path to save ONNX model
            input_names: Names for input nodes
            output_names: Names for output nodes
        """
        model.eval()
        
        if input_names is None:
            input_names = ['input']
        if output_names is None:
            output_names = ['output']
        
        torch.onnx.export(
            model,
            dummy_input,
            filepath,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Model exported to ONNX: {filepath}")


class ModelFactory:
    """
    Factory class for creating models with different configurations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model factory with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def create_model(self, model_name: str = None) -> nn.Module:
        """
        Create model based on configuration.
        
        Args:
            model_name: Override model name from config
            
        Returns:
            nn.Module: Initialized model
        """
        model_name = model_name or self.config.get('model', {}).get('name', 'simple_cnn')
        model_config = self.config.get('model', {})
        
        if model_name == 'simple_cnn':
            from .baseline import SimpleCNN
            return SimpleCNN(
                num_classes=model_config.get('num_classes', 5),
                dropout=model_config.get('dropout', 0.5)
            )
        elif model_name == 'resnet':
            from .baseline import ResNetBaseline
            return ResNetBaseline(
                num_classes=model_config.get('num_classes', 5),
                backbone=model_config.get('backbone', 'resnet18'),
                pretrained=model_config.get('pretrained', True),
                dropout=model_config.get('dropout', 0.5)
            )
        elif model_name == 'timeseries':
            from .advanced import TimeSeriesBloomNet
            return TimeSeriesBloomNet(
                num_classes=model_config.get('num_classes', 5),
                sequence_length=model_config.get('sequence_length', 5)
            )
        elif model_name == 'attention':
            from .advanced import AttentionBloomNet
            return AttentionBloomNet(
                num_classes=model_config.get('num_classes', 5)
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """
        Create optimizer for model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Optimizer instance
        """
        training_config = self.config.get('training', {})
        optimizer_name = training_config.get('optimizer', 'adam').lower()
        lr = training_config.get('learning_rate', 0.001)
        weight_decay = training_config.get('weight_decay', 0.0001)
        
        if optimizer_name == 'adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = training_config.get('momentum', 0.9)
            return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        elif optimizer_name == 'adamw':
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def create_scheduler(self, optimizer: optim.Optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler.
        
        Args:
            optimizer: Optimizer instance
            
        Returns:
            Optional scheduler instance
        """
        training_config = self.config.get('training', {})
        scheduler_name = training_config.get('scheduler', None)
        
        if scheduler_name is None:
            return None
        elif scheduler_name == 'step':
            step_size = training_config.get('step_size', 20)
            gamma = training_config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'cosine':
            T_max = training_config.get('epochs', 50)
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        elif scheduler_name == 'exponential':
            gamma = training_config.get('gamma', 0.95)
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def create_loss_function(self) -> nn.Module:
        """
        Create loss function based on configuration.
        
        Returns:
            Loss function
        """
        model_config = self.config.get('model', {})
        num_classes = model_config.get('num_classes', 5)
        
        # For now, use cross entropy as default
        # Can be extended to use focal loss or custom losses
        return nn.CrossEntropyLoss()


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = np.inf
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Current model
            
        Returns:
            bool: True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class MetricsCalculator:
    """
    Calculate various metrics for bloom stage classification.
    """
    
    @staticmethod
    def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate accuracy."""
        pred_classes = torch.argmax(predictions, dim=1)
        return (pred_classes == targets).float().mean().item()
    
    @staticmethod
    def top_k_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
        """Calculate top-k accuracy."""
        _, top_k_preds = predictions.topk(k, dim=1)
        return (top_k_preds == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    
    @staticmethod
    def confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> np.ndarray:
        """Calculate confusion matrix."""
        pred_classes = torch.argmax(predictions, dim=1)
        cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        
        for actual, predicted in zip(targets, pred_classes):
            cm[actual, predicted] += 1
        
        return cm.numpy()
    
    @staticmethod
    def precision_recall_f1(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Dict[str, np.ndarray]:
        """Calculate precision, recall, and F1 score per class."""
        cm = MetricsCalculator.confusion_matrix(predictions, targets, num_classes)
        
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        f1 = np.zeros(num_classes)
        
        for i in range(num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


def profile_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cpu',
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Profile model performance (inference time, memory usage).
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (including batch size)
        device: Device to run profiling on
        num_runs: Number of inference runs for timing
        
    Returns:
        Dict with profiling results
    """
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Time inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / num_runs
    
    # Memory usage (rough estimate)
    if device == 'cuda':
        torch.cuda.synchronize()
        memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        memory_used = 0  # CPU memory tracking is more complex
    
    return {
        'avg_inference_time_ms': avg_inference_time * 1000,
        'fps': 1.0 / avg_inference_time,
        'memory_used_mb': memory_used
    }