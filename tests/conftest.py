"""
Test configuration and fixtures for BloomWatch tests.

This module provides pytest configuration, shared fixtures,
and utility functions for testing the BloomWatch project.
"""

import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
import pandas as pd

# Test configuration
pytest_plugins = []


@pytest.fixture
def device():
    """Fixture providing the test device (CPU for consistency)."""
    return torch.device('cpu')


@pytest.fixture
def dummy_image():
    """Fixture providing a dummy RGB image."""
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(image_array)


@pytest.fixture
def dummy_image_tensor():
    """Fixture providing a dummy image tensor."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_config():
    """Fixture providing sample configuration dictionary."""
    return {
        'project': {
            'name': 'BloomWatch',
            'version': '0.1.0'
        },
        'data': {
            'batch_size': 32,
            'image_size': [224, 224],
            'num_workers': 2
        },
        'model': {
            'name': 'simple_cnn',
            'num_classes': 5,
            'dropout': 0.5
        },
        'training': {
            'epochs': 10,
            'learning_rate': 0.001,
            'weight_decay': 0.0001
        }
    }


@pytest.fixture
def temp_data_dir():
    """Fixture providing a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_annotations():
    """Fixture providing sample annotation data."""
    return pd.DataFrame({
        'image_path': [f'image_{i:03d}.jpg' for i in range(20)],
        'bloom_stage': ['bud', 'early_bloom', 'full_bloom', 'late_bloom', 'dormant'] * 4,
        'stage': ['train'] * 15 + ['val'] * 3 + ['test'] * 2,
        'plant_id': [f'plant_{i//4:02d}' for i in range(20)],
        'timestamp': [f'2024-01-{i+1:02d}' for i in range(20)]
    })


@pytest.fixture
def class_names():
    """Fixture providing bloom stage class names."""
    return ['bud', 'early_bloom', 'full_bloom', 'late_bloom', 'dormant']


@pytest.fixture
def dummy_predictions():
    """Fixture providing dummy prediction data."""
    np.random.seed(42)
    return {
        'predictions': np.random.randint(0, 5, 50),
        'targets': np.random.randint(0, 5, 50),
        'probabilities': np.random.rand(50, 5)
    }


@pytest.fixture(scope="session")
def test_model_path():
    """Fixture providing path for test model saving."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir) / "test_model.pth"


# Helper functions for tests
def create_dummy_dataset_files(data_dir: Path, annotations: pd.DataFrame):
    """Create dummy image files for testing."""
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy images
    for image_path in annotations['image_path']:
        img_file = data_dir / image_path
        dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(dummy_img).save(img_file)
    
    # Save annotations
    annotations.to_csv(data_dir / "annotations.csv", index=False)


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple):
    """Assert that tensor has expected shape."""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"


def assert_tensor_range(tensor: torch.Tensor, min_val: float = None, max_val: float = None):
    """Assert that tensor values are within expected range."""
    if min_val is not None:
        assert tensor.min() >= min_val, f"Tensor minimum {tensor.min()} below expected {min_val}"
    if max_val is not None:
        assert tensor.max() <= max_val, f"Tensor maximum {tensor.max()} above expected {max_val}"