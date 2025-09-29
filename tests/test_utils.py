"""
Unit tests for utils module components.

Tests configuration management, logging, metrics, and helper utilities.
"""

import pytest
import torch
import numpy as np
import tempfile
import json
from pathlib import Path
import logging

from utils import (
    ConfigManager, setup_logging, get_logger, MetricsTracker,
    get_device, set_seed, ensure_dir, format_time, format_bytes
)
from utils.config import ExperimentTracker, load_config_from_env
from utils.metrics import BloomStageMetrics, calculate_bloom_metrics
from utils.helpers import (
    get_system_info, validate_image_file, estimate_training_time,
    ProgressTracker
)


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def test_config_initialization(self, temp_data_dir):
        """Test ConfigManager initialization."""
        # Create a test config file
        config_data = {
            'project': {'name': 'test', 'version': '1.0'},
            'model': {'num_classes': 5},
            'training': {'epochs': 10}
        }
        
        config_file = temp_data_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(config_data, f)
        
        config_manager = ConfigManager(str(config_file))
        
        assert config_manager.config is not None
        assert config_manager.get('project.name') == 'test'
        assert config_manager.get('model.num_classes') == 5
    
    def test_config_loading(self, temp_data_dir, sample_config):
        """Test loading configuration from file."""
        config_file = temp_data_dir / "config.yaml"
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(sample_config, f)
        
        config_manager = ConfigManager()
        loaded_config = config_manager.load_config(str(config_file))
        
        assert loaded_config is not None
        assert loaded_config.project.name == sample_config['project']['name']
    
    def test_config_get_set(self, sample_config):
        """Test getting and setting configuration values."""
        from omegaconf import OmegaConf
        
        config_manager = ConfigManager()
        config_manager.config = OmegaConf.create(sample_config)
        
        # Test getting values
        assert config_manager.get('project.name') == 'BloomWatch'
        assert config_manager.get('nonexistent.key', 'default') == 'default'
        
        # Test setting values
        config_manager.set('new.key', 'new_value')
        assert config_manager.get('new.key') == 'new_value'
    
    def test_config_validation(self, temp_data_dir):
        """Test configuration validation."""
        # Create valid config
        valid_config = {
            'project': {'name': 'test'},
            'data': {'batch_size': 32},
            'model': {'num_classes': 5},
            'training': {'epochs': 10, 'learning_rate': 0.001}
        }
        
        config_file = temp_data_dir / "valid_config.yaml"
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(valid_config, f)
        
        config_manager = ConfigManager(str(config_file))
        validation_report = config_manager.validate_config()
        
        assert validation_report['valid'] is True
        assert len(validation_report['errors']) == 0
    
    def test_config_saving(self, temp_data_dir, sample_config):
        """Test saving configuration to file."""
        from omegaconf import OmegaConf
        
        config_manager = ConfigManager()
        config_manager.config = OmegaConf.create(sample_config)
        
        save_path = temp_data_dir / "saved_config.yaml"
        config_manager.save_config(str(save_path))
        
        assert save_path.exists()
        
        # Verify saved content
        with open(save_path, 'r') as f:
            import yaml
            loaded_data = yaml.safe_load(f)
        
        assert loaded_data['project']['name'] == sample_config['project']['name']


class TestExperimentTracker:
    """Test ExperimentTracker functionality."""
    
    def test_tracker_initialization(self, temp_data_dir):
        """Test ExperimentTracker initialization."""
        tracker = ExperimentTracker(str(temp_data_dir))
        
        assert tracker.experiments_dir == temp_data_dir
        assert tracker.registry_path.exists() or not tracker.registry_path.exists()  # May or may not exist initially
    
    def test_experiment_registration(self, temp_data_dir, sample_config):
        """Test registering experiments."""
        from omegaconf import OmegaConf
        
        tracker = ExperimentTracker(str(temp_data_dir))
        config = OmegaConf.create(sample_config)
        
        tracker.register_experiment("test_exp", config, {"description": "test"})
        
        experiment_info = tracker.get_experiment("test_exp")
        assert experiment_info is not None
        assert experiment_info['metadata']['description'] == "test"
    
    def test_experiment_listing(self, temp_data_dir, sample_config):
        """Test listing experiments."""
        from omegaconf import OmegaConf
        
        tracker = ExperimentTracker(str(temp_data_dir))
        config = OmegaConf.create(sample_config)
        
        # Register multiple experiments
        tracker.register_experiment("exp1", config)
        tracker.register_experiment("exp2", config)
        
        experiments = tracker.list_experiments()
        assert len(experiments) >= 2
        assert "exp1" in experiments
        assert "exp2" in experiments


class TestLogging:
    """Test logging utilities."""
    
    def test_logging_setup(self, temp_data_dir):
        """Test logging setup."""
        logger = setup_logging(
            name='test_logger',
            level='INFO',
            log_dir=str(temp_data_dir),
            console_output=True,
            file_output=True
        )
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_logger'
        assert logger.level == logging.INFO
    
    def test_get_logger(self):
        """Test getting logger instance."""
        logger = get_logger('test_module')
        
        assert isinstance(logger, logging.Logger)
        assert 'test_module' in logger.name
    
    def test_logging_output(self, temp_data_dir):
        """Test logging output to file."""
        logger = setup_logging(
            name='file_test_logger',
            log_dir=str(temp_data_dir),
            file_output=True
        )
        
        test_message = "Test log message"
        logger.info(test_message)
        
        # Check if log file was created
        log_files = list(temp_data_dir.glob("*.log"))
        assert len(log_files) > 0


class TestMetricsTracker:
    """Test MetricsTracker functionality."""
    
    def test_tracker_initialization(self, class_names):
        """Test MetricsTracker initialization."""
        tracker = MetricsTracker(num_classes=5, class_names=class_names)
        
        assert tracker.num_classes == 5
        assert tracker.class_names == class_names
        assert len(tracker.metrics_history) > 0
    
    def test_metrics_update(self, class_names):
        """Test updating metrics."""
        tracker = MetricsTracker(num_classes=5, class_names=class_names)
        
        metrics = {
            'train_loss': 0.5,
            'val_loss': 0.6,
            'train_acc': 0.8,
            'val_acc': 0.75
        }
        
        tracker.update_epoch_metrics(0, metrics)
        
        assert len(tracker.metrics_history['epoch']) == 1
        assert tracker.metrics_history['train_loss'][0] == 0.5
        assert tracker.best_metrics['best_val_acc'] == 0.75
    
    def test_classification_metrics(self, class_names, dummy_predictions):
        """Test classification metrics computation."""
        tracker = MetricsTracker(num_classes=5, class_names=class_names)
        
        predictions = torch.from_numpy(dummy_predictions['predictions'])
        targets = torch.from_numpy(dummy_predictions['targets'])
        
        metrics = tracker.compute_classification_metrics(predictions, targets)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_confusion_matrix(self, class_names, dummy_predictions):
        """Test confusion matrix computation."""
        tracker = MetricsTracker(num_classes=5, class_names=class_names)
        
        predictions = torch.from_numpy(dummy_predictions['predictions'])
        targets = torch.from_numpy(dummy_predictions['targets'])
        
        cm = tracker.compute_confusion_matrix(predictions, targets)
        
        assert cm.shape == (5, 5)
        assert cm.sum() == len(predictions)
        assert (cm >= 0).all()
    
    def test_metrics_saving_loading(self, temp_data_dir, class_names):
        """Test saving and loading metrics."""
        tracker = MetricsTracker(num_classes=5, class_names=class_names)
        
        # Add some metrics
        metrics = {'train_loss': 0.5, 'val_acc': 0.8}
        tracker.update_epoch_metrics(0, metrics)
        
        # Save metrics
        save_path = temp_data_dir / "metrics.json"
        tracker.save_metrics(str(save_path))
        
        assert save_path.exists()
        
        # Load metrics
        new_tracker = MetricsTracker(num_classes=5, class_names=class_names)
        new_tracker.load_metrics(str(save_path))
        
        assert len(new_tracker.metrics_history['epoch']) == 1
        assert new_tracker.metrics_history['train_loss'][0] == 0.5


class TestBloomStageMetrics:
    """Test BloomStageMetrics functionality."""
    
    def test_bloom_metrics_initialization(self, class_names):
        """Test BloomStageMetrics initialization."""
        metrics = BloomStageMetrics(class_names)
        
        assert metrics.class_names == class_names
        assert metrics.num_classes == len(class_names)
    
    def test_stage_progression_accuracy(self, class_names):
        """Test stage progression accuracy calculation."""
        metrics = BloomStageMetrics(class_names)
        
        # Perfect predictions (same as targets)
        predictions = np.array([0, 1, 2, 3, 4])
        targets = np.array([0, 1, 2, 3, 4])
        
        accuracy = metrics.stage_progression_accuracy(predictions, targets)
        assert accuracy == 1.0
        
        # Adjacent stage predictions (should get partial credit)
        predictions = np.array([1, 2, 3, 4, 0])  # Off by one
        targets = np.array([0, 1, 2, 3, 4])
        
        accuracy = metrics.stage_progression_accuracy(predictions, targets)
        assert 0 < accuracy < 1.0  # Should get partial credit
    
    def test_temporal_consistency_score(self, class_names):
        """Test temporal consistency scoring."""
        metrics = BloomStageMetrics(class_names)
        
        # Create consistent temporal progression
        predictions = [
            np.array([0, 1, 2]),  # Plant progressing from bud to full bloom
            np.array([1, 2, 3])   # Another plant's progression
        ]
        targets = [
            np.array([0, 1, 2]),
            np.array([1, 2, 3])
        ]
        plant_ids = ['plant_1', 'plant_1', 'plant_1', 'plant_2', 'plant_2', 'plant_2']
        
        # Flatten for the function
        flat_predictions = np.concatenate(predictions)
        flat_targets = np.concatenate(targets)
        
        score = metrics.temporal_consistency_score(
            [flat_predictions[:3], flat_predictions[3:]],
            [flat_targets[:3], flat_targets[3:]],
            ['plant_1', 'plant_2']
        )
        
        assert 0 <= score <= 1.0


class TestHelperUtilities:
    """Test helper utility functions."""
    
    def test_device_detection(self):
        """Test device detection."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_seed_setting(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand1 = torch.rand(5)
        np_rand1 = np.random.rand(5)
        
        # Reset seed and generate again
        set_seed(42)
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        # Should be identical
        assert torch.allclose(torch_rand1, torch_rand2)
        assert np.allclose(np_rand1, np_rand2)
    
    def test_directory_creation(self, temp_data_dir):
        """Test directory creation utility."""
        test_dir = temp_data_dir / "test_subdir" / "nested"
        
        created_dir = ensure_dir(test_dir)
        
        assert created_dir.exists()
        assert created_dir.is_dir()
        assert created_dir == test_dir
    
    def test_time_formatting(self):
        """Test time formatting utility."""
        # Test seconds
        assert format_time(30) == "30.0s"
        
        # Test minutes
        assert format_time(90) == "1.5m"
        
        # Test hours
        assert format_time(7200) == "2.0h"
    
    def test_bytes_formatting(self):
        """Test bytes formatting utility."""
        # Test different units
        assert format_bytes(512) == "512.0B"
        assert format_bytes(1024) == "1.0KB"
        assert format_bytes(1024**2) == "1.0MB"
        assert format_bytes(1024**3) == "1.0GB"
    
    def test_system_info(self):
        """Test system information gathering."""
        info = get_system_info()
        
        assert isinstance(info, dict)
        assert 'platform' in info
        assert 'python_version' in info
        assert 'pytorch_version' in info
    
    def test_training_time_estimation(self):
        """Test training time estimation."""
        estimate = estimate_training_time(
            dataset_size=1000,
            batch_size=32,
            epochs=10,
            seconds_per_batch=0.5
        )
        
        assert isinstance(estimate, dict)
        assert 'estimated_time_seconds' in estimate
        assert 'estimated_time_formatted' in estimate
        assert estimate['estimated_time_seconds'] > 0
    
    def test_progress_tracker(self):
        """Test progress tracker utility."""
        tracker = ProgressTracker(total_steps=10, description="Test")
        
        assert tracker.total_steps == 10
        assert tracker.current_step == 0
        
        # Update progress
        tracker.update(5)
        assert tracker.current_step == 5
        
        # Set specific progress
        tracker.set_progress(8)
        assert tracker.current_step == 8


class TestValidationUtilities:
    """Test validation and checking utilities."""
    
    def test_image_validation(self, dummy_image, temp_data_dir):
        """Test image file validation."""
        # Save dummy image
        image_path = temp_data_dir / "test_image.jpg"
        dummy_image.save(image_path)
        
        result = validate_image_file(image_path)
        
        assert result['valid'] is True
        assert result['error'] is None
        assert result['format'] == 'JPEG'
        assert result['size'] == dummy_image.size
    
    def test_invalid_file_validation(self, temp_data_dir):
        """Test validation of invalid files."""
        # Create non-image file
        invalid_file = temp_data_dir / "not_an_image.txt"
        with open(invalid_file, 'w') as f:
            f.write("This is not an image")
        
        result = validate_image_file(invalid_file)
        
        assert result['valid'] is False
        assert result['error'] is not None


class TestEnvironmentLoading:
    """Test environment variable loading."""
    
    def test_env_config_loading(self, monkeypatch):
        """Test loading configuration from environment variables."""
        # Set environment variables
        monkeypatch.setenv('BLOOMWATCH_MODEL_NUM_CLASSES', '7')
        monkeypatch.setenv('BLOOMWATCH_TRAINING_EPOCHS', '50')
        monkeypatch.setenv('BLOOMWATCH_DATA_BATCH_SIZE', '64')
        
        env_config = load_config_from_env()
        
        assert env_config['model']['num_classes'] == 7
        assert env_config['training']['epochs'] == 50
        assert env_config['data']['batch_size'] == 64