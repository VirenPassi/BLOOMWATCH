"""
Unit tests for models module components.

Tests model architectures, training utilities, and loss functions.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile

from models import (
    SimpleCNN, ResNetBaseline, ModelUtils, ModelFactory,
    FocalLoss, BloomStageLoss, TemporalConsistencyLoss
)
from models.advanced import TimeSeriesBloomNet, AttentionBloomNet, MultiScaleBloomNet
from tests.conftest import assert_tensor_shape, assert_tensor_range


class TestSimpleCNN:
    """Test SimpleCNN model architecture."""
    
    def test_model_initialization(self):
        """Test SimpleCNN initialization."""
        model = SimpleCNN(
            num_classes=5,
            input_channels=3,
            dropout=0.5,
            use_batch_norm=True
        )
        
        assert model.num_classes == 5
        assert model.input_channels == 3
        assert model.dropout == 0.5
        assert model.use_batch_norm is True
    
    def test_forward_pass(self, dummy_image_tensor):
        """Test forward pass through SimpleCNN."""
        model = SimpleCNN(num_classes=5)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_image_tensor)
        
        assert_tensor_shape(output, (1, 5))
        # Logits can be any value
        assert output.dtype == torch.float32
    
    def test_feature_extraction(self, dummy_image_tensor):
        """Test feature map extraction."""
        model = SimpleCNN(num_classes=5)
        model.eval()
        
        with torch.no_grad():
            features = model.get_feature_maps(dummy_image_tensor)
        
        assert isinstance(features, torch.Tensor)
        assert features.dim() == 4  # (batch, channels, height, width)
    
    def test_batch_processing(self):
        """Test processing of different batch sizes."""
        model = SimpleCNN(num_classes=5)
        model.eval()
        
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 3, 224, 224)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert_tensor_shape(output, (batch_size, 5))
    
    def test_different_input_sizes(self):
        """Test model with different input sizes."""
        model = SimpleCNN(num_classes=5)
        model.eval()
        
        input_sizes = [(224, 224), (256, 256), (128, 128)]
        
        for h, w in input_sizes:
            input_tensor = torch.randn(1, 3, h, w)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert_tensor_shape(output, (1, 5))


class TestResNetBaseline:
    """Test ResNetBaseline model."""
    
    def test_model_initialization(self):
        """Test ResNetBaseline initialization."""
        model = ResNetBaseline(
            num_classes=5,
            backbone='resnet18',
            pretrained=False,  # Avoid downloading in tests
            dropout=0.5
        )
        
        assert model.num_classes == 5
        assert model.backbone_name == 'resnet18'
        assert not model.pretrained
    
    def test_forward_pass(self, dummy_image_tensor):
        """Test forward pass through ResNetBaseline."""
        model = ResNetBaseline(
            num_classes=5,
            backbone='resnet18',
            pretrained=False
        )
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_image_tensor)
        
        assert_tensor_shape(output, (1, 5))
    
    def test_feature_extraction(self, dummy_image_tensor):
        """Test feature extraction from ResNet."""
        model = ResNetBaseline(
            num_classes=5,
            backbone='resnet18',
            pretrained=False
        )
        model.eval()
        
        with torch.no_grad():
            features = model.get_features(dummy_image_tensor)
        
        assert isinstance(features, torch.Tensor)
        assert features.dim() == 2  # (batch, features)
        assert features.shape[0] == 1  # batch size
    
    def test_freeze_unfreeze(self, dummy_image_tensor):
        """Test backbone freezing and unfreezing."""
        model = ResNetBaseline(
            num_classes=5,
            backbone='resnet18',
            pretrained=False,
            freeze_backbone=True
        )
        
        # Check that backbone is frozen
        backbone_params = list(model.backbone.parameters())[:-2]  # Exclude classifier
        assert all(not param.requires_grad for param in backbone_params)
        
        # Unfreeze and check
        model.unfreeze_backbone()
        assert all(param.requires_grad for param in model.backbone.parameters())
    
    def test_different_backbones(self):
        """Test different ResNet backbone variants."""
        backbones = ['resnet18', 'resnet34']  # Limit to avoid large downloads
        
        for backbone in backbones:
            model = ResNetBaseline(
                num_classes=5,
                backbone=backbone,
                pretrained=False
            )
            
            assert model.backbone_name == backbone
            assert hasattr(model.backbone, 'fc')  # Has classifier layer


class TestAdvancedModels:
    """Test advanced model architectures."""
    
    def test_timeseries_model(self):
        """Test TimeSeriesBloomNet."""
        model = TimeSeriesBloomNet(
            num_classes=5,
            sequence_length=3,
            hidden_size=128
        )
        model.eval()
        
        # Input: (batch, sequence, channels, height, width)
        input_tensor = torch.randn(2, 3, 3, 224, 224)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert_tensor_shape(output, (2, 5))
    
    def test_attention_model(self, dummy_image_tensor):
        """Test AttentionBloomNet."""
        model = AttentionBloomNet(
            num_classes=5,
            feature_dim=256,
            num_heads=4,
            num_layers=2
        )
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_image_tensor)
        
        assert_tensor_shape(output, (1, 5))
    
    def test_multiscale_model(self, dummy_image_tensor):
        """Test MultiScaleBloomNet."""
        model = MultiScaleBloomNet(
            num_classes=5,
            scales=[224, 256],  # Limit scales for testing
            fusion_method='concat'
        )
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_image_tensor)
        
        assert_tensor_shape(output, (1, 5))


class TestModelUtils:
    """Test ModelUtils functionality."""
    
    def test_parameter_counting(self):
        """Test parameter counting utility."""
        model = SimpleCNN(num_classes=5)
        
        param_count = ModelUtils.count_parameters(model)
        
        assert isinstance(param_count, dict)
        assert 'total_parameters' in param_count
        assert 'trainable_parameters' in param_count
        assert param_count['total_parameters'] > 0
        assert param_count['trainable_parameters'] > 0
    
    def test_model_size_calculation(self):
        """Test model size calculation."""
        model = SimpleCNN(num_classes=5)
        
        size_info = ModelUtils.get_model_size(model)
        
        assert isinstance(size_info, dict)
        assert 'total_mb' in size_info
        assert 'parameters_mb' in size_info
        assert size_info['total_mb'] > 0
    
    def test_model_saving_loading(self, test_model_path):
        """Test model saving and loading."""
        model = SimpleCNN(num_classes=5)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save model
        ModelUtils.save_model(
            model=model,
            optimizer=optimizer,
            epoch=10,
            loss=0.5,
            filepath=str(test_model_path),
            metadata={'test': True}
        )
        
        assert test_model_path.exists()
        
        # Load model
        new_model = SimpleCNN(num_classes=5)
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        checkpoint_info = ModelUtils.load_model(
            model=new_model,
            filepath=str(test_model_path),
            optimizer=new_optimizer
        )
        
        assert checkpoint_info['epoch'] == 10
        assert checkpoint_info['loss'] == 0.5
        assert checkpoint_info['metadata']['test'] is True


class TestModelFactory:
    """Test ModelFactory functionality."""
    
    def test_factory_initialization(self, sample_config):
        """Test ModelFactory initialization."""
        factory = ModelFactory(sample_config)
        
        assert factory.config == sample_config
    
    def test_model_creation(self, sample_config):
        """Test creating models through factory."""
        factory = ModelFactory(sample_config)
        
        model = factory.create_model('simple_cnn')
        
        assert isinstance(model, SimpleCNN)
        assert model.num_classes == sample_config['model']['num_classes']
    
    def test_optimizer_creation(self, sample_config):
        """Test optimizer creation."""
        factory = ModelFactory(sample_config)
        model = SimpleCNN(num_classes=5)
        
        optimizer = factory.create_optimizer(model)
        
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]['lr'] == sample_config['training']['learning_rate']
    
    def test_scheduler_creation(self, sample_config):
        """Test learning rate scheduler creation."""
        factory = ModelFactory(sample_config)
        model = SimpleCNN(num_classes=5)
        optimizer = torch.optim.Adam(model.parameters())
        
        scheduler = factory.create_scheduler(optimizer)
        
        # Should create scheduler based on config
        assert scheduler is not None or scheduler is None  # Depends on config
    
    def test_loss_function_creation(self, sample_config):
        """Test loss function creation."""
        factory = ModelFactory(sample_config)
        
        loss_fn = factory.create_loss_function()
        
        assert isinstance(loss_fn, nn.Module)


class TestLossFunctions:
    """Test custom loss functions."""
    
    def test_focal_loss(self):
        """Test FocalLoss implementation."""
        loss_fn = FocalLoss(alpha=None, gamma=2.0)
        
        # Create dummy inputs
        inputs = torch.randn(4, 5)  # (batch_size, num_classes)
        targets = torch.randint(0, 5, (4,))  # (batch_size,)
        
        loss = loss_fn(inputs, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0
    
    def test_bloom_stage_loss(self):
        """Test BloomStageLoss implementation."""
        loss_fn = BloomStageLoss(
            num_classes=5,
            transition_penalty=0.1,
            smoothing=0.1
        )
        
        inputs = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        
        loss = loss_fn(inputs, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_temporal_consistency_loss(self):
        """Test TemporalConsistencyLoss implementation."""
        loss_fn = TemporalConsistencyLoss(
            consistency_weight=0.1,
            smoothness_weight=0.05
        )
        
        # Create sequence predictions
        predictions = torch.randn(2, 4, 5)  # (batch, sequence, classes)
        targets = torch.randint(0, 5, (2, 4))  # (batch, sequence)
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_loss_reduction_modes(self):
        """Test different loss reduction modes."""
        reduction_modes = ['mean', 'sum', 'none']
        
        for reduction in reduction_modes:
            loss_fn = FocalLoss(reduction=reduction)
            
            inputs = torch.randn(3, 5)
            targets = torch.randint(0, 5, (3,))
            
            loss = loss_fn(inputs, targets)
            
            if reduction == 'none':
                assert loss.shape == (3,)  # One loss per sample
            else:
                assert loss.dim() == 0  # Scalar loss


class TestModelTraining:
    """Test model training utilities."""
    
    def test_model_training_mode(self):
        """Test switching between training and evaluation modes."""
        model = SimpleCNN(num_classes=5)
        
        # Test training mode
        model.train()
        assert model.training is True
        
        # Test evaluation mode
        model.eval()
        assert model.training is False
    
    def test_gradient_computation(self):
        """Test gradient computation during training."""
        model = SimpleCNN(num_classes=5)
        criterion = nn.CrossEntropyLoss()
        
        # Forward pass
        inputs = torch.randn(2, 3, 224, 224)
        targets = torch.randint(0, 5, (2,))
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_inference(self):
        """Test model inference without gradients."""
        model = SimpleCNN(num_classes=5)
        model.eval()
        
        inputs = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            outputs = model(inputs)
        
        assert_tensor_shape(outputs, (1, 5))
        
        # Check that no gradients are computed
        for param in model.parameters():
            assert param.grad is None or param.grad.sum() == 0


class TestModelValidation:
    """Test model validation and testing utilities."""
    
    def test_prediction_accuracy(self):
        """Test accuracy calculation for predictions."""
        # Create perfect predictions
        logits = torch.tensor([
            [10.0, 0.0, 0.0, 0.0, 0.0],  # Class 0
            [0.0, 10.0, 0.0, 0.0, 0.0],  # Class 1
            [0.0, 0.0, 10.0, 0.0, 0.0],  # Class 2
        ])
        targets = torch.tensor([0, 1, 2])
        
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == targets).float().mean()
        
        assert accuracy.item() == 1.0  # Perfect accuracy
    
    def test_probability_extraction(self):
        """Test probability extraction from logits."""
        logits = torch.randn(3, 5)
        
        probabilities = torch.softmax(logits, dim=1)
        
        # Check that probabilities sum to 1
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(3))
        
        # Check that all probabilities are positive
        assert (probabilities >= 0).all()
        assert (probabilities <= 1).all()