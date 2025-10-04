"""
Baseline CNN models for plant bloom stage detection.

This module contains simple and effective baseline models that can
serve as starting points for plant bloom classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Dict, Any

class SimpleCNN(nn.Module):
 """
 Simple CNN baseline for plant bloom stage classification.
 
 A lightweight convolutional neural network designed specifically
 for plant bloom stage detection with good performance on small datasets.
 """
 
 def __init__(
 self,
 num_classes: int = 5,
 input_channels: int = 3,
 dropout: float = 0.5,
 use_batch_norm: bool = True
 ):
 """
 Initialize SimpleCNN model.
 
 Args:
 num_classes: Number of bloom stage classes
 input_channels: Number of input channels (3 for RGB)
 dropout: Dropout rate for regularization
 use_batch_norm: Whether to use batch normalization
 """
 super(SimpleCNN, self).__init__()
 
 self.num_classes = num_classes
 self.input_channels = input_channels
 self.dropout = dropout
 self.use_batch_norm = use_batch_norm
 
 # Feature extraction layers
 self.features = self._make_feature_layers()
 
 # Adaptive pooling to handle variable input sizes
 self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
 
 # Classifier layers
 self.classifier = self._make_classifier_layers()
 
 # Initialize weights
 self._initialize_weights()
 
 def _make_feature_layers(self) -> nn.Sequential:
 """Create the feature extraction layers."""
 layers = []
 in_channels = self.input_channels
 
 # Configuration: (out_channels, kernel_size, stride, padding)
 layer_configs = [
 (32, 3, 1, 1),
 (32, 3, 1, 1),
 (64, 3, 1, 1),
 (64, 3, 1, 1),
 (128, 3, 1, 1),
 (128, 3, 1, 1),
 (256, 3, 1, 1),
 (256, 3, 1, 1),
 ]
 
 for i, (out_channels, kernel_size, stride, padding) in enumerate(layer_configs):
 # Convolution
 layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
 
 # Batch normalization
 if self.use_batch_norm:
 layers.append(nn.BatchNorm2d(out_channels))
 
 # Activation
 layers.append(nn.ReLU(inplace=True))
 
 # Max pooling after every 2 conv layers
 if i % 2 == 1:
 layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
 
 in_channels = out_channels
 
 return nn.Sequential(*layers)
 
 def _make_classifier_layers(self) -> nn.Sequential:
 """Create the classifier layers."""
 return nn.Sequential(
 nn.Dropout(self.dropout),
 nn.Linear(256 * 7 * 7, 512),
 nn.ReLU(inplace=True),
 nn.Dropout(self.dropout),
 nn.Linear(512, 128),
 nn.ReLU(inplace=True),
 nn.Dropout(self.dropout / 2), # Reduced dropout for final layer
 nn.Linear(128, self.num_classes)
 )
 
 def _initialize_weights(self):
 """Initialize model weights using Xavier/He initialization."""
 for m in self.modules():
 if isinstance(m, nn.Conv2d):
 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
 if m.bias is not None:
 nn.init.constant_(m.bias, 0)
 elif isinstance(m, nn.BatchNorm2d):
 nn.init.constant_(m.weight, 1)
 nn.init.constant_(m.bias, 0)
 elif isinstance(m, nn.Linear):
 nn.init.normal_(m.weight, 0, 0.01)
 nn.init.constant_(m.bias, 0)
 
 def forward(self, x: torch.Tensor) -> torch.Tensor:
 """
 Forward pass through the network.
 
 Args:
 x: Input tensor of shape (batch_size, channels, height, width)
 
 Returns:
 torch.Tensor: Logits of shape (batch_size, num_classes)
 """
 # Feature extraction
 x = self.features(x)
 
 # Adaptive pooling
 x = self.adaptive_pool(x)
 
 # Flatten
 x = torch.flatten(x, 1)
 
 # Classification
 x = self.classifier(x)
 
 return x
 
 def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
 """
 Extract feature maps for visualization.
 
 Args:
 x: Input tensor
 
 Returns:
 torch.Tensor: Feature maps from the last convolutional layer
 """
 return self.features(x)

class ResNetBaseline(nn.Module):
 """
 ResNet-based baseline model for plant bloom classification.
 
 Uses a pre-trained ResNet backbone with custom classifier head
 optimized for plant bloom stage detection.
 """
 
 def __init__(
 self,
 num_classes: int = 5,
 backbone: str = 'resnet18',
 pretrained: bool = True,
 dropout: float = 0.5,
 freeze_backbone: bool = False
 ):
 """
 Initialize ResNet baseline model.
 
 Args:
 num_classes: Number of bloom stage classes
 backbone: ResNet variant ('resnet18', 'resnet34', 'resnet50')
 pretrained: Whether to use ImageNet pre-trained weights
 dropout: Dropout rate for classifier
 freeze_backbone: Whether to freeze backbone weights
 """
 super(ResNetBaseline, self).__init__()
 
 self.num_classes = num_classes
 self.backbone_name = backbone
 self.pretrained = pretrained
 
 # Load backbone
 self.backbone = self._load_backbone(backbone, pretrained)
 
 # Get feature dimension
 feature_dim = self.backbone.fc.in_features
 
 # Replace classifier
 self.backbone.fc = self._create_classifier(feature_dim, dropout)
 
 # Optionally freeze backbone
 if freeze_backbone:
 self._freeze_backbone()
 
 def _load_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
 """Load the specified ResNet backbone."""
 backbone_dict = {
 'resnet18': models.resnet18,
 'resnet34': models.resnet34,
 'resnet50': models.resnet50,
 'resnet101': models.resnet101,
 'resnet152': models.resnet152
 }
 
 if backbone not in backbone_dict:
 raise ValueError(f"Unsupported backbone: {backbone}")
 
 return backbone_dict[backbone](pretrained=pretrained)
 
 def _create_classifier(self, feature_dim: int, dropout: float) -> nn.Module:
 """Create custom classifier head."""
 return nn.Sequential(
 nn.Dropout(dropout),
 nn.Linear(feature_dim, 256),
 nn.ReLU(inplace=True),
 nn.Dropout(dropout / 2),
 nn.Linear(256, 128),
 nn.ReLU(inplace=True),
 nn.Dropout(dropout / 2),
 nn.Linear(128, self.num_classes)
 )
 
 def _freeze_backbone(self):
 """Freeze backbone parameters for transfer learning."""
 for param in self.backbone.parameters():
 param.requires_grad = False
 
 # Unfreeze classifier
 for param in self.backbone.fc.parameters():
 param.requires_grad = True
 
 def unfreeze_backbone(self):
 """Unfreeze backbone for fine-tuning."""
 for param in self.backbone.parameters():
 param.requires_grad = True
 
 def forward(self, x: torch.Tensor) -> torch.Tensor:
 """
 Forward pass through the network.
 
 Args:
 x: Input tensor of shape (batch_size, channels, height, width)
 
 Returns:
 torch.Tensor: Logits of shape (batch_size, num_classes)
 """
 return self.backbone(x)
 
 def get_features(self, x: torch.Tensor) -> torch.Tensor:
 """
 Extract features before classification.
 
 Args:
 x: Input tensor
 
 Returns:
 torch.Tensor: Feature vector
 """
 # Forward through all layers except classifier
 x = self.backbone.conv1(x)
 x = self.backbone.bn1(x)
 x = self.backbone.relu(x)
 x = self.backbone.maxpool(x)
 
 x = self.backbone.layer1(x)
 x = self.backbone.layer2(x)
 x = self.backbone.layer3(x)
 x = self.backbone.layer4(x)
 
 x = self.backbone.avgpool(x)
 x = torch.flatten(x, 1)
 
 return x

class EfficientNetBaseline(nn.Module):
 """
 EfficientNet-based baseline for plant bloom classification.
 
 Uses EfficientNet backbone for improved efficiency and accuracy.
 Note: Requires timm library for EfficientNet implementation.
 """
 
 def __init__(
 self,
 num_classes: int = 5,
 model_name: str = 'efficientnet_b0',
 pretrained: bool = True,
 dropout: float = 0.3
 ):
 """
 Initialize EfficientNet baseline.
 
 Args:
 num_classes: Number of bloom stage classes
 model_name: EfficientNet variant
 pretrained: Whether to use pre-trained weights
 dropout: Dropout rate
 """
 super(EfficientNetBaseline, self).__init__()
 
 try:
 import timm
 except ImportError:
 raise ImportError("timm library required for EfficientNet. Install with: pip install timm")
 
 self.num_classes = num_classes
 
 # Load pre-trained EfficientNet
 self.backbone = timm.create_model(
 model_name,
 pretrained=pretrained,
 num_classes=0 # Remove classifier
 )
 
 # Get feature dimension
 feature_dim = self.backbone.num_features
 
 # Custom classifier
 self.classifier = nn.Sequential(
 nn.Dropout(dropout),
 nn.Linear(feature_dim, 256),
 nn.ReLU(inplace=True),
 nn.Dropout(dropout / 2),
 nn.Linear(256, num_classes)
 )
 
 def forward(self, x: torch.Tensor) -> torch.Tensor:
 """Forward pass through the network."""
 features = self.backbone(x)
 return self.classifier(features)

def create_baseline_model(
 model_type: str = 'simple_cnn',
 num_classes: int = 5,
 **kwargs
) -> nn.Module:
 """
 Factory function to create baseline models.
 
 Args:
 model_type: Type of model ('simple_cnn', 'resnet', 'efficientnet')
 num_classes: Number of output classes
 **kwargs: Additional model parameters
 
 Returns:
 nn.Module: Initialized model
 """
 model_dict = {
 'simple_cnn': SimpleCNN,
 'resnet': ResNetBaseline,
 'efficientnet': EfficientNetBaseline
 }
 
 if model_type not in model_dict:
 raise ValueError(f"Unknown model type: {model_type}")
 
 return model_dict[model_type](num_classes=num_classes, **kwargs)

def count_parameters(model: nn.Module) -> Dict[str, int]:
 """
 Count the number of parameters in a model.
 
 Args:
 model: PyTorch model
 
 Returns:
 Dict with parameter counts
 """
 total_params = sum(p.numel() for p in model.parameters())
 trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
 
 return {
 'total_parameters': total_params,
 'trainable_parameters': trainable_params,
 'non_trainable_parameters': total_params - trainable_params
 }