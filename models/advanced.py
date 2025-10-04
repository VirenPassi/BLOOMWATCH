"""
Advanced models for plant bloom detection.

This module contains more sophisticated architectures including
time-series models and attention mechanisms for improved performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math

class TimeSeriesBloomNet(nn.Module):
 """
 Time-series model for tracking plant bloom progression over time.
 
 Uses LSTM/GRU layers to model temporal dependencies in plant growth
 and blooming patterns from sequential images.
 """
 
 def __init__(
 self,
 feature_extractor: str = 'resnet18',
 num_classes: int = 5,
 sequence_length: int = 5,
 hidden_size: int = 256,
 num_layers: int = 2,
 rnn_type: str = 'lstm',
 dropout: float = 0.3,
 bidirectional: bool = True
 ):
 """
 Initialize TimeSeriesBloomNet.
 
 Args:
 feature_extractor: Backbone CNN for feature extraction
 num_classes: Number of bloom stage classes
 sequence_length: Length of input sequences
 hidden_size: Hidden size for RNN layers
 num_layers: Number of RNN layers
 rnn_type: Type of RNN ('lstm' or 'gru')
 dropout: Dropout rate
 bidirectional: Whether to use bidirectional RNN
 """
 super(TimeSeriesBloomNet, self).__init__()
 
 self.num_classes = num_classes
 self.sequence_length = sequence_length
 self.hidden_size = hidden_size
 self.num_layers = num_layers
 self.bidirectional = bidirectional
 
 # Feature extractor (CNN backbone)
 self.feature_extractor = self._create_feature_extractor(feature_extractor)
 feature_dim = self._get_feature_dim()
 
 # RNN for temporal modeling
 if rnn_type.lower() == 'lstm':
 self.rnn = nn.LSTM(
 input_size=feature_dim,
 hidden_size=hidden_size,
 num_layers=num_layers,
 dropout=dropout if num_layers > 1 else 0,
 bidirectional=bidirectional,
 batch_first=True
 )
 elif rnn_type.lower() == 'gru':
 self.rnn = nn.GRU(
 input_size=feature_dim,
 hidden_size=hidden_size,
 num_layers=num_layers,
 dropout=dropout if num_layers > 1 else 0,
 bidirectional=bidirectional,
 batch_first=True
 )
 else:
 raise ValueError(f"Unsupported RNN type: {rnn_type}")
 
 # Classifier
 classifier_input_size = hidden_size * (2 if bidirectional else 1)
 self.classifier = nn.Sequential(
 nn.Dropout(dropout),
 nn.Linear(classifier_input_size, 128),
 nn.ReLU(inplace=True),
 nn.Dropout(dropout / 2),
 nn.Linear(128, num_classes)
 )
 
 # Attention mechanism for sequence aggregation
 self.attention = nn.Sequential(
 nn.Linear(classifier_input_size, classifier_input_size // 2),
 nn.Tanh(),
 nn.Linear(classifier_input_size // 2, 1)
 )
 
 def _create_feature_extractor(self, backbone: str) -> nn.Module:
 """Create CNN feature extractor."""
 if backbone == 'resnet18':
 import torchvision.models as models
 resnet = models.resnet18(pretrained=True)
 # Remove final classification layer
 self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
 return self.feature_extractor
 else:
 # Placeholder for other backbones
 raise NotImplementedError(f"Backbone {backbone} not implemented yet")
 
 def _get_feature_dim(self) -> int:
 """Get the feature dimension from the backbone."""
 # For ResNet18
 return 512
 
 def forward(self, x: torch.Tensor) -> torch.Tensor:
 """
 Forward pass through the time-series model.
 
 Args:
 x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
 
 Returns:
 torch.Tensor: Logits of shape (batch_size, num_classes)
 """
 batch_size, seq_len, c, h, w = x.size()
 
 # Reshape for CNN processing
 x = x.view(batch_size * seq_len, c, h, w)
 
 # Extract features
 features = self.feature_extractor(x)
 features = features.view(batch_size, seq_len, -1)
 
 # RNN processing
 rnn_out, _ = self.rnn(features)
 
 # Apply attention mechanism
 attention_weights = self.attention(rnn_out)
 attention_weights = F.softmax(attention_weights, dim=1)
 
 # Weighted sum over sequence dimension
 context = torch.sum(rnn_out * attention_weights, dim=1)
 
 # Classification
 output = self.classifier(context)
 
 return output

class AttentionBloomNet(nn.Module):
 """
 Attention-based model for plant bloom detection.
 
 Uses self-attention mechanisms to focus on important regions
 of the plant image for bloom stage classification.
 """
 
 def __init__(
 self,
 num_classes: int = 5,
 feature_dim: int = 512,
 num_heads: int = 8,
 num_layers: int = 3,
 dropout: float = 0.1
 ):
 """
 Initialize AttentionBloomNet.
 
 Args:
 num_classes: Number of bloom stage classes
 feature_dim: Feature dimension for attention
 num_heads: Number of attention heads
 num_layers: Number of transformer layers
 dropout: Dropout rate
 """
 super(AttentionBloomNet, self).__init__()
 
 self.num_classes = num_classes
 self.feature_dim = feature_dim
 
 # CNN backbone for initial feature extraction
 self.backbone = self._create_backbone()
 
 # Positional encoding for spatial attention
 self.positional_encoding = PositionalEncoding2D(feature_dim)
 
 # Transformer encoder layers
 encoder_layer = nn.TransformerEncoderLayer(
 d_model=feature_dim,
 nhead=num_heads,
 dim_feedforward=feature_dim * 4,
 dropout=dropout,
 activation='relu'
 )
 self.transformer_encoder = nn.TransformerEncoder(
 encoder_layer,
 num_layers=num_layers
 )
 
 # Classification head
 self.classifier = nn.Sequential(
 nn.Dropout(dropout),
 nn.Linear(feature_dim, feature_dim // 2),
 nn.ReLU(inplace=True),
 nn.Dropout(dropout),
 nn.Linear(feature_dim // 2, num_classes)
 )
 
 # Global average pooling
 self.global_pool = nn.AdaptiveAvgPool1d(1)
 
 def _create_backbone(self) -> nn.Module:
 """Create CNN backbone for feature extraction."""
 import torchvision.models as models
 resnet = models.resnet18(pretrained=True)
 
 # Remove avgpool and fc layers
 backbone = nn.Sequential(*list(resnet.children())[:-2])
 
 # Add projection layer to match feature_dim
 projection = nn.Conv2d(512, self.feature_dim, kernel_size=1)
 
 return nn.Sequential(backbone, projection)
 
 def forward(self, x: torch.Tensor) -> torch.Tensor:
 """
 Forward pass through attention model.
 
 Args:
 x: Input tensor of shape (batch_size, channels, height, width)
 
 Returns:
 torch.Tensor: Logits of shape (batch_size, num_classes)
 """
 # Extract features
 features = self.backbone(x) # (B, feature_dim, H, W)
 
 B, C, H, W = features.size()
 
 # Reshape for transformer: (B, H*W, C)
 features = features.view(B, C, H * W).permute(0, 2, 1)
 
 # Add positional encoding
 features = self.positional_encoding(features, H, W)
 
 # Transformer processing: (seq_len, batch, feature_dim)
 features = features.permute(1, 0, 2)
 attended_features = self.transformer_encoder(features)
 
 # Back to (batch, seq_len, feature_dim)
 attended_features = attended_features.permute(1, 0, 2)
 
 # Global pooling and classification
 pooled = self.global_pool(attended_features.permute(0, 2, 1)).squeeze(-1)
 output = self.classifier(pooled)
 
 return output

class PositionalEncoding2D(nn.Module):
 """
 2D positional encoding for spatial attention in images.
 """
 
 def __init__(self, d_model: int, max_len: int = 10000):
 """
 Initialize 2D positional encoding.
 
 Args:
 d_model: Model dimension
 max_len: Maximum sequence length
 """
 super(PositionalEncoding2D, self).__init__()
 self.d_model = d_model
 
 def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
 """
 Add 2D positional encoding to input tensor.
 
 Args:
 x: Input tensor of shape (batch, height*width, d_model)
 height: Image height
 width: Image width
 
 Returns:
 torch.Tensor: Tensor with positional encoding added
 """
 batch_size = x.size(0)
 
 # Create position indices
 pos_h = torch.arange(height, dtype=torch.float, device=x.device).unsqueeze(1).repeat(1, width)
 pos_w = torch.arange(width, dtype=torch.float, device=x.device).unsqueeze(0).repeat(height, 1)
 
 pos_h = pos_h.flatten().unsqueeze(0).repeat(batch_size, 1) # (batch, H*W)
 pos_w = pos_w.flatten().unsqueeze(0).repeat(batch_size, 1) # (batch, H*W)
 
 # Generate encoding
 pe = torch.zeros(batch_size, height * width, self.d_model, device=x.device)
 
 div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=x.device) *
 -(math.log(10000.0) / self.d_model))
 
 # Height encoding
 pe[:, :, 0::4] = torch.sin(pos_h.unsqueeze(-1) * div_term[::2])
 pe[:, :, 1::4] = torch.cos(pos_h.unsqueeze(-1) * div_term[::2])
 
 # Width encoding
 pe[:, :, 2::4] = torch.sin(pos_w.unsqueeze(-1) * div_term[::2])
 pe[:, :, 3::4] = torch.cos(pos_w.unsqueeze(-1) * div_term[::2])
 
 return x + pe

class MultiScaleBloomNet(nn.Module):
 """
 Multi-scale model for capturing bloom features at different scales.
 
 Processes the input image at multiple resolutions to capture
 both fine-grained flower details and overall plant structure.
 """
 
 def __init__(
 self,
 num_classes: int = 5,
 scales: list = [224, 384, 512],
 backbone: str = 'resnet18',
 fusion_method: str = 'concat'
 ):
 """
 Initialize MultiScaleBloomNet.
 
 Args:
 num_classes: Number of bloom stage classes
 scales: List of input scales
 backbone: CNN backbone
 fusion_method: How to fuse multi-scale features ('concat', 'add', 'attention')
 """
 super(MultiScaleBloomNet, self).__init__()
 
 self.num_classes = num_classes
 self.scales = scales
 self.fusion_method = fusion_method
 
 # Create backbone for each scale
 self.backbones = nn.ModuleList([
 self._create_backbone(backbone) for _ in scales
 ])
 
 # Feature fusion
 if fusion_method == 'concat':
 fusion_dim = len(scales) * 512 # Assuming ResNet18 features
 elif fusion_method == 'add':
 fusion_dim = 512
 else: # attention
 fusion_dim = 512
 self.scale_attention = nn.Sequential(
 nn.Linear(512, 128),
 nn.ReLU(),
 nn.Linear(128, 1)
 )
 
 # Classifier
 self.classifier = nn.Sequential(
 nn.Dropout(0.5),
 nn.Linear(fusion_dim, 256),
 nn.ReLU(inplace=True),
 nn.Dropout(0.3),
 nn.Linear(256, num_classes)
 )
 
 def _create_backbone(self, backbone: str) -> nn.Module:
 """Create CNN backbone."""
 import torchvision.models as models
 if backbone == 'resnet18':
 model = models.resnet18(pretrained=True)
 return nn.Sequential(*list(model.children())[:-1])
 else:
 raise NotImplementedError(f"Backbone {backbone} not implemented")
 
 def forward(self, x: torch.Tensor) -> torch.Tensor:
 """
 Forward pass through multi-scale model.
 
 Args:
 x: Input tensor of shape (batch_size, channels, height, width)
 
 Returns:
 torch.Tensor: Logits of shape (batch_size, num_classes)
 """
 scale_features = []
 
 # Process at each scale
 for i, (scale, backbone) in enumerate(zip(self.scales, self.backbones)):
 # Resize input
 scaled_x = F.interpolate(x, size=(scale, scale), mode='bilinear', align_corners=False)
 
 # Extract features
 features = backbone(scaled_x).squeeze(-1).squeeze(-1) # (B, 512)
 scale_features.append(features)
 
 # Fuse features
 if self.fusion_method == 'concat':
 fused_features = torch.cat(scale_features, dim=1)
 elif self.fusion_method == 'add':
 fused_features = torch.stack(scale_features, dim=0).sum(dim=0)
 else: # attention
 # Compute attention weights for each scale
 attention_weights = []
 for features in scale_features:
 weight = self.scale_attention(features)
 attention_weights.append(weight)
 
 attention_weights = torch.cat(attention_weights, dim=1)
 attention_weights = F.softmax(attention_weights, dim=1)
 
 # Weighted combination
 fused_features = torch.zeros_like(scale_features[0])
 for i, features in enumerate(scale_features):
 fused_features += attention_weights[:, i:i+1] * features
 
 # Classification
 output = self.classifier(fused_features)
 return output