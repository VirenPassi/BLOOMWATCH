"""
Custom loss functions for plant bloom detection.

This module contains specialized loss functions designed for
plant bloom stage classification with class imbalance handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tensor


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in bloom stage classification.
    
    Focal Loss focuses learning on hard negatives by down-weighting
    well-classified examples, which is useful when some bloom stages
    are much more common than others.
    
    Reference: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    """
    
    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (optional)
            gamma: Focusing parameter (higher gamma = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Tensor: Focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BloomStageLoss(nn.Module):
    """
    Custom loss function for plant bloom stage classification.
    
    Combines cross-entropy loss with additional penalties for
    biologically implausible transitions between bloom stages.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        transition_penalty: float = 0.1,
        class_weights: Optional[Tensor] = None,
        smoothing: float = 0.0
    ):
        """
        Initialize BloomStageLoss.
        
        Args:
            num_classes: Number of bloom stage classes
            transition_penalty: Weight for transition penalty
            class_weights: Weights for each class
            smoothing: Label smoothing factor
        """
        super(BloomStageLoss, self).__init__()
        self.num_classes = num_classes
        self.transition_penalty = transition_penalty
        self.smoothing = smoothing
        
        # Define transition matrix (how "far" each stage is from others)
        self.register_buffer('transition_matrix', self._create_transition_matrix())
        
        # Class weights
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def _create_transition_matrix(self) -> Tensor:
        """
        Create transition penalty matrix for bloom stages.
        
        Assumes stages are ordered: [bud, early_bloom, full_bloom, late_bloom, dormant]
        """
        # Distance matrix between stages
        transition_matrix = torch.zeros(self.num_classes, self.num_classes)
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                # Biological distance between stages
                if i == j:
                    transition_matrix[i, j] = 0.0  # Correct prediction
                else:
                    # Penalize based on biological implausibility
                    distance = abs(i - j)
                    if distance == 1:
                        transition_matrix[i, j] = 0.5  # Adjacent stages (reasonable error)
                    elif distance == 2:
                        transition_matrix[i, j] = 1.0  # Two stages apart
                    else:
                        transition_matrix[i, j] = 2.0  # Very different stages
        
        return transition_matrix
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute bloom stage loss.
        
        Args:
            inputs: Predicted logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Tensor: Combined loss value
        """
        # Standard cross-entropy loss with optional label smoothing
        if self.smoothing > 0:
            ce_loss = self._label_smoothing_cross_entropy(inputs, targets)
        else:
            ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights)
        
        # Transition penalty
        if self.transition_penalty > 0:
            transition_loss = self._compute_transition_penalty(inputs, targets)
            total_loss = ce_loss + self.transition_penalty * transition_loss
        else:
            total_loss = ce_loss
        
        return total_loss
    
    def _label_smoothing_cross_entropy(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute cross-entropy with label smoothing."""
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # One-hot encoding with smoothing
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        if self.class_weights is not None:
            loss = loss * self.class_weights[targets]
        
        return loss.mean()
    
    def _compute_transition_penalty(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute penalty for biologically implausible predictions."""
        probs = F.softmax(inputs, dim=-1)
        
        # Compute expected transition cost
        penalty = 0.0
        for i in range(len(targets)):
            target_class = targets[i].item()
            for j in range(self.num_classes):
                penalty += probs[i, j] * self.transition_matrix[target_class, j]
        
        return penalty / len(targets)


class TemporalConsistencyLoss(nn.Module):
    """
    Loss function for enforcing temporal consistency in time-series predictions.
    
    Useful for time-series bloom detection where predictions should
    follow biologically plausible temporal patterns.
    """
    
    def __init__(
        self,
        consistency_weight: float = 0.1,
        smoothness_weight: float = 0.05
    ):
        """
        Initialize temporal consistency loss.
        
        Args:
            consistency_weight: Weight for consistency penalty
            smoothness_weight: Weight for smoothness penalty
        """
        super(TemporalConsistencyLoss, self).__init__()
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight
    
    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        base_loss_fn: nn.Module = nn.CrossEntropyLoss()
    ) -> Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            predictions: Predicted logits (batch_size, sequence_length, num_classes)
            targets: Ground truth labels (batch_size, sequence_length)
            base_loss_fn: Base loss function for classification
            
        Returns:
            Tensor: Combined loss with temporal penalties
        """
        batch_size, seq_len, num_classes = predictions.shape
        
        # Reshape for base loss computation
        pred_flat = predictions.view(-1, num_classes)
        target_flat = targets.view(-1)
        
        # Base classification loss
        base_loss = base_loss_fn(pred_flat, target_flat)
        
        # Temporal consistency penalty
        consistency_loss = 0.0
        if self.consistency_weight > 0 and seq_len > 1:
            for t in range(seq_len - 1):
                # Penalize large jumps between adjacent time steps
                curr_probs = F.softmax(predictions[:, t], dim=-1)
                next_probs = F.softmax(predictions[:, t + 1], dim=-1)
                
                # KL divergence between adjacent predictions
                consistency_loss += F.kl_div(
                    F.log_softmax(predictions[:, t + 1], dim=-1),
                    curr_probs,
                    reduction='batchmean'
                )
        
        # Smoothness penalty
        smoothness_loss = 0.0
        if self.smoothness_weight > 0 and seq_len > 2:
            for t in range(1, seq_len - 1):
                # Penalize non-smooth transitions
                prev_probs = F.softmax(predictions[:, t - 1], dim=-1)
                curr_probs = F.softmax(predictions[:, t], dim=-1)
                next_probs = F.softmax(predictions[:, t + 1], dim=-1)
                
                # Second-order difference
                second_diff = next_probs - 2 * curr_probs + prev_probs
                smoothness_loss += torch.mean(torch.sum(second_diff ** 2, dim=-1))
        
        # Combine losses
        total_loss = (base_loss + 
                     self.consistency_weight * consistency_loss + 
                     self.smoothness_weight * smoothness_loss)
        
        return total_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning discriminative features.
    
    Useful for learning better representations of different bloom stages
    by encouraging similar stages to be close and different stages to be far apart.
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for contrastive loss
            temperature: Temperature for similarity computation
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        """
        Compute contrastive loss.
        
        Args:
            features: Feature representations (batch_size, feature_dim)
            labels: Class labels (batch_size,)
            
        Returns:
            Tensor: Contrastive loss value
        """
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute pairwise similarities
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same class)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size, device=mask.device)
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        
        # Positive pairs
        pos_sim = exp_sim * mask
        pos_sum = pos_sim.sum(dim=1, keepdim=True)
        
        # All pairs
        all_sum = exp_sim.sum(dim=1, keepdim=True)
        
        # Contrastive loss
        loss = -torch.log(pos_sum / (all_sum + 1e-8))
        
        # Only consider samples that have positive pairs
        valid_samples = (mask.sum(dim=1) > 0).float()
        loss = (loss.squeeze() * valid_samples).sum() / (valid_samples.sum() + 1e-8)
        
        return loss


def create_loss_function(
    loss_type: str = 'cross_entropy',
    num_classes: int = 5,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss ('cross_entropy', 'focal', 'bloom_stage', 'temporal')
        num_classes: Number of classes
        **kwargs: Additional loss parameters
        
    Returns:
        nn.Module: Loss function
    """
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'bloom_stage':
        return BloomStageLoss(num_classes=num_classes, **kwargs)
    elif loss_type == 'temporal':
        return TemporalConsistencyLoss(**kwargs)
    elif loss_type == 'contrastive':
        return ContrastiveLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")