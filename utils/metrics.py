"""
Metrics calculation and tracking for plant bloom detection.

This module provides comprehensive metrics for evaluating
bloom stage classification models and tracking training progress.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime


class MetricsTracker:
    """
    Track and compute metrics for plant bloom detection experiments.
    
    Maintains history of metrics across epochs and provides
    utilities for visualization and analysis.
    """
    
    def __init__(self, num_classes: int = 5, class_names: Optional[List[str]] = None):
        """
        Initialize metrics tracker.
        
        Args:
            num_classes: Number of bloom stage classes
            class_names: Names of the classes
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        
        # Storage for metrics history
        self.metrics_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rate': []
        }
        
        # Best metrics tracking
        self.best_metrics = {
            'best_val_acc': 0.0,
            'best_val_f1': 0.0,
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }
        
        # Current epoch metrics
        self.current_epoch = 0
    
    def update_epoch_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Update metrics for current epoch.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics
        """
        self.current_epoch = epoch
        self.metrics_history['epoch'].append(epoch)
        
        # Update history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # Update best metrics
        val_acc = metrics.get('val_acc', 0.0)
        val_f1 = metrics.get('val_f1', 0.0)
        val_loss = metrics.get('val_loss', float('inf'))
        
        if val_acc > self.best_metrics['best_val_acc']:
            self.best_metrics['best_val_acc'] = val_acc
            self.best_metrics['best_epoch'] = epoch
        
        if val_f1 > self.best_metrics['best_val_f1']:
            self.best_metrics['best_val_f1'] = val_f1
        
        if val_loss < self.best_metrics['best_val_loss']:
            self.best_metrics['best_val_loss'] = val_loss
    
    def compute_classification_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth labels
            average: Averaging method for multi-class metrics
            
        Returns:
            Dictionary of computed metrics
        """
        # Convert to numpy
        if torch.is_tensor(predictions):
            if predictions.dim() > 1:
                pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
                pred_probs = F.softmax(predictions, dim=1).cpu().numpy()
            else:
                pred_classes = predictions.cpu().numpy()
                pred_probs = None
        else:
            pred_classes = predictions
            pred_probs = None
        
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        # Basic metrics
        accuracy = accuracy_score(targets, pred_classes)
        
        # Precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, pred_classes, average=average, zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Per-class metrics
        if average == 'weighted':
            per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
                targets, pred_classes, average=None, zero_division=0
            )
            
            for i, class_name in enumerate(self.class_names):
                metrics[f'{class_name}_precision'] = per_class_precision[i] if i < len(per_class_precision) else 0.0
                metrics[f'{class_name}_recall'] = per_class_recall[i] if i < len(per_class_recall) else 0.0
                metrics[f'{class_name}_f1'] = per_class_f1[i] if i < len(per_class_f1) else 0.0
        
        # AUC metrics (if probabilities available)
        if pred_probs is not None and pred_probs.shape[1] == self.num_classes:
            try:
                # One-vs-rest AUC
                targets_onehot = np.eye(self.num_classes)[targets]
                auc_ovr = roc_auc_score(targets_onehot, pred_probs, average=average, multi_class='ovr')
                metrics['auc_ovr'] = auc_ovr
                
                # Average precision
                avg_precision = average_precision_score(targets_onehot, pred_probs, average=average)
                metrics['avg_precision'] = avg_precision
                
            except ValueError:
                # Skip AUC if only one class present
                pass
        
        return metrics
    
    def compute_confusion_matrix(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        normalize: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            normalize: Normalization method ('true', 'pred', 'all', or None)
            
        Returns:
            Confusion matrix
        """
        if torch.is_tensor(predictions):
            if predictions.dim() > 1:
                pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
            else:
                pred_classes = predictions.cpu().numpy()
        else:
            pred_classes = predictions
        
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        cm = confusion_matrix(targets, pred_classes, labels=range(self.num_classes))
        
        if normalize:
            if normalize == 'true':
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            elif normalize == 'pred':
                cm = cm.astype('float') / cm.sum(axis=0)
            elif normalize == 'all':
                cm = cm.astype('float') / cm.sum()
        
        return cm
    
    def plot_metrics_history(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training metrics history.
        
        Args:
            save_path: Optional path to save plot
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics History', fontsize=16)
        
        epochs = self.metrics_history['epoch']
        
        # Loss plot
        axes[0, 0].plot(epochs, self.metrics_history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(epochs, self.metrics_history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.metrics_history['train_acc'], label='Train Acc', color='blue')
        axes[0, 1].plot(epochs, self.metrics_history['val_acc'], label='Val Acc', color='red')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score plot
        axes[1, 0].plot(epochs, self.metrics_history['train_f1'], label='Train F1', color='blue')
        axes[1, 0].plot(epochs, self.metrics_history['val_f1'], label='Val F1', color='red')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot
        if self.metrics_history['learning_rate']:
            axes[1, 1].plot(epochs, self.metrics_history['learning_rate'], color='green')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'No LR data', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: Optional[str] = None,
        title: str = 'Confusion Matrix'
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            save_path: Optional path to save plot
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if cm.dtype == np.float64 else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_metrics(self, save_path: str):
        """
        Save metrics history to JSON file.
        
        Args:
            save_path: Path to save metrics
        """
        save_data = {
            'metrics_history': self.metrics_history,
            'best_metrics': self.best_metrics,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'timestamp': datetime.now().isoformat()
        }
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def load_metrics(self, load_path: str):
        """
        Load metrics history from JSON file.
        
        Args:
            load_path: Path to load metrics from
        """
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        self.metrics_history = data['metrics_history']
        self.best_metrics = data['best_metrics']
        self.class_names = data['class_names']
        self.num_classes = data['num_classes']
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of metrics.
        
        Returns:
            Dictionary with metrics summary
        """
        if not self.metrics_history['epoch']:
            return {'message': 'No metrics recorded yet'}
        
        summary = {
            'total_epochs': len(self.metrics_history['epoch']),
            'best_metrics': self.best_metrics.copy(),
            'latest_metrics': {}
        }
        
        # Latest metrics
        for key, values in self.metrics_history.items():
            if values and key != 'epoch':
                summary['latest_metrics'][key] = values[-1]
        
        return summary


def calculate_bloom_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for bloom stage classification.
    
    Args:
        predictions: Model predictions (logits or class indices)
        targets: Ground truth labels
        class_names: Optional class names
        
    Returns:
        Dictionary with all computed metrics
    """
    num_classes = predictions.shape[1] if predictions.dim() > 1 else len(torch.unique(targets))
    
    if class_names is None:
        class_names = ['bud', 'early_bloom', 'full_bloom', 'late_bloom', 'dormant'][:num_classes]
    
    tracker = MetricsTracker(num_classes, class_names)
    
    # Classification metrics
    classification_metrics = tracker.compute_classification_metrics(predictions, targets)
    
    # Confusion matrix
    cm = tracker.compute_confusion_matrix(predictions, targets)
    cm_normalized = tracker.compute_confusion_matrix(predictions, targets, normalize='true')
    
    return {
        'classification_metrics': classification_metrics,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist(),
        'class_names': class_names
    }


class BloomStageMetrics:
    """
    Specialized metrics for bloom stage classification.
    
    Provides bloom-specific metrics that consider the biological
    relationships between different bloom stages.
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize bloom stage metrics.
        
        Args:
            class_names: List of bloom stage names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def stage_progression_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculate accuracy considering biological stage progression.
        
        Gives partial credit for predictions that are close to the true stage
        in the biological progression.
        
        Args:
            predictions: Predicted class indices
            targets: True class indices
            
        Returns:
            Stage progression accuracy (0-1)
        """
        total_score = 0.0
        
        for pred, true in zip(predictions, targets):
            distance = abs(pred - true)
            
            if distance == 0:
                score = 1.0  # Perfect prediction
            elif distance == 1:
                score = 0.5  # Adjacent stage (reasonable error)
            elif distance == 2:
                score = 0.25  # Two stages off
            else:
                score = 0.0  # Very different stages
            
            total_score += score
        
        return total_score / len(predictions)
    
    def temporal_consistency_score(
        self,
        predictions: List[np.ndarray],
        targets: List[np.ndarray],
        plant_ids: List[str]
    ) -> float:
        """
        Calculate temporal consistency score for time-series predictions.
        
        Measures how well predictions follow biologically plausible
        temporal patterns for the same plant.
        
        Args:
            predictions: List of prediction arrays for each time point
            targets: List of target arrays for each time point
            plant_ids: List of plant identifiers for each time point
            
        Returns:
            Temporal consistency score (0-1)
        """
        if len(predictions) < 2:
            return 1.0  # Cannot compute consistency with single time point
        
        # Group by plant ID
        plant_sequences = {}
        for i, plant_id in enumerate(plant_ids):
            if plant_id not in plant_sequences:
                plant_sequences[plant_id] = {'preds': [], 'targets': []}
            plant_sequences[plant_id]['preds'].append(predictions[i])
            plant_sequences[plant_id]['targets'].append(targets[i])
        
        consistency_scores = []
        
        for plant_id, sequences in plant_sequences.items():
            if len(sequences['preds']) < 2:
                continue
            
            plant_score = 0.0
            valid_transitions = 0
            
            for t in range(len(sequences['preds']) - 1):
                current_pred = sequences['preds'][t]
                next_pred = sequences['preds'][t + 1]
                
                # Check if transition is biologically plausible
                stage_diff = next_pred - current_pred
                
                if abs(stage_diff) <= 1:  # Stay same or move to adjacent stage
                    plant_score += 1.0
                elif abs(stage_diff) == 2:  # Skip one stage (less likely)
                    plant_score += 0.5
                # else: 0 score for implausible jumps
                
                valid_transitions += 1
            
            if valid_transitions > 0:
                consistency_scores.append(plant_score / valid_transitions)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def bloom_timing_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        timestamps: List[str]
    ) -> Dict[str, float]:
        """
        Calculate bloom timing accuracy metrics.
        
        Analyzes how well the model predicts bloom timing
        patterns throughout the growing season.
        
        Args:
            predictions: Predicted bloom stages
            targets: True bloom stages
            timestamps: Timestamp strings for each prediction
            
        Returns:
            Dictionary with timing metrics
        """
        # Convert timestamps to datetime objects
        from datetime import datetime
        dates = [datetime.fromisoformat(ts) for ts in timestamps]
        
        # Find peak bloom periods (full_bloom stage)
        full_bloom_class = 2  # Assuming full_bloom is class 2
        
        pred_peak_dates = [dates[i] for i, pred in enumerate(predictions) if pred == full_bloom_class]
        true_peak_dates = [dates[i] for i, true in enumerate(targets) if true == full_bloom_class]
        
        metrics = {}
        
        # Peak bloom detection accuracy
        if len(true_peak_dates) > 0:
            # Find closest predicted peak to each true peak
            timing_errors = []
            for true_date in true_peak_dates:
                if pred_peak_dates:
                    closest_pred_date = min(pred_peak_dates, key=lambda x: abs((x - true_date).days))
                    error_days = abs((closest_pred_date - true_date).days)
                    timing_errors.append(error_days)
            
            if timing_errors:
                metrics['mean_timing_error_days'] = np.mean(timing_errors)
                metrics['median_timing_error_days'] = np.median(timing_errors)
                metrics['timing_accuracy_7days'] = np.mean([e <= 7 for e in timing_errors])
                metrics['timing_accuracy_14days'] = np.mean([e <= 14 for e in timing_errors])
        
        return metrics