"""
Core plotting utilities for BloomWatch visualizations.

This module provides essential plotting functions for training metrics,
confusion matrices, and other standard ML visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import warnings

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_training_metrics(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    title: str = "Training Metrics"
) -> plt.Figure:
    """
    Plot comprehensive training metrics including loss, accuracy, and learning rate.
    
    Args:
        metrics_history: Dictionary containing metric histories
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    epochs = metrics_history.get('epoch', range(len(metrics_history.get('train_loss', []))))
    
    # Loss plot
    if 'train_loss' in metrics_history:
        axes[0, 0].plot(epochs, metrics_history['train_loss'], 
                       label='Training Loss', color='#1f77b4', linewidth=2)
    if 'val_loss' in metrics_history:
        axes[0, 0].plot(epochs, metrics_history['val_loss'], 
                       label='Validation Loss', color='#ff7f0e', linewidth=2)
    
    axes[0, 0].set_title('Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if 'train_acc' in metrics_history:
        axes[0, 1].plot(epochs, metrics_history['train_acc'], 
                       label='Training Accuracy', color='#2ca02c', linewidth=2)
    if 'val_acc' in metrics_history:
        axes[0, 1].plot(epochs, metrics_history['val_acc'], 
                       label='Validation Accuracy', color='#d62728', linewidth=2)
    
    axes[0, 1].set_title('Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score plot
    if 'train_f1' in metrics_history:
        axes[1, 0].plot(epochs, metrics_history['train_f1'], 
                       label='Training F1', color='#9467bd', linewidth=2)
    if 'val_f1' in metrics_history:
        axes[1, 0].plot(epochs, metrics_history['val_f1'], 
                       label='Validation F1', color='#8c564b', linewidth=2)
    
    axes[1, 0].set_title('F1 Score', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate plot
    if 'learning_rate' in metrics_history and metrics_history['learning_rate']:
        axes[1, 1].plot(epochs, metrics_history['learning_rate'], 
                       color='#e377c2', linewidth=2)
        axes[1, 1].set_title('Learning Rate', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Learning Rate Data', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=14, style='italic')
        axes[1, 1].set_title('Learning Rate', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Plot a beautiful confusion matrix with proper formatting.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        normalize: Whether to normalize the matrix
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title += ' (Normalized)'
    else:
        fmt = 'd'
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center", color=text_color, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_class_distribution(
    labels: Union[List, np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Class Distribution"
) -> plt.Figure:
    """
    Plot the distribution of classes in the dataset.
    
    Args:
        labels: Array of class labels
        class_names: Optional list of class names
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    labels = np.array(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique_labels]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Bar plot
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    bars = ax1.bar(class_names, counts, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Class Counts', fontweight='bold')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(counts, labels=class_names, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Class Proportions', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_feature_maps(
    feature_maps: torch.Tensor,
    num_maps: int = 16,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    title: str = "Feature Maps"
) -> plt.Figure:
    """
    Visualize feature maps from convolutional layers.
    
    Args:
        feature_maps: Feature maps tensor (C, H, W) or (B, C, H, W)
        num_maps: Number of feature maps to display
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    if feature_maps.dim() == 4:
        feature_maps = feature_maps[0]  # Take first batch
    
    num_channels = feature_maps.shape[0]
    num_maps = min(num_maps, num_channels)
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_maps)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier indexing
    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(grid_size * grid_size):
        ax = axes[i]
        
        if i < num_maps:
            # Convert to numpy and normalize
            fmap = feature_maps[i].detach().cpu().numpy()
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            
            im = ax.imshow(fmap, cmap='viridis')
            ax.set_title(f'Channel {i}', fontsize=10)
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_loss_landscape(
    loss_values: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Loss Landscape"
) -> plt.Figure:
    """
    Plot 2D loss landscape visualization.
    
    Args:
        loss_values: 2D array of loss values
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create contour plot
    contour = ax.contourf(loss_values, levels=50, cmap='RdYlBu_r')
    ax.contour(loss_values, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Loss Value', rotation=270, labelpad=20, fontweight='bold')
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Parameter Dimension 1')
    ax.set_ylabel('Parameter Dimension 2')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_learning_curves(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Learning Curves"
) -> plt.Figure:
    """
    Plot learning curves showing performance vs training set size.
    
    Args:
        train_sizes: Array of training set sizes
        train_scores: Training scores for each size
        val_scores: Validation scores for each size
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot curves
    ax.plot(train_sizes, train_mean, 'o-', color='#1f77b4', 
           label='Training Score', linewidth=2, markersize=6)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                   color='#1f77b4', alpha=0.2)
    
    ax.plot(train_sizes, val_mean, 's-', color='#ff7f0e', 
           label='Validation Score', linewidth=2, markersize=6)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                   color='#ff7f0e', alpha=0.2)
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_model_comparison(
    models_results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'f1', 'precision', 'recall'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Model Comparison"
) -> plt.Figure:
    """
    Compare multiple models across different metrics.
    
    Args:
        models_results: Dictionary with model names as keys and metrics as values
        metrics: List of metrics to compare
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    # Prepare data
    model_names = list(models_results.keys())
    metric_values = {metric: [models_results[model].get(metric, 0) for model in model_names] 
                    for metric in metrics}
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    
    for i, metric in enumerate(metrics[:4]):  # Limit to 4 metrics
        ax = axes[i]
        bars = ax.bar(model_names, metric_values[metric], color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_title(f'{metric.capitalize()}', fontweight='bold')
        ax.set_ylabel(metric.capitalize())
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(metric_values[metric]) * 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Hide unused subplots
    for i in range(len(metrics), 4):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def create_subplot_grid(
    data_list: List[np.ndarray],
    titles: List[str],
    rows: int,
    cols: int,
    figsize: Tuple[int, int] = (15, 10),
    suptitle: str = "",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a grid of subplots for visualizing multiple images or data.
    
    Args:
        data_list: List of data arrays to plot
        titles: List of titles for each subplot
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        figsize: Figure size tuple
        suptitle: Super title for the entire figure
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier indexing
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, (data, title) in enumerate(zip(data_list, titles)):
        if i < len(axes):
            ax = axes[i]
            
            if len(data.shape) == 3:  # Color image
                ax.imshow(data)
            elif len(data.shape) == 2:  # Grayscale image or 2D data
                ax.imshow(data, cmap='viridis')
            else:  # 1D data
                ax.plot(data)
            
            ax.set_title(title, fontweight='bold')
            ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(data_list), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig