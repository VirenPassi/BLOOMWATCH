"""
Data visualization utilities for exploring plant bloom datasets.

This module provides tools for visualizing dataset samples, bloom progressions,
and creating data exploration dashboards.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from PIL import Image
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import random
from matplotlib.gridspec import GridSpec

def plot_dataset_samples(
 dataset,
 num_samples: int = 16,
 samples_per_class: Optional[int] = None,
 save_path: Optional[str] = None,
 figsize: Tuple[int, int] = (15, 12),
 title: str = "Dataset Samples"
) -> plt.Figure:
 """
 Plot random samples from the dataset, optionally balanced by class.
 
 Args:
 dataset: PyTorch dataset object
 num_samples: Total number of samples to display
 samples_per_class: Number of samples per class (overrides num_samples)
 save_path: Optional path to save the plot
 figsize: Figure size tuple
 title: Plot title
 
 Returns:
 matplotlib Figure object
 """
 if samples_per_class:
 # Get balanced samples
 class_indices = {}
 for i in range(len(dataset)):
 _, label, _ = dataset[i]
 if isinstance(label, torch.Tensor):
 label = label.item()
 
 if label not in class_indices:
 class_indices[label] = []
 class_indices[label].append(i)
 
 # Sample from each class
 selected_indices = []
 for class_label, indices in class_indices.items():
 sampled = random.sample(indices, min(samples_per_class, len(indices)))
 selected_indices.extend(sampled)
 
 num_samples = len(selected_indices)
 else:
 # Random sampling
 selected_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
 
 # Calculate grid size
 grid_cols = int(np.ceil(np.sqrt(num_samples)))
 grid_rows = int(np.ceil(num_samples / grid_cols))
 
 fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
 fig.suptitle(title, fontsize=16, fontweight='bold')
 
 # Flatten axes for easier indexing
 if grid_rows == 1 and grid_cols == 1:
 axes = [axes]
 else:
 axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
 
 # Get class names if available
 class_names = getattr(dataset, 'class_names', None)
 if class_names is None and hasattr(dataset, 'STAGE_NAMES'):
 class_names = list(dataset.STAGE_NAMES.values())
 
 for i, idx in enumerate(selected_indices):
 if i >= len(axes):
 break
 
 image, label, metadata = dataset[idx]
 
 # Convert tensor to numpy if needed
 if torch.is_tensor(image):
 if image.dim() == 3 and image.shape[0] in [1, 3]: # CHW format
 image = image.permute(1, 2, 0)
 image = image.cpu().numpy()
 
 # Denormalize if needed (assuming ImageNet normalization)
 if image.min() < 0: # Likely normalized
 mean = np.array([0.485, 0.456, 0.406])
 std = np.array([0.229, 0.224, 0.225])
 image = image * std + mean
 image = np.clip(image, 0, 1)
 
 # Display image
 axes[i].imshow(image)
 
 # Create title with class information
 if isinstance(label, torch.Tensor):
 label = label.item()
 
 if class_names and label < len(class_names):
 class_name = class_names[label]
 else:
 class_name = f"Class {label}"
 
 # Add metadata if available
 plant_id = metadata.get('plant_id', '') if isinstance(metadata, dict) else ''
 title_text = f"{class_name}"
 if plant_id:
 title_text += f"\n{plant_id}"
 
 axes[i].set_title(title_text, fontsize=10)
 axes[i].axis('off')
 
 # Hide unused subplots
 for i in range(len(selected_indices), len(axes)):
 axes[i].axis('off')
 
 plt.tight_layout()
 
 if save_path:
 plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
 
 return fig

def plot_bloom_progression(
 dataset,
 plant_id: str,
 save_path: Optional[str] = None,
 figsize: Tuple[int, int] = (20, 6),
 title: Optional[str] = None
) -> plt.Figure:
 """
 Plot the bloom progression for a specific plant over time.
 
 Args:
 dataset: Dataset containing temporal data
 plant_id: ID of the plant to visualize
 save_path: Optional path to save the plot
 figsize: Figure size tuple
 title: Optional custom title
 
 Returns:
 matplotlib Figure object
 """
 # Find all samples for the specified plant
 plant_samples = []
 
 for i in range(len(dataset)):
 image, label, metadata = dataset[i]
 if isinstance(metadata, dict) and metadata.get('plant_id') == plant_id:
 plant_samples.append((i, image, label, metadata))
 
 if not plant_samples:
 raise ValueError(f"No samples found for plant_id: {plant_id}")
 
 # Sort by timestamp if available
 if all('timestamp' in sample[3] for sample in plant_samples):
 plant_samples.sort(key=lambda x: x[3]['timestamp'])
 
 num_samples = len(plant_samples)
 
 fig, axes = plt.subplots(2, num_samples, figsize=figsize, 
 gridspec_kw={'height_ratios': [4, 1]})
 
 if title is None:
 title = f"Bloom Progression - Plant {plant_id}"
 fig.suptitle(title, fontsize=16, fontweight='bold')
 
 # Handle single sample case
 if num_samples == 1:
 axes = axes.reshape(-1, 1)
 
 # Get class names
 class_names = getattr(dataset, 'class_names', None)
 if class_names is None and hasattr(dataset, 'STAGE_NAMES'):
 class_names = list(dataset.STAGE_NAMES.values())
 
 bloom_progression = []
 timestamps = []
 
 for i, (idx, image, label, metadata) in enumerate(plant_samples):
 # Process image
 if torch.is_tensor(image):
 if image.dim() == 3 and image.shape[0] in [1, 3]:
 image = image.permute(1, 2, 0)
 image = image.cpu().numpy()
 
 # Denormalize if needed
 if image.min() < 0:
 mean = np.array([0.485, 0.456, 0.406])
 std = np.array([0.229, 0.224, 0.225])
 image = image * std + mean
 image = np.clip(image, 0, 1)
 
 # Display image
 axes[0, i].imshow(image)
 
 # Get label info
 if isinstance(label, torch.Tensor):
 label = label.item()
 
 class_name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
 timestamp = metadata.get('timestamp', f'Time {i}')
 
 axes[0, i].set_title(f"{timestamp}\n{class_name}", fontsize=10)
 axes[0, i].axis('off')
 
 # Store for progression plot
 bloom_progression.append(label)
 timestamps.append(timestamp)
 
 # Plot bloom stage progression
 if num_samples > 1:
 # Use the full width for progression plot
 ax_prog = plt.subplot(2, 1, 2)
 ax_prog.plot(range(len(bloom_progression)), bloom_progression, 
 'o-', linewidth=3, markersize=8, color='#2E8B57')
 ax_prog.set_title('Bloom Stage Progression', fontweight='bold')
 ax_prog.set_xlabel('Time Point')
 ax_prog.set_ylabel('Bloom Stage')
 
 # Set y-axis labels
 if class_names:
 ax_prog.set_yticks(range(len(class_names)))
 ax_prog.set_yticklabels(class_names, rotation=45)
 
 ax_prog.grid(True, alpha=0.3)
 ax_prog.set_xticks(range(len(timestamps)))
 ax_prog.set_xticklabels([t.split('-')[-1] if '-' in t else t for t in timestamps], 
 rotation=45)
 else:
 # Hide the bottom subplot for single sample
 axes[1, 0].axis('off')
 
 plt.tight_layout()
 
 if save_path:
 plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
 
 return fig

def create_data_explorer(
 dataset,
 save_path: Optional[str] = None,
 figsize: Tuple[int, int] = (20, 15)
) -> plt.Figure:
 """
 Create a comprehensive data exploration dashboard.
 
 Args:
 dataset: Dataset to explore
 save_path: Optional path to save the plot
 figsize: Figure size tuple
 
 Returns:
 matplotlib Figure object
 """
 fig = plt.figure(figsize=figsize)
 gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
 
 fig.suptitle('Dataset Explorer Dashboard', fontsize=20, fontweight='bold')
 
 # 1. Class distribution
 ax1 = fig.add_subplot(gs[0, :2])
 
 # Get all labels
 all_labels = []
 for i in range(min(len(dataset), 1000)): # Sample for efficiency
 _, label, _ = dataset[i]
 if torch.is_tensor(label):
 label = label.item()
 all_labels.append(label)
 
 unique_labels, counts = np.unique(all_labels, return_counts=True)
 class_names = getattr(dataset, 'class_names', [f'Class {i}' for i in unique_labels])
 
 bars = ax1.bar(class_names, counts, color='skyblue', alpha=0.8, edgecolor='black')
 ax1.set_title('Class Distribution', fontweight='bold', fontsize=14)
 ax1.set_ylabel('Count')
 ax1.tick_params(axis='x', rotation=45)
 
 # Add count labels
 for bar, count in zip(bars, counts):
 height = bar.get_height()
 ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
 f'{count}', ha='center', va='bottom', fontweight='bold')
 
 # 2. Sample images grid
 ax2 = fig.add_subplot(gs[0, 2:])
 
 # Show sample images in a grid within this subplot
 sample_indices = random.sample(range(len(dataset)), min(8, len(dataset)))
 
 # Create mini grid for samples
 mini_rows, mini_cols = 2, 4
 for i, idx in enumerate(sample_indices):
 if i >= 8:
 break
 
 image, label, metadata = dataset[idx]
 
 # Process image
 if torch.is_tensor(image):
 if image.dim() == 3 and image.shape[0] in [1, 3]:
 image = image.permute(1, 2, 0)
 image = image.cpu().numpy()
 
 if image.min() < 0:
 mean = np.array([0.485, 0.456, 0.406])
 std = np.array([0.229, 0.224, 0.225])
 image = image * std + mean
 image = np.clip(image, 0, 1)
 
 # Calculate position in mini grid
 row = i // mini_cols
 col = i % mini_cols
 
 # Create small subplot
 left = col / mini_cols
 bottom = 1 - (row + 1) / mini_rows
 width = 1 / mini_cols
 height = 1 / mini_rows
 
 mini_ax = fig.add_axes([ax2.get_position().x0 + left * ax2.get_position().width,
 ax2.get_position().y0 + bottom * ax2.get_position().height,
 width * ax2.get_position().width * 0.9,
 height * ax2.get_position().height * 0.9])
 
 mini_ax.imshow(image)
 mini_ax.axis('off')
 
 if torch.is_tensor(label):
 label = label.item()
 class_name = class_names[label] if label < len(class_names) else f"Class {label}"
 mini_ax.set_title(class_name, fontsize=8)
 
 ax2.set_title('Random Samples', fontweight='bold', fontsize=14)
 ax2.axis('off')
 
 # 3. Dataset statistics
 ax3 = fig.add_subplot(gs[1, :2])
 
 stats_data = {
 'Total Samples': len(dataset),
 'Number of Classes': len(unique_labels),
 'Samples per Class (avg)': np.mean(counts),
 'Class Imbalance Ratio': max(counts) / min(counts) if min(counts) > 0 else 0
 }
 
 if hasattr(dataset, 'annotations'):
 unique_plants = dataset.annotations.get('plant_id', pd.Series()).nunique()
 if unique_plants > 0:
 stats_data['Unique Plants'] = unique_plants
 
 # Create text display for statistics
 stats_text = '\n'.join([f'{k}: {v:.2f}' if isinstance(v, float) else f'{k}: {v}' 
 for k, v in stats_data.items()])
 
 ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes, fontsize=12,
 verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
 ax3.set_title('Dataset Statistics', fontweight='bold', fontsize=14)
 ax3.axis('off')
 
 # 4. Class balance pie chart
 ax4 = fig.add_subplot(gs[1, 2:])
 
 colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
 wedges, texts, autotexts = ax4.pie(counts, labels=class_names, autopct='%1.1f%%', 
 colors=colors, startangle=90)
 ax4.set_title('Class Balance', fontweight='bold', fontsize=14)
 
 # 5. Sample augmentations (if available)
 ax5 = fig.add_subplot(gs[2, :])
 
 try:
 # Try to show augmentation examples
 sample_idx = random.randint(0, len(dataset) - 1)
 original_image, _, _ = dataset[sample_idx]
 
 # If dataset has augmentation transforms, apply them
 if hasattr(dataset, 'transform') and dataset.transform:
 augmented_images = []
 for _ in range(5):
 aug_image = dataset.transform(Image.fromarray((original_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)))
 if torch.is_tensor(aug_image):
 aug_image = aug_image.permute(1, 2, 0).numpy()
 augmented_images.append(aug_image)
 
 # Display original + augmentations
 for i, img in enumerate([original_image.permute(1, 2, 0).numpy()] + augmented_images):
 left = i / 6
 mini_ax = fig.add_axes([ax5.get_position().x0 + left * ax5.get_position().width,
 ax5.get_position().y0,
 ax5.get_position().width / 6 * 0.9,
 ax5.get_position().height * 0.9])
 
 if img.min() < 0:
 mean = np.array([0.485, 0.456, 0.406])
 std = np.array([0.229, 0.224, 0.225])
 img = img * std + mean
 img = np.clip(img, 0, 1)
 
 mini_ax.imshow(img)
 mini_ax.axis('off')
 mini_ax.set_title('Original' if i == 0 else f'Aug {i}', fontsize=8)
 
 ax5.set_title('Sample Augmentations', fontweight='bold', fontsize=14)
 else:
 ax5.text(0.5, 0.5, 'No augmentations available', ha='center', va='center',
 transform=ax5.transAxes, fontsize=14, style='italic')
 ax5.set_title('Augmentations', fontweight='bold', fontsize=14)
 
 ax5.axis('off')
 
 except Exception as e:
 ax5.text(0.5, 0.5, f'Error displaying augmentations: {str(e)}', 
 ha='center', va='center', transform=ax5.transAxes, fontsize=12)
 ax5.set_title('Augmentations', fontweight='bold', fontsize=14)
 ax5.axis('off')
 
 if save_path:
 plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
 
 return fig

def visualize_augmentations(
 dataset,
 sample_idx: int = 0,
 num_augmentations: int = 8,
 save_path: Optional[str] = None,
 figsize: Tuple[int, int] = (16, 8)
) -> plt.Figure:
 """
 Visualize different augmentations applied to a sample image.
 
 Args:
 dataset: Dataset with augmentation transforms
 sample_idx: Index of sample to augment
 num_augmentations: Number of augmented versions to show
 save_path: Optional path to save the plot
 figsize: Figure size tuple
 
 Returns:
 matplotlib Figure object
 """
 # Get original sample
 original_image, label, metadata = dataset[sample_idx]
 
 # Convert to PIL for augmentation
 if torch.is_tensor(original_image):
 if original_image.dim() == 3 and original_image.shape[0] in [1, 3]:
 original_image = original_image.permute(1, 2, 0)
 original_image = original_image.cpu().numpy()
 
 # Denormalize if needed
 if original_image.min() < 0:
 mean = np.array([0.485, 0.456, 0.406])
 std = np.array([0.229, 0.224, 0.225])
 original_image = original_image * std + mean
 original_image = np.clip(original_image, 0, 1)
 
 # Convert to PIL Image
 pil_image = Image.fromarray((original_image * 255).astype(np.uint8))
 
 fig, axes = plt.subplots(2, num_augmentations // 2 + 1, figsize=figsize)
 fig.suptitle(f'Augmentation Examples - Sample {sample_idx}', fontsize=16, fontweight='bold')
 
 axes = axes.flatten()
 
 # Show original
 axes[0].imshow(original_image)
 axes[0].set_title('Original', fontweight='bold')
 axes[0].axis('off')
 
 # Apply different augmentations
 if hasattr(dataset, 'transform') and dataset.transform:
 for i in range(1, min(len(axes), num_augmentations + 1)):
 try:
 augmented = dataset.transform(pil_image)
 
 if torch.is_tensor(augmented):
 if augmented.dim() == 3 and augmented.shape[0] in [1, 3]:
 augmented = augmented.permute(1, 2, 0)
 augmented = augmented.cpu().numpy()
 
 # Denormalize if needed
 if augmented.min() < 0:
 mean = np.array([0.485, 0.456, 0.406])
 std = np.array([0.229, 0.224, 0.225])
 augmented = augmented * std + mean
 augmented = np.clip(augmented, 0, 1)
 
 axes[i].imshow(augmented)
 axes[i].set_title(f'Augmented {i}', fontweight='bold')
 axes[i].axis('off')
 
 except Exception as e:
 axes[i].text(0.5, 0.5, f'Error: {str(e)[:20]}...', ha='center', va='center',
 transform=axes[i].transAxes, fontsize=10)
 axes[i].set_title(f'Augmented {i}', fontweight='bold')
 axes[i].axis('off')
 
 # Hide unused subplots
 for i in range(num_augmentations + 1, len(axes)):
 axes[i].axis('off')
 
 plt.tight_layout()
 
 if save_path:
 plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
 
 return fig