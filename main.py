#!/usr/bin/env python3
"""
BloomWatch Main Training Script

This script demonstrates the complete training pipeline for plant bloom detection.
It includes dummy data generation, model training, validation, and visualization.

Usage:
 python main.py
 python main.py --config configs/train_config.yaml
 python main.py --model simple_cnn --epochs 10
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import application components
# Create dummy dataset class for demonstration
class DummyPlantBloomDataset(Dataset):
 """Dummy dataset for plant bloom detection."""
 
 def __init__(self, num_samples=100, img_size=(224, 224), num_classes=5, transform=None):
 self.num_samples = num_samples
 self.img_size = img_size
 self.num_classes = num_classes
 self.transform = transform
 
 # Generate dummy data
 self.data = torch.randn(num_samples, 3, *img_size)
 self.targets = torch.randint(0, num_classes, (num_samples,))
 
 def __len__(self):
 return self.num_samples
 
 def __getitem__(self, idx):
 image = self.data[idx]
 target = self.targets[idx]
 return image, target

# Simple CNN model for demonstration
class SimpleCNN(nn.Module):
 """Simple CNN for plant bloom classification."""
 
 def __init__(self, num_classes=5):
 super(SimpleCNN, self).__init__()
 self.features = nn.Sequential(
 nn.Conv2d(3, 32, 3, padding=1),
 nn.ReLU(inplace=True),
 nn.MaxPool2d(2, 2),
 nn.Conv2d(32, 64, 3, padding=1),
 nn.ReLU(inplace=True),
 nn.MaxPool2d(2, 2),
 nn.Conv2d(64, 128, 3, padding=1),
 nn.ReLU(inplace=True),
 nn.AdaptiveAvgPool2d((7, 7))
 )
 self.classifier = nn.Sequential(
 nn.Linear(128 * 7 * 7, 256),
 nn.ReLU(inplace=True),
 nn.Dropout(0.5),
 nn.Linear(256, num_classes)
 )
 
 def forward(self, x):
 x = self.features(x)
 x = torch.flatten(x, 1)
 x = self.classifier(x)
 return x

def parse_arguments():
 """Parse command line arguments."""
 parser = argparse.ArgumentParser(description='BloomWatch Training Pipeline')
 parser.add_argument('--model', type=str, default='simple_cnn',
 choices=['simple_cnn'],
 help='Model architecture to use')
 parser.add_argument('--epochs', type=int, default=5,
 help='Number of training epochs')
 parser.add_argument('--batch_size', type=int, default=16,
 help='Batch size')
 parser.add_argument('--lr', type=float, default=0.001,
 help='Learning rate')
 parser.add_argument('--device', type=str, default='auto',
 choices=['cpu', 'cuda', 'auto'],
 help='Device to use for training')
 parser.add_argument('--output_dir', type=str, default='outputs',
 help='Output directory for results')
 parser.add_argument('--seed', type=int, default=42,
 help='Random seed for reproducibility')
 
 return parser.parse_args()

def set_seed(seed):
 """Set random seeds for reproducibility."""
 torch.manual_seed(seed)
 torch.cuda.manual_seed_all(seed)
 import random
 import numpy as np
 random.seed(seed)
 np.random.seed(seed)
 torch.backends.cudnn.deterministic = True
 torch.backends.cudnn.benchmark = False

def get_device(device_arg=None):
 """Get the appropriate device for training."""
 if device_arg == 'cpu':
 return torch.device('cpu')
 elif device_arg == 'cuda':
 if torch.cuda.is_available():
 return torch.device('cuda')
 else:
 logging.warning("CUDA requested but not available, using CPU")
 return torch.device('cpu')
 else: # auto or None
 return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_data_loaders(batch_size, num_workers=0):
 """Create training and validation data loaders."""
 # Create dummy datasets
 train_dataset = DummyPlantBloomDataset(
 num_samples=200,
 img_size=(224, 224),
 num_classes=5
 )
 
 val_dataset = DummyPlantBloomDataset(
 num_samples=50,
 img_size=(224, 224),
 num_classes=5
 )
 
 train_loader = DataLoader(
 train_dataset,
 batch_size=batch_size,
 shuffle=True,
 num_workers=num_workers
 )
 
 val_loader = DataLoader(
 val_dataset,
 batch_size=batch_size,
 shuffle=False,
 num_workers=num_workers
 )
 
 logging.info(f"Created data loaders: train={len(train_dataset)}, val={len(val_dataset)}")
 return train_loader, val_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
 """Train the model for one epoch."""
 model.train()
 total_loss = 0.0
 correct = 0
 total = 0
 
 for batch_idx, (data, target) in enumerate(train_loader):
 data, target = data.to(device), target.to(device)
 
 optimizer.zero_grad()
 output = model(data)
 loss = criterion(output, target)
 loss.backward()
 optimizer.step()
 
 total_loss += loss.item()
 pred = output.argmax(dim=1, keepdim=True)
 correct += pred.eq(target.view_as(pred)).sum().item()
 total += target.size(0)
 
 if batch_idx % 10 == 0:
 logging.debug(f'Train Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
 
 avg_loss = total_loss / len(train_loader)
 accuracy = 100. * correct / total
 
 return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, device):
 """Validate the model for one epoch."""
 model.eval()
 total_loss = 0.0
 correct = 0
 total = 0
 
 with torch.no_grad():
 for data, target in val_loader:
 data, target = data.to(device), target.to(device)
 output = model(data)
 loss = criterion(output, target)
 
 total_loss += loss.item()
 pred = output.argmax(dim=1, keepdim=True)
 correct += pred.eq(target.view_as(pred)).sum().item()
 total += target.size(0)
 
 avg_loss = total_loss / len(val_loader)
 accuracy = 100. * correct / total
 
 return avg_loss, accuracy

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
 """Plot training and validation curves."""
 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
 
 # Plot loss
 ax1.plot(train_losses, label='Train Loss')
 ax1.plot(val_losses, label='Val Loss')
 ax1.set_title('Training and Validation Loss')
 ax1.set_xlabel('Epoch')
 ax1.set_ylabel('Loss')
 ax1.legend()
 ax1.grid(True, alpha=0.3)
 
 # Plot accuracy
 ax2.plot(train_accs, label='Train Accuracy')
 ax2.plot(val_accs, label='Val Accuracy')
 ax2.set_title('Training and Validation Accuracy')
 ax2.set_xlabel('Epoch')
 ax2.set_ylabel('Accuracy (%)')
 ax2.legend()
 ax2.grid(True, alpha=0.3)
 
 plt.tight_layout()
 
 if save_path:
 plt.savefig(save_path, dpi=300, bbox_inches='tight')
 plt.close(fig)
 
 return fig

def plot_confusion_matrix(targets, predictions, class_names=None, save_path=None):
 """Plot confusion matrix."""
 from sklearn.metrics import confusion_matrix
 import seaborn as sns
 
 cm = confusion_matrix(targets, predictions)
 
 plt.figure(figsize=(8, 6))
 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
 xticklabels=class_names or range(len(cm)),
 yticklabels=class_names or range(len(cm)))
 plt.title('Confusion Matrix')
 plt.xlabel('Predicted')
 plt.ylabel('Actual')
 
 if save_path:
 plt.savefig(save_path, dpi=300, bbox_inches='tight')
 plt.close()
 
 return plt.gcf()

def plot_growth_curve(time_points, bloom_scores, save_path=None):
 """Plot a simple growth curve."""
 plt.figure(figsize=(10, 6))
 plt.plot(time_points, bloom_scores, 'o-', linewidth=2, markersize=6)
 plt.title('Plant Bloom Progression Curve', fontsize=14, fontweight='bold')
 plt.xlabel('Time (days)', fontweight='bold')
 plt.ylabel('Bloom Score', fontweight='bold')
 plt.grid(True, alpha=0.3)
 
 # Color code points based on bloom intensity
 colors = []
 for score in bloom_scores:
 if score < 0.2:
 colors.append('brown') # Bud stage
 elif score < 0.4:
 colors.append('green') # Early bloom
 elif score < 0.7:
 colors.append('pink') # Full bloom
 elif score < 0.9:
 colors.append('orange') # Late bloom
 else:
 colors.append('gray') # Dormant
 
 plt.scatter(time_points, bloom_scores, c=colors, s=100, edgecolor='black', linewidth=1, zorder=5)
 plt.tight_layout()
 
 if save_path:
 plt.savefig(save_path, dpi=300, bbox_inches='tight')
 plt.close()
 
 return plt.gcf()

def main():
 """Main training function."""
 # Parse arguments
 args = parse_arguments()
 
 # Set up output directory
 output_dir = Path(args.output_dir)
 output_dir.mkdir(exist_ok=True)
 
 # Set up logging
 logging.basicConfig(
 level=logging.INFO,
 format='%(asctime)s - %(levelname)s - %(message)s',
 handlers=[
 logging.FileHandler(output_dir / 'training.log'),
 logging.StreamHandler()
 ]
 )
 logger = logging.getLogger(__name__)
 
 # Set random seed
 set_seed(args.seed)
 logger.info(f"Set random seed to {args.seed}")
 
 logger.info("Configuration loaded successfully")
 logger.info(f"Training config: epochs={args.epochs}, "
 f"batch_size={args.batch_size}, "
 f"lr={args.lr}")
 
 # Get device
 device = get_device(args.device)
 logger.info(f"Using device: {device}")
 
 # Create data loaders
 train_loader, val_loader = create_data_loaders(args.batch_size)
 
 # Create model
 model = SimpleCNN(num_classes=5).to(device)
 logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
 
 # Create optimizer and loss function
 optimizer = optim.Adam(model.parameters(), lr=args.lr)
 criterion = nn.CrossEntropyLoss()
 
 # Training loop
 logger.info("Starting training...")
 
 train_losses = []
 val_losses = []
 train_accs = []
 val_accs = []
 
 for epoch in range(args.epochs):
 logger.info(f"Epoch {epoch+1}/{args.epochs}")
 
 # Training
 train_loss, train_acc = train_epoch(
 model, train_loader, criterion, optimizer, device
 )
 train_losses.append(train_loss)
 train_accs.append(train_acc)
 
 # Validation
 val_loss, val_acc = validate_epoch(
 model, val_loader, criterion, device
 )
 val_losses.append(val_loss)
 val_accs.append(val_acc)
 
 # Log epoch results
 logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
 logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
 
 # Generate visualizations
 logger.info("Generating visualizations...")
 
 # Plot training curves
 plot_training_curves(
 train_losses, 
 val_losses,
 train_accs, 
 val_accs,
 save_path=output_dir / 'training_curves.png'
 )
 
 # Plot dummy growth curve
 plot_growth_curve(
 time_points=list(range(30)), # 30 days
 bloom_scores=[0.1 + 0.03*i + 0.01*i**1.5 for i in range(30)], # Dummy growth
 save_path=output_dir / 'growth_curve.png'
 )
 
 logger.info(f"Training completed!")
 logger.info(f"Final validation accuracy: {val_accs[-1]:.2f}%")
 logger.info(f"Results saved to: {output_dir}")
 
 print("\n" + "="*60)
 print(" BLOOMWATCH TRAINING COMPLETE! ")
 print("="*60)
 print(f"Final Validation Accuracy: {val_accs[-1]:.2f}%")
 print(f"Results saved to: {output_dir}")
 print(f"View training curves: {output_dir / 'training_curves.png'}")
 print(f"View growth curve: {output_dir / 'growth_curve.png'}")
 print("="*60)

if __name__ == "__main__":
 main()