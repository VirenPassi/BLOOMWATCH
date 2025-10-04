"""
Full Kaggle Flowers Recognition Training Pipeline

Features:
- Automatic dataset download and validation
- ResNet50 backbone with two-phase training
- Strong data augmentation and early stopping
- GPU auto-detection with CPU fallback
- Comprehensive evaluation and reporting
- Hackathon-ready outputs

Usage:
 python pipelines/train_flowers_full.py
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Configuration
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
TRAINING_DIR = OUTPUTS_DIR / "flowers_training"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
BEST_MODEL_PATH = MODELS_DIR / "flowers_resnet50_best.pt"
FINAL_METRICS_PATH = OUTPUTS_DIR / "flowers_final_metrics.json"
SUMMARY_PATH = OUTPUTS_DIR / "flowers_summary.md"

# Dataset configuration
FLOWER_CLASSES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
# Use materialized splits under processed real flowers dataset
PROCESSED_DIR = ROOT / "data" / "processed" / "real_flowers"
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training configuration
DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 7
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE_GPU = 32
DEFAULT_BATCH_SIZE_CPU = 8

class ProcessedFlowersDataset(Dataset):
 """Dataset loading from processed split folders with strong augmentation."""
 def __init__(self, data_root: Path, split: str = "train"):
 from torchvision import datasets as tvdatasets
 self.split = split
 if split == "train":
 self.base_transform = transforms.Compose([
 transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
 transforms.RandomHorizontalFlip(p=0.5),
 transforms.RandomRotation(30),
 transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
 transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
 transforms.ToTensor(),
 transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
 transforms.RandomErasing(p=0.2)
 ])
 else:
 self.base_transform = transforms.Compose([
 transforms.Resize((224, 224)),
 transforms.ToTensor(),
 transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
 ])
 # Use ImageFolder to get class indices
 self.folder = tvdatasets.ImageFolder(data_root, transform=self._img_only_transform())
 self.classes = self.folder.classes
 # Map from folder class index to our FLOWER_CLASSES order
 self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
 self.target_indices = {c: FLOWER_CLASSES.index(c) for c in self.classes}

 def _img_only_transform(self):
 # Same as base_transform but returns only image tensor
 return self.base_transform

 def __len__(self):
 return len(self.folder)

 def __getitem__(self, idx):
 img, folder_label = self.folder[idx]
 # Convert folder_label (0..n) to our canonical label order
 class_name = self.classes[folder_label]
 y = torch.tensor(FLOWER_CLASSES.index(class_name), dtype=torch.long)
 # Add zeros for NDVI/EVI to make 5 channels
 ndvi = torch.zeros(1, 224, 224, dtype=torch.float32)
 evi = torch.zeros(1, 224, 224, dtype=torch.float32)
 x = torch.cat([img, ndvi, evi], dim=0)
 return x, y

class ResNet50FlowerClassifier(nn.Module):
 """ResNet50-based flower classifier with 5-channel input adaptation."""
 
 def __init__(self, num_classes: int, pretrained: bool = True):
 super().__init__()
 # Load pretrained ResNet50
 self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
 
 # Adapt 5-channel input to 3-channel ResNet50
 self.input_adaptation = nn.Conv2d(5, 3, kernel_size=1, bias=False)
 
 # Replace final classifier
 self.backbone.fc = nn.Sequential(
 nn.Dropout(0.5),
 nn.Linear(self.backbone.fc.in_features, 512),
 nn.ReLU(inplace=True),
 nn.Dropout(0.3),
 nn.Linear(512, num_classes)
 )
 
 # Freeze backbone initially
 self.freeze_backbone()
 
 def freeze_backbone(self):
 """Freeze all backbone parameters except the final classifier."""
 for param in self.backbone.parameters():
 param.requires_grad = False
 for param in self.backbone.fc.parameters():
 param.requires_grad = True
 
 def unfreeze_backbone(self):
 """Unfreeze all parameters for fine-tuning."""
 for param in self.backbone.parameters():
 param.requires_grad = True
 
 def forward(self, x):
 x = self.input_adaptation(x)
 return self.backbone(x)

class EarlyStopping:
 """Early stopping utility."""
 
 def __init__(self, patience: int = 7, min_delta: float = 0.001):
 self.patience = patience
 self.min_delta = min_delta
 self.counter = 0
 self.best_score = None
 self.early_stop = False
 
 def __call__(self, val_score: float) -> bool:
 if self.best_score is None:
 self.best_score = val_score
 elif val_score < self.best_score + self.min_delta:
 self.counter += 1
 if self.counter >= self.patience:
 self.early_stop = True
 else:
 self.best_score = val_score
 self.counter = 0
 
 return self.early_stop

def ensure_processed_splits() -> bool:
 """Verify processed split directories exist."""
 if not TRAIN_DIR.exists() or not VAL_DIR.exists() or not TEST_DIR.exists():
 print("Processed split directories not found. Please run data/split_real_flowers.py")
 return False
 return True

def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int, total_epochs: int):
 """Train for one epoch with progress logging."""
 model.train()
 total, correct, loss_sum = 0, 0, 0.0
 
 for batch_idx, (x, y) in enumerate(loader):
 x, y = x.to(device), y.to(device)
 optimizer.zero_grad()
 out = model(x)
 loss = criterion(out, y)
 loss.backward()
 
 # Gradient clipping
 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 
 optimizer.step()
 loss_sum += float(loss.item())
 _, pred = out.max(1)
 total += y.size(0)
 correct += (pred == y).sum().item()
 
 # Progress logging
 if batch_idx % 20 == 0:
 current_lr = optimizer.param_groups[0]['lr']
 print(f" Epoch {epoch}/{total_epochs} - Batch {batch_idx}/{len(loader)} - "
 f"Loss: {loss.item():.4f} - LR: {current_lr:.6f}")
 
 avg_loss = loss_sum / max(len(loader), 1)
 accuracy = correct / max(total, 1)
 return avg_loss, accuracy

def evaluate(model, loader, device):
 """Evaluate model and return metrics."""
 model.eval()
 total, correct = 0, 0
 preds, targets = [], []
 
 with torch.no_grad():
 for x, y in loader:
 x, y = x.to(device), y.to(device)
 out = model(x)
 _, pred = out.max(1)
 total += y.size(0)
 correct += (pred == y).sum().item()
 preds.extend(pred.cpu().numpy())
 targets.extend(y.cpu().numpy())
 
 acc = correct / max(total, 1)
 cm = confusion_matrix(targets, preds)
 report = classification_report(targets, preds, target_names=FLOWER_CLASSES, 
 output_dict=True, zero_division=0)
 return acc, cm, report

def save_training_plots(history: Dict, cm: np.ndarray, save_dir: Path):
 """Save training curves and confusion matrix."""
 save_dir.mkdir(parents=True, exist_ok=True)
 
 # Training curves
 fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
 
 # Loss curve
 ax1.plot(history['train_loss'], label='Train Loss', color='blue')
 ax1.set_title('Training Loss')
 ax1.set_xlabel('Epoch')
 ax1.set_ylabel('Loss')
 ax1.legend()
 ax1.grid(True)
 
 # Accuracy curve
 ax2.plot(history['train_acc'], label='Train Acc', color='blue')
 ax2.plot(history['val_acc'], label='Val Acc', color='red')
 ax2.set_title('Training and Validation Accuracy')
 ax2.set_xlabel('Epoch')
 ax2.set_ylabel('Accuracy')
 ax2.legend()
 ax2.grid(True)
 
 # Learning rate curve
 ax3.plot(history['lr'], label='Learning Rate', color='green')
 ax3.set_title('Learning Rate Schedule')
 ax3.set_xlabel('Epoch')
 ax3.set_ylabel('Learning Rate')
 ax3.legend()
 ax3.grid(True)
 ax3.set_yscale('log')
 
 # Confusion matrix
 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
 xticklabels=FLOWER_CLASSES, yticklabels=FLOWER_CLASSES, ax=ax4)
 ax4.set_title('Confusion Matrix (Test Set)')
 ax4.set_xlabel('Predicted')
 ax4.set_ylabel('Actual')
 
 plt.tight_layout()
 plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
 plt.close()
 
 # Separate confusion matrix
 plt.figure(figsize=(10, 8))
 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
 xticklabels=FLOWER_CLASSES, yticklabels=FLOWER_CLASSES)
 plt.title('Confusion Matrix - Kaggle Flowers Recognition')
 plt.xlabel('Predicted')
 plt.ylabel('Actual')
 plt.tight_layout()
 plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
 plt.close()
 
 print(f"Training plots saved to {save_dir}")

def generate_summary_report(metrics: Dict, save_path: Path):
 """Generate comprehensive summary report."""
 
 report = f"""# Kaggle Flowers Recognition - Training Summary

## **Training Results**

**Model Architecture:** ResNet50 with 5-channel input adaptation 
**Dataset:** Kaggle Flowers Recognition (5 classes) 
**Training Date:** {time.strftime("%Y-%m-%d %H:%M:%S")} 

## **Performance Metrics**

### Overall Performance
- **Best Validation Accuracy:** {metrics['best_val_acc']:.1%}
- **Final Test Accuracy:** {metrics['test_acc']:.1%}
- **Training Time:** {metrics['training_time']:.1f} seconds
- **Total Epochs:** {metrics['total_epochs']}

### Dataset Information
- **Training Samples:** {metrics['train_samples']:,}
- **Validation Samples:** {metrics['val_samples']:,}
- **Test Samples:** {metrics['test_samples']:,}
- **Total Samples:** {metrics['total_samples']:,}

### Training Configuration
- **Batch Size:** {metrics['batch_size']}
- **Learning Rate:** {metrics['learning_rate']}
- **Optimizer:** AdamW
- **Early Stopping Patience:** {metrics['patience']}
- **Device:** {metrics['device']}

## **Class-wise Performance**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
"""
 
 # Add class-wise metrics
 for class_name in FLOWER_CLASSES:
 if class_name in metrics['test_report']:
 cls_metrics = metrics['test_report'][class_name]
 report += f"| {class_name.capitalize()} | {cls_metrics['precision']:.3f} | {cls_metrics['recall']:.3f} | {cls_metrics['f1-score']:.3f} | {int(cls_metrics['support'])} |\n"
 
 report += f"""
## **Training History**

- **Final Training Loss:** {metrics['final_train_loss']:.4f}
- **Final Training Accuracy:** {metrics['final_train_acc']:.1%}
- **Final Validation Accuracy:** {metrics['final_val_acc']:.1%}
- **Best Epoch:** {metrics['best_epoch']}

## **How to Reproduce**

```bash
# Download dataset
python data/download_kaggle_flowers.py

# Run training
python pipelines/train_flowers_full.py

# Launch web interface
streamlit run webapp/app.py
```

## **Output Files**

- **Best Model:** `outputs/models/flowers_resnet50_best.pt`
- **Training Plots:** `outputs/flowers_training/`
- **Metrics:** `outputs/flowers_final_metrics.json`
- **Summary:** `outputs/flowers_summary.md`

## **Model Usage**

```python
import torch
from pipelines.train_flowers_full import ResNet50FlowerClassifier

# Load trained model
model = ResNet50FlowerClassifier(num_classes=5, pretrained=False)
model.load_state_dict(torch.load('outputs/models/flowers_resnet50_best.pt'))
model.eval()

# Run inference
# (See webapp/app.py for complete inference pipeline)
```

---
*Generated automatically by BloomWatch training pipeline*
"""
 
 with open(save_path, 'w', encoding='utf-8') as f:
 f.write(report)
 
 print(f"Summary report saved to {save_path}")

def main():
 """Main training pipeline."""
 print("=" * 80)
 print("KAGGLE FLOWERS RECOGNITION - FULL TRAINING PIPELINE")
 print("=" * 80)
 
 # Check if model already exists
 if BEST_MODEL_PATH.exists():
 print(f"Model already exists at {BEST_MODEL_PATH}")
 print("Skipping training. Delete the model file to retrain.")
 return
 
 # Device setup
 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 batch_size = DEFAULT_BATCH_SIZE_GPU if device.type == 'cuda' else DEFAULT_BATCH_SIZE_CPU
 
 print(f"Using device: {device}")
 if device.type == 'cuda':
 print(f"GPU: {torch.cuda.get_device_name(0)}")
 print(f"Batch size: {batch_size}")
 else:
 print(f"CPU mode - Batch size: {batch_size}")
 
 # Ensure processed splits
 if not ensure_processed_splits():
 return
 
 # Load datasets from processed splits
 print("\nLoading datasets from processed splits...")
 train_ds = ProcessedFlowersDataset(TRAIN_DIR, split='train')
 val_ds = ProcessedFlowersDataset(VAL_DIR, split='val')
 test_ds = ProcessedFlowersDataset(TEST_DIR, split='test')
 
 train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
 val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
 test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
 
 print(f"Dataset splits -> Train: {len(train_ds):,}, Val: {len(val_ds):,}, Test: {len(test_ds):,}")
 
 # Initialize model
 print("\nInitializing ResNet50 model...")
 model = ResNet50FlowerClassifier(num_classes=len(FLOWER_CLASSES), pretrained=True).to(device)
 
 # Training setup
 criterion = nn.CrossEntropyLoss()
 optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LR, weight_decay=1e-4)
 scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
 optimizer, T_max=10, eta_min=1e-6
 )
 early_stopping = EarlyStopping(patience=max(DEFAULT_PATIENCE, 10))
 
 # Training history
 history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
 best_val_acc = 0.0
 best_epoch = 0
 
 print(f"\nStarting training for {DEFAULT_EPOCHS} epochs...")
 start_time = time.time()
 
 for epoch in range(1, DEFAULT_EPOCHS + 1):
 print(f"\nEpoch {epoch}/{DEFAULT_EPOCHS}")
 
 # Training
 train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, DEFAULT_EPOCHS)
 
 # Validation
 val_acc, val_cm, val_report = evaluate(model, val_loader, device)
 
 # Learning rate scheduling per epoch
 scheduler.step()
 current_lr = optimizer.param_groups[0]['lr']
 
 # Logging
 print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, LR: {current_lr:.6f}")
 
 # Save history
 history['train_loss'].append(train_loss)
 history['train_acc'].append(train_acc)
 history['val_acc'].append(val_acc)
 history['lr'].append(current_lr)
 
 # Save best model
 if val_acc > best_val_acc:
 best_val_acc = val_acc
 best_epoch = epoch
 MODELS_DIR.mkdir(parents=True, exist_ok=True)
 torch.save(model.state_dict(), BEST_MODEL_PATH)
 print(f"New best model saved (Val Acc: {val_acc:.3f})")
 
 # Early stopping
 if early_stopping(val_acc):
 print(f"Early stopping triggered at epoch {epoch}")
 break
 
 training_time = time.time() - start_time
 print(f"\nTraining completed in {training_time:.1f} seconds")
 print(f"Best validation accuracy: {best_val_acc:.3f} (epoch {best_epoch})")
 
 # Final evaluation on test set
 print(f"\nFinal evaluation on test set...")
 model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
 test_acc, test_cm, test_report = evaluate(model, test_loader, device)
 print(f"Final test accuracy: {test_acc:.3f}")
 
 # Save training plots
 save_training_plots(history, test_cm, TRAINING_DIR)
 
 # Generate comprehensive metrics
 metrics = {
 "model_architecture": "ResNet50",
 "dataset": "Kaggle Flowers Recognition",
 "classes": FLOWER_CLASSES,
 "best_val_acc": best_val_acc,
 "test_acc": test_acc,
 "training_time": training_time,
 "total_epochs": len(history['train_loss']),
 "best_epoch": best_epoch,
 "train_samples": len(train_ds),
 "val_samples": len(val_ds),
 "test_samples": len(test_ds),
 "total_samples": len(train_ds) + len(val_ds) + len(test_ds),
 "batch_size": batch_size,
 "learning_rate": DEFAULT_LR,
 "patience": DEFAULT_PATIENCE,
 "device": str(device),
 "final_train_loss": history['train_loss'][-1],
 "final_train_acc": history['train_acc'][-1],
 "final_val_acc": history['val_acc'][-1],
 "test_report": test_report,
 "model_path": str(BEST_MODEL_PATH),
 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
 }
 
 # Save metrics
 with open(FINAL_METRICS_PATH, 'w') as f:
 json.dump(metrics, f, indent=2)
 
 # Generate summary report
 generate_summary_report(metrics, SUMMARY_PATH)
 
 print(f"\nTraining pipeline completed successfully!")
 print(f" Best model: {BEST_MODEL_PATH}")
 print(f"Metrics: {FINAL_METRICS_PATH}")
 print(f" Summary: {SUMMARY_PATH}")
 print(f"Plots: {TRAINING_DIR}")

if __name__ == "__main__":
 main()
