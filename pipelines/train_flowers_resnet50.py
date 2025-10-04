"""
Fully automated ResNet50 training pipeline for flower classification.

This script trains a ResNet50 model on the real-world flower dataset with the following features:
- Two-phase training (frozen backbone + fine-tuning)
- Advanced data augmentations for training
- Automatic GPU detection and utilization
- Real-time logging and metrics tracking
- Automatic evaluation and reporting
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F_nn
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random
import time
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data.dataset import PlantBloomDataset

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class AdvancedAugmentations:
 """Advanced data augmentations for flower images."""
 
 def __init__(self):
 pass
 
 def __call__(self, img):
 # RandomResizedCrop(224)
 img = transforms.RandomResizedCrop(224)(img)
 
 # RandomHorizontalFlip()
 img = transforms.RandomHorizontalFlip()(img)
 
 # RandomRotation(30°)
 img = transforms.RandomRotation(30)(img)
 
 # RandomAffine(degrees=30, translate=(0.1,0.1), scale=(0.8,1.2))
 img = transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2))(img)
 
 # ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
 img = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)(img)
 
 # Convert to tensor
 img = transforms.ToTensor()(img)
 
 # RandomErasing(p=0.2)
 img = transforms.RandomErasing(p=0.2)(img)
 
 return img

def create_data_loaders(batch_size=32):
 """Create data loaders for train, validation, and test sets."""
 
 # Training transforms with strong augmentations
 train_transform = transforms.Compose([
 AdvancedAugmentations(),
 transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
 ])
 
 # Validation/Test transforms (only resize and normalization)
 val_test_transform = transforms.Compose([
 transforms.Resize((224, 224)),
 transforms.ToTensor(),
 transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
 ])
 
 # Data directories
 data_root = project_root / "data" / "processed" / "real_flowers"
 train_dir = data_root / "train"
 val_dir = data_root / "val"
 test_dir = data_root / "test"
 
 # Create datasets
 train_dataset = CustomFlowerDataset(train_dir, transform=train_transform)
 val_dataset = CustomFlowerDataset(val_dir, transform=val_test_transform)
 test_dataset = CustomFlowerDataset(test_dir, transform=val_test_transform)
 
 # Create data loaders
 train_loader = DataLoader(
 train_dataset, 
 batch_size=batch_size, 
 shuffle=True, 
 num_workers=4, 
 pin_memory=True if torch.cuda.is_available() else False
 )
 
 val_loader = DataLoader(
 val_dataset, 
 batch_size=batch_size, 
 shuffle=False, 
 num_workers=4, 
 pin_memory=True if torch.cuda.is_available() else False
 )
 
 test_loader = DataLoader(
 test_dataset, 
 batch_size=batch_size, 
 shuffle=False, 
 num_workers=4, 
 pin_memory=True if torch.cuda.is_available() else False
 )
 
 return train_loader, val_loader, test_loader

class CustomFlowerDataset(torch.utils.data.Dataset):
 """Custom dataset for flower images organized in folders by class."""
 
 def __init__(self, root_dir, transform=None):
 self.root_dir = root_dir
 self.transform = transform
 self.classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
 self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
 
 # Collect all image paths and labels
 self.samples = []
 for class_name in self.classes:
 class_dir = root_dir / class_name
 for img_path in class_dir.glob("*"):
 if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
 self.samples.append((img_path, self.class_to_idx[class_name]))
 
 def __len__(self):
 return len(self.samples)
 
 def __getitem__(self, idx):
 img_path, label = self.samples[idx]
 image = Image.open(img_path).convert('RGB')
 
 if self.transform:
 image = self.transform(image)
 
 return image, label

def create_model(num_classes=5, pretrained=True):
 """Create ResNet50 model with specified number of classes."""
 model = models.resnet50(pretrained=pretrained)
 model.fc = nn.Linear(model.fc.in_features, num_classes)
 return model

class EarlyStopping:
 """Early stopping to prevent overfitting."""
 
 def __init__(self, patience=10, min_delta=0):
 self.patience = patience
 self.min_delta = min_delta
 self.counter = 0
 self.best_loss = None
 self.early_stop = False

 def __call__(self, val_loss):
 if self.best_loss is None:
 self.best_loss = val_loss
 elif val_loss > self.best_loss - self.min_delta:
 self.counter += 1
 if self.counter >= self.patience:
 self.early_stop = True
 else:
 self.best_loss = val_loss
 self.counter = 0

def calculate_metrics(y_true, y_pred, class_names):
 """Calculate comprehensive classification metrics."""
 # Get classification report as dict
 report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
 
 # Convert numpy arrays to lists for JSON serialization
 for key, value in report.items():
 if isinstance(value, dict):
 for sub_key, sub_value in value.items():
 if isinstance(sub_value, (np.ndarray, np.integer, np.floating)):
 report[key][sub_key] = sub_value.item() if hasattr(sub_value, 'item') else float(sub_value)
 elif isinstance(value, (np.ndarray, np.integer, np.floating)):
 report[key] = value.item() if hasattr(value, 'item') else float(value)
 
 return report

def train_phase1(model, train_loader, val_loader, device, epochs=20):
 """Phase 1: Train only the classifier head with frozen backbone."""
 print("Starting Phase 1: Training classifier head with frozen backbone...")
 
 # Freeze all backbone layers
 for param in model.parameters():
 param.requires_grad = False
 
 # Unfreeze classifier head
 for param in model.fc.parameters():
 param.requires_grad = True
 
 # Optimizer and scheduler for phase 1
 optimizer = optim.AdamW(model.fc.parameters(), lr=0.001, weight_decay=1e-4)
 scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
 criterion = nn.CrossEntropyLoss()
 
 # Gradient scaler for mixed precision training
 scaler = GradScaler() if torch.cuda.is_available() else None
 
 # Early stopping
 early_stopping = EarlyStopping(patience=7)
 
 best_val_acc = 0.0
 train_losses, val_losses = [], []
 train_accs, val_accs = [], []
 
 for epoch in range(epochs):
 # Training
 model.train()
 train_loss, correct, total = 0.0, 0, 0
 
 for batch_idx, (data, target) in enumerate(train_loader):
 data, target = data.to(device), target.to(device)
 
 optimizer.zero_grad()
 
 if scaler is not None:
 with autocast():
 output = model(data)
 loss = criterion(output, target)
 scaler.scale(loss).backward()
 scaler.step(optimizer)
 scaler.update()
 else:
 output = model(data)
 loss = criterion(output, target)
 loss.backward()
 optimizer.step()
 
 train_loss += loss.item()
 pred = output.argmax(dim=1, keepdim=True)
 correct += pred.eq(target.view_as(pred)).sum().item()
 total += target.size(0)
 
 if batch_idx % 10 == 0:
 print(f'Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{len(train_loader)} - '
 f'Loss: {loss.item():.6f}')
 
 train_acc = 100. * correct / total
 avg_train_loss = train_loss / len(train_loader)
 
 # Validation
 model.eval()
 val_loss, val_correct, val_total = 0.0, 0, 0
 
 with torch.no_grad():
 for data, target in val_loader:
 data, target = data.to(device), target.to(device)
 output = model(data)
 val_loss += criterion(output, target).item()
 pred = output.argmax(dim=1, keepdim=True)
 val_correct += pred.eq(target.view_as(pred)).sum().item()
 val_total += target.size(0)
 
 val_acc = 100. * val_correct / val_total
 avg_val_loss = val_loss / len(val_loader)
 
 # Update learning rate
 scheduler.step()
 current_lr = scheduler.get_last_lr()[0]
 
 # Store metrics
 train_losses.append(avg_train_loss)
 val_losses.append(avg_val_loss)
 train_accs.append(train_acc)
 val_accs.append(val_acc)
 
 print(f'Epoch {epoch+1}/{epochs}:')
 print(f' Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
 print(f' Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
 print(f' LR: {current_lr:.6f}')
 
 # Save best model
 if val_acc > best_val_acc:
 best_val_acc = val_acc
 torch.save(model.state_dict(), project_root / "outputs" / "models" / "flowers_resnet50_phase1_best.pt")
 print(f' Saved new best model with validation accuracy: {best_val_acc:.2f}%')
 
 # Early stopping check
 early_stopping(avg_val_loss)
 if early_stopping.early_stop:
 print("Early stopping triggered!")
 break
 
 return train_losses, val_losses, train_accs, val_accs

def train_phase2(model, train_loader, val_loader, device, epochs=30):
 """Phase 2: Fine-tune the entire network with smaller learning rate."""
 print("Starting Phase 2: Fine-tuning entire network...")
 
 # Unfreeze all parameters
 for param in model.parameters():
 param.requires_grad = True
 
 # Optimizer and scheduler for phase 2
 optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
 scheduler = optim.lr_scheduler.OneCycleLR(
 optimizer, 
 max_lr=0.001, 
 epochs=epochs, 
 steps_per_epoch=len(train_loader)
 )
 criterion = nn.CrossEntropyLoss()
 
 # Gradient scaler for mixed precision training
 scaler = GradScaler() if torch.cuda.is_available() else None
 
 # Gradient clipping
 max_norm = 1.0
 
 # Early stopping
 early_stopping = EarlyStopping(patience=10)
 
 best_val_acc = 0.0
 train_losses, val_losses = [], []
 train_accs, val_accs = [], []
 
 for epoch in range(epochs):
 # Training
 model.train()
 train_loss, correct, total = 0.0, 0, 0
 
 for batch_idx, (data, target) in enumerate(train_loader):
 data, target = data.to(device), target.to(device)
 
 optimizer.zero_grad()
 
 if scaler is not None:
 with autocast():
 output = model(data)
 loss = criterion(output, target)
 scaler.scale(loss).backward()
 # Gradient clipping
 scaler.unscale_(optimizer)
 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
 scaler.step(optimizer)
 scaler.update()
 else:
 output = model(data)
 loss = criterion(output, target)
 loss.backward()
 # Gradient clipping
 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
 optimizer.step()
 
 scheduler.step() # Update learning rate each batch for OneCycleLR
 
 train_loss += loss.item()
 pred = output.argmax(dim=1, keepdim=True)
 correct += pred.eq(target.view_as(pred)).sum().item()
 total += target.size(0)
 
 if batch_idx % 10 == 0:
 current_lr = scheduler.get_last_lr()[0]
 print(f'Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{len(train_loader)} - '
 f'Loss: {loss.item():.6f}, LR: {current_lr:.6f}')
 
 train_acc = 100. * correct / total
 avg_train_loss = train_loss / len(train_loader)
 
 # Validation
 model.eval()
 val_loss, val_correct, val_total = 0.0, 0, 0
 
 with torch.no_grad():
 for data, target in val_loader:
 data, target = data.to(device), target.to(device)
 output = model(data)
 val_loss += criterion(output, target).item()
 pred = output.argmax(dim=1, keepdim=True)
 val_correct += pred.eq(target.view_as(pred)).sum().item()
 val_total += target.size(0)
 
 val_acc = 100. * val_correct / val_total
 avg_val_loss = val_loss / len(val_loader)
 current_lr = scheduler.get_last_lr()[0]
 
 # Store metrics
 train_losses.append(avg_train_loss)
 val_losses.append(avg_val_loss)
 train_accs.append(train_acc)
 val_accs.append(val_acc)
 
 print(f'Epoch {epoch+1}/{epochs}:')
 print(f' Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
 print(f' Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
 print(f' LR: {current_lr:.6f}')
 
 # Save best model
 if val_acc > best_val_acc:
 best_val_acc = val_acc
 torch.save(model.state_dict(), project_root / "outputs" / "models" / "flowers_resnet50_best.pt")
 print(f' Saved new best model with validation accuracy: {best_val_acc:.2f}%')
 
 # Early stopping check
 early_stopping(avg_val_loss)
 if early_stopping.early_stop:
 print("Early stopping triggered!")
 break
 
 return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader, device, class_names):
 """Evaluate the model on test set and generate metrics."""
 print("Evaluating model on test set...")
 
 model.eval()
 all_preds = []
 all_targets = []
 all_probs = []
 
 criterion = nn.CrossEntropyLoss()
 test_loss = 0.0
 correct = 0
 total = 0
 
 with torch.no_grad():
 for data, target in test_loader:
 data, target = data.to(device), target.to(device)
 output = model(data)
 test_loss += criterion(output, target).item()
 pred = output.argmax(dim=1, keepdim=True)
 correct += pred.eq(target.view_as(pred)).sum().item()
 total += target.size(0)
 
 # Store predictions and targets for metrics
 all_preds.extend(pred.cpu().numpy().flatten())
 all_targets.extend(target.cpu().numpy())
 all_probs.extend(torch.softmax(output, dim=1).cpu().numpy())
 
 test_acc = 100. * correct / total
 avg_test_loss = test_loss / len(test_loader)
 
 print(f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%')
 
 # Calculate detailed metrics
 metrics = calculate_metrics(np.array(all_targets), np.array(all_preds), class_names)
 metrics['test_accuracy'] = test_acc
 metrics['test_loss'] = avg_test_loss
 
 # Confusion matrix
 cm = confusion_matrix(all_targets, all_preds)
 
 return metrics, cm, all_probs

def plot_confusion_matrix(cm, class_names, save_path):
 """Plot and save confusion matrix."""
 plt.figure(figsize=(10, 8))
 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
 xticklabels=class_names, yticklabels=class_names)
 plt.title('Confusion Matrix')
 plt.xlabel('Predicted Label')
 plt.ylabel('True Label')
 plt.tight_layout()
 plt.savefig(save_path, dpi=300, bbox_inches='tight')
 plt.close()

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
 """Plot training and validation curves."""
 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
 
 # Loss curves
 ax1.plot(train_losses, label='Train Loss')
 ax1.plot(val_losses, label='Validation Loss')
 ax1.set_title('Training and Validation Loss')
 ax1.set_xlabel('Epoch')
 ax1.set_ylabel('Loss')
 ax1.legend()
 ax1.grid(True)
 
 # Accuracy curves
 ax2.plot(train_accs, label='Train Accuracy')
 ax2.plot(val_accs, label='Validation Accuracy')
 ax2.set_title('Training and Validation Accuracy')
 ax2.set_xlabel('Epoch')
 ax2.set_ylabel('Accuracy (%)')
 ax2.legend()
 ax2.grid(True)
 
 plt.tight_layout()
 plt.savefig(save_path, dpi=300, bbox_inches='tight')
 plt.close()

def generate_summary_report(metrics, class_names, save_path, device_info, batch_size, epochs_phase1, epochs_phase2):
 """Generate a markdown summary report."""
 report = f"""# Flower Classification Training Report

## Dataset Information
- Dataset: Real-world flower dataset
- Split: 70% train, 20% validation, 10% test
- Classes: {', '.join(class_names)}

## Model Information
- Architecture: ResNet50
- Pretrained: ImageNet weights
- Training Phases:
 1. Frozen backbone + trainable classifier head
 2. Full network fine-tuning

## Training Configuration
- Device: {device_info}
- Batch Size: {batch_size}
- Phase 1 Epochs: {epochs_phase1}
- Phase 2 Epochs: {epochs_phase2}
- Optimizer: AdamW
- Learning Rate Scheduler: CosineAnnealingLR (Phase 1), OneCycleLR (Phase 2)
- Gradient Clipping: Max norm = 1.0

## Results
- Best Test Accuracy: {metrics['test_accuracy']:.2f}%
- Test Loss: {metrics['test_loss']:.4f}

## Per-Class Metrics
"""
 
 for i, class_name in enumerate(class_names):
 report += f"- **{class_name}**: Precision={metrics['precision'][i]:.3f}, Recall={metrics['recall'][i]:.3f}, F1-Score={metrics['f1-score'][i]:.3f}\n"
 
 report += f"""
## Overall Metrics
- Macro Average Precision: {metrics['macro avg']['precision']:.3f}
- Macro Average Recall: {metrics['macro avg']['recall']:.3f}
- Macro Average F1-Score: {metrics['macro avg']['f1-score']:.3f}
- Weighted Average Precision: {metrics['weighted avg']['precision']:.3f}
- Weighted Average Recall: {metrics['weighted avg']['recall']:.3f}
- Weighted Average F1-Score: {metrics['weighted avg']['f1-score']:.3f}

## Instructions to Reproduce
1. Run the training pipeline: `python pipelines/train_flowers_resnet50.py`
2. The best model checkpoint will be saved at: `outputs/models/flowers_resnet50_best.pt`
3. Evaluation results and plots are saved in: `outputs/flowers_training/`

*This report was automatically generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
 
 with open(save_path, 'w', encoding='utf-8') as f:
 f.write(report)

def main():
 """Main training function."""
 print("Starting ResNet50 flower classification training pipeline...")
 
 # Create output directories if they don't exist
 (project_root / "outputs" / "models").mkdir(parents=True, exist_ok=True)
 (project_root / "outputs" / "flowers_training").mkdir(parents=True, exist_ok=True)
 
 # Detect device
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 device_info = f"{device} ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
 print(f"Using device: {device_info}")
 
 # Set batch size based on device
 batch_size = 32 if torch.cuda.is_available() else 16
 print(f"Using batch size: {batch_size}")
 
 # Create data loaders
 print("Loading datasets...")
 train_loader, val_loader, test_loader = create_data_loaders(batch_size=batch_size)
 print(f"Dataset loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test samples")
 
 # Get class names
 class_names = train_loader.dataset.classes
 num_classes = len(class_names)
 print(f"Classes: {class_names}")
 
 # Create model
 print("Creating ResNet50 model...")
 model = create_model(num_classes=num_classes, pretrained=True)
 model = model.to(device)
 print(f"Model created with {num_classes} classes")
 
 # Phase 1: Train classifier head with frozen backbone
 epochs_phase1 = 20
 train_losses1, val_losses1, train_accs1, val_accs1 = train_phase1(
 model, train_loader, val_loader, device, epochs=epochs_phase1
 )
 
 # Load best model from phase 1
 model.load_state_dict(torch.load(project_root / "outputs" / "models" / "flowers_resnet50_phase1_best.pt"))
 
 # Phase 2: Fine-tune entire network
 epochs_phase2 = 30
 train_losses2, val_losses2, train_accs2, val_accs2 = train_phase2(
 model, train_loader, val_loader, device, epochs=epochs_phase2
 )
 
 # Combine metrics from both phases for plotting
 all_train_losses = train_losses1 + train_losses2
 all_val_losses = val_losses1 + val_losses2
 all_train_accs = train_accs1 + train_accs2
 all_val_accs = val_accs1 + val_accs2
 
 # Plot training curves
 plot_training_curves(
 all_train_losses, all_val_losses, all_train_accs, all_val_accs,
 project_root / "outputs" / "flowers_training" / "training_curves.png"
 )
 print("Training curves saved to outputs/flowers_training/training_curves.png")
 
 # Load best model for final evaluation
 model.load_state_dict(torch.load(project_root / "outputs" / "models" / "flowers_resnet50_best.pt"))
 
 # Evaluate on test set
 metrics, cm, _ = evaluate_model(model, test_loader, device, class_names)
 
 # Save metrics
 with open(project_root / "outputs" / "flowers_final_metrics.json", 'w') as f:
 json.dump(metrics, f, indent=2)
 print("Metrics saved to outputs/flowers_final_metrics.json")
 
 # Plot and save confusion matrix
 plot_confusion_matrix(
 cm, class_names,
 project_root / "outputs" / "flowers_training" / "confusion_matrix.png"
 )
 print("Confusion matrix saved to outputs/flowers_training/confusion_matrix.png")
 
 # Save confusion matrix as JSON
 cm_dict = {
 "confusion_matrix": cm.tolist(),
 "class_names": class_names
 }
 with open(project_root / "outputs" / "confusion_matrix.json", 'w') as f:
 json.dump(cm_dict, f, indent=2)
 
 # Generate summary report
 generate_summary_report(
 metrics, class_names,
 project_root / "outputs" / "flowers_summary.md",
 device_info, batch_size, epochs_phase1, epochs_phase2
 )
 print("Summary report saved to outputs/flowers_summary.md")
 
 print("\nTraining pipeline completed successfully!")
 print(f"Final test accuracy: {metrics['test_accuracy']:.2f}%")
 print("Model saved to: outputs/models/flowers_resnet50_best.pt")
 print("All outputs saved to: outputs/flowers_training/")

if __name__ == "__main__":
 main()