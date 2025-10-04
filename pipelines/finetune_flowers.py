"""
Upgraded fine-tuning script for BloomWatch on real flower datasets.

Features:
- Real Kaggle Flowers Recognition dataset with synthetic fallback
- Strong data augmentation for training
- ResNet50 backbone with two-phase training
- Early stopping and learning rate scheduling
- Comprehensive evaluation and logging

Usage:
 python pipelines/finetune_flowers.py --data-root data/raw/flowers_recognition/flowers --epochs 30
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
DEFAULT_CKPT = MODELS_DIR / "stage2_transfer_learning_bloomwatch.pt"

FLOWER_CLASSES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# ImageNet normalization stats for pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class FlowersDataset(Dataset):
 """Enhanced dataset with proper train/val/test splits and strong augmentation.
 
 Dataset split: 70% train, 20% validation, 10% test
 """

 def __init__(self, data_root: Path, split: str = "train", seed: int = 42, is_training: bool = True):
 self.data_root = Path(data_root)
 self.split = split
 self.is_training = is_training
 
 # Define transforms based on split and training mode
 if is_training and split == "train":
 # Strong augmentation for training
 self.transform = transforms.Compose([
 transforms.Resize((256, 256)),
 transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
 transforms.RandomHorizontalFlip(p=0.5),
 transforms.RandomRotation(30),
 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
 transforms.ToTensor(),
 transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
 ])
 else:
 # Simple resize + normalize for validation/test
 self.transform = transforms.Compose([
 transforms.Resize((224, 224)),
 transforms.ToTensor(),
 transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
 ])
 
 # Build index of all images
 items = []
 for cls in FLOWER_CLASSES:
 class_dir = self.data_root / cls
 if class_dir.exists():
 for ext in ["*.jpg", "*.png", "*.jpeg"]:
 for p in class_dir.glob(ext):
 items.append((p, cls))
 
 if not items:
 raise ValueError(f"No images found in {self.data_root}")
 
 # Shuffle and split dataset: 70% train, 20% val, 10% test
 rng = np.random.default_rng(seed)
 rng.shuffle(items)
 n = len(items)
 n_train = int(n * 0.7)
 n_val = int(n * 0.2)
 
 if split == "train":
 self.items = items[:n_train]
 elif split == "val":
 self.items = items[n_train:n_train + n_val]
 elif split == "test":
 self.items = items[n_train + n_val:]
 else:
 raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
 
 print(f"Loaded {len(self.items)} samples for {split} split")

 def __len__(self):
 return len(self.items)

 def __getitem__(self, idx):
 path, cls = self.items[idx]
 img = Image.open(path).convert("RGB")
 x3 = self.transform(img) # 3x224x224 normalized
 # Add zero NDVI/EVI channels to match 5-channel input
 ndvi = torch.zeros(1, 224, 224, dtype=torch.float32)
 evi = torch.zeros(1, 224, 224, dtype=torch.float32)
 x = torch.cat([x3, ndvi, evi], dim=0) # 5x224x224
 y = torch.tensor(FLOWER_CLASSES.index(cls), dtype=torch.long)
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
 
 # Freeze backbone initially (will be unfrozen in phase 2)
 self.freeze_backbone()
 
 def freeze_backbone(self):
 """Freeze all backbone parameters except the final classifier."""
 for param in self.backbone.parameters():
 param.requires_grad = False
 # Unfreeze the final classifier
 for param in self.backbone.fc.parameters():
 param.requires_grad = True
 
 def unfreeze_backbone(self):
 """Unfreeze all parameters for fine-tuning."""
 for param in self.backbone.parameters():
 param.requires_grad = True
 
 def forward(self, x):
 x = self.input_adaptation(x)
 return self.backbone(x)

def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int, total_epochs: int):
 """Enhanced training function with progress logging."""
 model.train()
 total, correct, loss_sum = 0, 0, 0.0
 
 for batch_idx, (x, y) in enumerate(loader):
 x, y = x.to(device), y.to(device)
 optimizer.zero_grad()
 out = model(x)
 loss = criterion(out, y)
 loss.backward()
 
 # Gradient clipping for stability
 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 
 optimizer.step()
 loss_sum += float(loss.item())
 _, pred = out.max(1)
 total += y.size(0)
 correct += (pred == y).sum().item()
 
 # Print progress every 10 batches
 if batch_idx % 10 == 0:
 current_lr = optimizer.param_groups[0]['lr']
 print(f" Epoch {epoch}/{total_epochs} - Batch {batch_idx}/{len(loader)} - "
 f"Loss: {loss.item():.4f} - LR: {current_lr:.6f}")
 
 avg_loss = loss_sum / max(len(loader), 1)
 accuracy = correct / max(total, 1)
 return avg_loss, accuracy

def evaluate(model, loader, device):
 """Enhanced evaluation function with detailed metrics."""
 model.eval()
 total, correct = 0, 0
 preds, targs = [], []
 all_probs = []
 
 with torch.no_grad():
 for x, y in loader:
 x, y = x.to(device), y.to(device)
 out = model(x)
 probs = torch.softmax(out, dim=1)
 _, pred = out.max(1)
 total += y.size(0)
 correct += (pred == y).sum().item()
 preds.extend(pred.cpu().numpy())
 targs.extend(y.cpu().numpy())
 all_probs.extend(probs.cpu().numpy())
 
 acc = correct / max(total, 1)
 cm = confusion_matrix(targs, preds)
 report = classification_report(targs, preds, target_names=FLOWER_CLASSES, output_dict=True, zero_division=0)
 return acc, cm, report, all_probs

class EarlyStopping:
 """Early stopping utility to prevent overfitting."""
 
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

def save_confusion(cm: np.ndarray, title: str, path: Path):
 path.parent.mkdir(parents=True, exist_ok=True)
 plt.figure(figsize=(8, 6))
 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=FLOWER_CLASSES, yticklabels=FLOWER_CLASSES)
 plt.title(title)
 plt.xlabel('Predicted')
 plt.ylabel('Actual')
 plt.tight_layout()
 plt.savefig(path, dpi=300)
 plt.close()

def main():
 import argparse
 parser = argparse.ArgumentParser(description="Upgraded BloomWatch fine-tuning on flowers dataset")
 parser.add_argument('--data-root', type=str, default=str(ROOT / 'data' / 'raw' / 'flowers_recognition' / 'flowers'))
 parser.add_argument('--epochs', type=int, default=30, help="Total epochs for training")
 parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
 parser.add_argument('--lr-head', type=float, default=1e-3, help="Learning rate for classifier head")
 parser.add_argument('--lr-backbone', type=float, default=1e-5, help="Learning rate for backbone fine-tuning")
 parser.add_argument('--patience', type=int, default=10, help="Early stopping patience")
 parser.add_argument('--checkpoint', type=str, default=str(DEFAULT_CKPT), help="Base model checkpoint")
 parser.add_argument('--phase1-epochs', type=int, default=15, help="Epochs for phase 1 (head only)")
 args = parser.parse_args()

 # Device setup
 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 print(f"Using device: {device}")
 if device.type == 'cuda':
 print(f"GPU: {torch.cuda.get_device_name(0)}")

 # Data loading with proper splits
 print("Loading datasets...")
 train_ds = FlowersDataset(Path(args.data_root), split='train', is_training=True)
 val_ds = FlowersDataset(Path(args.data_root), split='val', is_training=False)
 test_ds = FlowersDataset(Path(args.data_root), split='test', is_training=False)
 
 train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
 val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
 test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
 
 print(f"Dataset splits -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

 # Model initialization
 print("Initializing ResNet50 model...")
 model = ResNet50FlowerClassifier(num_classes=len(FLOWER_CLASSES), pretrained=True).to(device)
 
 # Try to load base checkpoint if available
 if Path(args.checkpoint).exists():
 try:
 state = torch.load(args.checkpoint, map_location=device)
 # Only load compatible weights
 model_dict = model.state_dict()
 pretrained_dict = {k: v for k, v in state.items() if k in model_dict and model_dict[k].shape == v.shape}
 model_dict.update(pretrained_dict)
 model.load_state_dict(model_dict)
 print(f"Loaded compatible weights from: {args.checkpoint}")
 except Exception as e:
 print(f"Could not load checkpoint {args.checkpoint}: {e}")
 print("Training from ImageNet pretrained weights only")
 else:
 print("No base checkpoint found, using ImageNet pretrained weights")

 # Loss and optimizers
 criterion = nn.CrossEntropyLoss()
 
 # Phase 1: Train classifier head only
 print(f"\n{'='*60}")
 print("PHASE 1: Training classifier head (backbone frozen)")
 print(f"{'='*60}")
 
 optimizer_phase1 = torch.optim.AdamW(
 [p for p in model.parameters() if p.requires_grad], 
 lr=args.lr_head, 
 weight_decay=1e-4
 )
 scheduler_phase1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
 optimizer_phase1, mode='max', factor=0.5, patience=3
 )
 early_stopping = EarlyStopping(patience=args.patience)
 
 best_val_acc = 0.0
 training_history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
 
 start_time = time.time()
 
 for epoch in range(1, args.phase1_epochs + 1):
 print(f"\nEpoch {epoch}/{args.phase1_epochs}")
 
 # Training
 train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_phase1, device, epoch, args.phase1_epochs)
 
 # Validation
 val_acc, val_cm, val_report, _ = evaluate(model, val_loader, device)
 
 # Learning rate scheduling
 scheduler_phase1.step(val_acc)
 current_lr = optimizer_phase1.param_groups[0]['lr']
 
 # Logging
 print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, LR: {current_lr:.6f}")
 
 # Save training history
 training_history['train_loss'].append(train_loss)
 training_history['train_acc'].append(train_acc)
 training_history['val_acc'].append(val_acc)
 training_history['lr'].append(current_lr)
 
 # Save best model
 if val_acc > best_val_acc:
 best_val_acc = val_acc
 out_path = MODELS_DIR / 'stage2_real_finetuned.pt'
 out_path.parent.mkdir(parents=True, exist_ok=True)
 torch.save(model.state_dict(), out_path)
 print(f" New best model saved (Val Acc: {val_acc:.3f})")
 
 # Early stopping check
 if early_stopping(val_acc):
 print(f"Early stopping triggered at epoch {epoch}")
 break
 
 phase1_time = time.time() - start_time
 print(f"\nPhase 1 completed in {phase1_time:.1f} seconds")
 print(f"Best validation accuracy in Phase 1: {best_val_acc:.3f}")
 
 # Phase 2: Fine-tune entire model
 print(f"\n{'='*60}")
 print("PHASE 2: Fine-tuning entire model (backbone unfrozen)")
 print(f"{'='*60}")
 
 # Unfreeze backbone
 model.unfreeze_backbone()
 
 # New optimizer with lower learning rate for backbone
 optimizer_phase2 = torch.optim.AdamW(
 model.parameters(), 
 lr=args.lr_backbone, 
 weight_decay=1e-4
 )
 scheduler_phase2 = torch.optim.lr_scheduler.CosineAnnealingLR(
 optimizer_phase2, T_max=args.epochs - args.phase1_epochs, eta_min=1e-7
 )
 early_stopping_phase2 = EarlyStopping(patience=args.patience // 2)
 
 start_time_phase2 = time.time()
 
 for epoch in range(args.phase1_epochs + 1, args.epochs + 1):
 print(f"\nEpoch {epoch}/{args.epochs}")
 
 # Training
 train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_phase2, device, epoch, args.epochs)
 
 # Validation
 val_acc, val_cm, val_report, _ = evaluate(model, val_loader, device)
 
 # Learning rate scheduling
 scheduler_phase2.step()
 current_lr = optimizer_phase2.param_groups[0]['lr']
 
 # Logging
 print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, LR: {current_lr:.6f}")
 
 # Save training history
 training_history['train_loss'].append(train_loss)
 training_history['train_acc'].append(train_acc)
 training_history['val_acc'].append(val_acc)
 training_history['lr'].append(current_lr)
 
 # Save best model
 if val_acc > best_val_acc:
 best_val_acc = val_acc
 out_path = MODELS_DIR / 'stage2_real_finetuned.pt'
 torch.save(model.state_dict(), out_path)
 print(f" New best model saved (Val Acc: {val_acc:.3f})")
 
 # Early stopping check
 if early_stopping_phase2(val_acc):
 print(f"Early stopping triggered at epoch {epoch}")
 break
 
 phase2_time = time.time() - start_time_phase2
 total_time = time.time() - start_time
 print(f"\nPhase 2 completed in {phase2_time:.1f} seconds")
 print(f"Total training time: {total_time:.1f} seconds")
 print(f"Final best validation accuracy: {best_val_acc:.3f}")

 # Final evaluation on test set
 print(f"\n{'='*60}")
 print("FINAL EVALUATION ON TEST SET")
 print(f"{'='*60}")
 
 # Load best model
 best_model_path = MODELS_DIR / 'stage2_real_finetuned.pt'
 if best_model_path.exists():
 model.load_state_dict(torch.load(best_model_path, map_location=device))
 print("Loaded best model for final evaluation")
 
 test_acc, test_cm, test_report, test_probs = evaluate(model, test_loader, device)
 print(f"Test Accuracy: {test_acc:.3f}")
 
 # Save results
 save_confusion(test_cm, 'Flowers Test Set - Final Model', OUTPUTS_DIR / 'flowers_confusion.png')
 
 # Create comprehensive metrics
 metrics = {
 "dataset": "kaggle_flowers_recognition",
 "model_architecture": "ResNet50",
 "training_phases": {
 "phase1_epochs": args.phase1_epochs,
 "phase2_epochs": args.epochs - args.phase1_epochs,
 "total_epochs": len(training_history['train_loss'])
 },
 "dataset_splits": {
 "train_samples": len(train_ds),
 "val_samples": len(val_ds),
 "test_samples": len(test_ds)
 },
 "hyperparameters": {
 "batch_size": args.batch_size,
 "lr_head": args.lr_head,
 "lr_backbone": args.lr_backbone,
 "patience": args.patience
 },
 "performance": {
 "best_val_accuracy": best_val_acc,
 "final_test_accuracy": test_acc,
 "training_time_seconds": total_time
 },
 "test_report": test_report,
 "training_history": training_history,
 "model_path": str(best_model_path),
 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
 }
 
 with open(OUTPUTS_DIR / 'flowers_metrics.json', 'w') as f:
 json.dump(metrics, f, indent=2)
 
 print(f"\n{'='*60}")
 print("TRAINING COMPLETED SUCCESSFULLY")
 print(f"{'='*60}")
 print(f"Best model saved to: {best_model_path}")
 print(f"Metrics saved to: {OUTPUTS_DIR / 'flowers_metrics.json'}")
 print(f"Confusion matrix saved to: {OUTPUTS_DIR / 'flowers_confusion.png'}")
 print(f"Final test accuracy: {test_acc:.3f}")

if __name__ == "__main__":
 main()

