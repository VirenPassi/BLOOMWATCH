"""
Automated Flowers Training + Evaluation Pipeline

- Checks for existing checkpoint: outputs/models/flowers_resnet50_best.pt
- If missing: trains with two-phase fine-tuning on processed splits
- Then evaluates on test set and generates metrics, confusion matrix, and UTF-8 summary

Usage:
  python pipelines/train_flowers_auto.py
"""

import json
import time
from pathlib import Path
import argparse
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "flowers_training"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = MODELS_DIR / "flowers_resnet50_best.pt"
SUMMARY_MD = OUTPUTS_DIR / "flowers_summary.md"
METRICS_JSON = OUTPUTS_DIR / "flowers_final_metrics.json"

PROCESSED_DIR = ROOT / "data" / "processed" / "real_flowers"
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"

CLASSES: List[str] = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 10
DEFAULT_LR = 1e-4
BATCH_GPU = 32
BATCH_CPU = 8


class ResNet50FlowerClassifier(nn.Module):
	def __init__(self, num_classes: int, pretrained: bool = True):
		super().__init__()
		self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
		self.input_adaptation = nn.Conv2d(5, 3, kernel_size=1, bias=False)
		self.backbone.fc = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(self.backbone.fc.in_features, 512),
			nn.ReLU(inplace=True),
			nn.Dropout(0.3),
			nn.Linear(512, num_classes)
		)
	def freeze_backbone(self):
		for p in self.backbone.parameters():
			p.requires_grad = False
		for p in self.backbone.fc.parameters():
			p.requires_grad = True
	def unfreeze_backbone(self):
		for p in self.backbone.parameters():
			p.requires_grad = True
	def forward(self, x):
		x = self.input_adaptation(x)
		return self.backbone(x)


def get_dataloaders(device):
	train_tf = transforms.Compose([
		transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomRotation(30),
		transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
		transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
		transforms.ToTensor(),
		transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
		transforms.RandomErasing(p=0.2)
	])
	val_tf = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
	])

	def to_five_channels(ds):
		class FiveChan(torch.utils.data.Dataset):
			def __init__(self, base):
				self.base = base
				self.classes = base.classes
			def __len__(self):
				return len(self.base)
			def __getitem__(self, idx):
				img, y = self.base[idx]
				ndvi = torch.zeros(1, 224, 224)
				evi = torch.zeros(1, 224, 224)
				x = torch.cat([img, ndvi, evi], dim=0)
				return x, y
		return FiveChan(ds)

	train_base = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
	val_base = datasets.ImageFolder(VAL_DIR, transform=val_tf)
	test_base = datasets.ImageFolder(TEST_DIR, transform=val_tf)

	batch = BATCH_GPU if device.type == 'cuda' else BATCH_CPU
	train_loader = DataLoader(to_five_channels(train_base), batch_size=batch, shuffle=True, num_workers=0)
	val_loader = DataLoader(to_five_channels(val_base), batch_size=batch, shuffle=False, num_workers=0)
	test_loader = DataLoader(to_five_channels(test_base), batch_size=batch, shuffle=False, num_workers=0)
	return train_loader, val_loader, test_loader


def train_two_phase(model, train_loader, val_loader, device):
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LR, weight_decay=1e-4)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
	best_val = 0.0
	best_state = None
	patience = DEFAULT_PATIENCE
	wait = 0

	def run_epoch(train: bool):
		model.train(train)
		loss_sum, correct, total = 0.0, 0, 0
		loader = train_loader if train else val_loader
		for imgs, labels in loader:
			imgs, labels = imgs.to(device), labels.to(device)
			optimizer.zero_grad()
			with torch.set_grad_enabled(train):
				out = model(imgs)
				loss = criterion(out, labels)
				if train:
					loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
					optimizer.step()
			loss_sum += float(loss.item())
			pred = out.argmax(1)
			correct += (pred == labels).sum().item()
			total += labels.size(0)
		return (loss_sum / max(len(loader), 1)), (correct / max(total, 1))

	# Phase 1: train head
	model.freeze_backbone()
	for epoch in range(1, DEFAULT_EPOCHS + 1):
		tr_loss, tr_acc = run_epoch(True)
		val_loss, val_acc = run_epoch(False)
		scheduler.step()
		print(f"[Phase 1] Epoch {epoch}/{DEFAULT_EPOCHS} - train_loss={tr_loss:.4f} val_acc={val_acc:.3f}")
		if val_acc > best_val:
			best_val = val_acc
			best_state = model.state_dict()
			MODELS_DIR.mkdir(parents=True, exist_ok=True)
			torch.save(best_state, CHECKPOINT)
			print("Saved new best checkpoint (phase 1)")
		else:
			wait += 1
			if wait >= patience:
				print("Early stopping (phase 1)")
				break

	# Phase 2: fine-tune all layers
	model.unfreeze_backbone()
	wait = 0
	for epoch in range(1, DEFAULT_EPOCHS + 1):
		tr_loss, tr_acc = run_epoch(True)
		val_loss, val_acc = run_epoch(False)
		scheduler.step()
		print(f"[Phase 2] Epoch {epoch}/{DEFAULT_EPOCHS} - train_loss={tr_loss:.4f} val_acc={val_acc:.3f}")
		if val_acc > best_val:
			best_val = val_acc
			best_state = model.state_dict()
			torch.save(best_state, CHECKPOINT)
			print("Saved new best checkpoint (phase 2)")
		else:
			wait += 1
			if wait >= patience:
				print("Early stopping (phase 2)")
				break

	if best_state is not None:
		model.load_state_dict(best_state)
	return best_val


def evaluate(model, test_loader, device):
	model.eval()
	targets, preds = [], []
	with torch.no_grad():
		for imgs, labels in test_loader:
			imgs, labels = imgs.to(device), labels.to(device)
			out = model(imgs)
			pred = out.argmax(1)
			targets.extend(labels.cpu().numpy().tolist())
			preds.extend(pred.cpu().numpy().tolist())
	acc = (np.array(targets) == np.array(preds)).mean().item()
	cm = confusion_matrix(targets, preds)
	report = classification_report(targets, preds, target_names=CLASSES, output_dict=True, zero_division=0)
	return acc, cm, report


def save_outputs(acc, cm, report, device):
	plt.figure(figsize=(10, 8))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
	plt.title('Confusion Matrix - Flowers Test Set')
	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	plt.tight_layout()
	plt.savefig(PLOTS_DIR / 'confusion_matrix.png', dpi=200)
	plt.close()

	metrics = {
		"accuracy": acc,
		"classification_report": report,
		"classes": CLASSES,
		"checkpoint": str(CHECKPOINT),
		"device": str(device),
		"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
	}
	with open(METRICS_JSON, 'w', encoding='utf-8') as f:
		json.dump(metrics, f, indent=2)

	summary = f"""# Flowers Model - Summary

- Checkpoint: `{CHECKPOINT}`
- Device: {device}
- Test Accuracy: {acc:.3f}

## Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
"""
	for cls in CLASSES:
		m = report.get(cls, {})
		summary += f"| {cls} | {m.get('precision', 0):.3f} | {m.get('recall', 0):.3f} | {m.get('f1-score', 0):.3f} | {int(m.get('support', 0))} |\n"
	with open(SUMMARY_MD, 'w', encoding='utf-8') as f:
		f.write(summary)


def main():
    parser = argparse.ArgumentParser(description="Auto-train flowers model with two-phase fine-tuning")
    parser.add_argument("--force-train", action="store_true", help="Force training even if checkpoint exists")
    args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	batch = BATCH_GPU if device.type == 'cuda' else BATCH_CPU
	print(f"Using device: {device} (batch={batch})")

	# Data
	if not TRAIN_DIR.exists() or not VAL_DIR.exists() or not TEST_DIR.exists():
		print("Processed split directories not found. Run data/split_real_flowers.py first.")
		return
	train_loader, val_loader, test_loader = get_dataloaders(device)

    # Train if needed
    if CHECKPOINT.exists() and not args.force_train:
        print(f"Checkpoint exists: {CHECKPOINT} -> skipping training (use --force-train to retrain)")
        model = ResNet50FlowerClassifier(num_classes=len(CLASSES), pretrained=False).to(device)
        state = torch.load(CHECKPOINT, map_location=device)
        model.load_state_dict(state, strict=False)
    else:
        if CHECKPOINT.exists() and args.force_train:
            print("--force-train specified: retraining and overwriting best checkpoint")
        else:
            print("No checkpoint found -> starting training")
        model = ResNet50FlowerClassifier(num_classes=len(CLASSES), pretrained=True).to(device)
        best_val = train_two_phase(model, train_loader, val_loader, device)
        print(f"Best val acc: {best_val:.3f}")

	# Evaluate
	acc, cm, report = evaluate(model, test_loader, device)
	print(f"Test accuracy: {acc:.3f}")
	save_outputs(acc, cm, report, device)
	print("Done. Outputs written to outputs/ directory.")


if __name__ == '__main__':
	main()
