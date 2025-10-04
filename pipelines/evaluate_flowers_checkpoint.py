"""
Evaluate ResNet50 flowers checkpoint on processed test split and generate outputs.

- Loads: outputs/models/flowers_resnet50_best.pt
- Test data: data/processed/real_flowers/test/<class>/
- Saves:
 - outputs/flowers_final_metrics.json
 - outputs/flowers_training/confusion_matrix.png
 - outputs/flowers_summary.md (UTF-8)

Usage:
 python pipelines/evaluate_flowers_checkpoint.py
"""

import json
import time
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "real_flowers"
TEST_DIR = DATA_DIR / "test"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "flowers_training"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = MODELS_DIR / "flowers_resnet50_best.pt"
METRICS_JSON = OUTPUTS_DIR / "flowers_final_metrics.json"
SUMMARY_MD = OUTPUTS_DIR / "flowers_summary.md"

CLASSES: List[str] = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class ResNet50FlowerClassifier(nn.Module):
	def __init__(self, num_classes: int):
		super().__init__()
		self.backbone = models.resnet50(weights=None)
		self.input_adaptation = nn.Conv2d(5, 3, kernel_size=1, bias=False)
		self.backbone.fc = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(self.backbone.fc.in_features, 512),
			nn.ReLU(inplace=True),
			nn.Dropout(0.3),
			nn.Linear(512, num_classes)
		)
	def forward(self, x):
		x = self.input_adaptation(x)
		return self.backbone(x)

def main():
	if not CHECKPOINT.exists():
		print(f"Checkpoint not found: {CHECKPOINT}")
		return
	if not TEST_DIR.exists():
		print(f"Test directory not found: {TEST_DIR}")
		return

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	batch_size = 32 if device.type == 'cuda' else 8

	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
	])
	test_ds = datasets.ImageFolder(TEST_DIR, transform=transform)
	test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

	model = ResNet50FlowerClassifier(num_classes=len(CLASSES)).to(device)
	state = torch.load(CHECKPOINT, map_location=device)
	model.load_state_dict(state, strict=False)
	model.eval()

	all_targets, all_preds = [], []
	with torch.no_grad():
		for imgs, labels in test_loader:
			imgs = imgs.to(device)
			labels = labels.to(device)
			# add zero NDVI/EVI channels to match 5-channel
			ndvi = torch.zeros(imgs.size(0), 1, 224, 224, device=device)
			evi = torch.zeros(imgs.size(0), 1, 224, 224, device=device)
			x = torch.cat([imgs, ndvi, evi], dim=1)
			logits = model(x)
			preds = logits.argmax(dim=1)
			all_targets.extend(labels.cpu().numpy().tolist())
			all_preds.extend(preds.cpu().numpy().tolist())

	acc = (np.array(all_targets) == np.array(all_preds)).mean().item()
	cm = confusion_matrix(all_targets, all_preds)
	report = classification_report(all_targets, all_preds, target_names=CLASSES, output_dict=True, zero_division=0)

	# Save confusion matrix plot
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
		"batch_size": batch_size,
		"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
	}
	with open(METRICS_JSON, 'w', encoding='utf-8') as f:
		json.dump(metrics, f, indent=2)

	# Markdown summary
	summary = f"""# Flowers Model - Evaluation Summary

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

	summary += f"""

## Confusion Matrix

Saved to: `outputs/flowers_training/confusion_matrix.png`

"""
	with open(SUMMARY_MD, 'w', encoding='utf-8') as f:
		f.write(summary)

	print(f"Evaluation complete. Accuracy={acc:.3f}")
	print(f"Metrics saved to {METRICS_JSON}")
	print(f"Summary saved to {SUMMARY_MD}")
	print(f"Confusion matrix saved to {PLOTS_DIR / 'confusion_matrix.png'}")

if __name__ == '__main__':
	main()
