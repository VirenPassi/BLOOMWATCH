"""
Mini BloomWatch end-to-end pipeline.

This script:
- Supports both tiny synthetic dataset and larger real dataset
- Handles NDVI/EVI arrays and labeled plant images
- Applies data augmentation for improved training
- Builds a combined dataset (RGB + NDVI + EVI as 5-channel tensor)
- Trains a CNN with configurable epochs and batch size
- Runs predictions and outputs BloomWatch-style JSON

It avoids heavyweight deps (GDAL/torchvision) and runs on CPU.
"""

import os
import json
import time
from pathlib import Path
import sys
from typing import List, Tuple, Dict
import warnings

import numpy as np
from PIL import Image
import os as _os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
# Disable TorchDynamo to avoid importing torch.onnx/transformers in this minimal pipeline
_os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
_os.environ.setdefault('PYTORCH_DISABLE_JIT', '1')
_os.environ.setdefault('TORCH_ONNX_CHEAPER_CHECK', '1')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# Paths
# ------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed" / "MINI"
IMG_DIR = DATA_DIR / "mini_images"
ANNOTATIONS = DATA_DIR / "mini_annotations.csv"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"

# Expanded dataset paths
EXPANDED_DATASET_DIR = DATA_DIR / "expanded_dataset"
EXPANDED_NDVI_DIR = EXPANDED_DATASET_DIR / "ndvi"
EXPANDED_PLANT_DIR = EXPANDED_DATASET_DIR / "plant_images"
EXPANDED_METADATA = EXPANDED_DATASET_DIR / "metadata.csv"

def ensure_dirs():
 for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, IMG_DIR, OUTPUTS_DIR, MODELS_DIR, 
 EXPANDED_DATASET_DIR, EXPANDED_NDVI_DIR, EXPANDED_PLANT_DIR]:
 d.mkdir(parents=True, exist_ok=True)

# Ensure project root is importable for 'app' package
if str(ROOT) not in sys.path:
 sys.path.insert(0, str(ROOT))

# ------------------------------
# Step 1: Create tiny labeled image set + Data Augmentation
# ------------------------------
CLASSES = ["bud", "early_bloom", "full_bloom", "late_bloom", "dormant"]

class EnhancedDataAugmentation:
 """Enhanced data augmentation for plant images with more transformations."""
 
 @staticmethod
 def random_flip_horizontal(image: Image.Image, p: float = 0.5) -> Image.Image:
 if np.random.random() < p:
 return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
 return image
 
 @staticmethod
 def random_flip_vertical(image: Image.Image, p: float = 0.3) -> Image.Image:
 if np.random.random() < p:
 return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
 return image
 
 @staticmethod
 def random_rotation(image: Image.Image, max_angle: float = 20.0, p: float = 0.6) -> Image.Image:
 if np.random.random() < p:
 angle = np.random.uniform(-max_angle, max_angle)
 return image.rotate(angle, fillcolor=(128, 128, 128))
 return image
 
 @staticmethod
 def random_brightness_contrast(image: Image.Image, brightness_range: tuple = (0.7, 1.3), 
 contrast_range: tuple = (0.8, 1.2), p: float = 0.6) -> Image.Image:
 if np.random.random() < p:
 # Brightness adjustment
 brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
 arr = np.array(image, dtype=np.float32)
 arr = np.clip(arr * brightness_factor, 0, 255)
 
 # Contrast adjustment
 contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
 mean = np.mean(arr)
 arr = np.clip((arr - mean) * contrast_factor + mean, 0, 255)
 
 return Image.fromarray(arr.astype(np.uint8))
 return image
 
 @staticmethod
 def random_saturation(image: Image.Image, saturation_range: tuple = (0.7, 1.3), p: float = 0.4) -> Image.Image:
 if np.random.random() < p:
 # Convert to HSV for saturation adjustment
 arr = np.array(image, dtype=np.float32)
 hsv = arr.copy()
 
 # Adjust saturation channel
 saturation_factor = np.random.uniform(saturation_range[0], saturation_range[1])
 hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
 
 # Convert back to RGB (simplified)
 return Image.fromarray(hsv.astype(np.uint8))
 return image
 
 @staticmethod
 def gaussian_blur(image: Image.Image, blur_range: tuple = (0.5, 2.0), p: float = 0.3) -> Image.Image:
 if np.random.random() < p:
 # Simple blur using resize down and up
 blur_factor = np.random.uniform(blur_range[0], blur_range[1])
 w, h = image.size
 new_w, new_h = int(w / blur_factor), int(h / blur_factor)
 
 # Resize down and up for blur effect
 blurred = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
 blurred = blurred.resize((w, h), Image.Resampling.LANCZOS)
 return blurred
 return image
 
 @staticmethod
 def random_resize_crop(image: Image.Image, target_size: tuple = (224, 224), 
 scale_range: tuple = (0.7, 1.0), p: float = 0.7) -> Image.Image:
 if np.random.random() < p:
 w, h = image.size
 scale = np.random.uniform(scale_range[0], scale_range[1])
 new_w, new_h = int(w * scale), int(h * scale)
 
 # Resize
 image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
 
 # Random crop
 if new_w > target_size[1] and new_h > target_size[0]:
 left = np.random.randint(0, new_w - target_size[1])
 top = np.random.randint(0, new_h - target_size[0])
 image = image.crop((left, top, left + target_size[1], top + target_size[0]))
 else:
 image = image.resize(target_size, Image.Resampling.LANCZOS)
 else:
 image = image.resize(target_size, Image.Resampling.LANCZOS)
 return image
 
 @staticmethod
 def elastic_transform(image: Image.Image, alpha: float = 100, sigma: float = 10, p: float = 0.3) -> Image.Image:
 """Simple elastic transformation for plant images."""
 if np.random.random() < p:
 arr = np.array(image)
 h, w = arr.shape[:2]
 
 # Generate random displacement fields
 dx = np.random.randn(h, w) * alpha
 dy = np.random.randn(h, w) * alpha
 
 # Apply Gaussian filter to displacement
 from scipy.ndimage import gaussian_filter
 dx = gaussian_filter(dx, sigma)
 dy = gaussian_filter(dy, sigma)
 
 # Create coordinate grids
 x, y = np.meshgrid(np.arange(w), np.arange(h))
 x_new = np.clip(x + dx, 0, w-1).astype(np.float32)
 y_new = np.clip(y + dy, 0, h-1).astype(np.float32)
 
 # Apply transformation
 from scipy.ndimage import map_coordinates
 transformed = np.zeros_like(arr)
 for i in range(arr.shape[2]):
 transformed[:, :, i] = map_coordinates(arr[:, :, i], [y_new, x_new], order=1, mode='reflect')
 
 return Image.fromarray(transformed.astype(np.uint8))
 return image
 
 @classmethod
 def apply_enhanced_augmentation(cls, image: Image.Image, is_training: bool = True, 
 augmentation_strength: str = 'medium') -> Image.Image:
 """Apply enhanced augmentations with configurable strength."""
 if not is_training:
 return image.resize((224, 224), Image.Resampling.LANCZOS)
 
 # Configure augmentation strength
 if augmentation_strength == 'light':
 probs = {'flip_h': 0.3, 'flip_v': 0.1, 'rotation': 0.4, 'brightness': 0.4, 
 'saturation': 0.2, 'blur': 0.1, 'resize_crop': 0.5, 'elastic': 0.1}
 elif augmentation_strength == 'medium':
 probs = {'flip_h': 0.5, 'flip_v': 0.2, 'rotation': 0.6, 'brightness': 0.6, 
 'saturation': 0.3, 'blur': 0.2, 'resize_crop': 0.7, 'elastic': 0.2}
 else: # strong
 probs = {'flip_h': 0.6, 'flip_v': 0.3, 'rotation': 0.7, 'brightness': 0.7, 
 'saturation': 0.4, 'blur': 0.3, 'resize_crop': 0.8, 'elastic': 0.3}
 
 # Apply augmentations
 if np.random.random() < probs['flip_h']:
 image = cls.random_flip_horizontal(image)
 if np.random.random() < probs['flip_v']:
 image = cls.random_flip_vertical(image)
 if np.random.random() < probs['rotation']:
 image = cls.random_rotation(image)
 if np.random.random() < probs['brightness']:
 image = cls.random_brightness_contrast(image)
 if np.random.random() < probs['saturation']:
 image = cls.random_saturation(image)
 if np.random.random() < probs['blur']:
 image = cls.gaussian_blur(image)
 if np.random.random() < probs['resize_crop']:
 image = cls.random_resize_crop(image)
 if np.random.random() < probs['elastic']:
 image = cls.elastic_transform(image)
 
 return image

def synthesize_image(width: int = 224, height: int = 224, seed: int = 0) -> Image.Image:
 rng = np.random.default_rng(seed)
 # Create a simple plant-like image: greenish blob on background
 img = np.zeros((height, width, 3), dtype=np.uint8)
 img[..., 1] = rng.integers(60, 180, size=(height, width)) # green channel
 # Add a circular brighter region to mimic a plant/flower area
 cy, cx = rng.integers(height // 3, 2 * height // 3), rng.integers(width // 3, 2 * width // 3)
 rr, cc = np.ogrid[:height, :width]
 mask = (rr - cy) ** 2 + (cc - cx) ** 2 <= (min(height, width) // 6) ** 2
 img[mask, 1] = np.clip(img[mask, 1] + rng.integers(40, 60), 0, 255) # boost green in circle
 # Add some red/blue noise
 img[..., 0] = rng.integers(0, 40, size=(height, width))
 img[..., 2] = rng.integers(0, 40, size=(height, width))
 return Image.fromarray(img)

def prepare_mini_images(num_images: int = 10) -> None:
 if any(IMG_DIR.iterdir()):
 return
 IMG_DIR.mkdir(parents=True, exist_ok=True)
 entries: List[Dict[str, str]] = []
 rng = np.random.default_rng(42)
 for i in range(num_images):
 cls = CLASSES[i % len(CLASSES)]
 img = synthesize_image(seed=i)
 cls_dir = IMG_DIR / cls
 cls_dir.mkdir(exist_ok=True)
 fname = f"img_{i:03d}.png"
 (cls_dir / fname).parent.mkdir(parents=True, exist_ok=True)
 img.save(cls_dir / fname)
 entries.append({
 "image_path": str(Path(cls).joinpath(fname)).replace("\\", "/"),
 "bloom_stage": cls,
 "stage": "train" if i < int(0.8 * num_images) else ("val" if i < int(0.9 * num_images) else "test"),
 "plant_id": f"plant_{i%2}",
 "timestamp": f"2024-01-{(i%10)+1:02d}"
 })
 # Write annotations
 import csv
 with open(ANNOTATIONS, "w", newline="") as f:
 writer = csv.DictWriter(f, fieldnames=["image_path", "bloom_stage", "stage", "plant_id", "timestamp"])
 writer.writeheader()
 writer.writerows(entries)

# ------------------------------
# Step 2: Synthesize small NDVI/EVI timeseries
# ------------------------------
def synthesize_ndvi_evi(height: int = 224, width: int = 224, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
 rng = np.random.default_rng(seed)
 # NDVI/EVI range ~[-1, 1], we clamp to [-0.2, 0.9] to look realistic
 ndvi = rng.normal(loc=0.4, scale=0.15, size=(height, width)).astype(np.float32)
 evi = rng.normal(loc=0.3, scale=0.2, size=(height, width)).astype(np.float32)
 ndvi = np.clip(ndvi, -0.2, 0.9)
 evi = np.clip(evi, -0.2, 0.9)
 return ndvi, evi

def prepare_mini_modis(num_times: int = 3, locations: List[str] = None) -> List[Path]:
 if locations is None:
 locations = ["tile_h31v08", "tile_h32v08"]
 saved: List[Path] = []
 for loc in locations:
 for t in range(num_times):
 ndvi, evi = synthesize_ndvi_evi(seed=1000 + t)
 base = PROCESSED_DIR / f"{loc}_t{t}"
 PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
 np.save(base.with_suffix(".ndvi.npy"), ndvi)
 np.save(base.with_suffix(".evi.npy"), evi)
 saved.append(base)
 return saved

# ------------------------------
# Step 3: Combined dataset (RGB + NDVI + EVI) with augmentation support
# ------------------------------
class MODISPlantBloomDataset(Dataset):
 BLOOM_STAGES = {"bud": 0, "early_bloom": 1, "full_bloom": 2, "late_bloom": 3, "dormant": 4}

 def __init__(self, image_root: Path, annotations_csv: Path, modis_dir: Path, stage: str = "train", 
 use_augmentation: bool = True, use_expanded_dataset: bool = False, 
 augmentation_strength: str = 'medium', balance_classes: bool = True):
 import pandas as pd
 self.image_root = Path(image_root)
 self.modis_dir = Path(modis_dir)
 self.stage = stage
 self.use_augmentation = use_augmentation and (stage == "train")
 self.use_expanded_dataset = use_expanded_dataset
 self.augmentation_strength = augmentation_strength
 self.balance_classes = balance_classes
 
 # Load annotations
 if use_expanded_dataset and EXPANDED_METADATA.exists():
 df = pd.read_csv(EXPANDED_METADATA)
 self.image_root = EXPANDED_PLANT_DIR
 self.modis_dir = EXPANDED_NDVI_DIR
 else:
 df = pd.read_csv(annotations_csv)
 
 if stage in ["train", "val", "test"]:
 df = df[df["stage"] == stage].reset_index(drop=True)
 
 # Class balancing for training set
 if self.balance_classes and stage == "train":
 df = self._balance_classes(df)
 
 self.df = df
 
 # Find NDVI/EVI files
 if use_expanded_dataset:
 # Look for .npy files in expanded dataset
 self.modis_bases = sorted({p.with_suffix("") for p in self.modis_dir.glob("*.npy")})
 else:
 # Simple mapping: alternate NDVI/EVI files across samples
 self.modis_bases = sorted({p.with_suffix("") for p in self.modis_dir.glob("*.ndvi.npy")})
 
 if not self.modis_bases:
 raise RuntimeError("No MODIS NDVI/EVI files found. Run synthesis step first.")
 
 def _balance_classes(self, df):
 """Balance classes by oversampling minority classes."""
 import pandas as pd
 class_counts = df['bloom_stage'].value_counts()
 max_count = class_counts.max()
 
 balanced_dfs = []
 for class_name in self.BLOOM_STAGES.keys():
 class_df = df[df['bloom_stage'] == class_name]
 current_count = len(class_df)
 
 if current_count < max_count:
 # Oversample minority class
 oversample_count = max_count - current_count
 oversampled = class_df.sample(n=oversample_count, replace=True, random_state=42)
 balanced_dfs.append(pd.concat([class_df, oversampled]))
 else:
 balanced_dfs.append(class_df)
 
 balanced_df = pd.concat(balanced_dfs, ignore_index=True)
 return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle
 
 def get_class_weights(self):
 """Calculate class weights for weighted loss."""
 class_counts = self.df['bloom_stage'].value_counts()
 total_samples = len(self.df)
 
 weights = []
 for class_name in self.BLOOM_STAGES.keys():
 count = class_counts.get(class_name, 1)
 weight = total_samples / (len(self.BLOOM_STAGES) * count)
 weights.append(weight)
 
 return torch.FloatTensor(weights)

 def __len__(self) -> int:
 return len(self.df)

 def _load_image_tensor(self, path: Path) -> torch.Tensor:
 # Load and apply enhanced augmentation if training
 img = Image.open(path).convert("RGB")
 if self.use_augmentation:
 img = EnhancedDataAugmentation.apply_enhanced_augmentation(
 img, is_training=True, augmentation_strength=self.augmentation_strength
 )
 else:
 img = img.resize((224, 224), Image.Resampling.LANCZOS)
 
 # Convert to tensor and normalize
 arr = np.array(img, dtype=np.float32) / 255.0
 t = torch.from_numpy(arr).permute(2, 0, 1)
 mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
 std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
 t = (t - mean) / std
 return t

 def _load_modis_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
 if self.use_expanded_dataset:
 # For expanded dataset, load separate NDVI and EVI files
 ndvi_files = sorted(self.modis_dir.glob("ndvi_*.npy"))
 evi_files = sorted(self.modis_dir.glob("evi_*.npy"))
 
 if not ndvi_files or not evi_files:
 raise RuntimeError("No NDVI/EVI files found in expanded dataset")
 
 # Use modulo to cycle through available files
 ndvi_idx = idx % len(ndvi_files)
 evi_idx = idx % len(evi_files)
 
 ndvi = np.load(ndvi_files[ndvi_idx])
 evi = np.load(evi_files[evi_idx])
 else:
 # For mini dataset, load paired files
 base = self.modis_bases[idx % len(self.modis_bases)]
 ndvi = np.load(base.with_suffix(".ndvi.npy"))
 evi = np.load(base.with_suffix(".evi.npy"))
 
 # Normalize NDVI/EVI to [0,1]
 ndvi01 = ((ndvi + 1.0) / 2.0).astype(np.float32)
 evi01 = ((evi + 1.0) / 2.0).astype(np.float32)
 # Ensure 224x224
 ndvi01 = _resize_array(ndvi01, (224, 224))
 evi01 = _resize_array(evi01, (224, 224))
 return torch.from_numpy(ndvi01), torch.from_numpy(evi01)

 def __getitem__(self, idx: int):
 row = self.df.iloc[idx]
 img_path = self.image_root / row["image_path"]
 x_rgb = self._load_image_tensor(img_path) # (3, 224, 224)
 ndvi, evi = self._load_modis_pair(idx) # (224,224) each
 # Stack into 5-channel tensor
 x = torch.cat([x_rgb, ndvi.unsqueeze(0), evi.unsqueeze(0)], dim=0) # (5,224,224)
 y = self.BLOOM_STAGES[row["bloom_stage"]]
 meta = {"image_path": str(row["image_path"]), "plant_id": row.get("plant_id", "unknown"), "timestamp": row.get("timestamp", "unknown")}
 return x, y, meta

def _resize_array(arr: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
 if arr.shape == target_hw:
 return arr
 # PIL resize for simplicity
 im = Image.fromarray((arr * 255).astype(np.uint8))
 im = im.resize((target_hw[1], target_hw[0]), Image.Resampling.BILINEAR)
 out = np.array(im).astype(np.float32) / 255.0
 return out

# ------------------------------
# Step 4: Enhanced Models with Transfer Learning
# ------------------------------
class TransferLearningCNN(nn.Module):
 """Transfer learning model using pretrained backbone for 5-channel input."""
 
 def __init__(self, num_classes: int = len(CLASSES), backbone: str = 'mobilenet', 
 pretrained: bool = True, freeze_backbone: bool = False):
 super().__init__()
 self.num_classes = num_classes
 self.backbone_name = backbone
 
 # Create 5-channel input adapter
 self.input_adapter = nn.Conv2d(5, 3, kernel_size=1, bias=False)
 
 # Load pretrained backbone
 self.backbone = self._create_backbone(backbone, pretrained)
 
 # Get feature dimension
 if backbone == 'mobilenet':
 feature_dim = 1280 # MobileNetV2
 elif backbone == 'resnet18':
 feature_dim = 512 # ResNet18
 else:
 feature_dim = 512 # Default
 
 # Enhanced classifier with regularization
 self.classifier = nn.Sequential(
 nn.Dropout(0.3),
 nn.Linear(feature_dim, 256),
 nn.BatchNorm1d(256),
 nn.ReLU(inplace=True),
 nn.Dropout(0.2),
 nn.Linear(256, 128),
 nn.BatchNorm1d(128),
 nn.ReLU(inplace=True),
 nn.Dropout(0.1),
 nn.Linear(128, num_classes)
 )
 
 # Freeze backbone if requested
 if freeze_backbone:
 self._freeze_backbone()
 
 def _create_backbone(self, backbone: str, pretrained: bool):
 """Create pretrained backbone."""
 if backbone == 'mobilenet':
 # Use MobileNetV2 for efficiency
 import torchvision.models as models
 model = models.mobilenet_v2(pretrained=pretrained)
 # Remove classifier
 return nn.Sequential(*list(model.children())[:-1])
 
 elif backbone == 'resnet18':
 import torchvision.models as models
 model = models.resnet18(pretrained=pretrained)
 # Remove final layers
 return nn.Sequential(*list(model.children())[:-2])
 
 else:
 raise ValueError(f"Unsupported backbone: {backbone}")
 
 def _freeze_backbone(self):
 """Freeze backbone parameters."""
 for param in self.backbone.parameters():
 param.requires_grad = False
 
 def unfreeze_backbone(self):
 """Unfreeze backbone for fine-tuning."""
 for param in self.backbone.parameters():
 param.requires_grad = True
 
 def forward(self, x: torch.Tensor) -> torch.Tensor:
 # Adapt 5-channel input to 3-channel
 x = self.input_adapter(x)
 
 # Extract features
 features = self.backbone(x)
 
 # Global average pooling
 if self.backbone_name == 'mobilenet':
 features = torch.mean(features, dim=[2, 3]) # MobileNet already has adaptive pooling
 else:
 features = torch.mean(features, dim=[2, 3]) # ResNet
 
 # Classification
 return self.classifier(features)

class EnhancedFiveChannelCNN(nn.Module):
 """Enhanced CNN with better architecture for 5-channel input."""
 
 def __init__(self, num_classes: int = len(CLASSES), dropout: float = 0.3):
 super().__init__()
 
 # Enhanced feature extractor
 self.features = nn.Sequential(
 # First block
 nn.Conv2d(5, 64, 3, padding=1),
 nn.BatchNorm2d(64),
 nn.ReLU(inplace=True),
 nn.Conv2d(64, 64, 3, padding=1),
 nn.BatchNorm2d(64),
 nn.ReLU(inplace=True),
 nn.MaxPool2d(2),
 nn.Dropout2d(0.1),
 
 # Second block
 nn.Conv2d(64, 128, 3, padding=1),
 nn.BatchNorm2d(128),
 nn.ReLU(inplace=True),
 nn.Conv2d(128, 128, 3, padding=1),
 nn.BatchNorm2d(128),
 nn.ReLU(inplace=True),
 nn.MaxPool2d(2),
 nn.Dropout2d(0.2),
 
 # Third block
 nn.Conv2d(128, 256, 3, padding=1),
 nn.BatchNorm2d(256),
 nn.ReLU(inplace=True),
 nn.Conv2d(256, 256, 3, padding=1),
 nn.BatchNorm2d(256),
 nn.ReLU(inplace=True),
 nn.MaxPool2d(2),
 nn.Dropout2d(0.2),
 )
 
 # Global pooling
 self.pool = nn.AdaptiveAvgPool2d((1, 1))
 
 # Enhanced classifier
 self.classifier = nn.Sequential(
 nn.Dropout(dropout),
 nn.Linear(256, 128),
 nn.BatchNorm1d(128),
 nn.ReLU(inplace=True),
 nn.Dropout(dropout * 0.5),
 nn.Linear(128, 64),
 nn.BatchNorm1d(64),
 nn.ReLU(inplace=True),
 nn.Dropout(dropout * 0.25),
 nn.Linear(64, num_classes)
 )
 
 def forward(self, x: torch.Tensor) -> torch.Tensor:
 x = self.features(x)
 x = self.pool(x)
 x = x.view(x.size(0), -1)
 return self.classifier(x)

# Legacy model for compatibility
class MiniFiveChannelCNN(EnhancedFiveChannelCNN):
 """Legacy model - now uses enhanced architecture."""
 def __init__(self, num_classes: int = len(CLASSES)):
 super().__init__(num_classes, dropout=0.2)

# ------------------------------
# Step 5: Enhanced training loop with CPU optimization
# ------------------------------
def train_enhanced(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
 device: torch.device, epochs: int = 15, use_expanded_dataset: bool = False,
 model_type: str = 'enhanced', use_class_weights: bool = True) -> Dict:
 model.to(device)
 
 # Use weighted loss if class weights available
 if use_class_weights and hasattr(train_loader.dataset, 'get_class_weights'):
 class_weights = train_loader.dataset.get_class_weights().to(device)
 criterion = nn.CrossEntropyLoss(weight=class_weights)
 print(f"Using weighted loss with weights: {class_weights.cpu().numpy()}")
 else:
 criterion = nn.CrossEntropyLoss()
 
 # Optimized hyperparameters based on model type
 if model_type == 'transfer_learning':
 # Transfer learning: lower LR, more epochs
 optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
 scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
 batch_size_factor = 1.0
 elif model_type == 'enhanced':
 # Enhanced CNN: moderate LR
 optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
 scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
 batch_size_factor = 1.0
 else:
 # Legacy: original settings
 optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
 scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
 batch_size_factor = 1.0
 
 best_val_acc = 0.0
 patience = 7 if use_expanded_dataset else 5
 patience_counter = 0
 
 # Track learning curves
 train_losses = []
 train_accs = []
 val_losses = []
 val_accs = []
 epoch_numbers = []
 
 print(f"Training {model_type} model for {epochs} epochs on {device}")
 print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
 print(f"Optimizer: {type(optimizer).__name__}, Scheduler: {type(scheduler).__name__}")
 
 for epoch in range(epochs):
 model.train()
 total, correct, loss_sum = 0, 0, 0.0
 
 for batch_idx, (xb, yb, _) in enumerate(train_loader):
 xb = xb.to(device)
 yb = yb.to(device)
 
 optimizer.zero_grad()
 logits = model(xb)
 loss = criterion(logits, yb)
 
 # Gradient clipping for stability
 loss.backward()
 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 optimizer.step()
 
 loss_sum += loss.item() * xb.size(0)
 preds = logits.argmax(dim=1)
 correct += (preds == yb).sum().item()
 total += xb.size(0)
 
 # Progress indicator for larger datasets
 if use_expanded_dataset and batch_idx % 15 == 0:
 print(f" Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
 
 train_acc = correct / max(1, total)
 val_acc = evaluate(model, val_loader, device)
 
 # Track metrics for learning curves
 avg_train_loss = loss_sum / max(1, total)
 train_losses.append(avg_train_loss)
 train_accs.append(train_acc)
 val_losses.append(0.0) # We'll compute this properly
 val_accs.append(val_acc)
 epoch_numbers.append(epoch + 1)
 
 # Learning rate scheduling
 if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
 scheduler.step(val_acc)
 else:
 scheduler.step()
 
 current_lr = optimizer.param_groups[0]['lr']
 
 # Early stopping with improvement tracking
 if val_acc > best_val_acc:
 best_val_acc = val_acc
 patience_counter = 0
 # Save best model
 torch.save(model.state_dict(), MODELS_DIR / "best_model.pt")
 else:
 patience_counter += 1
 
 print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.3f} - Val Acc: {val_acc:.3f} - Best: {best_val_acc:.3f} - LR: {current_lr:.2e}")
 
 if patience_counter >= patience and epoch > 5:
 print(f"Early stopping at epoch {epoch+1}")
 break
 
 # Load best model
 if (MODELS_DIR / "best_model.pt").exists():
 model.load_state_dict(torch.load(MODELS_DIR / "best_model.pt"))
 print("Loaded best model for final evaluation")
 
 # Return training history
 return {
 "train_losses": train_losses,
 "train_accs": train_accs,
 "val_losses": val_losses,
 "val_accs": val_accs,
 "epochs": epoch_numbers,
 "best_val_acc": best_val_acc,
 "final_train_acc": train_accs[-1] if train_accs else 0.0,
 "final_val_acc": val_accs[-1] if val_accs else 0.0
 }

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
 model.eval()
 total, correct = 0, 0
 with torch.no_grad():
 for xb, yb, _ in loader:
 xb = xb.to(device)
 yb = yb.to(device)
 logits = model(xb)
 preds = logits.argmax(dim=1)
 correct += (preds == yb).sum().item()
 total += xb.size(0)
 return correct / max(1, total)

# ------------------------------
# Step 6: Sample prediction -> BloomWatch JSON format
# ------------------------------
def prediction_to_bloomwatch_json(logits: torch.Tensor) -> Dict[str, object]:
 probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()
 pred_idx = int(np.argmax(probs))
 return {
 "predicted_class": pred_idx,
 "confidence": float(max(probs)),
 "class_confidences": {name: float(probs[i]) for i, name in enumerate(CLASSES)},
 "probabilities": probs,
 }

# ------------------------------
# Quality Assurance Functions
# ------------------------------
def check_dataset_leakage(train_ds, val_ds, test_ds=None):
 """Check for dataset leakage by examining plant_id overlap between splits."""
 import pandas as pd
 
 # Get plant_ids from each split
 train_ids = set(train_ds.df['plant_id'].unique())
 val_ids = set(val_ds.df['plant_id'].unique())
 test_ids = set(test_ds.df['plant_id'].unique()) if test_ds else set()
 
 # Check for overlaps
 train_val_overlap = train_ids.intersection(val_ids)
 train_test_overlap = train_ids.intersection(test_ids) if test_ds else set()
 val_test_overlap = val_ids.intersection(test_ids) if test_ds else set()
 
 leakage_detected = len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0
 
 results = {
 "leakage_detected": leakage_detected,
 "train_plant_ids": len(train_ids),
 "val_plant_ids": len(val_ids),
 "test_plant_ids": len(test_ids) if test_ds else 0,
 "train_val_overlap": list(train_val_overlap),
 "train_test_overlap": list(train_test_overlap),
 "val_test_overlap": list(val_test_overlap),
 "overlap_counts": {
 "train_val": len(train_val_overlap),
 "train_test": len(train_test_overlap),
 "val_test": len(val_test_overlap)
 }
 }
 
 # Save results
 with open(OUTPUTS_DIR / "dataset_check.json", "w") as f:
 json.dump(results, f, indent=2)
 
 # Print warnings if leakage detected
 if leakage_detected:
 print(" DATASET LEAKAGE DETECTED!")
 if train_val_overlap:
 print(f" Train-Val overlap: {len(train_val_overlap)} plant_ids")
 if train_test_overlap:
 print(f" Train-Test overlap: {len(train_test_overlap)} plant_ids")
 if val_test_overlap:
 print(f" Val-Test overlap: {len(val_test_overlap)} plant_ids")
 else:
 print(" No dataset leakage detected")
 
 return results

def plot_learning_curves(train_losses, train_accs, val_losses, val_accs, epochs):
 """Plot and save learning curves."""
 plt.style.use('default')
 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
 
 # Loss curves
 ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
 ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
 ax1.set_xlabel('Epoch')
 ax1.set_ylabel('Loss')
 ax1.set_title('Learning Curves - Loss')
 ax1.legend()
 ax1.grid(True, alpha=0.3)
 
 # Accuracy curves
 ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
 ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
 ax2.set_xlabel('Epoch')
 ax2.set_ylabel('Accuracy')
 ax2.set_title('Learning Curves - Accuracy')
 ax2.legend()
 ax2.grid(True, alpha=0.3)
 ax2.set_ylim(0, 1)
 
 plt.tight_layout()
 plt.savefig(OUTPUTS_DIR / "learning_curves.png", dpi=150, bbox_inches='tight')
 plt.close()
 print(f" Learning curves saved to: {OUTPUTS_DIR / 'learning_curves.png'}")

def compute_confusion_matrix(model, val_loader, device, class_names):
 """Compute and save confusion matrix."""
 model.eval()
 all_preds = []
 all_labels = []
 
 with torch.no_grad():
 for xb, yb, _ in val_loader:
 xb = xb.to(device)
 yb = yb.to(device)
 logits = model(xb)
 preds = logits.argmax(dim=1)
 
 all_preds.extend(preds.cpu().numpy())
 all_labels.extend(yb.cpu().numpy())
 
 # Compute confusion matrix
 cm = confusion_matrix(all_labels, all_preds)
 
 # Save as JSON
 cm_dict = {
 "confusion_matrix": cm.tolist(),
 "class_names": class_names,
 "accuracy": np.trace(cm) / np.sum(cm),
 "per_class_accuracy": (cm.diagonal() / cm.sum(axis=1)).tolist()
 }
 
 with open(OUTPUTS_DIR / "confusion_matrix.json", "w") as f:
 json.dump(cm_dict, f, indent=2)
 
 # Create and save visualization
 plt.figure(figsize=(8, 6))
 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
 xticklabels=class_names, yticklabels=class_names)
 plt.title('Confusion Matrix')
 plt.xlabel('Predicted')
 plt.ylabel('Actual')
 plt.tight_layout()
 plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches='tight')
 plt.close()
 
 print(f" Confusion matrix saved to: {OUTPUTS_DIR / 'confusion_matrix.png'}")
 print(f" Confusion matrix JSON saved to: {OUTPUTS_DIR / 'confusion_matrix.json'}")
 
 return cm_dict

def check_suspicious_accuracy(train_acc, val_acc, threshold_diff=0.1):
 """Check if validation accuracy is suspiciously high compared to training."""
 diff = val_acc - train_acc
 suspicious = diff > threshold_diff
 
 if suspicious:
 print(f" SUSPICIOUS ACCURACY: Val ({val_acc:.3f}) >> Train ({train_acc:.3f})")
 print(f" Difference: {diff:.3f} (threshold: {threshold_diff})")
 return True
 return False

def resplit_dataset_by_plant_id(metadata_path, output_path):
 """Re-split dataset by plant_id to avoid leakage."""
 import pandas as pd
 
 df = pd.read_csv(metadata_path)
 
 # Get unique plant_ids
 plant_ids = df['plant_id'].unique()
 np.random.shuffle(plant_ids)
 
 # Split plant_ids (not individual samples)
 n_plants = len(plant_ids)
 train_plants = plant_ids[:int(0.7 * n_plants)]
 val_plants = plant_ids[int(0.7 * n_plants):int(0.85 * n_plants)]
 test_plants = plant_ids[int(0.85 * n_plants):]
 
 # Assign splits based on plant_id
 def assign_split(plant_id):
 if plant_id in train_plants:
 return 'train'
 elif plant_id in val_plants:
 return 'val'
 else:
 return 'test'
 
 df['stage'] = df['plant_id'].apply(assign_split)
 
 # Save new split
 df.to_csv(output_path, index=False)
 
 print(f" Dataset re-split by plant_id:")
 print(f" Train plants: {len(train_plants)}")
 print(f" Val plants: {len(val_plants)}")
 print(f" Test plants: {len(test_plants)}")
 print(f" Saved to: {output_path}")
 
 return df

# ------------------------------
# Orchestration with dataset detection
# ------------------------------
def main():
 t0 = time.time()
 ensure_dirs()
 
 # Check if expanded dataset exists
 use_expanded = EXPANDED_METADATA.exists() and EXPANDED_PLANT_DIR.exists() and EXPANDED_NDVI_DIR.exists()
 
 # Model selection based on dataset size
 if use_expanded:
 print("Using expanded dataset with enhanced training...")
 model_type = 'transfer_learning' # Use transfer learning for larger dataset
 augmentation_strength = 'medium'
 batch_size = 6 # Optimized for CPU
 epochs = 25
 else:
 print("Using mini synthetic dataset...")
 model_type = 'enhanced' # Use enhanced CNN for small dataset
 augmentation_strength = 'light'
 batch_size = 4
 epochs = 15
 
 if use_expanded:
 # Load expanded dataset with enhanced augmentation
 train_ds = MODISPlantBloomDataset(EXPANDED_PLANT_DIR, EXPANDED_METADATA, EXPANDED_NDVI_DIR, 
 stage="train", use_augmentation=True, use_expanded_dataset=True,
 augmentation_strength=augmentation_strength, balance_classes=True)
 val_ds = MODISPlantBloomDataset(EXPANDED_PLANT_DIR, EXPANDED_METADATA, EXPANDED_NDVI_DIR, 
 stage="val", use_augmentation=False, use_expanded_dataset=True,
 balance_classes=False)
 else:
 # Prepare mini dataset
 prepare_mini_images(num_images=12)
 bases = prepare_mini_modis(num_times=3, locations=["tile_h31v08"])
 
 train_ds = MODISPlantBloomDataset(IMG_DIR, ANNOTATIONS, PROCESSED_DIR, 
 stage="train", use_augmentation=True, 
 augmentation_strength=augmentation_strength, balance_classes=True)
 val_ds = MODISPlantBloomDataset(IMG_DIR, ANNOTATIONS, PROCESSED_DIR, 
 stage="val", use_augmentation=False, balance_classes=False)
 
 # Create data loaders with optimized settings
 train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, 
 pin_memory=False, drop_last=True)
 val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0, pin_memory=False)

 # Model selection
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 if model_type == 'transfer_learning':
 # Use transfer learning for better accuracy
 model = TransferLearningCNN(num_classes=len(CLASSES), backbone='mobilenet', 
 pretrained=True, freeze_backbone=False)
 print("Using MobileNetV2 transfer learning model")
 elif model_type == 'enhanced':
 # Use enhanced CNN
 model = EnhancedFiveChannelCNN(num_classes=len(CLASSES), dropout=0.3)
 print("Using enhanced 5-channel CNN")
 else:
 # Fallback to legacy model
 model = MiniFiveChannelCNN(num_classes=len(CLASSES))
 print("Using legacy model")
 
 # Check for dataset leakage before training
 print("\n Checking for dataset leakage...")
 leakage_results = check_dataset_leakage(train_ds, val_ds)
 
 # Train with enhanced settings
 print("\n Starting training...")
 training_history = train_enhanced(model, train_loader, val_loader, device, epochs=epochs, 
 use_expanded_dataset=use_expanded, model_type=model_type, use_class_weights=True)
 
 # Check for suspicious accuracy patterns
 print("\n Checking for suspicious accuracy patterns...")
 suspicious = check_suspicious_accuracy(training_history["final_train_acc"], 
 training_history["final_val_acc"])
 
 # Generate learning curves
 print("\n Generating learning curves...")
 plot_learning_curves(training_history["train_losses"], training_history["train_accs"],
 training_history["val_losses"], training_history["val_accs"],
 training_history["epochs"])
 
 # Compute confusion matrix
 print("\n Computing confusion matrix...")
 cm_results = compute_confusion_matrix(model, val_loader, device, CLASSES)
 
 # Auto re-split if issues detected
 if leakage_results["leakage_detected"] or suspicious:
 print("\n Issues detected - attempting automatic re-split...")
 if use_expanded:
 metadata_path = EXPANDED_METADATA
 backup_path = EXPANDED_METADATA.with_suffix('.backup.csv')
 else:
 metadata_path = ANNOTATIONS
 backup_path = ANNOTATIONS.with_suffix('.backup.csv')
 
 # Backup original
 import shutil
 shutil.copy2(metadata_path, backup_path)
 
 # Re-split by plant_id
 resplit_dataset_by_plant_id(metadata_path, metadata_path)
 
 print(" Re-splitting complete. Consider re-running the pipeline.")
 print(f" Original split backed up to: {backup_path}")

 # Save model with descriptive name
 model_name = f"{model_type}_{'expanded' if use_expanded else 'mini'}_bloomwatch.pt"
 model_path = MODELS_DIR / model_name
 torch.save(model.state_dict(), model_path)

 # Generate prediction
 if len(val_ds) == 0:
 sample = train_ds[0]
 else:
 sample = val_ds[0]
 x, y, meta = sample
 model.eval()
 with torch.no_grad():
 logits = model(x.unsqueeze(0).to(device))
 result = prediction_to_bloomwatch_json(logits.cpu())
 result.update({
 "true_label": int(y),
 "class_names": CLASSES,
 "inference_sample": meta,
 "model_path": str(model_path),
 "dataset_type": "expanded" if use_expanded else "mini",
 "model_type": model_type,
 "augmentation_strength": augmentation_strength,
 "training_epochs": epochs,
 "batch_size": batch_size,
 "quality_assurance": {
 "leakage_detected": leakage_results["leakage_detected"],
 "suspicious_accuracy": suspicious,
 "final_train_acc": training_history["final_train_acc"],
 "final_val_acc": training_history["final_val_acc"],
 "best_val_acc": training_history["best_val_acc"],
 "confusion_matrix_accuracy": cm_results["accuracy"]
 }
 })

 # Write JSON artifact
 out_json = OUTPUTS_DIR / f"{model_type}_{'expanded' if use_expanded else 'mini'}_prediction.json"
 with open(out_json, "w") as f:
 json.dump(result, f, indent=2)

 print(f"\nBloomWatch pipeline complete ({model_type} model, {'expanded' if use_expanded else 'mini'} dataset).")
 print(f"Model saved to: {model_path}")
 print(f"Sample prediction JSON: {out_json}")
 print(json.dumps(result, indent=2))
 print(f"Total time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
 main()

