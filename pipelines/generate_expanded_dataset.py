"""
Generate expanded synthetic dataset for testing the enhanced pipeline.

This script creates a larger synthetic dataset with:
- More plant images (100-200 per class)
- More NDVI/EVI arrays
- Proper directory structure for expanded dataset
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image
import csv

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXPANDED_DATASET_DIR = DATA_DIR / "expanded_dataset"
EXPANDED_NDVI_DIR = EXPANDED_DATASET_DIR / "ndvi"
EXPANDED_PLANT_DIR = EXPANDED_DATASET_DIR / "plant_images"
EXPANDED_METADATA = EXPANDED_DATASET_DIR / "metadata.csv"

CLASSES = ["bud", "early_bloom", "full_bloom", "late_bloom", "dormant"]

def ensure_dirs():
 for d in [EXPANDED_DATASET_DIR, EXPANDED_NDVI_DIR, EXPANDED_PLANT_DIR]:
 d.mkdir(parents=True, exist_ok=True)
 for cls in CLASSES:
 (EXPANDED_PLANT_DIR / cls).mkdir(exist_ok=True)

def synthesize_plant_image(width: int = 224, height: int = 224, seed: int = 0, class_type: str = "bud") -> Image.Image:
 """Generate a more realistic plant image based on bloom stage."""
 rng = np.random.default_rng(seed)
 
 # Base green background
 img = np.zeros((height, width, 3), dtype=np.uint8)
 img[..., 1] = rng.integers(80, 160, size=(height, width)) # green channel
 
 # Add plant-like features based on bloom stage
 if class_type == "bud":
 # Small circular regions (buds)
 num_buds = rng.integers(3, 8)
 for _ in range(num_buds):
 cy, cx = rng.integers(height//4, 3*height//4), rng.integers(width//4, 3*width//4)
 radius = rng.integers(5, 15)
 rr, cc = np.ogrid[:height, :width]
 mask = (rr - cy) ** 2 + (cc - cx) ** 2 <= radius ** 2
 img[mask, 1] = np.clip(img[mask, 1] + rng.integers(30, 60), 0, 255)
 img[mask, 0] = rng.integers(0, 20) # slight red tint
 img[mask, 2] = rng.integers(0, 20) # slight blue tint
 
 elif class_type == "early_bloom":
 # Larger circular regions with more color
 num_flowers = rng.integers(2, 5)
 for _ in range(num_flowers):
 cy, cx = rng.integers(height//4, 3*height//4), rng.integers(width//4, 3*width//4)
 radius = rng.integers(10, 25)
 rr, cc = np.ogrid[:height, :width]
 mask = (rr - cy) ** 2 + (cc - cx) ** 2 <= radius ** 2
 img[mask, 1] = np.clip(img[mask, 1] + rng.integers(40, 80), 0, 255)
 img[mask, 0] = rng.integers(10, 40) # more red
 img[mask, 2] = rng.integers(5, 30) # some blue
 
 elif class_type == "full_bloom":
 # Large, bright, colorful regions
 num_flowers = rng.integers(3, 6)
 for _ in range(num_flowers):
 cy, cx = rng.integers(height//4, 3*height//4), rng.integers(width//4, 3*width//4)
 radius = rng.integers(15, 35)
 rr, cc = np.ogrid[:height, :width]
 mask = (rr - cy) ** 2 + (cc - cx) ** 2 <= radius ** 2
 img[mask, 1] = np.clip(img[mask, 1] + rng.integers(50, 100), 0, 255)
 img[mask, 0] = rng.integers(20, 80) # bright red
 img[mask, 2] = rng.integers(10, 50) # bright blue
 
 elif class_type == "late_bloom":
 # Fading colors, some brown
 num_flowers = rng.integers(2, 4)
 for _ in range(num_flowers):
 cy, cx = rng.integers(height//4, 3*height//4), rng.integers(width//4, 3*width//4)
 radius = rng.integers(12, 28)
 rr, cc = np.ogrid[:height, :width]
 mask = (rr - cy) ** 2 + (cc - cx) ** 2 <= radius ** 2
 img[mask, 1] = np.clip(img[mask, 1] + rng.integers(20, 50), 0, 255)
 img[mask, 0] = rng.integers(30, 60) # brownish red
 img[mask, 2] = rng.integers(5, 25) # muted blue
 
 else: # dormant
 # Mostly green, minimal color
 img[..., 1] = rng.integers(60, 120, size=(height, width))
 img[..., 0] = rng.integers(0, 10)
 img[..., 2] = rng.integers(0, 10)
 
 return Image.fromarray(img)

def synthesize_ndvi_evi(height: int = 224, width: int = 224, seed: int = 0) -> tuple:
 """Generate more realistic NDVI/EVI arrays."""
 rng = np.random.default_rng(seed)
 
 # Create spatial patterns
 y, x = np.ogrid[:height, :width]
 center_y, center_x = height // 2, width // 2
 
 # Distance from center
 dist = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
 
 # NDVI: higher in center, lower at edges
 ndvi_base = 0.3 + 0.4 * np.exp(-dist / (min(height, width) * 0.3))
 ndvi_noise = rng.normal(0, 0.1, (height, width))
 ndvi = np.clip(ndvi_base + ndvi_noise, -0.2, 0.9).astype(np.float32)
 
 # EVI: similar pattern but different values
 evi_base = 0.2 + 0.3 * np.exp(-dist / (min(height, width) * 0.4))
 evi_noise = rng.normal(0, 0.08, (height, width))
 evi = np.clip(evi_base + evi_noise, -0.2, 0.9).astype(np.float32)
 
 return ndvi, evi

def generate_expanded_dataset():
 """Generate the expanded synthetic dataset."""
 print("Generating expanded synthetic dataset...")
 ensure_dirs()
 
 # Generate plant images
 images_per_class = 50 # 50 images per class = 250 total
 entries = []
 
 for class_idx, class_name in enumerate(CLASSES):
 print(f"Generating {images_per_class} images for class: {class_name}")
 
 for i in range(images_per_class):
 # Generate image
 img = synthesize_plant_image(seed=class_idx * 1000 + i, class_type=class_name)
 
 # Save image
 img_path = f"{class_name}/img_{i:03d}.png"
 img.save(EXPANDED_PLANT_DIR / img_path)
 
 # Determine stage (train/val/test split)
 if i < int(0.7 * images_per_class):
 stage = "train"
 elif i < int(0.85 * images_per_class):
 stage = "val"
 else:
 stage = "test"
 
 # Create rng for this iteration
 rng = np.random.default_rng(class_idx * 1000 + i)
 entries.append({
 "image_path": img_path,
 "bloom_stage": class_name,
 "stage": stage,
 "plant_id": f"plant_{class_idx}_{i//10:03d}",
 "timestamp": f"2024-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}"
 })
 
 # Generate NDVI/EVI arrays
 print("Generating NDVI/EVI arrays...")
 num_ndvi_arrays = 20 # 20 NDVI/EVI pairs
 
 for i in range(num_ndvi_arrays):
 ndvi, evi = synthesize_ndvi_evi(seed=5000 + i)
 
 # Save as .npy files
 np.save(EXPANDED_NDVI_DIR / f"ndvi_{i:03d}.npy", ndvi)
 np.save(EXPANDED_NDVI_DIR / f"evi_{i:03d}.npy", evi)
 
 # Write metadata CSV
 print("Writing metadata...")
 with open(EXPANDED_METADATA, "w", newline="") as f:
 writer = csv.DictWriter(f, fieldnames=["image_path", "bloom_stage", "stage", "plant_id", "timestamp"])
 writer.writeheader()
 writer.writerows(entries)
 
 # Create dataset info
 info = {
 "total_images": len(entries),
 "images_per_class": images_per_class,
 "classes": CLASSES,
 "ndvi_evi_pairs": num_ndvi_arrays,
 "train_samples": len([e for e in entries if e["stage"] == "train"]),
 "val_samples": len([e for e in entries if e["stage"] == "val"]),
 "test_samples": len([e for e in entries if e["stage"] == "test"]),
 "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
 }
 
 with open(EXPANDED_DATASET_DIR / "dataset_info.json", "w") as f:
 json.dump(info, f, indent=2)
 
 print(f"Expanded dataset generated successfully!")
 print(f"Total images: {info['total_images']}")
 print(f"Train: {info['train_samples']}, Val: {info['val_samples']}, Test: {info['test_samples']}")
 print(f"NDVI/EVI pairs: {info['ndvi_evi_pairs']}")
 print(f"Dataset location: {EXPANDED_DATASET_DIR}")

if __name__ == "__main__":
 generate_expanded_dataset()
