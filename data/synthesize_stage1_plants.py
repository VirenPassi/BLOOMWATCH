"""
Stage-1 Plant Image Synthesis for BloomWatch Pipeline

Generates 2-3k realistic plant images balanced across 5 bloom stages.
Uses advanced augmentation and style variations for diversity.
"""

import os
import sys
import numpy as np
from pathlib import Path
import json
import csv
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFilter
import random

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Configuration
STAGE1_PLANT_DIR = ROOT / "data" / "expanded_dataset" / "plant_images"
STAGE1_METADATA = ROOT / "data" / "expanded_dataset" / "metadata.csv"
STAGE1_PROCESSED_DIR = ROOT / "data" / "processed" / "MODIS" / "stage1"

CLASSES = ["bud", "early_bloom", "full_bloom", "late_bloom", "dormant"]
IMAGES_PER_CLASS = 600  # 600 * 5 = 3000 total images


class AdvancedPlantSynthesizer:
    """Advanced plant image synthesizer with realistic variations."""
    
    def __init__(self, width: int = 224, height: int = 224):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(42)  # Reproducible results
    
    def create_plant_base(self, class_type: str) -> Image.Image:
        """Create base plant structure based on bloom stage."""
        img = Image.new('RGB', (self.width, self.height), (135, 206, 235))  # Sky blue background
        
        # Add ground
        ground_height = self.height // 3
        ground_color = (139, 69, 19)  # Brown ground
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, self.height - ground_height, self.width, self.height], 
                      fill=ground_color)
        
        # Add plant stem
        stem_width = random.randint(3, 8)
        stem_x = self.width // 2
        stem_top = self.height - ground_height - random.randint(50, 100)
        
        draw.rectangle([stem_x - stem_width//2, stem_top, 
                       stem_x + stem_width//2, self.height - ground_height], 
                      fill=(34, 139, 34))  # Green stem
        
        return img, draw, stem_x, stem_top
    
    def add_bloom_features(self, img: Image.Image, draw: ImageDraw.Draw, 
                          stem_x: int, stem_top: int, class_type: str) -> Image.Image:
        """Add bloom-specific features to the plant."""
        
        if class_type == "bud":
            # Small circular buds
            num_buds = random.randint(3, 8)
            for _ in range(num_buds):
                bud_x = stem_x + random.randint(-30, 30)
                bud_y = stem_top + random.randint(-20, 20)
                bud_size = random.randint(8, 15)
                
                # Bud color (greenish)
                bud_color = (random.randint(50, 100), random.randint(100, 150), 
                           random.randint(20, 60))
                draw.ellipse([bud_x - bud_size, bud_y - bud_size, 
                             bud_x + bud_size, bud_y + bud_size], 
                            fill=bud_color)
        
        elif class_type == "early_bloom":
            # Small flowers with some color
            num_flowers = random.randint(2, 5)
            for _ in range(num_flowers):
                flower_x = stem_x + random.randint(-40, 40)
                flower_y = stem_top + random.randint(-30, 30)
                flower_size = random.randint(12, 20)
                
                # Early bloom colors (pale)
                colors = [(255, 182, 193), (255, 218, 185), (255, 239, 213)]
                flower_color = random.choice(colors)
                draw.ellipse([flower_x - flower_size, flower_y - flower_size, 
                             flower_x + flower_size, flower_y + flower_size], 
                            fill=flower_color)
        
        elif class_type == "full_bloom":
            # Large, vibrant flowers
            num_flowers = random.randint(3, 6)
            for _ in range(num_flowers):
                flower_x = stem_x + random.randint(-50, 50)
                flower_y = stem_top + random.randint(-40, 40)
                flower_size = random.randint(15, 30)
                
                # Full bloom colors (vibrant)
                colors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), 
                         (0, 255, 0), (0, 0, 255), (255, 0, 255)]
                flower_color = random.choice(colors)
                draw.ellipse([flower_x - flower_size, flower_y - flower_size, 
                             flower_x + flower_size, flower_y + flower_size], 
                            fill=flower_color)
        
        elif class_type == "late_bloom":
            # Fading flowers with brown tints
            num_flowers = random.randint(2, 4)
            for _ in range(num_flowers):
                flower_x = stem_x + random.randint(-45, 45)
                flower_y = stem_top + random.randint(-35, 35)
                flower_size = random.randint(12, 25)
                
                # Late bloom colors (fading/brownish)
                colors = [(139, 69, 19), (160, 82, 45), (205, 133, 63)]
                flower_color = random.choice(colors)
                draw.ellipse([flower_x - flower_size, flower_y - flower_size, 
                             flower_x + flower_size, flower_y + flower_size], 
                            fill=flower_color)
        
        else:  # dormant
            # Mostly bare, some dead leaves
            num_leaves = random.randint(1, 3)
            for _ in range(num_leaves):
                leaf_x = stem_x + random.randint(-25, 25)
                leaf_y = stem_top + random.randint(-20, 20)
                leaf_size = random.randint(5, 12)
                
                # Dead leaf colors
                leaf_color = (101, 67, 33)  # Brown
                draw.ellipse([leaf_x - leaf_size, leaf_y - leaf_size, 
                             leaf_x + leaf_size, leaf_y + leaf_size], 
                            fill=leaf_color)
        
        return img
    
    def apply_environmental_variations(self, img: Image.Image) -> Image.Image:
        """Apply environmental variations (lighting, weather, etc.)."""
        # Random lighting variations
        brightness_factor = random.uniform(0.7, 1.3)
        contrast_factor = random.uniform(0.8, 1.2)
        
        # Convert to numpy for processing
        img_array = np.array(img, dtype=np.float32)
        
        # Brightness adjustment
        img_array = img_array * brightness_factor
        
        # Contrast adjustment
        mean = np.mean(img_array)
        img_array = (img_array - mean) * contrast_factor + mean
        
        # Clamp values
        img_array = np.clip(img_array, 0, 255)
        
        # Convert back to PIL
        img = Image.fromarray(img_array.astype(np.uint8))
        
        # Random blur (wind effect)
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        # Random noise
        if random.random() < 0.1:
            noise = np.random.normal(0, 10, img_array.shape)
            img_array = np.array(img) + noise
            img_array = np.clip(img_array, 0, 255)
            img = Image.fromarray(img_array.astype(np.uint8))
        
        return img
    
    def synthesize_plant_image(self, class_type: str, seed: int) -> Image.Image:
        """Synthesize a complete plant image."""
        random.seed(seed)
        np.random.seed(seed)
        
        # Create base plant
        img, draw, stem_x, stem_top = self.create_plant_base(class_type)
        
        # Add bloom features
        img = self.add_bloom_features(img, draw, stem_x, stem_top, class_type)
        
        # Apply environmental variations
        img = self.apply_environmental_variations(img)
        
        return img


def ensure_dirs():
    """Create necessary directories."""
    STAGE1_PLANT_DIR.mkdir(parents=True, exist_ok=True)
    for class_name in CLASSES:
        (STAGE1_PLANT_DIR / class_name).mkdir(exist_ok=True)


def get_ndvi_files() -> List[Path]:
    """Get available NDVI files for metadata linking."""
    if not STAGE1_PROCESSED_DIR.exists():
        return []
    
    ndvi_files = list(STAGE1_PROCESSED_DIR.glob("*_ndvi.npy"))
    return sorted(ndvi_files)


def generate_stage1_dataset():
    """Generate Stage-1 plant image dataset."""
    print("🌱 Starting Stage-1 plant image synthesis...")
    print(f"📊 Target: {IMAGES_PER_CLASS} images per class ({len(CLASSES)} classes)")
    print(f"📁 Output directory: {STAGE1_PLANT_DIR}")
    
    ensure_dirs()
    
    synthesizer = AdvancedPlantSynthesizer()
    ndvi_files = get_ndvi_files()
    
    print(f"📡 Found {len(ndvi_files)} NDVI files for metadata linking")
    
    entries = []
    total_generated = 0
    
    for class_idx, class_name in enumerate(CLASSES):
        print(f"\n🌿 Generating {IMAGES_PER_CLASS} images for class: {class_name}")
        
        for i in range(IMAGES_PER_CLASS):
            # Generate image
            seed = class_idx * 10000 + i
            img = synthesizer.synthesize_plant_image(class_name, seed)
            
            # Save image
            img_filename = f"{class_name}_stage1_{i:04d}.png"
            img_path = STAGE1_PLANT_DIR / class_name / img_filename
            img.save(img_path)
            
            # Determine stage (train/val/test split)
            if i < int(0.7 * IMAGES_PER_CLASS):
                stage = "train"
            elif i < int(0.85 * IMAGES_PER_CLASS):
                stage = "val"
            else:
                stage = "test"
            
            # Link to NDVI file (cycle through available files)
            ndvi_file = ndvi_files[i % len(ndvi_files)] if ndvi_files else None
            ndvi_path = ndvi_file.name if ndvi_file else f"synthetic_ndvi_{i % 10:03d}.npy"
            
            # Generate realistic timestamp
            year = 2023
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            timestamp = f"{year}-{month:02d}-{day:02d}"
            
            entries.append({
                "image_path": f"{class_name}/{img_filename}",
                "bloom_stage": class_name,
                "stage": stage,
                "plant_id": f"stage1_plant_{class_idx}_{i//50:03d}",  # Group by 50
                "timestamp": timestamp,
                "ndvi_file": ndvi_path,
                "is_synthetic": True,
                "synthesis_seed": seed
            })
            
            total_generated += 1
            
            if (i + 1) % 100 == 0:
                print(f"   Generated {i + 1}/{IMAGES_PER_CLASS} images")
    
    # Write metadata CSV
    print(f"\n📝 Writing metadata for {total_generated} images...")
    with open(STAGE1_METADATA, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_path", "bloom_stage", "stage", "plant_id", "timestamp", 
            "ndvi_file", "is_synthetic", "synthesis_seed"
        ])
        writer.writeheader()
        writer.writerows(entries)
    
    # Create dataset info
    info = {
        "total_images": total_generated,
        "images_per_class": IMAGES_PER_CLASS,
        "classes": CLASSES,
        "train_samples": len([e for e in entries if e["stage"] == "train"]),
        "val_samples": len([e for e in entries if e["stage"] == "val"]),
        "test_samples": len([e for e in entries if e["stage"] == "test"]),
        "ndvi_files_linked": len(ndvi_files),
        "is_synthetic": True,
        "generated_at": str(np.datetime64('now')),
        "output_directory": str(STAGE1_PLANT_DIR)
    }
    
    info_path = ROOT / "outputs" / "stage1_synthesis_report.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"🎉 Stage-1 synthesis complete!")
    print(f"📊 Total images: {info['total_images']}")
    print(f"📊 Train: {info['train_samples']}, Val: {info['val_samples']}, Test: {info['test_samples']}")
    print(f"📁 Dataset location: {STAGE1_PLANT_DIR}")
    print(f"📋 Report saved to: {info_path}")
    
    return info


def main():
    """Main synthesis function."""
    generate_stage1_dataset()


if __name__ == "__main__":
    main()
