"""
Stage-2 Plant Image Synthesis for BloomWatch Pipeline

Generates 6-8k realistic plant images with seasonal variations and enhanced diversity.
Uses advanced augmentation, style variations, and climate-specific characteristics.
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
from datetime import datetime, timedelta

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Configuration
STAGE2_PLANT_DIR = ROOT / "data" / "expanded_dataset" / "plant_images"
STAGE2_METADATA = ROOT / "data" / "expanded_dataset" / "metadata.csv"
STAGE2_PROCESSED_DIR = ROOT / "data" / "processed" / "MODIS" / "stage2"

CLASSES = ["bud", "early_bloom", "full_bloom", "late_bloom", "dormant"]
IMAGES_PER_CLASS = 1500  # 1500 * 5 = 7500 total images

# Seasonal variations
SEASONS = {
    "spring": {"months": [3, 4, 5], "growth_factor": 1.2, "color_intensity": 0.8},
    "summer": {"months": [6, 7, 8], "growth_factor": 1.5, "color_intensity": 1.0},
    "autumn": {"months": [9, 10, 11], "growth_factor": 0.8, "color_intensity": 0.6},
    "winter": {"months": [12, 1, 2], "growth_factor": 0.3, "color_intensity": 0.4}
}

# Climate-specific characteristics
CLIMATE_CHARS = {
    "california_central_valley": {"base_size": 1.0, "color_saturation": 0.8, "growth_rate": 1.0},
    "iowa_corn_belt": {"base_size": 1.2, "color_saturation": 1.0, "growth_rate": 1.1},
    "amazon_basin": {"base_size": 1.5, "color_saturation": 1.2, "growth_rate": 1.3},
    "great_plains": {"base_size": 0.8, "color_saturation": 0.7, "growth_rate": 0.8},
    "southeast_us": {"base_size": 1.1, "color_saturation": 1.1, "growth_rate": 1.2}
}


class AdvancedPlantSynthesizerStage2:
    """Enhanced plant image synthesizer with seasonal and climate variations."""
    
    def __init__(self, width: int = 224, height: int = 224):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(42)  # Reproducible results
    
    def get_seasonal_params(self, month: int) -> Dict:
        """Get seasonal parameters based on month."""
        for season, params in SEASONS.items():
            if month in params["months"]:
                return params
        return SEASONS["spring"]  # Default
    
    def get_climate_params(self, aoi: str) -> Dict:
        """Get climate-specific parameters."""
        return CLIMATE_CHARS.get(aoi, CLIMATE_CHARS["california_central_valley"])
    
    def create_plant_base_enhanced(self, class_type: str, season_params: Dict, climate_params: Dict) -> Image.Image:
        """Create enhanced plant structure with seasonal and climate variations."""
        # Background with seasonal variation
        if season_params["growth_factor"] > 1.0:  # Summer
            bg_color = (135, 206, 235)  # Sky blue
        elif season_params["growth_factor"] < 0.5:  # Winter
            bg_color = (176, 196, 222)  # Light steel blue
        else:
            bg_color = (135, 206, 235)  # Default sky blue
        
        img = Image.new('RGB', (self.width, self.height), bg_color)
        
        # Add ground with seasonal variation
        ground_height = int(self.height // 3 * season_params["growth_factor"])
        ground_height = max(ground_height, self.height // 4)
        
        if season_params["growth_factor"] < 0.5:  # Winter
            ground_color = (139, 69, 19)  # Brown ground
        else:
            ground_color = (34, 139, 34)  # Green ground
        
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, self.height - ground_height, self.width, self.height], 
                      fill=ground_color)
        
        # Add plant stem with climate variation
        stem_width = int(random.randint(3, 8) * climate_params["base_size"])
        stem_x = self.width // 2
        stem_top = self.height - ground_height - int(random.randint(50, 100) * season_params["growth_factor"])
        
        stem_color = (34, 139, 34) if season_params["growth_factor"] > 0.5 else (101, 67, 33)
        draw.rectangle([stem_x - stem_width//2, stem_top, 
                       stem_x + stem_width//2, self.height - ground_height], 
                      fill=stem_color)
        
        return img, draw, stem_x, stem_top
    
    def add_bloom_features_enhanced(self, img: Image.Image, draw: ImageDraw.Draw, 
                                  stem_x: int, stem_top: int, class_type: str,
                                  season_params: Dict, climate_params: Dict) -> Image.Image:
        """Add bloom-specific features with seasonal and climate variations."""
        
        # Apply seasonal growth factor
        size_multiplier = season_params["growth_factor"] * climate_params["growth_rate"]
        color_intensity = season_params["color_intensity"] * climate_params["color_saturation"]
        
        if class_type == "bud":
            # Small circular buds
            num_buds = int(random.randint(3, 8) * size_multiplier)
            for _ in range(num_buds):
                bud_x = stem_x + random.randint(-30, 30)
                bud_y = stem_top + random.randint(-20, 20)
                bud_size = int(random.randint(8, 15) * size_multiplier)
                
                # Bud color with seasonal variation
                bud_color = (int(50 * color_intensity), int(100 * color_intensity), 
                           int(20 * color_intensity))
                draw.ellipse([bud_x - bud_size, bud_y - bud_size, 
                             bud_x + bud_size, bud_y + bud_size], 
                            fill=bud_color)
        
        elif class_type == "early_bloom":
            # Small flowers with some color
            num_flowers = int(random.randint(2, 5) * size_multiplier)
            for _ in range(num_flowers):
                flower_x = stem_x + random.randint(-40, 40)
                flower_y = stem_top + random.randint(-30, 30)
                flower_size = int(random.randint(12, 20) * size_multiplier)
                
                # Early bloom colors with seasonal variation
                colors = [(255, 182, 193), (255, 218, 185), (255, 239, 213)]
                flower_color = random.choice(colors)
                # Apply seasonal color intensity
                flower_color = tuple(int(c * color_intensity) for c in flower_color)
                draw.ellipse([flower_x - flower_size, flower_y - flower_size, 
                             flower_x + flower_size, flower_y + flower_size], 
                            fill=flower_color)
        
        elif class_type == "full_bloom":
            # Large, vibrant flowers
            num_flowers = int(random.randint(3, 6) * size_multiplier)
            for _ in range(num_flowers):
                flower_x = stem_x + random.randint(-50, 50)
                flower_y = stem_top + random.randint(-40, 40)
                flower_size = int(random.randint(15, 30) * size_multiplier)
                
                # Full bloom colors with seasonal variation
                colors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), 
                         (0, 255, 0), (0, 0, 255), (255, 0, 255)]
                flower_color = random.choice(colors)
                # Apply seasonal color intensity
                flower_color = tuple(int(c * color_intensity) for c in flower_color)
                draw.ellipse([flower_x - flower_size, flower_y - flower_size, 
                             flower_x + flower_size, flower_y + flower_size], 
                            fill=flower_color)
        
        elif class_type == "late_bloom":
            # Fading flowers with brown tints
            num_flowers = int(random.randint(2, 4) * size_multiplier)
            for _ in range(num_flowers):
                flower_x = stem_x + random.randint(-45, 45)
                flower_y = stem_top + random.randint(-35, 35)
                flower_size = int(random.randint(12, 25) * size_multiplier)
                
                # Late bloom colors with seasonal variation
                colors = [(139, 69, 19), (160, 82, 45), (205, 133, 63)]
                flower_color = random.choice(colors)
                # Apply seasonal color intensity
                flower_color = tuple(int(c * color_intensity) for c in flower_color)
                draw.ellipse([flower_x - flower_size, flower_y - flower_size, 
                             flower_x + flower_size, flower_y + flower_size], 
                            fill=flower_color)
        
        else:  # dormant
            # Mostly bare, some dead leaves
            num_leaves = int(random.randint(1, 3) * size_multiplier)
            for _ in range(num_leaves):
                leaf_x = stem_x + random.randint(-25, 25)
                leaf_y = stem_top + random.randint(-20, 20)
                leaf_size = int(random.randint(5, 12) * size_multiplier)
                
                # Dead leaf colors
                leaf_color = (101, 67, 33)  # Brown
                draw.ellipse([leaf_x - leaf_size, leaf_y - leaf_size, 
                             leaf_x + leaf_size, leaf_y + leaf_size], 
                            fill=leaf_color)
        
        return img
    
    def apply_environmental_variations_enhanced(self, img: Image.Image, season_params: Dict) -> Image.Image:
        """Apply enhanced environmental variations based on season."""
        # Seasonal lighting variations
        brightness_factor = 0.7 + (season_params["growth_factor"] - 0.3) * 0.6
        contrast_factor = 0.8 + (season_params["color_intensity"] - 0.4) * 0.4
        
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
        
        # Seasonal blur effects
        if season_params["growth_factor"] < 0.5:  # Winter - more wind
            if random.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
        else:
            if random.random() < 0.2:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        # Seasonal noise
        if season_params["growth_factor"] < 0.5:  # Winter - more atmospheric effects
            if random.random() < 0.15:
                noise = np.random.normal(0, 15, img_array.shape)
                img_array = np.array(img) + noise
                img_array = np.clip(img_array, 0, 255)
                img = Image.fromarray(img_array.astype(np.uint8))
        
        return img
    
    def synthesize_plant_image_enhanced(self, class_type: str, seed: int, aoi: str = "california_central_valley") -> Image.Image:
        """Synthesize a complete plant image with seasonal and climate variations."""
        random.seed(seed)
        np.random.seed(seed)
        
        # Generate realistic timestamp
        year = random.randint(2023, 2024)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        
        # Get seasonal and climate parameters
        season_params = self.get_seasonal_params(month)
        climate_params = self.get_climate_params(aoi)
        
        # Create base plant
        img, draw, stem_x, stem_top = self.create_plant_base_enhanced(class_type, season_params, climate_params)
        
        # Add bloom features
        img = self.add_bloom_features_enhanced(img, draw, stem_x, stem_top, class_type, season_params, climate_params)
        
        # Apply environmental variations
        img = self.apply_environmental_variations_enhanced(img, season_params)
        
        return img, {"year": year, "month": month, "day": day, "season": season_params, "climate": climate_params}


def ensure_dirs():
    """Create necessary directories."""
    STAGE2_PLANT_DIR.mkdir(parents=True, exist_ok=True)
    for class_name in CLASSES:
        (STAGE2_PLANT_DIR / class_name).mkdir(exist_ok=True)


def get_ndvi_files_stage2() -> List[Path]:
    """Get available NDVI files for metadata linking."""
    if not STAGE2_PROCESSED_DIR.exists():
        return []
    
    ndvi_files = list(STAGE2_PROCESSED_DIR.glob("*_ndvi.npy"))
    return sorted(ndvi_files)


def generate_stage2_dataset():
    """Generate Stage-2 plant image dataset."""
    print("Starting Stage-2 plant image synthesis...")
    print(f"Target: {IMAGES_PER_CLASS} images per class ({len(CLASSES)} classes)")
    print(f"Total target: {IMAGES_PER_CLASS * len(CLASSES)} images")
    print(f"Output directory: {STAGE2_PLANT_DIR}")
    
    ensure_dirs()
    
    synthesizer = AdvancedPlantSynthesizerStage2()
    ndvi_files = get_ndvi_files_stage2()
    
    print(f"Found {len(ndvi_files)} NDVI files for metadata linking")
    
    entries = []
    total_generated = 0
    
    # Available AOIs for climate variation
    available_aois = ["california_central_valley", "iowa_corn_belt", "amazon_basin", "great_plains", "southeast_us"]
    
    for class_idx, class_name in enumerate(CLASSES):
        print(f"\nGenerating {IMAGES_PER_CLASS} images for class: {class_name}")
        
        for i in range(IMAGES_PER_CLASS):
            # Select AOI for climate variation
            aoi = available_aois[i % len(available_aois)]
            
            # Generate image
            seed = class_idx * 10000 + i
            img, timestamp_info = synthesizer.synthesize_plant_image_enhanced(class_name, seed, aoi)
            
            # Save image
            img_filename = f"{class_name}_stage2_{i:04d}.png"
            img_path = STAGE2_PLANT_DIR / class_name / img_filename
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
            
            # Generate timestamp
            timestamp = f"{timestamp_info['year']}-{timestamp_info['month']:02d}-{timestamp_info['day']:02d}"
            
            # Determine season from month
            month = timestamp_info['month']
            if month in [12, 1, 2]:
                season = "winter"
            elif month in [3, 4, 5]:
                season = "spring"
            elif month in [6, 7, 8]:
                season = "summer"
            else:  # 9, 10, 11
                season = "autumn"
            
            entries.append({
                "image_path": f"{class_name}/{img_filename}",
                "bloom_stage": class_name,
                "stage": stage,
                "plant_id": f"stage2_plant_{class_idx}_{i//75:03d}",  # Group by 75
                "timestamp": timestamp,
                "ndvi_file": ndvi_path,
                "aoi": aoi,
                "season": season,
                "is_synthetic": True,
                "synthesis_seed": seed,
                "climate_params": timestamp_info["climate"]
            })
            
            total_generated += 1
            
            if (i + 1) % 200 == 0:
                print(f"   Generated {i + 1}/{IMAGES_PER_CLASS} images")
    
    # Write metadata CSV
    print(f"\nWriting metadata for {total_generated} images...")
    with open(STAGE2_METADATA, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_path", "bloom_stage", "stage", "plant_id", "timestamp", 
            "ndvi_file", "aoi", "season", "is_synthetic", "synthesis_seed", "climate_params"
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
        "aois_used": list(set(e["aoi"] for e in entries)),
        "seasons_used": list(set(e["season"] for e in entries)),
        "is_synthetic": True,
        "generated_at": datetime.now().isoformat(),
        "output_directory": str(STAGE2_PLANT_DIR)
    }
    
    info_path = ROOT / "outputs" / "stage2_synthesis_report.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"Stage-2 synthesis complete!")
    print(f"Total images: {info['total_images']}")
    print(f"Train: {info['train_samples']}, Val: {info['val_samples']}, Test: {info['test_samples']}")
    print(f"AOIs used: {', '.join(info['aois_used'])}")
    print(f"Seasons used: {', '.join(info['seasons_used'])}")
    print(f"Dataset location: {STAGE2_PLANT_DIR}")
    print(f"Report saved to: {info_path}")
    
    return info


def main():
    """Main synthesis function."""
    generate_stage2_dataset()


if __name__ == "__main__":
    main()
