#!/usr/bin/env python3
"""
Create synthetic MODIS data that matches real NASA data characteristics
"""

import os
import numpy as np
from pathlib import Path
import warnings

def create_realistic_ndvi_evi(height=224, width=224, seed=0):
    """Create realistic NDVI/EVI data based on real MODIS characteristics"""
    rng = np.random.default_rng(seed)
    
    # Create base patterns that look like real vegetation
    y, x = np.ogrid[:height, :width]
    
    # Create some spatial patterns
    center_y, center_x = height // 2, width // 2
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # NDVI patterns (more realistic)
    ndvi_base = 0.3 + 0.4 * np.exp(-dist_from_center / (height * 0.3))
    ndvi_noise = rng.normal(0, 0.1, (height, width))
    ndvi = np.clip(ndvi_base + ndvi_noise, -0.2, 0.9).astype(np.float32)
    
    # EVI patterns (slightly different from NDVI)
    evi_base = 0.2 + 0.3 * np.exp(-dist_from_center / (height * 0.4))
    evi_noise = rng.normal(0, 0.08, (height, width))
    evi = np.clip(evi_base + evi_noise, -0.2, 0.9).astype(np.float32)
    
    # Add some seasonal variation
    seasonal_factor = 0.5 + 0.5 * np.sin(seed * 0.1)
    ndvi *= seasonal_factor
    evi *= seasonal_factor
    
    return ndvi, evi

def main():
    print("Creating Synthetic MODIS Data")
    print("=" * 40)
    
    # Output directory
    output_dir = Path("./data/processed/MODIS/stage2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create multiple time series of NDVI/EVI data
    num_times = 10  # Create 10 time points
    locations = ["h07v05", "h08v04", "h08v05"]  # Based on your real files
    
    print(f"Creating {num_times} time points for {len(locations)} locations")
    print(f"Output directory: {output_dir}")
    
    created_files = []
    
    for loc in locations:
        for t in range(num_times):
            # Create realistic NDVI/EVI data
            ndvi, evi = create_realistic_ndvi_evi(seed=hash(f"{loc}_{t}") % 10000)
            
            # Save NDVI
            ndvi_file = output_dir / f"ndvi_{loc}_t{t:02d}.npy"
            np.save(ndvi_file, ndvi)
            created_files.append(ndvi_file)
            
            # Save EVI
            evi_file = output_dir / f"evi_{loc}_t{t:02d}.npy"
            np.save(evi_file, evi)
            created_files.append(evi_file)
            
            print(f"Created: {ndvi_file.name}, {evi_file.name}")
    
    print(f"\nCreated {len(created_files)} files total")
    
    # Verify files
    ndvi_files = list(output_dir.glob("ndvi_*.npy"))
    evi_files = list(output_dir.glob("evi_*.npy"))
    
    print(f"NDVI files: {len(ndvi_files)}")
    print(f"EVI files: {len(evi_files)}")
    
    if ndvi_files and evi_files:
        # Test load one file
        test_ndvi = np.load(ndvi_files[0])
        test_evi = np.load(evi_files[0])
        
        print(f"\nTest file shapes: NDVI {test_ndvi.shape}, EVI {test_evi.shape}")
        print(f"NDVI range: [{test_ndvi.min():.3f}, {test_ndvi.max():.3f}]")
        print(f"EVI range: [{test_evi.min():.3f}, {test_evi.max():.3f}]")
        
        print("\nSuccess! Synthetic MODIS data is ready for training.")
        print("You can now run the Stage 2 training pipeline.")
    else:
        print("\nError: No files were created.")

if __name__ == "__main__":
    main()
