#!/usr/bin/env python3
"""
Setup Manual MODIS Data Organization
"""

import os
from pathlib import Path

def main():
    print("Setting up Manual MODIS Data Organization")
    print("=" * 60)
    
    # Create the main MODIS data directory structure
    modis_base_dir = Path("data/NASA_MODIS_Manual")
    modis_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create year-based subdirectories
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    
    for year in years:
        year_dir = modis_base_dir / str(year)
        year_dir.mkdir(exist_ok=True)
        print(f"Created directory: {year_dir}")
    
    # Create processed data directory
    processed_dir = Path("data/processed/MODIS/manual")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDirectory structure created:")
    print(f"Base directory: {modis_base_dir}")
    print(f"Processed directory: {processed_dir}")
    
    print(f"\nInstructions for organizing your data:")
    print(f"1. Copy your 2025 MODIS files to: {modis_base_dir}/2025/")
    print(f"2. Copy your 2024 MODIS files to: {modis_base_dir}/2024/")
    print(f"3. Copy your 2023 MODIS files to: {modis_base_dir}/2023/")
    print(f"4. Copy your 2022 MODIS files to: {modis_base_dir}/2022/")
    print(f"5. Copy your 2021 MODIS files to: {modis_base_dir}/2021/")
    print(f"6. Copy your 2020 MODIS files to: {modis_base_dir}/2020/")
    
    print(f"\nExpected file structure:")
    print(f"Each year directory should contain HDF files like:")
    print(f"  MOD13Q1.A2025001.h08v05.061.2025010154321.hdf")
    print(f"  MOD13Q1.A2025001.h08v04.061.2025010154321.hdf")
    print(f"  MOD13Q1.A2025001.h07v05.061.2025010154321.hdf")
    
    print(f"\nNext steps:")
    print(f"1. Copy your MODIS files to the year directories")
    print(f"2. Run: python process_manual_modis.py")
    print(f"3. Run: python train_manual_modis.py")
    
    print(f"\nSetup complete! Ready for your manual MODIS data.")

if __name__ == "__main__":
    main()
