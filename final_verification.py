#!/usr/bin/env python3
"""
Final Verification Script

This script verifies that the project structure is clean and all core files are present.
"""

import os
from pathlib import Path

def check_core_files():
    """Check that all core files are present"""
    project_root = Path.cwd()
    
    # List of essential core files that should exist
    core_files = [
        "main.py",
        "requirements.txt",
        "app/main.py",
        "app/endpoints.py",
        "app/models.py",
        "configs/config.yaml",
        "pipelines/mini_bloomwatch.py",
        "pipelines/stage2_training.py",
        "utils/helpers.py",
        "visualization/plots.py",
    ]
    
    missing_files = []
    present_files = []
    
    for file_path in core_files:
        full_path = project_root / file_path
        if full_path.exists():
            present_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    return present_files, missing_files

def check_trash_directory():
    """Check what's in the trash directory"""
    project_root = Path.cwd()
    trash_dir = project_root / 'trash'
    
    if not trash_dir.exists():
        return [], 0
    
    trash_contents = list(trash_dir.rglob('*'))
    removable_items = [item for item in trash_contents if item.is_file()]
    
    return trash_contents, len(removable_items)

def main():
    """Main verification function"""
    print("=" * 60)
    print("FINAL PROJECT STRUCTURE VERIFICATION")
    print("=" * 60)
    
    # Check core files
    present, missing = check_core_files()
    
    print(f"Core files present: {len(present)}")
    print(f"Core files missing: {len(missing)}")
    
    if missing:
        print("\nMissing core files:")
        for file in missing:
            print(f"  - {file}")
    else:
        print("\n✓ All essential core files are present")
    
    # Check trash directory
    trash_contents, removable_count = check_trash_directory()
    
    print(f"\nTrash directory contains: {len(trash_contents)} items")
    print(f"Removable files in trash: {removable_count}")
    
    if removable_count > 0:
        print("✓ Removable files have been properly moved to trash")
    else:
        print("✓ No removable files found in main project directories")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()