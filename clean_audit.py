#!/usr/bin/env python3
"""
Clean Audit Script for BloomWatch

This script performs a conservative audit of the BloomWatch project,
identifying only clearly removable files (caches, temp files, etc.)
and moving them to a trash directory.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List

def create_trash_directory(project_root: Path) -> Path:
    """Create the trash directory if it doesn't exist."""
    trash_dir = project_root / "trash"
    trash_dir.mkdir(exist_ok=True)
    return trash_dir

def is_clearly_removable(file_path: Path, relative_path: str) -> bool:
    """Determine if a file is clearly removable (caches, temp files, etc.)."""
    filename = file_path.name
    
    # Python cache files
    if file_path.suffix in ['.pyc', '.pyo', '.pyd']:
        return True
    
    # Cache directories
    if filename == '__pycache__':
        return True
    
    # System files
    system_files = {'.DS_Store', 'Thumbs.db'}
    if filename in system_files:
        return True
    
    # Log files
    if file_path.suffix == '.log':
        return True
    
    # Temp files
    temp_extensions = {'.tmp', '.bak', '.cache'}
    if file_path.suffix in temp_extensions:
        return True
    
    # Generated submission ZIP files (these were created during development)
    if file_path.suffix == '.zip' and 'Submission' in filename and '20251001' in filename:
        return True
    
    # Virtual environment (if it exists at root level)
    if relative_path == '.venv':
        return True
    
    return False

def audit_and_clean_project():
    """Audit and clean the project."""
    project_root = Path(__file__).parent.resolve()
    print(f"Auditing project at: {project_root}")
    
    # Create trash directory
    trash_dir = create_trash_directory(project_root)
    print(f"Trash directory: {trash_dir}")
    
    # Find removable files
    removable_files = []
    
    for root, dirs, files in os.walk(project_root):
        # Skip the trash directory itself
        if 'trash' in root.split(os.sep):
            continue
            
        # Check directories
        dirs_to_remove = []
        for dir_name in dirs:
            dir_path = Path(root) / dir_name
            relative_path = str(dir_path.relative_to(project_root))
            
            if is_clearly_removable(dir_path, relative_path):
                removable_files.append(dir_path)
                dirs_to_remove.append(dir_name)
        
        # Remove directories from walk (so we don't traverse into them)
        for dir_name in dirs_to_remove:
            dirs.remove(dir_name)
        
        # Check files
        for file_name in files:
            file_path = Path(root) / file_name
            relative_path = str(file_path.relative_to(project_root))
            
            # Skip this script itself
            if file_name == 'clean_audit.py':
                continue
                
            if is_clearly_removable(file_path, relative_path):
                removable_files.append(file_path)
    
    # Report findings
    print(f"\n=== AUDIT RESULTS ===")
    print(f"Found {len(removable_files)} clearly removable files/directories:")
    
    for file_path in removable_files:
        relative_path = file_path.relative_to(project_root)
        print(f"  - {relative_path}")
    
    # Move removable files
    if removable_files:
        print(f"\n=== MOVING REMOVABLE FILES ===")
        moved_count = 0
        
        for file_path in removable_files:
            try:
                relative_path = file_path.relative_to(project_root)
                trash_file_path = trash_dir / relative_path
                trash_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                if file_path.is_file():
                    shutil.move(str(file_path), str(trash_file_path))
                elif file_path.is_dir():
                    shutil.move(str(file_path), str(trash_file_path))
                
                print(f"Moved to trash: {relative_path}")
                moved_count += 1
                
            except Exception as e:
                print(f"Error moving {file_path}: {e}")
        
        print(f"\n✅ Moved {moved_count} files/directories to trash.")
    else:
        print("\n✅ No removable files found.")
    
    print(f"\n✅ Audit and cleaning complete!")
    print(f"✅ Project structure is now clean!")

if __name__ == "__main__":
    audit_and_clean_project()