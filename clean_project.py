#!/usr/bin/env python3
"""
Clean Project Script

This script moves removable files to the trash directory to maintain a clean project structure.
"""

import os
import shutil
from pathlib import Path

def is_removable_file(file_path):
 """Determine if a file is removable (cache, compiled files, etc.)"""
 path = Path(file_path)
 name = path.name
 
 # Cache directories
 if '__pycache__' in str(path):
 return True
 
 # Compiled Python files
 if name.endswith('.pyc') or name.endswith('.pyo') or name.endswith('.pyd'):
 return True
 
 # Virtual environments
 if path.is_dir() and name in ['venv', '.venv', 'env']:
 return True
 
 # System files
 if name in ['.DS_Store', 'Thumbs.db']:
 return True
 
 # Log files
 if name.endswith('.log'):
 return True
 
 # Temporary files
 if name.endswith('.tmp') or name.startswith('tmp'):
 return True
 
 return False

def move_removable_files():
 """Move removable files to trash directory"""
 project_root = Path.cwd()
 trash_dir = project_root / 'trash'
 trash_dir.mkdir(exist_ok=True)
 
 moved_files = []
 
 # Walk through all files and directories
 for item in project_root.rglob('*'):
 # Skip trash directory itself
 if 'trash' in str(item):
 continue
 
 # Skip .git directory
 if '.git' in str(item):
 continue
 
 if is_removable_file(item):
 try:
 # Create relative path in trash
 rel_path = item.relative_to(project_root)
 dest_path = trash_dir / rel_path
 dest_path.parent.mkdir(parents=True, exist_ok=True)
 
 # Move the file or directory
 shutil.move(str(item), str(dest_path))
 moved_files.append(str(rel_path))
 print(f"Moved: {rel_path}")
 except Exception as e:
 print(f"Error moving {item}: {e}")
 
 return moved_files

def main():
 """Main function"""
 print("Cleaning project structure...")
 moved_files = move_removable_files()
 
 print(f"\nMoved {len(moved_files)} removable items to trash/")
 if moved_files:
 print("Moved files:")
 for f in moved_files[:10]: # Show first 10
 print(f" {f}")
 if len(moved_files) > 10:
 print(f" ... and {len(moved_files) - 10} more")

if __name__ == "__main__":
 main()