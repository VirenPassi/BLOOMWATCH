#!/usr/bin/env python3
"""
Simple Project Structure Auditor and Cleaner
"""

import os
import shutil
from pathlib import Path

def classify_file(file_path, project_root):
 """
 Classify a file into CORE, OPTIONAL, or REMOVABLE
 """
 path = Path(file_path)
 rel_path = path.relative_to(project_root)
 name = path.name
 ext = path.suffix.lower()
 
 # Core files that are absolutely required
 core_files = [
 'main.py', 'requirements.txt', 'README.md', 'config.py', 
 'models.py', 'endpoints.py', 'utils.py', '__init__.py',
 'run_app.bat', 'run_app.ps1'
 ]
 
 # Core directories
 core_dirs = [
 'app', 'configs', 'models', 'pipelines', 'utils', 'visualization', 'webapp'
 ]
 
 # Check if in core directory
 for part in rel_path.parts:
 if part in core_dirs:
 # But still check for removable content within core dirs
 if '__pycache__' in str(rel_path) or name.endswith('.pyc'):
 return 'REMOVABLE'
 return 'CORE'
 
 # Explicit core files
 if name in core_files:
 return 'CORE'
 
 # Removable patterns
 removable_patterns = [
 '.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe', '.log', '.tmp'
 ]
 
 removable_names = [
 '.DS_Store', 'Thumbs.db'
 ]
 
 removable_dirs = [
 '__pycache__', '.pytest_cache', '.mypy_cache'
 ]
 
 # Check for removable items
 if name in removable_names or ext in removable_patterns:
 return 'REMOVABLE'
 
 if path.is_dir() and name in removable_dirs:
 return 'REMOVABLE'
 
 # Virtual environments
 if path.is_dir() and name in ['venv', '.venv', 'env']:
 return 'REMOVABLE'
 
 # Documentation (mostly optional)
 if ext == '.md' and name not in ['README.md']:
 return 'OPTIONAL'
 
 # Tests (optional)
 if 'test' in name.lower() or 'Test' in name:
 return 'OPTIONAL'
 
 # By default, consider as CORE if it's a Python file or config
 if ext in ['.py', '.yaml', '.yml', '.json', '.toml', '.ini', '.sh', '.bat', '.ps1']:
 return 'CORE'
 
 # Everything else is optional
 return 'OPTIONAL'

def scan_project(root_path):
 """
 Scan project and classify all items
 """
 classifications = {'CORE': [], 'OPTIONAL': [], 'REMOVABLE': []}
 root = Path(root_path)
 
 # Walk through all files and directories
 for item in root.rglob('*'):
 # Skip trash directory
 if 'trash' in str(item):
 continue
 
 classification = classify_file(item, root)
 classifications[classification].append(str(item))
 
 return classifications

def main():
 """Main audit function"""
 project_root = Path.cwd()
 print(f"Auditing: {project_root}")
 
 # Scan project
 results = scan_project(project_root)
 
 # Create trash directory
 trash_dir = project_root / 'trash'
 trash_dir.mkdir(exist_ok=True)
 
 # Move removable files
 moved_count = 0
 if results['REMOVABLE']:
 print(f"\nMoving {len(results['REMOVABLE'])} removable items to trash...")
 for item_path in results['REMOVABLE']:
 try:
 item = Path(item_path)
 if item.exists():
 # Create relative path in trash
 rel_path = item.relative_to(project_root)
 dest_path = trash_dir / rel_path
 dest_path.parent.mkdir(parents=True, exist_ok=True)
 
 shutil.move(str(item), str(dest_path))
 moved_count += 1
 print(f" Moved: {rel_path}")
 except Exception as e:
 print(f" Error moving {item_path}: {e}")
 
 # Print report
 print("\n" + "="*50)
 print("PROJECT AUDIT REPORT")
 print("="*50)
 print(f"Total items scanned: {sum(len(results[k]) for k in results)}")
 print(f"CORE files: {len(results['CORE'])}")
 print(f"OPTIONAL files: {len(results['OPTIONAL'])}")
 print(f"REMOVABLE items: {len(results['REMOVABLE'])}")
 print(f"Items moved to trash: {moved_count}")
 print("="*50)

if __name__ == "__main__":
 main()