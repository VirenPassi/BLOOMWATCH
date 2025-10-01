#!/usr/bin/env python3
"""
Targeted Project Structure Auditor
"""

import os
from pathlib import Path

def is_core_file(file_path, project_root):
    """
    Determine if a file is core (required for project to run)
    """
    path = Path(file_path)
    rel_path = path.relative_to(project_root)
    name = path.name
    ext = path.suffix.lower()
    
    # Core directories that contain essential files
    core_dirs = {
        'app', 'configs', 'models', 'pipelines', 'utils', 'visualization', 'webapp'
    }
    
    # Check if file is in a core directory
    for part in rel_path.parts[:-1]:  # All parts except filename
        if part in core_dirs:
            # But exclude cache files even in core directories
            if '__pycache__' in str(rel_path) or name.endswith('.pyc'):
                return False
            return True
    
    # Explicitly core files in root
    core_root_files = {
        'main.py', 'requirements.txt', 'README.md', '__init__.py',
        'run_app.bat', 'run_app.ps1', 'run_app_final.ps1', 
        'run_app_minimal.ps1', 'run_app_simple.ps1'
    }
    
    if name in core_root_files:
        return True
    
    # Configuration files are core
    config_exts = {'.yaml', '.yml', '.json', '.ini', '.toml'}
    if ext in config_exts:
        return True
    
    # Script files are core
    script_exts = {'.sh', '.bat', '.ps1'}
    if ext in script_exts:
        return True
    
    return False

def is_removable_file(file_path):
    """
    Determine if a file is removable (caches, compiled files, etc.)
    """
    path = Path(file_path)
    name = path.name
    ext = path.suffix.lower()
    
    # Cache directories
    if '__pycache__' in str(path):
        return True
    
    # Compiled Python files
    if ext == '.pyc' or name.endswith('.pyc'):
        return True
    
    # Virtual environments
    if path.is_dir() and name in ['venv', '.venv', 'env']:
        return True
    
    # System files
    if name in ['.DS_Store', 'Thumbs.db']:
        return True
    
    # Log files
    if ext == '.log':
        return True
    
    # Temporary files
    if ext == '.tmp' or name.startswith('tmp'):
        return True
    
    return False

def scan_project():
    """
    Scan the project and classify files
    """
    project_root = Path.cwd()
    print(f"Scanning project: {project_root}")
    
    core_files = []
    optional_files = []
    removable_files = []
    
    # Walk through all files
    for item in project_root.rglob('*'):
        # Skip trash directory
        if 'trash' in str(item):
            continue
            
        # Skip .git directory
        if '.git' in str(item):
            continue
            
        if item.is_file():
            if is_removable_file(item):
                removable_files.append(str(item.relative_to(project_root)))
            elif is_core_file(item, project_root):
                core_files.append(str(item.relative_to(project_root)))
            else:
                optional_files.append(str(item.relative_to(project_root)))
    
    return core_files, optional_files, removable_files

def main():
    """Main function"""
    core, optional, removable = scan_project()
    
    print("\n" + "="*60)
    print("PROJECT STRUCTURE AUDIT")
    print("="*60)
    print(f"Total files scanned: {len(core) + len(optional) + len(removable)}")
    print(f"Core files:          {len(core)}")
    print(f"Optional files:      {len(optional)}")
    print(f"Removable files:     {len(removable)}")
    
    if removable:
        print("\nRemovable files:")
        print("-" * 30)
        for f in sorted(removable):
            print(f"  {f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()