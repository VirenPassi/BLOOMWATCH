#!/usr/bin/env python3
"""
Comprehensive Project Structure Auditor and Cleaner

This script scans the entire project and classifies every file into three categories:
1. CORE FILES (absolutely required for the project to run)
2. OPTIONAL FILES (helpful for development but not strictly needed at runtime)
3. REMOVABLE FILES (caches, compiled files, temp logs, unused venvs, etc.)

Rules:
- Never delete CORE or OPTIONAL files
- Move REMOVABLE files to a trash/ folder at project root
- Generate a structured report of the audit results
"""

import os
import shutil
from pathlib import Path

def classify_file(file_path):
    """
    Classify a file into CORE, OPTIONAL, or REMOVABLE
    
    Returns:
        str: 'CORE', 'OPTIONAL', or 'REMOVABLE'
    """
    path = Path(file_path)
    name = path.name
    ext = path.suffix.lower()
    
    # Core files - absolutely required for project to run
    core_patterns = {
        # Essential Python files
        '.py', '.yaml', '.yml', '.json', '.toml', '.ini',
        # Essential documentation
        'requirements.txt', 'README.md', 'LICENSE',
        # Essential scripts
        '.sh', '.bat', '.ps1'
    }
    
    # Core filenames that must be preserved
    core_filenames = {
        'main.py', '__init__.py', 'config.py', 'models.py', 'endpoints.py',
        'utils.py', 'run_app.py', 'app.py', 'train.py', 'predict.py',
        'bloomwatch_temporal_workflow.py', 'TEMPORAL_WORKFLOW.md'
    }
    
    # Core directories that must be preserved
    core_directories = {
        'app', 'pipelines', 'models', 'configs', 'utils', 'visualization'
    }
    
    # Check if file is in a core directory
    for part in path.parts:
        if part in core_directories:
            # But still check for removable files within core directories
            if name in ['.DS_Store']:
                return 'REMOVABLE'
            if '__pycache__' in path.parts:
                return 'REMOVABLE'
            if ext in ['.pyc', '.pyo', '.pyd']:
                return 'REMOVABLE'
            if name.endswith('.pyc'):
                return 'REMOVABLE'
            return 'CORE'
    
    # Explicitly check for core filenames
    if name in core_filenames:
        return 'CORE'
    
    # Check for removable patterns
    removable_patterns = {
        '.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe', '.log', '.tmp',
        '.cache', '.DS_Store'
    }
    
    removable_filenames = {
        '.DS_Store', 'Thumbs.db', '.coverage', '.pytest_cache'
    }
    
    removable_directories = {
        '__pycache__', '.pytest_cache', '.mypy_cache', '.tox', '.eggs'
    }
    
    # Check for removable files
    if name in removable_filenames:
        return 'REMOVABLE'
    
    if ext in removable_patterns:
        return 'REMOVABLE'
    
    if name.startswith('.') and 'cache' in name.lower():
        return 'REMOVABLE'
    
    # Check for removable directories
    if path.is_dir() and name in removable_directories:
        return 'REMOVABLE'
    
    # Virtual environments
    if path.is_dir() and name in ['venv', '.venv', 'env']:
        return 'REMOVABLE'
    
    # Build artifacts
    build_artifacts = {
        'build', 'dist', '.egg-info', 'node_modules'
    }
    
    if path.is_dir() and name in build_artifacts:
        return 'REMOVABLE'
    
    # Temporary files
    if name.startswith('temp') or name.startswith('tmp'):
        return 'REMOVABLE'
    
    # Log files
    if ext == '.log':
        return 'REMOVABLE'
    
    # Check for core extensions
    if ext in core_patterns:
        # But exclude certain patterns
        if name.endswith('_test.py') or name.endswith('_tests.py'):
            return 'OPTIONAL'  # Tests are optional for runtime
        return 'CORE'
    
    # Documentation files (optional for runtime)
    doc_extensions = {'.md', '.rst', '.txt'}
    if ext in doc_extensions:
        # But some documentation is core
        core_docs = {
            'README.md', 'requirements.txt'
        }
        if name in core_docs:
            return 'CORE'
        return 'OPTIONAL'
    
    # Configuration files (usually core)
    config_extensions = {'.yaml', '.yml', '.json', '.ini', '.cfg', '.toml'}
    if ext in config_extensions:
        return 'CORE'
    
    # Script files (usually core)
    script_extensions = {'.sh', '.bat', '.ps1', '.bash', '.zsh'}
    if ext in script_extensions:
        return 'CORE'
    
    # Everything else that's not explicitly removable is optional
    return 'OPTIONAL'

def scan_project(root_path):
    """
    Scan the entire project and classify all files
    
    Args:
        root_path (str): Root path of the project
        
    Returns:
        dict: Dictionary with lists of CORE, OPTIONAL, and REMOVABLE files
    """
    classifications = {
        'CORE': [],
        'OPTIONAL': [],
        'REMOVABLE': []
    }
    
    root = Path(root_path)
    
    # Walk through all files and directories
    for path in root.rglob('*'):
        # Skip the trash directory itself
        if 'trash' in path.parts:
            continue
            
        classification = classify_file(path)
        classifications[classification].append(str(path))
    
    return classifications

def create_trash_directory(root_path):
    """
    Create the trash directory if it doesn't exist
    
    Args:
        root_path (str): Root path of the project
        
    Returns:
        Path: Path to the trash directory
    """
    trash_path = Path(root_path) / 'trash'
    trash_path.mkdir(exist_ok=True)
    return trash_path

def move_removable_files(removable_files, trash_path):
    """
    Move removable files to the trash directory
    
    Args:
        removable_files (list): List of removable file paths
        trash_path (Path): Path to the trash directory
        
    Returns:
        list: List of successfully moved files
    """
    moved_files = []
    
    for file_path in removable_files:
        try:
            path = Path(file_path)
            # Maintain directory structure in trash
            relative_path = path.relative_to(Path(file_path).parents[0])
            destination = trash_path / relative_path
            
            # Create parent directories if needed
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the file or directory
            if path.exists():
                shutil.move(str(path), str(destination))
                moved_files.append(file_path)
        except Exception as e:
            print(f"Error moving {file_path}: {e}")
    
    return moved_files

def generate_report(classifications, moved_files):
    """
    Generate a structured report of the audit results
    
    Args:
        classifications (dict): Dictionary with classified files
        moved_files (list): List of moved files
    """
    total_files = sum(len(files) for files in classifications.values())
    
    print("=" * 60)
    print("PROJECT STRUCTURE AUDIT REPORT")
    print("=" * 60)
    print(f"Total files scanned: {total_files}")
    print()
    
    print("CLASSIFICATION SUMMARY:")
    print("-" * 30)
    for category, files in classifications.items():
        print(f"{category:12}: {len(files):4} files")
    print()
    
    if moved_files:
        print("FILES MOVED TO trash/:")
        print("-" * 30)
        for file_path in moved_files:
            print(f"  - {file_path}")
        print()
    
    print("=" * 60)

def main():
    """Main function to run the audit"""
    # Get the project root (assuming script is in project root)
    project_root = Path(__file__).parent.absolute()
    
    print(f"Auditing project at: {project_root}")
    
    # Scan the project
    classifications = scan_project(project_root)
    
    # Create trash directory
    trash_path = create_trash_directory(project_root)
    
    # Move removable files
    removable_files = classifications['REMOVABLE']
    moved_files = move_removable_files(removable_files, trash_path)
    
    # Generate report
    generate_report(classifications, moved_files)
    
    print("Audit and cleaning completed successfully!")

if __name__ == "__main__":
    main()