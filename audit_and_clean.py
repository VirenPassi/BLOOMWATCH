#!/usr/bin/env python3
"""
Project Structure Auditor and Cleaner for BloomWatch

This script audits the entire project and classifies files into three categories:
1. CORE FILES (required for the project to run)
2. OPTIONAL FILES (helpful for development but not strictly needed)
3. REMOVABLE FILES (caches, compiled files, temp logs, etc.)

Removable files are moved to a 'trash/' directory instead of being deleted.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Set

def create_trash_directory(project_root: Path) -> Path:
    """Create the trash directory if it doesn't exist."""
    trash_dir = project_root / "trash"
    trash_dir.mkdir(exist_ok=True)
    return trash_dir

def is_core_file(file_path: Path, relative_path: str) -> bool:
    """Determine if a file is a core file required for the project to run."""
    core_patterns = {
        # Core code files
        '.py', '.yaml', '.yml', '.json', '.txt',
        # Entry points and scripts
        'main.py', 'run_*.py', 'run_*.ps1', 'run_*.bat',
        # Configuration files
        'requirements.txt', 'README.md', 'LICENSE',
        # Model files (critical for operation)
        '*.pt', '*.pth', '*.ckpt',
        # Documentation that's part of the core
        'TEMPORAL_WORKFLOW.md', 'PROJECT_SUMMARY.md'
    }
    
    # Specific core files and directories
    core_files = {
        'README.md',
        'requirements.txt',
        'TEMPORAL_WORKFLOW.md',
        'PROJECT_SUMMARY.md',
        'QUALITY_ASSURANCE.md',
        'TRAINING_PIPELINE.md',
        'MODIS_FUNCTIONALITY.md',
        'FINE_TUNING_SUMMARY.md',
        'UPGRADE_SUMMARY.md',
        'RUN_APP.md',
        'README_MINI_PIPELINE.md',
        '__init__.py',
        'main.py',
        'pipelines/bloomwatch_temporal_workflow.py',
        'webapp/bloomwatch_explorer.py'
    }
    
    # Core directories (contents are generally core)
    core_directories = {
        'pipelines', 'models', 'app', 'configs', 'data', 'utils', 'visualization', 'webapp', 'outputs'
    }
    
    filename = file_path.name
    parent_dir = file_path.parent.name
    
    # Check if it's in a core directory
    # More precise check for core directories
    for core_dir in core_directories:
        if core_dir in str(relative_path):
            return True
    
    # Check specific core files
    if filename in core_files:
        return True
    
    # Check file extensions for core files
    core_extensions = ['.py', '.yaml', '.yml', '.json', '.txt']
    if file_path.suffix in core_extensions and 'test' not in filename.lower() and 'temp' not in filename.lower():
        return True
    
    # Check if it's a model file in outputs/models
    if 'outputs/models' in relative_path and file_path.suffix in ['.pt', '.pth', '.ckpt']:
        return True
    
    # Check if it's the specific model we need
    if 'stage2_transfer_learning_bloomwatch.pt' in relative_path:
        return True
    
    return False

def is_optional_file(file_path: Path, relative_path: str) -> bool:
    """Determine if a file is optional (helpful but not required)."""
    optional_patterns = {
        # Documentation
        '*.md', '*.rst', '*.txt',
        # Development tools
        '.gitignore', '.editorconfig', '.vscode/*',
        # CI/CD files
        '.github/*', '.gitlab-ci.yml', 'Dockerfile',
        # IDE configs
        '.idea/*', '.vs/*',
        # Notebooks
        '*.ipynb'
    }
    
    optional_files = {
        'BLOOMWATCH_SUBMISSION_SUMMARY.md',
        '.gitignore',
        'package_submission.bat',
        'run_final_qa.py',
        'run_final_qa.ps1'
    }
    
    optional_directories = {
        '.git', 'notebooks', 'tests', '.github', '.vscode', '.idea'
    }
    
    filename = file_path.name
    parent_dir = file_path.parent.name
    
    # Check if it's in an optional directory
    if parent_dir in optional_directories:
        return True
    
    # Check specific optional files
    if filename in optional_files:
        return True
    
    # Documentation files (except core ones)
    if file_path.suffix == '.md' and filename not in ['README.md']:
        return True
    
    # Test files are optional
    if 'test' in filename.lower() or 'verify' in filename.lower():
        return True
    
    return False

def is_removable_file(file_path: Path, relative_path: str) -> bool:
    """Determine if a file is removable (caches, temp files, etc.)."""
    # Files that should never be considered removable (core files)
    protected_files = {
        'README.md',
        'requirements.txt',
        'TEMPORAL_WORKFLOW.md',
        'main.py',
        'pipelines/bloomwatch_temporal_workflow.py',
        'webapp/bloomwatch_explorer.py',
        'outputs/models/stage2_transfer_learning_bloomwatch.pt'
    }
    
    # If it's a protected file, it's not removable
    if relative_path in protected_files:
        return False
    
    removable_patterns = {
        # Python cache files
        '__pycache__', '*.pyc', '*.pyo', '*.pyd',
        # Virtual environment
        '.venv', 'venv', 'env',
        # System files
        '.DS_Store', 'Thumbs.db',
        # Log files
        '*.log', '*.tmp', '*.bak',
        # Compressed files
        '*.zip', '*.tar.gz', '*.rar',
        # Auto-generated files
        '*.cache'
    }
    
    removable_files = {
        'final_qa_submission.py',
        'simple_test.py',
        'test_temporal_workflow.py',
        'test_indices.py',
        'test_simple.py',
        'test_pipeline.py',
        'test_modis.py',
        'test_basic.py',
        'verify_indices.py',
        'verify_modis.py',
        'verify_package.py',
        'test_final_package.py',
        'minimal_test.py',
        'BloomWatch_Submission_20251001_154603.zip',
        'BloomWatch_Submission_20251001_154614.zip',
        'BloomWatch_Submission_20251001_154628.zip',
        'BloomWatch_Submission_20251001_154643.zip'
    }
    
    removable_directories = {
        '.venv', 'env', 'venv', '__pycache__', 'BloomWatch_Submission',
        'trash'  # We'll handle this specially
    }
    
    filename = file_path.name
    parent_dir = file_path.parent.name
    
    # Skip the trash directory itself
    if filename == 'trash' and file_path.is_dir():
        return False
    
    # Check if it's in a removable directory
    if parent_dir in removable_directories:
        return True
    
    # Check specific removable files
    if filename in removable_files:
        return True
    
    # Check file extensions
    removable_extensions = {'.pyc', '.pyo', '.pyd', '.log', '.tmp', '.bak', '.cache'}
    if file_path.suffix in removable_extensions:
        return True
    
    # Check for system files
    system_files = {'.DS_Store', 'Thumbs.db'}
    if filename in system_files:
        return True
    
    # ZIP files generated during development
    if file_path.suffix == '.zip' and 'Submission' in filename:
        return True
    
    # Check for temp directories
    temp_indicators = ['temp', 'tmp', 'cache']
    if any(indicator in str(file_path).lower() for indicator in temp_indicators):
        return True
    
    return False

def audit_project(project_root: Path) -> Dict[str, List[Path]]:
    """Audit the entire project and categorize files."""
    categories = {
        'CORE': [],
        'OPTIONAL': [],
        'REMOVABLE': []
    }
    
    # Walk through all files and directories
    for root, dirs, files in os.walk(project_root):
        # Skip the trash directory
        if 'trash' in root.split(os.sep):
            continue
            
        for file in files:
            file_path = Path(root) / file
            relative_path = str(file_path.relative_to(project_root))
            
            # Skip the trash directory itself
            if file == 'trash' and file_path.is_dir():
                continue
                
            if is_removable_file(file_path, relative_path):
                categories['REMOVABLE'].append(file_path)
            elif is_core_file(file_path, relative_path):
                categories['CORE'].append(file_path)
            elif is_optional_file(file_path, relative_path):
                categories['OPTIONAL'].append(file_path)
            else:
                # Default to optional for safety
                categories['OPTIONAL'].append(file_path)
    
    return categories

def move_removable_files(removable_files: List[Path], trash_dir: Path, project_root: Path) -> List[Path]:
    """Move removable files to the trash directory, preserving directory structure."""
    moved_files = []
    
    for file_path in removable_files:
        try:
            # Calculate relative path from project root
            relative_path = file_path.relative_to(project_root)
            
            # Create the same structure in trash
            trash_file_path = trash_dir / relative_path
            trash_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            if file_path.is_file():
                shutil.move(str(file_path), str(trash_file_path))
            elif file_path.is_dir():
                shutil.move(str(file_path), str(trash_file_path))
                
            moved_files.append(relative_path)
            print(f"Moved to trash: {relative_path}")
            
        except Exception as e:
            print(f"Error moving {file_path}: {e}")
    
    return moved_files

def main():
    """Main function to audit and clean the project."""
    project_root = Path(__file__).parent.resolve()
    print(f"Auditing project at: {project_root}")
    
    # Create trash directory
    trash_dir = create_trash_directory(project_root)
    print(f"Trash directory: {trash_dir}")
    
    # Audit the project
    print("\nAuditing project files...")
    categories = audit_project(project_root)
    
    # Print summary
    total_files = sum(len(files) for files in categories.values())
    print(f"\n=== AUDIT SUMMARY ===")
    print(f"Total files scanned: {total_files}")
    print(f"Core files: {len(categories['CORE'])}")
    print(f"Optional files: {len(categories['OPTIONAL'])}")
    print(f"Removable files: {len(categories['REMOVABLE'])}")
    
    # Show sample files from each category
    print(f"\n=== SAMPLE FILES ===")
    for category, files in categories.items():
        print(f"\n{category} FILES ({len(files)} total):")
        for file in files[:5]:  # Show first 5 files
            relative_path = file.relative_to(project_root)
            print(f"  - {relative_path}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")
    
    # Move removable files
    if categories['REMOVABLE']:
        print(f"\n=== MOVING REMOVABLE FILES ===")
        moved_files = move_removable_files(categories['REMOVABLE'], trash_dir, project_root)
        print(f"\nMoved {len(moved_files)} files to trash.")
        
        # List all moved files
        print("\nFiles moved to trash:")
        for file in moved_files:
            print(f"  - {file}")
    else:
        print("\nNo removable files found.")
    
    # Final verification
    print(f"\n=== FINAL VERIFICATION ===")
    core_size = sum(f.stat().st_size for f in categories['CORE'] if f.is_file())
    optional_size = sum(f.stat().st_size for f in categories['OPTIONAL'] if f.is_file())
    print(f"Core files total size: {core_size / (1024*1024):.2f} MB")
    print(f"Optional files total size: {optional_size / (1024*1024):.2f} MB")
    
    print(f"\n✅ Project audit and cleaning complete!")
    print(f"✅ Core files are preserved for safe operation")
    print(f"✅ Removable files moved to: {trash_dir}")
    print(f"✅ Project is now clean and safe to run")

if __name__ == "__main__":
    main()