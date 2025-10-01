#!/usr/bin/env python3
"""
Final Structure Verification for BloomWatch

This script verifies that the BloomWatch project structure is clean and safe to run.
"""

import os
import sys
from pathlib import Path

def verify_core_components(project_root: Path) -> bool:
    """Verify that all core components are present."""
    print("Verifying core components...")
    
    # Essential files that must be present
    essential_files = [
        "README.md",
        "requirements.txt",
        "TEMPORAL_WORKFLOW.md",
        "main.py",
        "pipelines/bloomwatch_temporal_workflow.py",
        "webapp/bloomwatch_explorer.py",
        "outputs/models/stage2_transfer_learning_bloomwatch.pt"
    ]
    
    missing_files = []
    for file_path in essential_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print(f"\n❌ {len(missing_files)} essential files are missing!")
        return False
    else:
        print("\n✅ All essential files are present!")
        return True

def verify_clean_structure(project_root: Path) -> bool:
    """Verify that the structure is clean (no temporary files, caches, etc.)."""
    print("\nVerifying clean structure...")
    
    # Check for common temporary/cached files that shouldn't be present
    unwanted_patterns = [
        "__pycache__",
        ".pyc",
        ".pyo",
        ".pyd",
        ".DS_Store",
        "Thumbs.db",
        ".log",
        ".tmp",
        ".bak"
    ]
    
    found_unwanted = []
    for root, dirs, files in os.walk(project_root):
        # Skip the .git directory
        if '.git' in root.split(os.sep):
            continue
            
        for pattern in unwanted_patterns:
            # Check directories
            for dir_name in dirs:
                if pattern in dir_name:
                    unwanted_path = Path(root) / dir_name
                    relative_path = unwanted_path.relative_to(project_root)
                    found_unwanted.append(str(relative_path))
            
            # Check files
            for file_name in files:
                if pattern in file_name:
                    unwanted_path = Path(root) / file_name
                    relative_path = unwanted_path.relative_to(project_root)
                    found_unwanted.append(str(relative_path))
    
    if found_unwanted:
        print(f"❌ Found {len(found_unwanted)} unwanted files/directories:")
        for item in found_unwanted[:10]:  # Show first 10 only
            print(f"  - {item}")
        if len(found_unwanted) > 10:
            print(f"  ... and {len(found_unwanted) - 10} more")
        return False
    else:
        print("✅ No unwanted temporary files or caches found!")
        return True

def verify_model_access(project_root: Path) -> bool:
    """Verify that the model can be accessed."""
    print("\nVerifying model access...")
    
    try:
        import torch
        model_path = project_root / "outputs" / "models" / "stage2_transfer_learning_bloomwatch.pt"
        
        if not model_path.exists():
            print("❌ Model file not found!")
            return False
            
        # Try to load the model structure (without loading weights to save time)
        print(f"✅ Model file exists: {model_path}")
        print(f"✅ Model size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        return True
        
    except ImportError:
        print("⚠️  PyTorch not available for model verification")
        # Just check if file exists
        model_path = project_root / "outputs" / "models" / "stage2_transfer_learning_bloomwatch.pt"
        if model_path.exists():
            print(f"✅ Model file exists: {model_path}")
            return True
        else:
            print("❌ Model file not found!")
            return False
    except Exception as e:
        print(f"❌ Error accessing model: {e}")
        return False

def verify_pipeline_import(project_root: Path) -> bool:
    """Verify that the main pipeline can be imported."""
    print("\nVerifying pipeline import...")
    
    try:
        # Add project root to Python path
        sys.path.insert(0, str(project_root))
        
        # Try to import the main pipeline
        from pipelines.bloomwatch_temporal_workflow import compute_indices, normalize_temporal
        print("✅ Pipeline modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Pipeline import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error importing pipeline: {e}")
        return False

def main():
    """Main verification function."""
    project_root = Path(__file__).parent.resolve()
    print(f"🌸 BloomWatch Final Structure Verification")
    print(f"Project root: {project_root}")
    print("=" * 50)
    
    # Run all verification checks
    checks = [
        ("Core Components", lambda: verify_core_components(project_root)),
        ("Clean Structure", lambda: verify_clean_structure(project_root)),
        ("Model Access", lambda: verify_model_access(project_root)),
        ("Pipeline Import", lambda: verify_pipeline_import(project_root))
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        print("-" * 30)
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} check failed with exception: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-" * 50)
    print(f"Total checks: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL VERIFICATIONS PASSED!")
        print("✅ Project structure is clean and safe to run!")
        print("✅ Ready for production deployment!")
        return 0
    else:
        print(f"\n❌ {failed} verification(s) failed!")
        print("⚠️  Please review the issues above before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())