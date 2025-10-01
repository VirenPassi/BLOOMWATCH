#!/usr/bin/env python3
"""
Run the final QA, reporting, cleanup, and submission process for BloomWatch.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the final QA process."""
    # Change to the project directory
    project_dir = Path(__file__).parent
    print(f"Running final QA process in: {project_dir}")
    
    # Run the final QA script
    try:
        result = subprocess.run([
            sys.executable, 
            str(project_dir / "final_qa_submission.py")
        ], cwd=project_dir, check=True)
        
        print("\n✅ Final QA process completed successfully!")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Final QA process failed with exit code {e.returncode}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error running final QA process: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())