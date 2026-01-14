"""
DVC Initialization Script 
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run command with user feedback"""
    print(f"{description}...", end=" ")
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True)
        print("OK")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n   Error: {e.stderr.decode() if e.stderr else 'Command failed'}")
        return False


def init_dvc():
    """Initialize DVC with feedback"""
    print("\n" + "="*50)
    print("Initializing DVC")
    print("="*50 + "\n")
    
    try:
        subprocess.run(["dvc", "--version"], check=True, capture_output=True)
        print("DVC already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        if not run_command("pip install dvc", "Installing DVC"):
            return False
    
    if not Path(".dvc").exists():
        if not run_command("dvc init", "Initializing DVC repository"):
            return False
    else:
        print("DVC already initialized")
    
    print("\nCreating data directories:")
    for d in ["data/raw", "data/processed", "data/features"]:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"   {d}")
    
    dvcignore = Path(".dvcignore")
    if not dvcignore.exists():
        dvcignore.write_text(
            "*.pyc\n__pycache__/\n.git/\n.pytest_cache/\n*.log\n"
        )
        print("Created .dvcignore")
    
    print("\n" + "="*50)
    print("DVC Initialization Complete")
    print("="*50)
    print("\nNext steps:")
    print("   1. Add data: dvc add data/raw/dataset.csv")
    print("   2. Track: git add data/raw/dataset.csv.dvc")
    print("   3. Commit: git commit -m 'Add dataset'")
    
    return True


if __name__ == "__main__":
    sys.exit(0 if init_dvc() else 1)
