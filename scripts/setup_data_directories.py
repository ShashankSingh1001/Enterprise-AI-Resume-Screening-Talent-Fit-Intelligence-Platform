"""
Data Directory Setup Script
Creates required folder structure for Phase 3: Data Ingestion
"""

import os
from pathlib import Path

def create_data_directories():
    """Create all required data directories"""
    
    base_dirs = [
        "data/raw/resumes",
        "data/raw/job_descriptions",
        "data/raw/labels",
        "data/processed",
        "data/features",
        "data/external",
        "data/interim",
    ]
    
    for dir_path in base_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")
    
    for dir_path in base_dirs:
        gitkeep_path = Path(dir_path) / ".gitkeep"
        gitkeep_path.touch(exist_ok=True)
    
    print("\nData directory structure created successfully!")
    print("\nNext steps:")
    print("1. Download Kaggle datasets to data/external/")
    print("2. Run data collection scripts")
    print("3. Initialize DVC tracking")

if __name__ == "__main__":
    create_data_directories()