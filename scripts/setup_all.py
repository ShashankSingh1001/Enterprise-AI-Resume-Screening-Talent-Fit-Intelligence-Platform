"""
Project Setup Verification Script
Verifies all components are properly configured
"""

import os
import sys
from pathlib import Path
import subprocess


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60)


def check_python_version():
    """Check Python version"""
    print_header("Python Version Check")
    
    version = sys.version_info
    required_major, required_minor = 3, 11
    
    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == required_major and version.minor >= required_minor:
        print(f"Python version is compatible (>= {required_major}.{required_minor})")
        return True
    else:
        print(f"Python {required_major}.{required_minor}+ is required")
        return False


def check_directory_structure():
    """Check if all required directories exist"""
    print_header("Directory Structure Check")
    
    required_dirs = [
        "src/ingestion",
        "src/nlp",
        "src/features",
        "src/similarity",
        "src/training",
        "src/inference",
        "src/explainability",
        "src/bias_audit",
        "src/mlops",
        "src/utils",
        "src/exceptions",
        "src/logging",
        "api",
        "dashboard",
        "data/raw",
        "data/processed",
        "data/features",
        "models",
        "notebooks",
        "tests",
        "logs",
        "config",
        "scripts"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(dir_path)
        else:
            print(f"{dir_path} (missing)")
            all_exist = False
    
    return all_exist


def check_config_files():
    """Check if all configuration files exist"""
    print_header("Configuration Files Check")
    
    required_files = [
        "requirements.txt",
        "setup.py",
        ".gitignore",
        ".dockerignore",
        ".env.example",
        "config/config.yaml",
        "config/__init__.py",
        "Dockerfile",
        "docker-compose.yml"
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(file_path)
        else:
            print(f"{file_path} (missing)")
            all_exist = False
    
    return all_exist


def check_environment_file():
    """Check if .env file exists"""
    print_header("Environment File Check")
    
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if env_path.exists():
        print(".env file exists")
        return True
    elif env_example_path.exists():
        print(".env file not found")
        print("Please copy .env.example to .env")
        return False
    else:
        print("Neither .env nor .env.example found")
        return False


def check_installed_packages():
    """Check if key packages are installed"""
    print_header("Required Packages Check")
    
    required_packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "spacy",
        "sentence-transformers",
        "fastapi",
        "streamlit",
        "mlflow",
        "dvc",
        "shap",
        "fairlearn"
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(package)
        except ImportError:
            print(f"{package} (not installed)")
            all_installed = False
    
    return all_installed


def check_spacy_model():
    """Check if spaCy model is downloaded"""
    print_header("spaCy Model Check")
    
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
            print("spaCy model 'en_core_web_sm' is installed")
            return True
        except OSError:
            print("spaCy model 'en_core_web_sm' not found")
            print("Install with: python -m spacy download en_core_web_sm")
            return False
    except ImportError:
        print("spaCy is not installed")
        return False


def check_docker():
    """Check if Docker is installed and running"""
    print_header("Docker Check")
    
    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
        print("Docker installed")
        
        try:
            subprocess.run(["docker", "ps"], capture_output=True, check=True)
            print("Docker daemon is running")
            return True
        except subprocess.CalledProcessError:
            print("Docker installed but daemon is not running")
            return False
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Docker is not installed")
        return False


def check_git():
    """Check if Git is initialized"""
    print_header("Git Repository Check")
    
    git_dir = Path(".git")
    if git_dir.exists():
        print("Git repository initialized")
        try:
            subprocess.run(["git", "status"], capture_output=True, check=True)
            print("Git is working")
            return True
        except subprocess.CalledProcessError:
            print("Git repository exists but may have issues")
            return False
    else:
        print("Git repository not initialized")
        return False


def check_config_loader():
    """Check if config module can be loaded"""
    print_header("Config Loader Check")
    
    try:
        from config import get_config
        app_name = get_config('app.name')
        print("Config loader working")
        print(f"App name: {app_name}")
        return True
    except Exception as e:
        print(f"Config loader failed: {e}")
        return False


def print_summary(results):
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    for check_name, status in results.items():
        print(f"{check_name}: {'OK' if status else 'FAILED'}")
    
    print(f"\nTotal: {passed_checks}/{total_checks} checks passed")
    return passed_checks == total_checks


def main():
    print("\n" + "="*60)
    print("Resume AI - Project Setup Verification")
    print("="*60)
    
    results = {}
    results["Python Version"] = check_python_version()
    results["Directory Structure"] = check_directory_structure()
    results["Configuration Files"] = check_config_files()
    results["Environment File"] = check_environment_file()
    results["Installed Packages"] = check_installed_packages()
    results["spaCy Model"] = check_spacy_model()
    results["Docker"] = check_docker()
    results["Git Repository"] = check_git()
    results["Config Loader"] = check_config_loader()
    
    return print_summary(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
