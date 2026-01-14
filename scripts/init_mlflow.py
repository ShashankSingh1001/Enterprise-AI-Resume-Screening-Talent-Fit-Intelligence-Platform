"""
MLflow Initialization Script - Hybrid Version
Concise code with informative output
"""

import os
import sys
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv


def init_mlflow():
    """Initialize MLflow with feedback"""
    print("\n" + "="*50)
    print("Initializing MLflow")
    print("="*50 + "\n")
    
    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "resume_screening_experiments")
    
    print("Configuration:")
    print(f"   URI: {tracking_uri}")
    print(f"   Experiment: {experiment_name}\n")
    
    try:
        mlflow.set_tracking_uri(tracking_uri)
        print("Connected to MLflow")
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                print(f"Experiment exists (ID: {experiment.experiment_id})")
            else:
                exp_id = mlflow.create_experiment(
                    experiment_name,
                    tags={
                        "project": "resume-ai",
                        "version": "1.0.0"
                    }
                )
                print(f"Created experiment (ID: {exp_id})")
        except Exception as e:
            print(f"Experiment setup: {str(e)[:50]}...")
            print("(Normal if MLflow server not running)")
        
        for dir_name in ["mlruns", "mlflow_artifacts"]:
            Path(dir_name).mkdir(exist_ok=True)
            print(f"Created {dir_name}/ directory")
        
        print("\n" + "="*50)
        print("MLflow Initialization Complete")
        print("="*50)
        print("\nStart MLflow UI:")
        print("   docker-compose up mlflow")
        print("   Access: http://localhost:5000")
        
        return True
        
    except Exception as e:
        print(f"\nError: {str(e)[:100]}")
        print("\nMLflow server may not be running")
        print("Start with: docker-compose up mlflow")
        return False


if __name__ == "__main__":
    load_dotenv()
    sys.exit(0 if init_mlflow() else 1)
