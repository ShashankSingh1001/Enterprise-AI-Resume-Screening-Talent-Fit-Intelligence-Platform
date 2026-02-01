"""
Kaggle Dataset Downloader Utility

This module provides functionality to download datasets from Kaggle using the Kaggle API.
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from dotenv import load_dotenv

from src.logging import get_logger
from src.exceptions import FileProcessingError

logger = get_logger(__name__)

load_dotenv()


class KaggleDownloader:
    """
    Handles downloading and extracting datasets from Kaggle.
    
    Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables.
    """
    
    def __init__(self, data_dir: str = "data/external"):
        """
        Initialize Kaggle downloader.
        
        Args:
            data_dir: Directory to save downloaded datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.username = os.getenv("KAGGLE_USERNAME")
        self.api_key = os.getenv("KAGGLE_KEY")
        
    def _verify_credentials(self) -> bool:
        """
        Verify Kaggle credentials are available.
        
        Returns:
            bool: True if credentials exist
            
        Raises:
            FileProcessingError: If credentials are missing
        """
        if not self.username or not self.api_key:
            raise FileProcessingError(
                "Kaggle credentials not found in .env file. "
                "Please set KAGGLE_USERNAME and KAGGLE_KEY",
                sys.exc_info()
            )
        
        # Set environment variables for subprocess
        os.environ["KAGGLE_USERNAME"] = self.username
        os.environ["KAGGLE_KEY"] = self.api_key
        
        logger.info("Kaggle credentials verified")
        return True
    
    def _check_kaggle_cli(self) -> bool:
        """
        Check if Kaggle CLI is installed and accessible.
        
        Returns:
            bool: True if CLI is available
        """
        try:
            result = subprocess.run(
                ["kaggle", "--version"],
                capture_output=True,
                text=True,
                env=os.environ.copy()
            )
            
            if result.returncode == 0:
                logger.info(f"Kaggle CLI available: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"Kaggle CLI error: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.error("Kaggle CLI not found. Install with: pip install kaggle")
            return False
        except Exception as e:
            logger.error(f"Error checking Kaggle CLI: {str(e)}")
            return False
    
    def download_dataset_cli(
        self, 
        dataset_slug: str, 
        download_path: Optional[Path] = None
    ) -> Tuple[bool, str]:
        """
        Download a Kaggle dataset using CLI (subprocess method).
        
        Args:
            dataset_slug: Kaggle dataset identifier (e.g., 'username/dataset-name')
            download_path: Optional custom download path
            
        Returns:
            Tuple[bool, str]: (Success status, Message or error)
        """
        try:
            download_path = download_path or self.data_dir
            download_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading dataset via CLI: {dataset_slug}")
            
            # Prepare environment with credentials
            env = os.environ.copy()
            env["KAGGLE_USERNAME"] = self.username
            env["KAGGLE_KEY"] = self.api_key
            
            # Run kaggle download command
            cmd = [
                "kaggle", 
                "datasets", 
                "download", 
                "-d", dataset_slug, 
                "-p", str(download_path),
                "--unzip"  # Auto-unzip after download
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully downloaded: {dataset_slug}")
                return True, f"Downloaded {dataset_slug}"
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                logger.error(f"Download failed: {error_msg}")
                return False, f"Download failed: {error_msg}"
                
        except FileNotFoundError:
            error_msg = "Kaggle CLI not found. Install with: pip install kaggle"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Download error: {error_msg}")
            return False, f"Download error: {error_msg}"
    
    def download_dataset_api(
        self, 
        dataset_slug: str, 
        download_path: Optional[Path] = None
    ) -> Tuple[bool, str]:
        """
        Download a Kaggle dataset using Python API (fallback method).
        
        Args:
            dataset_slug: Kaggle dataset identifier
            download_path: Optional custom download path
            
        Returns:
            Tuple[bool, str]: (Success status, Message or error)
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            download_path = download_path or self.data_dir
            download_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading dataset via API: {dataset_slug}")
            
            # Initialize and authenticate API
            api = KaggleApi()
            api.authenticate()
            
            # Download and unzip
            api.dataset_download_files(
                dataset_slug,
                path=str(download_path),
                unzip=True,
                quiet=False
            )
            
            logger.info(f"Successfully downloaded: {dataset_slug}")
            return True, f"Downloaded {dataset_slug}"
            
        except ImportError:
            error_msg = "Kaggle package not installed. Run: pip install kaggle"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = str(e)
            logger.error(f"API download error: {error_msg}")
            return False, f"API error: {error_msg}"
    
    def download_dataset(
        self, 
        dataset_slug: str, 
        download_path: Optional[Path] = None
    ) -> Tuple[bool, str]:
        """
        Download a Kaggle dataset using best available method.
        Tries CLI first, then falls back to Python API.
        
        Args:
            dataset_slug: Kaggle dataset identifier
            download_path: Optional custom download path
            
        Returns:
            Tuple[bool, str]: (Success status, Message or error)
        """
        # Try CLI method first (more reliable)
        if self._check_kaggle_cli():
            success, message = self.download_dataset_cli(dataset_slug, download_path)
            if success:
                return True, message
            else:
                logger.warning(f"CLI download failed, trying API method...")
        
        # Fallback to API method
        return self.download_dataset_api(dataset_slug, download_path)
    
    def download_resume_datasets(self) -> Tuple[bool, str]:
        """
        Download both resume and JD datasets for Phase 3.
        
        Returns:
            Tuple[bool, str]: Success status and message
        """
        try:
            self._verify_credentials()
            
            logger.info("Downloading required datasets for Phase 3...")
            
            # # Dataset 1: Resume Dataset
            # resume_dataset = "gauravduttakiit/resume-dataset"
            # logger.info(f"[1/2] Downloading resume dataset...")
            
            # success1, msg1 = self.download_dataset(resume_dataset, self.data_dir)
            
            # if not success1:
            #     logger.error(f"Resume dataset failed: {msg1}")
            #     return False, f"Resume dataset download failed: {msg1}"
            
            # Dataset 2: JD Dataset
            jd_dataset = "ravindrasinghrana/job-description-dataset"
            logger.info(f"[2/2] Downloading JD dataset...")
            
            success2, msg2 = self.download_dataset(jd_dataset, self.data_dir)
            
            if not success2:
                logger.error(f"JD dataset failed: {msg2}")
                return False, f"JD dataset download failed: {msg2}"
            
            logger.info("All datasets downloaded and extracted successfully")
            return True, "Both datasets downloaded successfully"
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Download failed: {error_msg}")
            return False, f"Download failed: {error_msg}"
    
    def verify_datasets(self) -> Tuple[bool, int]:
        """
        Verify that datasets have been downloaded.
        
        Returns:
            Tuple[bool, int]: (All datasets present, Number of CSV files found)
        """
        csv_files = list(self.data_dir.glob("*.csv"))
        num_files = len(csv_files)
        
        if num_files < 2:
            logger.warning(f"Found only {num_files} CSV files, expected 2+")
            return False, num_files
        
        logger.info(f"Found {num_files} dataset files")
        for csv_file in csv_files:
            logger.info(f"  - {csv_file.name}")
        
        return True, num_files


def download_kaggle_datasets() -> bool:
    """
    Convenience function to download all required datasets.
    
    Returns:
        bool: True if successful
    """
    try:
        downloader = KaggleDownloader()
        success, message = downloader.download_resume_datasets()
        
        if success:
            logger.info(message)
            return True
        else:
            logger.error(message)
            return False
            
    except Exception as e:
        logger.error(f"Failed to download datasets: {str(e)}")
        return False