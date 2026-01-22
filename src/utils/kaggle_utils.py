"""
Kaggle Dataset Download Utilities
Handles automated download and extraction of Kaggle datasets
"""

import os
import sys
import zipfile
from pathlib import Path
from typing import Optional, List, Tuple
import subprocess

from src.logging import get_logger
from src.exceptions import FileProcessingError

logger = get_logger(__name__)


class KaggleDownloader:
    """
    Automates Kaggle dataset download and extraction
    """
    
    def __init__(self, output_dir: str = 'data/external'):
        """
        Initialize Kaggle downloader
        
        Args:
            output_dir: Directory to save downloaded datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._verify_credentials()
    
    def _verify_credentials(self) -> bool:
        """
        Verify Kaggle API credentials are set
        
        Returns:
            True if credentials found
            
        Raises:
            FileProcessingError: If credentials missing
        """
        username = os.getenv('KAGGLE_USERNAME')
        key = os.getenv('KAGGLE_KEY')
        
        if not username or not key:
            error_msg = (
                "Kaggle credentials not found in environment variables.\n"
                "Please set KAGGLE_USERNAME and KAGGLE_KEY in .env file.\n"
                "Get credentials from: https://www.kaggle.com/settings -> API"
            )
            logger.error(error_msg)
            raise FileProcessingError(error_msg, error_detail=sys.exc_info())
        
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = key
        
        logger.info("Kaggle credentials verified")
        return True
    
    def _check_kaggle_cli(self) -> bool:
        """
        Check if Kaggle CLI is installed
        
        Returns:
            True if installed
        """
        try:
            result = subprocess.run(
                ['kaggle', '--version'],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                logger.info(f"Kaggle CLI version: {result.stdout.strip()}")
                return True
            else:
                return False
        except FileNotFoundError:
            return False
    
    def download_dataset(
        self,
        dataset_slug: str,
        force_download: bool = False
    ) -> Optional[Path]:
        """
        Download a Kaggle dataset
        
        Args:
            dataset_slug: Dataset identifier (e.g., 'username/dataset-name')
            force_download: Re-download even if exists
            
        Returns:
            Path to downloaded file or None if failed
        """
        logger.info(f"Downloading dataset: {dataset_slug}")
        
        if not self._check_kaggle_cli():
            error_msg = (
                "Kaggle CLI not found. Install with: pip install kaggle"
            )
            logger.error(error_msg)
            raise FileProcessingError(error_msg, error_detail=sys.exc_info())
        
        dataset_name = dataset_slug.split('/')[-1]
        zip_file = self.output_dir / f"{dataset_name}.zip"
        
        if zip_file.exists() and not force_download:
            logger.info(f"Dataset already exists: {zip_file}")
            return zip_file
        
        try:
            cmd = [
                'kaggle', 'datasets', 'download',
                '-d', dataset_slug,
                '-p', str(self.output_dir),
                '--force' if force_download else '--quiet'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Download completed: {dataset_slug}")
            logger.debug(f"Output: {result.stdout}")
            
            return zip_file
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed: {e.stderr}")
            raise FileProcessingError(f"Kaggle download failed: {e.stderr}", error_detail=sys.exc_info())
    
    def extract_dataset(
        self,
        zip_file: Path,
        extract_to: Optional[Path] = None,
        remove_zip: bool = True
    ) -> List[Path]:
        """
        Extract downloaded zip file
        
        Args:
            zip_file: Path to zip file
            extract_to: Extraction directory (default: same as zip)
            remove_zip: Delete zip after extraction
            
        Returns:
            List of extracted file paths
        """
        if not zip_file.exists():
            raise FileProcessingError(f"Zip file not found: {zip_file}", error_detail=sys.exc_info())
        
        if extract_to is None:
            extract_to = zip_file.parent
        
        extract_to.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting: {zip_file.name}")
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                extracted_files = [
                    extract_to / name for name in zip_ref.namelist()
                ]
            
            logger.info(f"Extracted {len(extracted_files)} files to {extract_to}")
            
            if remove_zip:
                zip_file.unlink()
                logger.info(f"Removed zip file: {zip_file.name}")
            
            return extracted_files
            
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid zip file: {e}")
            raise FileProcessingError(f"Zip extraction failed: {e}", error_detail=sys.exc_info())
    
    def download_and_extract(
        self,
        dataset_slug: str,
        force_download: bool = False,
        remove_zip: bool = True
    ) -> List[Path]:
        """
        Download and extract in one step
        
        Args:
            dataset_slug: Dataset identifier
            force_download: Re-download even if exists
            remove_zip: Delete zip after extraction
            
        Returns:
            List of extracted file paths
        """
        zip_file = self.download_dataset(dataset_slug, force_download)
        
        if zip_file:
            return self.extract_dataset(zip_file, remove_zip=remove_zip)
        
        return []
    
    def download_resume_datasets(
        self,
        force_download: bool = False
    ) -> Tuple[List[Path], List[Path]]:
        """
        Download both required datasets for Phase 3
        
        Args:
            force_download: Re-download even if exists
            
        Returns:
            Tuple of (resume_files, jd_files)
        """
        logger.info("Downloading required datasets for Phase 3...")
        
        resume_dataset = 'gauravduttakiit/resume-dataset'
        jd_dataset = 'ravindrasinghrana/job-description-dataset'
        
        resume_files = self.download_and_extract(
            resume_dataset,
            force_download=force_download
        )
        
        jd_files = self.download_and_extract(
            jd_dataset,
            force_download=force_download
        )
        
        logger.info(f"Resume dataset: {len(resume_files)} files")
        logger.info(f"JD dataset: {len(jd_files)} files")
        
        return resume_files, jd_files
    
    def verify_datasets(self) -> bool:
        """
        Verify required datasets exist
        
        Returns:
            True if all datasets found
        """
        csv_files = list(self.output_dir.glob('*.csv'))
        
        if len(csv_files) < 2:
            logger.warning(f"Found only {len(csv_files)} CSV files, expected 2+")
            return False
        
        logger.info(f"Found {len(csv_files)} dataset files")
        for csv_file in csv_files:
            logger.info(f"  - {csv_file.name}")
        
        return True


def main():
    """Download datasets for Phase 3"""
    from dotenv import load_dotenv
    
    load_dotenv()
    
    downloader = KaggleDownloader()
    
    try:
        resume_files, jd_files = downloader.download_resume_datasets()
        
        print("\nDownload Summary:")
        print(f"Resume files: {len(resume_files)}")
        print(f"JD files: {len(jd_files)}")
        
        if downloader.verify_datasets():
            print("\nAll datasets downloaded successfully!")
        else:
            print("\nWarning: Some datasets may be missing")
            
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print("1. Kaggle credentials are set in .env file")
        print("2. Kaggle CLI is installed: pip install kaggle")
        print("3. You have internet connection")


if __name__ == "__main__":
    main()