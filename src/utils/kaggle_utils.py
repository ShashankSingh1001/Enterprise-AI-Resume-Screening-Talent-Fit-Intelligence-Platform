"""
Kaggle Dataset Download Utilities
Handles automated download and extraction of Kaggle datasets
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

from src.logging import get_logger
from src.exceptions import FileProcessingError

logger = get_logger(__name__)


class KaggleDownloader:
    """
    Automates Kaggle dataset download and extraction
    """

    def __init__(self, output_dir: str = "data/external"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._verify_credentials()

    # =====================================================
    # CREDENTIALS & ENVIRONMENT
    # =====================================================

    def _verify_credentials(self) -> bool:
        """
        Verify Kaggle API credentials are set
        """
        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")

        if not username or not key:
            error_msg = (
                "Kaggle credentials not found.\n"
                "Set KAGGLE_USERNAME and KAGGLE_KEY in environment variables.\n"
                "Get them from: https://www.kaggle.com/settings -> API"
            )
            logger.error(error_msg)
            raise FileProcessingError(
                message=error_msg,
                error_detail=None
            )

        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key

        logger.info("Kaggle credentials verified")
        return True

    def _check_kaggle_cli(self) -> bool:
        """
        Check if Kaggle CLI is installed
        """
        try:
            result = subprocess.run(
                ["kaggle", "--version"],
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    # =====================================================
    # DOWNLOAD
    # =====================================================

    def download_dataset(
        self,
        dataset_slug: str,
        force_download: bool = False
    ) -> Optional[Path]:
        """
        Download a Kaggle dataset
        """
        logger.info(f"Downloading dataset: {dataset_slug}")

        if not self._check_kaggle_cli():
            raise FileProcessingError(
                message="Kaggle CLI not found. Install with: pip install kaggle",
                error_detail=None
            )

        dataset_name = dataset_slug.split("/")[-1]
        zip_file = self.output_dir / f"{dataset_name}.zip"

        if zip_file.exists() and not force_download:
            logger.info(f"Dataset already exists: {zip_file}")
            return zip_file

        cmd = [
            "kaggle", "datasets", "download",
            "-d", dataset_slug,
            "-p", str(self.output_dir)
        ]

        if force_download:
            cmd.append("--force")
        else:
            cmd.append("--quiet")

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Download completed: {dataset_slug}")
            return zip_file

        except subprocess.CalledProcessError as e:
            logger.error(f"Kaggle download failed: {e.stderr}")
            raise FileProcessingError(
                message=f"Kaggle download failed: {e.stderr}",
                error_detail=sys.exc_info()
            )

    # =====================================================
    # EXTRACTION
    # =====================================================

    def extract_dataset(
        self,
        zip_file: Path,
        extract_to: Optional[Path] = None,
        remove_zip: bool = True
    ) -> List[Path]:
        """
        Extract downloaded zip file
        """
        if not zip_file.exists():
            raise FileProcessingError(
                message=f"Zip file not found: {zip_file}",
                error_detail=None
            )

        extract_to = extract_to or zip_file.parent
        extract_to.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting dataset: {zip_file.name}")

        try:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(extract_to)
                extracted_files = [
                    extract_to / name for name in zip_ref.namelist()
                ]

            logger.info(f"Extracted {len(extracted_files)} files")

            if remove_zip:
                zip_file.unlink()
                logger.info(f"Removed zip file: {zip_file.name}")

            return extracted_files

        except zipfile.BadZipFile as e:
            logger.error(f"Invalid zip file: {e}")
            raise FileProcessingError(
                message=f"Zip extraction failed: {e}",
                error_detail=sys.exc_info()
            )

    # =====================================================
    # PIPELINE HELPERS
    # =====================================================

    def download_and_extract(
        self,
        dataset_slug: str,
        force_download: bool = False,
        remove_zip: bool = True
    ) -> List[Path]:
        """
        Download and extract dataset
        """
        zip_file = self.download_dataset(dataset_slug, force_download)
        return self.extract_dataset(zip_file, remove_zip=remove_zip) if zip_file else []

    def download_resume_datasets(
        self,
        force_download: bool = False
    ) -> Tuple[List[Path], List[Path]]:
        """
        Download resume + JD datasets
        """
        logger.info("Downloading Phase-3 datasets")

        resume_dataset = "gauravduttakiit/resume-dataset"
        jd_dataset = "ravindrasinghrana/job-description-dataset"

        resume_files = self.download_and_extract(resume_dataset, force_download)
        jd_files = self.download_and_extract(jd_dataset, force_download)

        logger.info(f"Resume files: {len(resume_files)}")
        logger.info(f"JD files: {len(jd_files)}")

        return resume_files, jd_files

    def verify_datasets(self) -> bool:
        """
        Verify dataset presence
        """
        csv_files = list(self.output_dir.rglob("*.csv"))

        if len(csv_files) < 2:
            logger.warning("Expected at least 2 CSV files")
            return False

        logger.info(f"Found {len(csv_files)} CSV files")
        for f in csv_files:
            logger.info(f"  - {f.name}")

        return True
