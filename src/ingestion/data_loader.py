"""
Data Loader Module
Handles loading, validation, and versioning of datasets with DVC integration
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from src.logging import get_logger
from src.exceptions import FileProcessingError
from src.utils.validation import validate_resume_data

logger = get_logger(__name__)


class DataLoader:
    """
    Data loader with DVC versioning and validation
    """
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize data loader
        
        Args:
            data_dir: Base data directory
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.external_dir = self.data_dir / 'external'
        
        for dir_path in [self.raw_dir, self.processed_dir, self.external_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_csv(
        self, 
        file_path: str, 
        encoding: str = 'utf-8',
        **kwargs
    ) -> pd.DataFrame:
        """
        Load CSV file with error handling
        
        Args:
            file_path: Path to CSV file
            encoding: File encoding
            **kwargs: Additional pandas read_csv arguments
            
        Returns:
            DataFrame
        """
        try:
            logger.info(f"Loading CSV: {file_path}")
            
            try:
                df = pd.read_csv(file_path, encoding=encoding, **kwargs)
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 failed, trying latin-1 for {file_path}")
                df = pd.read_csv(file_path, encoding='latin-1', **kwargs)
            
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load CSV {file_path}: {str(e)}")
            raise FileProcessingError(f"CSV load failed: {str(e)}", error_detail=sys.exc_info())
    
    def load_excel(self, file_path: str, sheet_name: str = 0) -> pd.DataFrame:
        """
        Load Excel file
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name or index
            
        Returns:
            DataFrame
        """
        try:
            logger.info(f"Loading Excel: {file_path}")
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to load Excel {file_path}: {str(e)}")
            raise FileProcessingError(f"Excel load failed: {str(e)}", error_detail=sys.exc_info())
    
    def load_kaggle_resume_dataset(self) -> pd.DataFrame:
        """
        Load and process Kaggle resume dataset (CSV format)
        
        Returns:
            Processed DataFrame
        """
        logger.info("Loading Kaggle resume dataset...")
        
        csv_files = list(self.external_dir.glob('*[Rr]esume*.csv'))
        csv_files.extend(list(self.external_dir.glob('*data*.csv')))
        
        if not csv_files:
            error_msg = (
                "Kaggle resume dataset not found in data/external/. "
                "Expected CSV file with resume data. "
                "Please ensure download completed successfully."
            )
            logger.error(error_msg)
            raise FileProcessingError(error_msg, error_detail=sys.exc_info())
        
        logger.info(f"Found CSV file: {csv_files[0].name}")
        df = self.load_csv(str(csv_files[0]))
        
        if 'Resume' in df.columns:
            df = df.rename(columns={'Resume': 'resume_text'})
        if 'Category' in df.columns:
            df = df.rename(columns={'Category': 'category'})
        
        df['source'] = 'kaggle_resume_dataset'
        df['loaded_at'] = datetime.now().isoformat()
        
        initial_count = len(df)
        df = df.drop_duplicates(subset=['resume_text'], keep='first')
        duplicates_removed = initial_count - len(df)
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate resumes")
        
        logger.info(f"Loaded {len(df)} unique resumes from Kaggle dataset")
        return df
    
    def load_jd_dataset(self) -> pd.DataFrame:
        """
        Load job description dataset and filter for IT/Tech jobs
        
        Returns:
            Processed DataFrame
        """
        logger.info("Loading JD dataset...")
        
        csv_files = list(self.external_dir.glob('*job*.csv'))
        
        if not csv_files:
            error_msg = (
                "JD dataset not found in data/external/. "
                "Expected CSV file with job descriptions. "
                "Please ensure download completed successfully."
            )
            logger.error(error_msg)
            raise FileProcessingError(error_msg, error_detail=sys.exc_info())
        
        df = self.load_csv(str(csv_files[0]))
        initial_count = len(df)
        logger.info(f"Initial JD count: {initial_count:,}")
        
        column_mapping = {
            'Job Description': 'jd_text',
            'Position': 'job_title',
            'Company': 'company',
            'location': 'location',
            'Job.Description': 'jd_text',
            'Job Title': 'job_title',
            'position_title': 'job_title',
            'description': 'jd_text'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        if 'jd_text' not in df.columns:
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            if len(text_cols) > 0:
                longest_col = max(text_cols, key=lambda col: df[col].astype(str).str.len().mean())
                df['jd_text'] = df[longest_col]
                logger.info(f"Using column '{longest_col}' as jd_text")
            else:
                logger.warning("No text columns found, using first column")
                df['jd_text'] = df.iloc[:, 0].astype(str)
        
        it_keywords = [
            'software', 'developer', 'engineer', 'programmer', 'python', 'java',
            'javascript', 'data', 'analyst', 'scientist', 'machine learning',
            'artificial intelligence', 'ai', 'ml', 'devops', 'cloud', 'aws',
            'azure', 'react', 'angular', 'node', 'backend', 'frontend',
            'full stack', 'database', 'sql', 'nosql', 'api', 'microservices',
            'kubernetes', 'docker', 'ci/cd', 'agile', 'scrum', 'git',
            'web development', 'mobile', 'android', 'ios', 'ui/ux', 'technology'
        ]
        
        logger.info("Filtering for IT/Tech jobs...")
        df['jd_text_filled'] = df['jd_text'].fillna('').astype(str)
        df['jd_lower'] = df['jd_text_filled'].str.lower()
        
        pattern = '|'.join(it_keywords)
        mask = df['jd_lower'].str.contains(pattern, case=False, na=False, regex=True)
        
        df_filtered = df[mask].copy()
        df_filtered = df_filtered.drop(['jd_lower', 'jd_text_filled'], axis=1)
        
        filtered_count = len(df_filtered)
        logger.info(f"After keyword filtering: {filtered_count:,} IT/Tech jobs ({filtered_count/initial_count*100:.1f}%)")
        
        if filtered_count > 100000:
            df_filtered = df_filtered.sample(n=100000, random_state=42)
            logger.info(f"Sampled down to {len(df_filtered):,} jobs for manageable size")
        
        before_dedup = len(df_filtered)
        df_filtered = df_filtered.drop_duplicates(subset=['jd_text'], keep='first')
        after_dedup = len(df_filtered)
        
        if before_dedup > after_dedup:
            logger.info(f"Removed {before_dedup - after_dedup} duplicate JDs")
        
        df_filtered['source'] = 'kaggle_jd_dataset'
        df_filtered['loaded_at'] = datetime.now().isoformat()
        
        logger.info(f"Final JD count: {len(df_filtered):,} unique IT/Tech job descriptions")
        logger.info(f"Dataset reduced: {initial_count:,} -> {len(df_filtered):,} ({len(df_filtered)/initial_count*100:.2f}%)")
        
        return df_filtered
    
    def validate_dataframe(
        self, 
        df: pd.DataFrame, 
        required_columns: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame structure
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Tuple of (is_valid, missing_columns)
        """
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            logger.warning(f"Missing columns: {missing}")
            return False, missing
        
        logger.info("DataFrame validation passed")
        return True, []
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Perform data quality checks
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'column_types': df.dtypes.to_dict()
        }
        
        quality_report['missing_percentage'] = {
            col: (count / len(df) * 100) 
            for col, count in quality_report['missing_values'].items()
        }
        
        logger.info(f"Data quality report generated: {len(df)} rows, "
                   f"{quality_report['duplicate_rows']} duplicates")
        
        return quality_report
    
    def save_with_dvc(
        self, 
        df: pd.DataFrame, 
        file_path: str,
        message: str = "Update dataset"
    ) -> str:
        """
        Save DataFrame and track with DVC
        
        Args:
            df: DataFrame to save
            file_path: Where to save
            message: DVC commit message
            
        Returns:
            Path to saved file
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {len(df)} rows to {file_path}")
        
        try:
            subprocess.run(['dvc', 'add', file_path], check=True, capture_output=True)
            logger.info(f"Added {file_path} to DVC tracking")
            
            dvc_file = f"{file_path}.dvc"
            subprocess.run(['git', 'add', dvc_file], check=False, capture_output=True)
            
            logger.info(f"DVC tracking enabled for {file_path}")
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"DVC tracking failed (this is okay if DVC not initialized): {e}")
        except FileNotFoundError:
            logger.warning("DVC not installed or not in PATH")
        
        return file_path
    
    def load_and_merge_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load all datasets and prepare for training
        
        Returns:
            Tuple of (resumes_df, jds_df)
        """
        logger.info("Loading and merging all datasets...")
        
        resumes_df = self.load_kaggle_resume_dataset()
        jds_df = self.load_jd_dataset()
        
        processed_resumes = self.processed_dir / 'resumes_parsed.csv'
        processed_jds = self.processed_dir / 'jds_parsed.csv'
        
        if processed_resumes.exists():
            logger.info("Loading processed resumes...")
            processed_resumes_df = self.load_csv(str(processed_resumes))
            resumes_df = pd.concat([resumes_df, processed_resumes_df], ignore_index=True)
        
        if processed_jds.exists():
            logger.info("Loading processed JDs...")
            processed_jds_df = self.load_csv(str(processed_jds))
            jds_df = pd.concat([jds_df, processed_jds_df], ignore_index=True)
        
        logger.info(f"Total resumes: {len(resumes_df)}")
        logger.info(f"Total JDs: {len(jds_df)}")
        
        return resumes_df, jds_df


def main():
    """Example usage"""
    loader = DataLoader()
    
    resumes_df, jds_df = loader.load_and_merge_datasets()
    
    resume_quality = loader.check_data_quality(resumes_df)
    jd_quality = loader.check_data_quality(jds_df)
    
    print(f"\nResume Dataset Quality:")
    print(f"  Total: {resume_quality['total_rows']:,} rows")
    print(f"  Duplicates: {resume_quality['duplicate_rows']:,}")
    print(f"  Memory: {resume_quality['memory_usage_mb']:.2f} MB")
    
    print(f"\nJD Dataset Quality:")
    print(f"  Total: {jd_quality['total_rows']:,} rows")
    print(f"  Duplicates: {jd_quality['duplicate_rows']:,}")
    print(f"  Memory: {jd_quality['memory_usage_mb']:.2f} MB")
    
    loader.save_with_dvc(
        resumes_df,
        'data/processed/all_resumes.csv',
        message="Merged resume datasets"
    )
    
    loader.save_with_dvc(
        jds_df,
        'data/processed/all_jds.csv',
        message="Merged JD datasets"
    )


if __name__ == "__main__":
    main()