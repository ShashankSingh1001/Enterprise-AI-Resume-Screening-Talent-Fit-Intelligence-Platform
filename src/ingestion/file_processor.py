"""
File Processor Module
Handles batch processing of resumes and job descriptions
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm

from src.utils.file_utils import (
    read_pdf, read_docx, read_txt, 
    validate_file_format, validate_file_size,
    get_file_metadata
)
from src.utils.text_utils import clean_text, extract_email, extract_phone
from src.logging import get_logger
from src.exceptions import FileProcessingError

logger = get_logger(__name__)


class FileProcessor:
    """
    Batch file processor for resumes and job descriptions
    """
    
    def __init__(self, max_file_size_mb: int = 10):
        """
        Initialize file processor
        
        Args:
            max_file_size_mb: Maximum file size in MB
        """
        self.max_file_size_mb = max_file_size_mb
        self.processed_count = 0
        self.failed_count = 0
        self.failed_files = []
        
    def process_single_file(self, file_path: str) -> Optional[Dict]:
        """
        Process a single file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with extracted data or None if failed
        """
        try:
            if not validate_file_format(file_path):
                logger.warning(f"Unsupported file format: {file_path}")
                return None
            
            if not validate_file_size(file_path, self.max_file_size_mb):
                logger.warning(f"File too large: {file_path}")
                return None
            
            metadata = get_file_metadata(file_path)
            
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.pdf':
                text = read_pdf(file_path)
            elif file_ext in ['.docx', '.doc']:
                text = read_docx(file_path)
            elif file_ext == '.txt':
                text = read_txt(file_path)
            else:
                raise FileProcessingError(f"Unsupported extension: {file_ext}")
            
            if not text or len(text.strip()) < 50:
                logger.warning(f"Insufficient text extracted from: {file_path}")
                return None
            
            clean = clean_text(text)
            email = extract_email(clean)
            phone = extract_phone(clean)
            
            return {
                'file_path': file_path,
                'file_name': metadata['file_name'],
                'file_size': metadata['file_size'],
                'raw_text': text,
                'clean_text': clean,
                'email': email,
                'phone': phone,
                'word_count': len(clean.split()),
                'char_count': len(clean),
                'created_date': metadata['created_date'],
                'modified_date': metadata['modified_date']
            }
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}", exc_info=True)
            self.failed_files.append((file_path, str(e)))
            return None
    
    def process_directory(
        self, 
        directory: str, 
        file_type: str = 'resume',
        recursive: bool = True
    ) -> pd.DataFrame:
        """
        Process all files in a directory
        
        Args:
            directory: Path to directory
            file_type: 'resume' or 'jd'
            recursive: Search subdirectories
            
        Returns:
            DataFrame with processed data
        """
        logger.info(f"Processing {file_type}s from: {directory}")
        
        path = Path(directory)
        if recursive:
            files = list(path.rglob('*.*'))
        else:
            files = list(path.glob('*.*'))
        
        supported = ['.pdf', '.docx', '.doc', '.txt']
        files = [f for f in files if f.suffix.lower() in supported]
        
        logger.info(f"Found {len(files)} files to process")
        
        results = []
        for file_path in tqdm(files, desc=f"Processing {file_type}s"):
            result = self.process_single_file(str(file_path))
            if result:
                result['document_type'] = file_type
                results.append(result)
                self.processed_count += 1
            else:
                self.failed_count += 1
        
        df = pd.DataFrame(results)
        
        logger.info(f"Processed: {self.processed_count}, Failed: {self.failed_count}")
        
        return df
    
    def process_batch(
        self, 
        resume_dir: str, 
        jd_dir: str,
        output_dir: str = 'data/processed'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process both resumes and JDs in batch
        
        Args:
            resume_dir: Directory with resumes
            jd_dir: Directory with job descriptions
            output_dir: Where to save processed data
            
        Returns:
            Tuple of (resumes_df, jds_df)
        """
        logger.info("Starting batch processing...")
        
        resumes_df = self.process_directory(resume_dir, file_type='resume')
        jds_df = self.process_directory(jd_dir, file_type='jd')
        
        os.makedirs(output_dir, exist_ok=True)
        
        resume_path = os.path.join(output_dir, 'resumes_parsed.csv')
        jd_path = os.path.join(output_dir, 'jds_parsed.csv')
        
        resumes_df.to_csv(resume_path, index=False)
        jds_df.to_csv(jd_path, index=False)
        
        logger.info(f"Saved resumes to: {resume_path}")
        logger.info(f"Saved JDs to: {jd_path}")
        
        if self.failed_files:
            logger.warning(f"Failed files ({len(self.failed_files)}):")
            for file_path, error in self.failed_files[:10]:
                logger.warning(f"  {file_path}: {error}")
        
        return resumes_df, jds_df
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            'total_processed': self.processed_count,
            'total_failed': self.failed_count,
            'success_rate': (
                self.processed_count / (self.processed_count + self.failed_count)
                if (self.processed_count + self.failed_count) > 0 else 0
            ),
            'failed_files': self.failed_files
        }


def main():
    """Example usage"""
    processor = FileProcessor(max_file_size_mb=10)
    
    df = processor.process_directory(
        directory='data/raw/resumes',
        file_type='resume'
    )
    
    print(f"\nProcessed {len(df)} files")
    print(f"\nSample data:\n{df.head()}")
    
    stats = processor.get_processing_stats()
    print(f"\nProcessing Stats: {stats}")


if __name__ == "__main__":
    main()