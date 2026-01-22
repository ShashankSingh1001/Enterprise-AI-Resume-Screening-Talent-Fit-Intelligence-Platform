"""
Data Ingestion Pipeline
End-to-end pipeline for Phase 3: Data collection, processing, and labeling
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

from src.ingestion import FileProcessor, DataLoader, SyntheticLabelGenerator
from src.utils.kaggle_utils import KaggleDownloader
from src.logging import get_logger

logger = get_logger(__name__)


def main():
    """
    Execute complete data ingestion pipeline
    
    Steps:
    1. Setup directories
    2. Download Kaggle datasets (if needed)
    3. Load datasets
    4. Process uploaded files
    5. Generate synthetic labels
    6. Create training dataset
    7. Save with DVC tracking
    """
    
    logger.info("="*60)
    logger.info("DATA INGESTION PIPELINE - PHASE 3")
    logger.info("="*60)
    
    logger.info("\n[1/7] Setting up directories...")
    from scripts.setup_data_directories import create_data_directories
    create_data_directories()
    
    logger.info("\n[2/7] Checking for Kaggle datasets...")
    downloader = KaggleDownloader()
    
    if not downloader.verify_datasets():
        logger.info("Datasets not found. Downloading from Kaggle...")
        try:
            resume_files, jd_files = downloader.download_resume_datasets()
            logger.info("Kaggle datasets downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download datasets: {e}")
            logger.info("\nManual download instructions:")
            logger.info("1. Resume Dataset: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset")
            logger.info("2. JD Dataset: https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset")
            logger.info("3. Place CSV files in: data/external/")
            logger.info("\nOr set KAGGLE_USERNAME and KAGGLE_KEY in .env file")
            return
    else:
        logger.info("Kaggle datasets already available")
    
    logger.info("\n[3/7] Loading datasets...")
    loader = DataLoader()
    
    try:
        resumes_df, jds_df = loader.load_and_merge_datasets()
        logger.info(f"Loaded {len(resumes_df):,} resumes")
        logger.info(f"Loaded {len(jds_df):,} job descriptions")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return
    
    logger.info("\n[4/7] Checking for additional resume files...")
    processor = FileProcessor(max_file_size_mb=10)
    
    resume_dir = 'data/raw/resumes'
    jd_dir = 'data/raw/job_descriptions'
    
    if os.path.exists(resume_dir) and len(list(Path(resume_dir).rglob('*.*'))) > 0:
        logger.info("Processing additional resumes...")
        extra_resumes = processor.process_directory(resume_dir, file_type='resume')
        logger.info(f"Processed {len(extra_resumes)} additional resumes")
    else:
        logger.info("No additional resume files found")
    
    if os.path.exists(jd_dir) and len(list(Path(jd_dir).rglob('*.*'))) > 0:
        logger.info("Processing additional JDs...")
        extra_jds = processor.process_directory(jd_dir, file_type='jd')
        logger.info(f"Processed {len(extra_jds)} additional JDs")
    else:
        logger.info("No additional JD files found")
    
    logger.info("\n[5/7] Running data quality checks...")
    
    resume_quality = loader.check_data_quality(resumes_df)
    jd_quality = loader.check_data_quality(jds_df)
    
    logger.info(f"Resume dataset: {resume_quality['total_rows']:,} rows, "
               f"{resume_quality['duplicate_rows']} duplicates, "
               f"{resume_quality['memory_usage_mb']:.2f} MB")
    
    logger.info(f"JD dataset: {jd_quality['total_rows']:,} rows, "
               f"{jd_quality['duplicate_rows']} duplicates, "
               f"{jd_quality['memory_usage_mb']:.2f} MB")
    
    logger.info("\n[6/7] Generating synthetic hiring labels...")
    
    label_generator = SyntheticLabelGenerator(random_seed=42)
    
    training_df = label_generator.generate_labels_for_dataset(
        resumes_df=resumes_df,
        jds_df=jds_df,
        n_samples=2000,
        resume_col='resume_text' if 'resume_text' in resumes_df.columns else resumes_df.columns[0],
        jd_col='jd_text' if 'jd_text' in jds_df.columns else jds_df.columns[0]
    )
    
    logger.info(f"Generated {len(training_df):,} labeled samples")
    logger.info(f"Selection rate: {training_df['selected'].mean():.1%}")
    logger.info(f"Average score: {training_df['final_score'].mean():.3f}")
    
    logger.info("\n[7/7] Saving datasets...")
    
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
    
    loader.save_with_dvc(
        training_df,
        'data/processed/training_data.csv',
        message="Generated synthetic labels for training"
    )
    
    logger.info("All datasets saved successfully")
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 3 COMPLETE")
    logger.info("="*60)
    logger.info(f"\nSummary:")
    logger.info(f"  Total Resumes: {len(resumes_df):,}")
    logger.info(f"  Total JDs: {len(jds_df):,}")
    logger.info(f"  Training Samples: {len(training_df):,}")
    logger.info(f"  Selected Candidates: {training_df['selected'].sum():,}")
    logger.info(f"  Rejected Candidates: {(~training_df['selected'].astype(bool)).sum():,}")
    
    logger.info(f"\nFiles Created:")
    logger.info(f"  data/processed/all_resumes.csv")
    logger.info(f"  data/processed/all_jds.csv")
    logger.info(f"  data/processed/training_data.csv")
    
    logger.info(f"\nNext Steps:")
    logger.info(f"  1. Run EDA notebook: notebooks/01_eda_analysis.ipynb")
    logger.info(f"  2. Initialize DVC: bash scripts/setup_dvc.sh")
    logger.info(f"  3. Start Phase 4: NLP Parsing")
    
    logger.info("\n" + "="*60)


if __name__ == "__main__":
    main()