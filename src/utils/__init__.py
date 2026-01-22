"""
Utilities Module
Reusable helper functions for Resume AI Platform
"""

# File utilities
from .file_utils import (
    read_pdf,
    read_docx,
    read_txt,
    validate_file_format,
    validate_file_size,
    save_uploaded_file,
    get_file_metadata,
    delete_file,
    ALLOWED_FORMATS,
    DEFAULT_MAX_SIZE_MB
)

# Text utilities
from .text_utils import (
    clean_text,
    extract_email,
    extract_phone,
    extract_urls,
    remove_stopwords,
    lemmatize_text,
    tokenize,
    extract_keywords,
    calculate_text_similarity,
    extract_sentences,
    remove_special_characters,
    normalize_whitespace,
    truncate_text
)

# Validation utilities
from .validation import (
    validate_resume_data,
    validate_jd_data,
    validate_experience_years,
    validate_education_level,
    validate_skills_list,
    sanitize_input,
    validate_model_config,
    validate_email,
    validate_phone_number,
    validate_file_path,
    validate_score_range,
    ALLOWED_EDUCATION_LEVELS,
    MIN_EXPERIENCE_YEARS,
    MAX_EXPERIENCE_YEARS
)

# Date utilities
from .date_utils import (
    parse_date,
    calculate_experience_duration,
    calculate_total_experience,
    format_date,
    is_date_valid,
    is_future_date,
    extract_year,
    extract_month_year,
    standardize_date_format,
    calculate_age,
    get_date_range_overlap,
    get_current_timestamp,
    get_current_date,
    DATE_FORMATS,
    CURRENT_KEYWORDS
)

# Model utilities
from .model_utils import (
    save_model,
    load_model,
    get_model_metadata,
    validate_model_input,
    preprocess_features,
    calculate_feature_importance,
    predict_with_validation,
    handle_missing_values,
    create_feature_dict,
    get_model_info,
    SUPPORTED_MODEL_FORMATS
)

# Kaggle utilities
from .kaggle_utils import (
    KaggleDownloader
)


__all__ = [
    # File utilities
    'read_pdf',
    'read_docx',
    'read_txt',
    'validate_file_format',
    'validate_file_size',
    'save_uploaded_file',
    'get_file_metadata',
    'delete_file',
    'ALLOWED_FORMATS',
    'DEFAULT_MAX_SIZE_MB',
    
    # Text utilities
    'clean_text',
    'extract_email',
    'extract_phone',
    'extract_urls',
    'remove_stopwords',
    'lemmatize_text',
    'tokenize',
    'extract_keywords',
    'calculate_text_similarity',
    'extract_sentences',
    'remove_special_characters',
    'normalize_whitespace',
    'truncate_text',
    
    # Validation utilities
    'validate_resume_data',
    'validate_jd_data',
    'validate_experience_years',
    'validate_education_level',
    'validate_skills_list',
    'sanitize_input',
    'validate_model_config',
    'validate_email',
    'validate_phone_number',
    'validate_file_path',
    'validate_score_range',
    'ALLOWED_EDUCATION_LEVELS',
    'MIN_EXPERIENCE_YEARS',
    'MAX_EXPERIENCE_YEARS',
    
    # Date utilities
    'parse_date',
    'calculate_experience_duration',
    'calculate_total_experience',
    'format_date',
    'is_date_valid',
    'is_future_date',
    'extract_year',
    'extract_month_year',
    'standardize_date_format',
    'calculate_age',
    'get_date_range_overlap',
    'get_current_timestamp',
    'get_current_date',
    'DATE_FORMATS',
    'CURRENT_KEYWORDS',
    
    # Model utilities
    'save_model',
    'load_model',
    'get_model_metadata',
    'validate_model_input',
    'preprocess_features',
    'calculate_feature_importance',
    'predict_with_validation',
    'handle_missing_values',
    'create_feature_dict',
    'get_model_info',
    'SUPPORTED_MODEL_FORMATS',
    
    # Kaggle utilities
    'KaggleDownloader',
]