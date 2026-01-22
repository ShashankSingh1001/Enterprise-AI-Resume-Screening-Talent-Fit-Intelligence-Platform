"""
Validation Utilities for Resume AI Platform
Provides input validation, sanitization, and data integrity checks
"""

import re
import sys
from typing import Dict, List, Any, Optional
import bleach

from src.logging import get_logger
from src.exceptions import ValidationError

logger = get_logger(__name__)

# Constants for validation
ALLOWED_EDUCATION_LEVELS = [
    'high_school',
    'associate',
    'bachelor',
    'master',
    'phd',
    'diploma',
    'certification'
]

MIN_EXPERIENCE_YEARS = 0
MAX_EXPERIENCE_YEARS = 50

REQUIRED_RESUME_FIELDS = [
    'resume_text',
    'file_name'
]

REQUIRED_JD_FIELDS = [
    'jd_text',
    'job_title'
]

REQUIRED_MODEL_CONFIG_FIELDS = [
    'model_type',
    'hyperparameters'
]

# SQL injection patterns
SQL_INJECTION_PATTERNS = [
    r"(\s*([\0\b\'\"\n\r\t\%\_\\]*\s*(((select\s*.+\s*from\s*.+)|(insert\s*.+\s*into\s*.+)|(update\s*.+\s*set\s*.+)|(delete\s*.+\s*from\s*.+)|(drop\s*.+)|(truncate\s*.+)|(alter\s*.+)|(exec\s*.+)|(\s*(all|any|not|and|between|in|like|or|some|contains|containsall|containskey)\s*.+[\=\>\<=\!\~]+.+)|(let\s+.+[\=]\s*.*)|(begin\s*.*\s*end)|(\s*[\/\*]+\s*.*\s*[\*\/]+)|(\s*(\-\-)\s*.*\s+)|(\s*(contains|containsall|containskey)\s+.*)))(\s*[\;]\s*)*)+)",
    r"((\%27)|(\')|(\-\-)|(\%23)|(#))",
    r"((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(\%3B)|(;))",
    r"\w*((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))",
]


def validate_resume_data(data: Dict[str, Any]) -> bool:
    """
    Validate resume data structure and content.
    
    Args:
        data: Resume data dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        logger.debug("Validating resume data")
        
        # Check if data is a dictionary
        if not isinstance(data, dict):
            raise ValidationError(
                Exception("Resume data must be a dictionary"),
                sys
            )
        
        # Check required fields
        missing_fields = [field for field in REQUIRED_RESUME_FIELDS if field not in data]
        if missing_fields:
            raise ValidationError(
                Exception(f"Missing required fields: {', '.join(missing_fields)}"),
                sys
            )
        
        # Validate resume_text is not empty
        if not data.get('resume_text') or not isinstance(data['resume_text'], str):
            raise ValidationError(
                Exception("resume_text must be a non-empty string"),
                sys
            )
        
        if len(data['resume_text'].strip()) < 50:
            raise ValidationError(
                Exception("resume_text is too short (minimum 50 characters)"),
                sys
            )
        
        # Validate file_name
        if not data.get('file_name') or not isinstance(data['file_name'], str):
            raise ValidationError(
                Exception("file_name must be a non-empty string"),
                sys
            )
        
        # Validate optional fields if present
        if 'experience_years' in data:
            validate_experience_years(data['experience_years'])
        
        if 'education_level' in data:
            validate_education_level(data['education_level'])
        
        if 'skills' in data:
            validate_skills_list(data['skills'])
        
        logger.info("Resume data validation successful")
        return True
    
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Resume validation failed: {str(e)}")
        raise ValidationError(e, sys)


def validate_jd_data(data: Dict[str, Any]) -> bool:
    """
    Validate job description data structure and content.
    
    Args:
        data: JD data dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        logger.debug("Validating JD data")
        
        # Check if data is a dictionary
        if not isinstance(data, dict):
            raise ValidationError(
                Exception("JD data must be a dictionary"),
                sys
            )
        
        # Check required fields
        missing_fields = [field for field in REQUIRED_JD_FIELDS if field not in data]
        if missing_fields:
            raise ValidationError(
                Exception(f"Missing required fields: {', '.join(missing_fields)}"),
                sys
            )
        
        # Validate jd_text is not empty
        if not data.get('jd_text') or not isinstance(data['jd_text'], str):
            raise ValidationError(
                Exception("jd_text must be a non-empty string"),
                sys
            )
        
        if len(data['jd_text'].strip()) < 100:
            raise ValidationError(
                Exception("jd_text is too short (minimum 100 characters)"),
                sys
            )
        
        # Validate job_title
        if not data.get('job_title') or not isinstance(data['job_title'], str):
            raise ValidationError(
                Exception("job_title must be a non-empty string"),
                sys
            )
        
        # Validate optional fields
        if 'required_skills' in data:
            validate_skills_list(data['required_skills'])
        
        if 'min_experience_years' in data:
            validate_experience_years(data['min_experience_years'])
        
        logger.info("JD data validation successful")
        return True
    
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"JD validation failed: {str(e)}")
        raise ValidationError(e, sys)


def validate_experience_years(years: Any) -> bool:
    """
    Validate experience years is within reasonable range.
    
    Args:
        years: Experience in years
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Convert to float for validation
        try:
            years_float = float(years)
        except (ValueError, TypeError):
            raise ValidationError(
                Exception(f"Experience years must be a number, got: {type(years).__name__}"),
                sys
            )
        
        # Check range
        if not (MIN_EXPERIENCE_YEARS <= years_float <= MAX_EXPERIENCE_YEARS):
            raise ValidationError(
                Exception(
                    f"Experience years must be between {MIN_EXPERIENCE_YEARS} and {MAX_EXPERIENCE_YEARS}, "
                    f"got: {years_float}"
                ),
                sys
            )
        
        # Check for negative values
        if years_float < 0:
            raise ValidationError(
                Exception(f"Experience years cannot be negative: {years_float}"),
                sys
            )
        
        logger.debug(f"Experience years validated: {years_float}")
        return True
    
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Experience validation failed: {str(e)}")
        raise ValidationError(e, sys)


def validate_education_level(level: str) -> bool:
    """
    Validate education level is in allowed list.
    
    Args:
        level: Education level string
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        if not isinstance(level, str):
            raise ValidationError(
                Exception(f"Education level must be a string, got: {type(level).__name__}"),
                sys
            )
        
        level_lower = level.lower().strip()
        
        if level_lower not in ALLOWED_EDUCATION_LEVELS:
            raise ValidationError(
                Exception(
                    f"Invalid education level: '{level}'. "
                    f"Allowed values: {', '.join(ALLOWED_EDUCATION_LEVELS)}"
                ),
                sys
            )
        
        logger.debug(f"Education level validated: {level_lower}")
        return True
    
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Education validation failed: {str(e)}")
        raise ValidationError(e, sys)


def validate_skills_list(skills: Any) -> bool:
    """
    Validate skills list format and content.
    
    Args:
        skills: List of skills
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Check if it's a list
        if not isinstance(skills, list):
            raise ValidationError(
                Exception(f"Skills must be a list, got: {type(skills).__name__}"),
                sys
            )
        
        # Check if list is empty
        if len(skills) == 0:
            raise ValidationError(
                Exception("Skills list cannot be empty"),
                sys
            )
        
        # Validate each skill is a non-empty string
        for i, skill in enumerate(skills):
            if not isinstance(skill, str):
                raise ValidationError(
                    Exception(f"Skill at index {i} must be a string, got: {type(skill).__name__}"),
                    sys
                )
            
            if not skill.strip():
                raise ValidationError(
                    Exception(f"Skill at index {i} cannot be empty or whitespace"),
                    sys
                )
            
            # Check skill length
            if len(skill) > 100:
                raise ValidationError(
                    Exception(f"Skill at index {i} is too long (max 100 characters): {skill[:50]}..."),
                    sys
                )
        
        logger.debug(f"Skills list validated: {len(skills)} skills")
        return True
    
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Skills validation failed: {str(e)}")
        raise ValidationError(e, sys)


def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent SQL injection and XSS attacks.
    
    Args:
        text: User input text
        
    Returns:
        Sanitized text
    """
    try:
        if not isinstance(text, str):
            logger.warning(f"Non-string input provided for sanitization: {type(text).__name__}")
            return str(text)
        
        # Check for SQL injection patterns
        for pattern in SQL_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Potential SQL injection detected and blocked")
                raise ValidationError(
                    Exception("Input contains potentially malicious SQL patterns"),
                    sys
                )
        
        # Remove HTML/JavaScript for XSS prevention
        sanitized = bleach.clean(
            text,
            tags=[],  # Remove all HTML tags
            strip=True
        )
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Limit length to prevent DoS
        MAX_INPUT_LENGTH = 100000  # 100KB
        if len(sanitized) > MAX_INPUT_LENGTH:
            logger.warning(f"Input too long, truncating from {len(sanitized)} to {MAX_INPUT_LENGTH}")
            sanitized = sanitized[:MAX_INPUT_LENGTH]
        
        if sanitized != text:
            logger.info("Input was sanitized")
        
        return sanitized
    
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Input sanitization failed: {str(e)}")
        raise ValidationError(e, sys)


def validate_model_config(config: Dict[str, Any]) -> bool:
    """
    Validate machine learning model configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        logger.debug("Validating model configuration")
        
        # Check if config is a dictionary
        if not isinstance(config, dict):
            raise ValidationError(
                Exception("Model config must be a dictionary"),
                sys
            )
        
        # Check required fields
        missing_fields = [field for field in REQUIRED_MODEL_CONFIG_FIELDS if field not in config]
        if missing_fields:
            raise ValidationError(
                Exception(f"Missing required config fields: {', '.join(missing_fields)}"),
                sys
            )
        
        # Validate model_type
        allowed_model_types = ['xgboost', 'lightgbm', 'random_forest', 'logistic_regression', 'neural_network']
        if config.get('model_type') not in allowed_model_types:
            raise ValidationError(
                Exception(
                    f"Invalid model_type: '{config.get('model_type')}'. "
                    f"Allowed: {', '.join(allowed_model_types)}"
                ),
                sys
            )
        
        # Validate hyperparameters is a dictionary
        if not isinstance(config.get('hyperparameters'), dict):
            raise ValidationError(
                Exception("hyperparameters must be a dictionary"),
                sys
            )
        
        # Validate specific hyperparameters for XGBoost
        if config['model_type'] == 'xgboost':
            hyper = config['hyperparameters']
            
            # Learning rate validation
            if 'learning_rate' in hyper:
                lr = hyper['learning_rate']
                if not (0 < lr <= 1):
                    raise ValidationError(
                        Exception(f"learning_rate must be between 0 and 1, got: {lr}"),
                        sys
                    )
            
            # Max depth validation
            if 'max_depth' in hyper:
                md = hyper['max_depth']
                if not isinstance(md, int) or md < 1 or md > 20:
                    raise ValidationError(
                        Exception(f"max_depth must be an integer between 1 and 20, got: {md}"),
                        sys
                    )
            
            # N estimators validation
            if 'n_estimators' in hyper:
                ne = hyper['n_estimators']
                if not isinstance(ne, int) or ne < 1 or ne > 10000:
                    raise ValidationError(
                        Exception(f"n_estimators must be an integer between 1 and 10000, got: {ne}"),
                        sys
                    )
        
        logger.info("Model config validation successful")
        return True
    
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Model config validation failed: {str(e)}")
        raise ValidationError(e, sys)


def validate_email(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email: Email address string
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(email_pattern, email):
            raise ValidationError(
                Exception(f"Invalid email format: {email}"),
                sys
            )
        
        logger.debug(f"Email validated: {email}")
        return True
    
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Email validation failed: {str(e)}")
        raise ValidationError(e, sys)


def validate_phone_number(phone: str) -> bool:
    """
    Validate phone number format.
    
    Args:
        phone: Phone number string
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)
        
        # Check if it's 10 digits (US format) or more (international)
        if len(digits) < 10:
            raise ValidationError(
                Exception(f"Phone number too short: {phone}"),
                sys
            )
        
        if len(digits) > 15:
            raise ValidationError(
                Exception(f"Phone number too long: {phone}"),
                sys
            )
        
        logger.debug(f"Phone number validated: {phone}")
        return True
    
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Phone validation failed: {str(e)}")
        raise ValidationError(e, sys)


def validate_file_path(file_path: str) -> bool:
    """
    Validate file path for security (prevent path traversal attacks).
    
    Args:
        file_path: File path string
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Check for path traversal attempts
        dangerous_patterns = ['..', '~/', '//']
        
        for pattern in dangerous_patterns:
            if pattern in file_path:
                raise ValidationError(
                    Exception(f"Invalid file path (path traversal detected): {file_path}"),
                    sys
                )
        
        # Check for absolute paths (should be relative)
        if file_path.startswith('/') or (len(file_path) > 1 and file_path[1] == ':'):
            logger.warning(f"Absolute path provided: {file_path}")
        
        logger.debug(f"File path validated: {file_path}")
        return True
    
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"File path validation failed: {str(e)}")
        raise ValidationError(e, sys)


def validate_score_range(score: float, min_score: float = 0.0, max_score: float = 1.0) -> bool:
    """
    Validate that a score is within valid range.
    
    Args:
        score: Score value
        min_score: Minimum allowed score
        max_score: Maximum allowed score
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        if not isinstance(score, (int, float)):
            raise ValidationError(
                Exception(f"Score must be a number, got: {type(score).__name__}"),
                sys
            )
        
        if not (min_score <= score <= max_score):
            raise ValidationError(
                Exception(f"Score {score} out of range [{min_score}, {max_score}]"),
                sys
            )
        
        logger.debug(f"Score validated: {score}")
        return True
    
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Score validation failed: {str(e)}")
        raise ValidationError(e, sys)