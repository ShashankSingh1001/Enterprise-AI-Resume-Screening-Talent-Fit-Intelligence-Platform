"""
Validation Utilities for Resume AI Platform
Provides input validation, sanitization, and data integrity checks
"""

import re
import sys
from typing import Dict, List, Any
import bleach

from src.logging import get_logger
from src.exceptions import ValidationError

logger = get_logger(__name__)

# =====================================================
# CONSTANTS
# =====================================================

ALLOWED_EDUCATION_LEVELS = {
    "high_school",
    "associate",
    "bachelor",
    "master",
    "phd",
    "diploma",
    "certification",
}

MIN_EXPERIENCE_YEARS = 0
MAX_EXPERIENCE_YEARS = 50

REQUIRED_RESUME_FIELDS = {"resume_text", "file_name"}
REQUIRED_JD_FIELDS = {"jd_text", "job_title"}
REQUIRED_MODEL_CONFIG_FIELDS = {"model_type", "hyperparameters"}

# SQL injection detection patterns
SQL_INJECTION_PATTERNS = [
    r"\b(select|insert|update|delete|drop|truncate|alter|exec)\b",
    r"(--|#|/\*|\*/)",
    r"\bor\b\s+.+=",
    r";",
]


# =====================================================
# CORE VALIDATORS
# =====================================================

def validate_resume_data(data: Dict[str, Any]) -> bool:
    try:
        if not isinstance(data, dict):
            raise ValidationError(message="Resume data must be a dictionary")

        missing = REQUIRED_RESUME_FIELDS - data.keys()
        if missing:
            raise ValidationError(
                message=f"Missing required fields: {', '.join(missing)}"
            )

        resume_text = data.get("resume_text")
        if not isinstance(resume_text, str) or len(resume_text.strip()) < 50:
            raise ValidationError(
                message="resume_text must be a non-empty string (min 50 characters)"
            )

        file_name = data.get("file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValidationError(message="file_name must be a non-empty string")

        if "experience_years" in data:
            validate_experience_years(data["experience_years"])

        if "education_level" in data:
            validate_education_level(data["education_level"])

        if "skills" in data:
            validate_skills_list(data["skills"])

        logger.debug("Resume data validation successful")
        return True

    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(message=str(e), error_detail=sys.exc_info())


def validate_jd_data(data: Dict[str, Any]) -> bool:
    try:
        if not isinstance(data, dict):
            raise ValidationError(message="JD data must be a dictionary")

        missing = REQUIRED_JD_FIELDS - data.keys()
        if missing:
            raise ValidationError(
                message=f"Missing required fields: {', '.join(missing)}"
            )

        jd_text = data.get("jd_text")
        if not isinstance(jd_text, str) or len(jd_text.strip()) < 100:
            raise ValidationError(
                message="jd_text must be a non-empty string (min 100 characters)"
            )

        job_title = data.get("job_title")
        if not isinstance(job_title, str) or not job_title.strip():
            raise ValidationError(message="job_title must be a non-empty string")

        if "required_skills" in data:
            validate_skills_list(data["required_skills"])

        if "min_experience_years" in data:
            validate_experience_years(data["min_experience_years"])

        logger.debug("JD data validation successful")
        return True

    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(message=str(e), error_detail=sys.exc_info())


# =====================================================
# FIELD VALIDATORS
# =====================================================

def validate_experience_years(years: Any) -> bool:
    try:
        years = float(years)
        if not (MIN_EXPERIENCE_YEARS <= years <= MAX_EXPERIENCE_YEARS):
            raise ValidationError(
                message=f"Experience years must be between {MIN_EXPERIENCE_YEARS} and {MAX_EXPERIENCE_YEARS}"
            )
        return True
    except ValueError:
        raise ValidationError(message="Experience years must be numeric")
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(message=str(e), error_detail=sys.exc_info())


def validate_education_level(level: str) -> bool:
    if not isinstance(level, str):
        raise ValidationError(message="Education level must be a string")

    level = level.lower().strip()
    if level not in ALLOWED_EDUCATION_LEVELS:
        raise ValidationError(message=f"Invalid education level: {level}")

    return True


def validate_skills_list(skills: Any) -> bool:
    if not isinstance(skills, list) or not skills:
        raise ValidationError(message="Skills must be a non-empty list")

    for idx, skill in enumerate(skills):
        if not isinstance(skill, str) or not skill.strip():
            raise ValidationError(message=f"Invalid skill at index {idx}")
        if len(skill) > 100:
            raise ValidationError(message=f"Skill too long at index {idx}")

    return True


# =====================================================
# SECURITY / SANITIZATION  ✅ FIXED
# =====================================================

def sanitize_input(text: str) -> str:
    try:
        if not isinstance(text, str):
            return str(text)

        # 1️⃣ REMOVE HTML / JS FIRST (XSS prevention)
        cleaned = bleach.clean(text, tags=[], strip=True)
        cleaned = cleaned.replace("\x00", "")

        # 2️⃣ SQL INJECTION CHECK ON CLEANED TEXT
        for pattern in SQL_INJECTION_PATTERNS:
            if re.search(pattern, cleaned, re.IGNORECASE):
                raise ValidationError(message="Potential SQL injection detected")

        # 3️⃣ Length guard
        return cleaned[:100_000]

    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(message=str(e), error_detail=sys.exc_info())


# =====================================================
# MODEL CONFIG VALIDATION
# =====================================================

def validate_model_config(config: Dict[str, Any]) -> bool:
    if not isinstance(config, dict):
        raise ValidationError(message="Model config must be a dictionary")

    missing = REQUIRED_MODEL_CONFIG_FIELDS - config.keys()
    if missing:
        raise ValidationError(
            message=f"Missing model config fields: {', '.join(missing)}"
        )

    allowed_models = {
        "xgboost",
        "lightgbm",
        "random_forest",
        "logistic_regression",
        "neural_network",
    }

    model_type = config["model_type"]
    hyper = config["hyperparameters"]

    if model_type not in allowed_models:
        raise ValidationError(message="Invalid model_type")

    if not isinstance(hyper, dict):
        raise ValidationError(message="hyperparameters must be a dictionary")

    # =====================================================
    # XGBoost-specific validation (REQUIRED by tests)
    # =====================================================
    if model_type == "xgboost":
        if "learning_rate" in hyper:
            lr = hyper["learning_rate"]
            if not isinstance(lr, (int, float)) or not (0 < lr <= 1):
                raise ValidationError(
                    message="learning_rate must be between 0 and 1"
                )

        if "max_depth" in hyper:
            md = hyper["max_depth"]
            if not isinstance(md, int) or not (1 <= md <= 20):
                raise ValidationError(
                    message="max_depth must be an integer between 1 and 20"
                )

        if "n_estimators" in hyper:
            ne = hyper["n_estimators"]
            if not isinstance(ne, int) or not (1 <= ne <= 10000):
                raise ValidationError(
                    message="n_estimators must be an integer between 1 and 10000"
                )

    return True


# =====================================================
# SIMPLE VALIDATORS
# =====================================================

def validate_email(email: str) -> bool:
    """
    Validate email format.
    Supports modern email formats including '+' and subdomains.
    """
    if not isinstance(email, str):
        raise ValidationError(message="Email must be a string")

    # RFC 5322–compatible (practical subset)
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

    if not re.match(pattern, email):
        raise ValidationError(message="Invalid email format")

    return True



def validate_phone_number(phone: str) -> bool:
    digits = re.sub(r"\D", "", phone)
    if not (10 <= len(digits) <= 15):
        raise ValidationError(message="Invalid phone number length")
    return True


def validate_file_path(file_path: str) -> bool:
    if any(p in file_path for p in ("..", "~", "//")):
        raise ValidationError(message="Invalid file path")
    return True


def validate_score_range(
    score: float,
    min_score: float = 0.0,
    max_score: float = 1.0
) -> bool:
    if not isinstance(score, (int, float)):
        raise ValidationError(message="Score must be numeric")

    if not (min_score <= score <= max_score):
        raise ValidationError(message="Score out of range")

    return True
