"""
Phase 2 Tests
Validates core infrastructure and utilities.
"""

import sys
from pathlib import Path
import tempfile
import pytest

# -------------------------------------------------
# Ensure project root is on PYTHONPATH
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------------------------------
# Imports
# -------------------------------------------------
from src.logging import get_logger
from src.exceptions import (
    ResumeParsingError,
    FileProcessingError,
    ValidationError
)
from src.utils.text_utils import (
    clean_text,
    extract_email,
    extract_phone,
    extract_urls
)
from src.utils.date_utils import parse_date
from src.utils.validation import (
    validate_email,
    validate_phone_number,
    validate_score_range
)
from src.utils.file_utils import (
    validate_file_format,
    validate_file_size
)

# -------------------------------------------------
# Logging
# -------------------------------------------------
def test_logger_initialization():
    logger = get_logger("phase2_test")
    logger.info("Logger initialized")
    assert logger is not None

# -------------------------------------------------
# Exceptions
# -------------------------------------------------
def test_custom_exceptions():
    with pytest.raises(ResumeParsingError):
        raise ResumeParsingError(message="Resume parsing failed", resume_id="R1")

    with pytest.raises(FileProcessingError):
        raise FileProcessingError(message="File error")

    with pytest.raises(ValidationError):
        raise ValidationError(message="Validation failed")

# -------------------------------------------------
# Text utilities
# -------------------------------------------------
def test_clean_text():
    assert clean_text(" Hello   World!! ") == "hello world"

def test_email_extraction():
    emails = extract_email("Email me at test.user@example.com")
    assert "test.user@example.com" in emails

def test_phone_extraction():
    phones = extract_phone("Call +1-234-567-8901")
    assert len(phones) > 0

def test_url_extraction():
    urls = extract_urls("GitHub: https://github.com/user")
    assert "https://github.com/user" in urls


# -------------------------------------------------
# Date utilities
# -------------------------------------------------
def test_parse_date():
    assert parse_date("Jan 2020") is not None
    assert parse_date("Present") is not None

# -------------------------------------------------
# Validation utilities (exception-based)
# -------------------------------------------------
def test_validate_email():
    assert validate_email("user@test.com") is True
    with pytest.raises(ValidationError):
        validate_email("invalid-email")

def test_validate_phone_number():
    assert validate_phone_number("+1-555-123-4567") is True
    with pytest.raises(ValidationError):
        validate_phone_number("abc")

def test_validate_score_range():
    assert validate_score_range(0.5) is True
    with pytest.raises(ValidationError):
        validate_score_range(1.1)

# -------------------------------------------------
# File utilities
# -------------------------------------------------
def test_file_format_validation():
    assert validate_file_format("resume.pdf") is True
    assert validate_file_format("file.exe") is False

def test_file_size_validation():
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(b"test")
        tmp.flush()
        assert validate_file_size(tmp.name, max_size_mb=1) is True
