"""
Test Suite for validation.py
Tests all validation functions with positive and negative cases
"""

import sys
import pytest
from src.utils.validation import (
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
    ALLOWED_EDUCATION_LEVELS
)
from src.exceptions import ValidationError


# ============================================
# Test validate_resume_data()
# ============================================

def test_validate_resume_data_valid():
    """Test valid resume data"""
    data = {
        'resume_text': 'This is a valid resume with more than 50 characters to pass validation checks.',
        'file_name': 'john_doe_resume.pdf'
    }
    assert validate_resume_data(data) == True


def test_validate_resume_data_with_optional_fields():
    """Test valid resume data with optional fields"""
    data = {
        'resume_text': 'This is a valid resume with more than 50 characters to pass validation checks.',
        'file_name': 'john_doe_resume.pdf',
        'experience_years': 5,
        'education_level': 'bachelor',
        'skills': ['Python', 'Machine Learning', 'SQL']
    }
    assert validate_resume_data(data) == True


def test_validate_resume_data_missing_fields():
    """Test resume data with missing required fields"""
    data = {
        'resume_text': 'Valid text here with enough characters to pass minimum length requirement.'
    }
    with pytest.raises(ValidationError):
        validate_resume_data(data)


def test_validate_resume_data_empty_text():
    """Test resume data with empty text"""
    data = {
        'resume_text': '',
        'file_name': 'resume.pdf'
    }
    with pytest.raises(ValidationError):
        validate_resume_data(data)


def test_validate_resume_data_text_too_short():
    """Test resume data with text too short"""
    data = {
        'resume_text': 'Short',
        'file_name': 'resume.pdf'
    }
    with pytest.raises(ValidationError):
        validate_resume_data(data)


def test_validate_resume_data_not_dict():
    """Test resume data that is not a dictionary"""
    with pytest.raises(ValidationError):
        validate_resume_data("not a dictionary")


# ============================================
# Test validate_jd_data()
# ============================================

def test_validate_jd_data_valid():
    """Test valid JD data"""
    data = {
        'jd_text': 'This is a valid job description with more than 100 characters to ensure it passes the minimum length validation requirement.',
        'job_title': 'Senior Software Engineer'
    }
    assert validate_jd_data(data) == True


def test_validate_jd_data_with_optional_fields():
    """Test valid JD data with optional fields"""
    data = {
        'jd_text': 'This is a valid job description with more than 100 characters to ensure it passes the minimum length validation requirement.',
        'job_title': 'Senior Software Engineer',
        'required_skills': ['Python', 'Django', 'AWS'],
        'min_experience_years': 3
    }
    assert validate_jd_data(data) == True


def test_validate_jd_data_missing_fields():
    """Test JD data with missing required fields"""
    data = {
        'jd_text': 'This is a valid job description with more than 100 characters to ensure it passes validation.'
    }
    with pytest.raises(ValidationError):
        validate_jd_data(data)


def test_validate_jd_data_text_too_short():
    """Test JD data with text too short"""
    data = {
        'jd_text': 'Short description',
        'job_title': 'Engineer'
    }
    with pytest.raises(ValidationError):
        validate_jd_data(data)


# ============================================
# Test validate_experience_years()
# ============================================

def test_validate_experience_years_valid():
    """Test valid experience years"""
    assert validate_experience_years(5) == True
    assert validate_experience_years(0) == True
    assert validate_experience_years(25.5) == True
    assert validate_experience_years(50) == True


def test_validate_experience_years_invalid_negative():
    """Test negative experience years"""
    with pytest.raises(ValidationError):
        validate_experience_years(-1)


def test_validate_experience_years_invalid_too_high():
    """Test experience years too high"""
    with pytest.raises(ValidationError):
        validate_experience_years(51)


def test_validate_experience_years_invalid_type():
    """Test invalid type for experience years"""
    with pytest.raises(ValidationError):
        validate_experience_years("five")


# ============================================
# Test validate_education_level()
# ============================================

def test_validate_education_level_valid():
    """Test valid education levels"""
    for level in ALLOWED_EDUCATION_LEVELS:
        assert validate_education_level(level) == True


def test_validate_education_level_case_insensitive():
    """Test education level validation is case insensitive"""
    assert validate_education_level('BACHELOR') == True
    assert validate_education_level('Master') == True
    assert validate_education_level('PhD') == True


def test_validate_education_level_invalid():
    """Test invalid education level"""
    with pytest.raises(ValidationError):
        validate_education_level('elementary')


def test_validate_education_level_not_string():
    """Test education level that is not a string"""
    with pytest.raises(ValidationError):
        validate_education_level(123)


# ============================================
# Test validate_skills_list()
# ============================================

def test_validate_skills_list_valid():
    """Test valid skills list"""
    skills = ['Python', 'Machine Learning', 'SQL', 'Docker']
    assert validate_skills_list(skills) == True


def test_validate_skills_list_empty():
    """Test empty skills list"""
    with pytest.raises(ValidationError):
        validate_skills_list([])


def test_validate_skills_list_not_list():
    """Test skills that is not a list"""
    with pytest.raises(ValidationError):
        validate_skills_list("Python, SQL")


def test_validate_skills_list_non_string_element():
    """Test skills list with non-string element"""
    with pytest.raises(ValidationError):
        validate_skills_list(['Python', 123, 'SQL'])


def test_validate_skills_list_empty_string():
    """Test skills list with empty string"""
    with pytest.raises(ValidationError):
        validate_skills_list(['Python', '', 'SQL'])


def test_validate_skills_list_too_long():
    """Test skills list with skill name too long"""
    long_skill = 'x' * 101
    with pytest.raises(ValidationError):
        validate_skills_list(['Python', long_skill])


# ============================================
# Test sanitize_input()
# ============================================

def test_sanitize_input_clean():
    """Test sanitizing clean input"""
    text = "This is a clean input"
    assert sanitize_input(text) == text


def test_sanitize_input_html_removed():
    """Test HTML tags are removed"""
    text = "<script>alert('xss')</script>Hello"
    result = sanitize_input(text)
    assert '<script>' not in result
    assert 'Hello' in result


def test_sanitize_input_sql_injection_detected():
    """Test SQL injection patterns are detected"""
    malicious_inputs = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'--",
        "' UNION SELECT * FROM users--"
    ]
    
    for malicious in malicious_inputs:
        with pytest.raises(ValidationError):
            sanitize_input(malicious)


def test_sanitize_input_null_bytes_removed():
    """Test null bytes are removed"""
    text = "Hello\x00World"
    result = sanitize_input(text)
    assert '\x00' not in result


def test_sanitize_input_non_string():
    """Test sanitizing non-string input"""
    result = sanitize_input(123)
    assert result == "123"


# ============================================
# Test validate_model_config()
# ============================================

def test_validate_model_config_valid_xgboost():
    """Test valid XGBoost model config"""
    config = {
        'model_type': 'xgboost',
        'hyperparameters': {
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 100
        }
    }
    assert validate_model_config(config) == True


def test_validate_model_config_valid_lightgbm():
    """Test valid LightGBM model config"""
    config = {
        'model_type': 'lightgbm',
        'hyperparameters': {
            'num_leaves': 31,
            'learning_rate': 0.05
        }
    }
    assert validate_model_config(config) == True


def test_validate_model_config_missing_fields():
    """Test model config with missing fields"""
    config = {
        'model_type': 'xgboost'
    }
    with pytest.raises(ValidationError):
        validate_model_config(config)


def test_validate_model_config_invalid_model_type():
    """Test invalid model type"""
    config = {
        'model_type': 'invalid_model',
        'hyperparameters': {}
    }
    with pytest.raises(ValidationError):
        validate_model_config(config)


def test_validate_model_config_invalid_learning_rate():
    """Test invalid learning rate"""
    config = {
        'model_type': 'xgboost',
        'hyperparameters': {
            'learning_rate': 1.5  # Invalid: > 1
        }
    }
    with pytest.raises(ValidationError):
        validate_model_config(config)


def test_validate_model_config_invalid_max_depth():
    """Test invalid max depth"""
    config = {
        'model_type': 'xgboost',
        'hyperparameters': {
            'max_depth': 25  # Invalid: > 20
        }
    }
    with pytest.raises(ValidationError):
        validate_model_config(config)


# ============================================
# Test validate_email()
# ============================================

def test_validate_email_valid():
    """Test valid email addresses"""
    valid_emails = [
        'user@example.com',
        'john.doe@company.org',
        'test+tag@domain.co.uk',
        'name123@test-domain.com'
    ]
    
    for email in valid_emails:
        assert validate_email(email) == True


def test_validate_email_invalid():
    """Test invalid email addresses"""
    invalid_emails = [
        'not-an-email',
        '@example.com',
        'user@',
        'user@domain',
        'user domain@example.com'
    ]
    
    for email in invalid_emails:
        with pytest.raises(ValidationError):
            validate_email(email)


# ============================================
# Test validate_phone_number()
# ============================================

def test_validate_phone_number_valid():
    """Test valid phone numbers"""
    valid_phones = [
        '1234567890',
        '(123) 456-7890',
        '123-456-7890',
        '+1-123-456-7890',
        '+44 20 1234 5678'
    ]
    
    for phone in valid_phones:
        assert validate_phone_number(phone) == True


def test_validate_phone_number_too_short():
    """Test phone number too short"""
    with pytest.raises(ValidationError):
        validate_phone_number('12345')


def test_validate_phone_number_too_long():
    """Test phone number too long"""
    with pytest.raises(ValidationError):
        validate_phone_number('1234567890123456')


# ============================================
# Test validate_file_path()
# ============================================

def test_validate_file_path_valid():
    """Test valid file paths"""
    valid_paths = [
        'data/resumes/resume.pdf',
        'uploads/file.docx',
        'files/document.txt'
    ]
    
    for path in valid_paths:
        assert validate_file_path(path) == True


def test_validate_file_path_traversal_attack():
    """Test path traversal attack prevention"""
    malicious_paths = [
        '../../../etc/passwd',
        'data/../../../secret.txt',
        '~/secret/file.txt',
        'uploads//malicious.exe'
    ]
    
    for path in malicious_paths:
        with pytest.raises(ValidationError):
            validate_file_path(path)


# ============================================
# Test validate_score_range()
# ============================================

def test_validate_score_range_valid():
    """Test valid scores"""
    assert validate_score_range(0.0) == True
    assert validate_score_range(0.5) == True
    assert validate_score_range(1.0) == True
    assert validate_score_range(0.75, 0.0, 1.0) == True


def test_validate_score_range_custom_range():
    """Test score with custom range"""
    assert validate_score_range(50, 0, 100) == True
    assert validate_score_range(0.5, 0, 1) == True


def test_validate_score_range_out_of_bounds():
    """Test score out of valid range"""
    with pytest.raises(ValidationError):
        validate_score_range(1.5)
    
    with pytest.raises(ValidationError):
        validate_score_range(-0.1)


def test_validate_score_range_invalid_type():
    """Test score with invalid type"""
    with pytest.raises(ValidationError):
        validate_score_range("0.5")


# ============================================
# Run Tests
# ============================================

if __name__ == '__main__':
    # Run all tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])
    
    print("\n" + "="*50)
    print("✅ All validation tests completed!")
    print("="*50)