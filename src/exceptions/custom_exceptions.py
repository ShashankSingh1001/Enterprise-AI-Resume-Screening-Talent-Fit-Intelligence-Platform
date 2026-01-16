"""
Custom Exception Classes for Resume AI Platform
Provides detailed error tracking with file, line, and function information
"""

import sys
from typing import Optional


def get_detailed_error_message(error: Exception, error_detail: sys) -> str:
    """
    Extract detailed error information including file, line, and function.
    
    Args:
        error: The exception object
        error_detail: sys module for traceback extraction
        
    Returns:
        Formatted error message with location details
    """
    _, _, exc_tb = error_detail.exc_info()
    
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    function_name = exc_tb.tb_frame.f_code.co_name
    
    error_message = (
        f"Error occurred in module [{file_name}] "
        f"at line [{line_number}] "
        f"in function [{function_name}]: "
        f"{str(error)}"
    )
    
    return error_message


class ResumeAIException(Exception):
    """
    Base exception class for Resume AI Platform.
    Captures detailed error location information for debugging.
    """
    
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(error_message)
        self.error_message = get_detailed_error_message(error_message, error_detail)
    
    def __str__(self) -> str:
        return self.error_message
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: {self.error_message}"


class FileProcessingError(ResumeAIException):
    """
    Raised when file processing operations fail.
    Includes reading PDFs, DOCX files, or file validation errors.
    """
    pass


class ResumeParsingError(ResumeAIException):
    """
    Raised when resume parsing fails.
    Occurs during NLP extraction of skills, education, or experience from resumes.
    """
    
    def __init__(self, error_message: Exception, error_detail: sys, resume_id: Optional[str] = None):
        super().__init__(error_message, error_detail)
        self.resume_id = resume_id


class JDParsingError(ResumeAIException):
    """
    Raised when job description parsing fails.
    Occurs during NLP extraction of requirements and qualifications from JDs.
    """
    pass


class ModelPredictionError(ResumeAIException):
    """
    Raised when model prediction or inference fails.
    Includes errors during feature extraction, model loading, or prediction execution.
    """
    
    def __init__(self, error_message: Exception, error_detail: sys, model_name: Optional[str] = None):
        super().__init__(error_message, error_detail)
        self.model_name = model_name


class BiasDetectionError(ResumeAIException):
    """
    Raised during fairness auditing and bias detection.
    Separate handling required for compliance and reporting purposes.
    """
    
    def __init__(self, error_message: Exception, error_detail: sys, 
                 sensitive_feature: Optional[str] = None):
        super().__init__(error_message, error_detail)
        self.sensitive_feature = sensitive_feature


class MLflowTrackingError(ResumeAIException):
    """
    Raised when MLflow experiment tracking or model registry operations fail.
    Includes logging metrics, parameters, or artifact storage errors.
    """
    pass


class DatabaseConnectionError(ResumeAIException):
    """
    Raised when database operations fail.
    Includes connection errors, query failures, or transaction issues.
    """
    pass


class ValidationError(ResumeAIException):
    """
    Raised when input validation fails.
    Used for API request validation and data integrity checks.
    """
    pass


# Exception hierarchy documentation
__all__ = [
    'ResumeAIException',
    'FileProcessingError',
    'ResumeParsingError',
    'JDParsingError',
    'ModelPredictionError',
    'BiasDetectionError',
    'MLflowTrackingError',
    'DatabaseConnectionError',
    'ValidationError',
    'get_detailed_error_message'
]