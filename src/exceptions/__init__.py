"""
Exception Module
Custom exception classes for Resume AI Platform
"""

from .custom_exceptions import (
    ResumeAIException,
    FileProcessingError,
    ResumeParsingError,
    JDParsingError,
    ModelPredictionError,
    BiasDetectionError,
    MLflowTrackingError,
    DatabaseConnectionError,
    ValidationError,
    get_detailed_error_message
)

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