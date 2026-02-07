"""
Custom Exceptions for Resume Screening Platform
Keyword-only parameters for safety
"""

from typing import Optional


class ResumeScreeningError(Exception):
    """
    Base exception for all resume screening errors.
    Enforces keyword-only parameters to avoid positional misuse.
    """

    def __init__(self, *, message: str, error_detail: Optional[object] = None):
        self.message = message
        self.error_detail = error_detail
        super().__init__(self.message)

    def __repr__(self):
        return f"{self.__class__.__name__}(message={self.message!r})"


class ResumeParsingError(ResumeScreeningError):
    """
    Raised when resume parsing fails.
    """

    def __init__(
        self,
        *,
        message: str,
        resume_id: Optional[str] = None,
        error_detail: Optional[object] = None
    ):
        self.resume_id = resume_id
        super().__init__(message=message, error_detail=error_detail)

    def __str__(self):
        if self.resume_id:
            return f"Resume {self.resume_id}: {self.message}"
        return self.message


class JDParsingError(ResumeScreeningError):
    """
    Raised when job description parsing fails.
    """

    def __init__(
        self,
        *,
        message: str,
        jd_id: Optional[str] = None,
        error_detail: Optional[object] = None
    ):
        self.jd_id = jd_id
        super().__init__(message=message, error_detail=error_detail)

    def __str__(self):
        if self.jd_id:
            return f"JD {self.jd_id}: {self.message}"
        return self.message


class DataValidationError(ResumeScreeningError):
    """
    Raised when data validation fails.
    """
    pass


class FeatureExtractionError(ResumeScreeningError):
    """
    Raised when feature extraction fails.
    """
    pass


class ModelError(ResumeScreeningError):
    """
    Raised when ML model operations fail.
    """
    pass


__all__ = [
    "ResumeScreeningError",
    "ResumeParsingError",
    "JDParsingError",
    "DataValidationError",
    "FeatureExtractionError",
    "ModelError",
]

# =====================================================
# Backward-compatible aliases (REQUIRED)
# =====================================================

# Validation
ValidationError = DataValidationError

# File note:
# FileProcessingError is intentionally mapped to ResumeScreeningError
FileProcessingError = ResumeScreeningError

# Model prediction
ModelPredictionError = ModelError
