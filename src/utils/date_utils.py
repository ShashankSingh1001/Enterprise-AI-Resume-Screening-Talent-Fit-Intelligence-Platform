"""
Date and Time Utilities for Resume AI Platform
Provides date parsing, formatting, and experience calculation functions
"""

import re
import sys
from datetime import datetime
from typing import Optional, Tuple
from dateutil import parser
from dateutil.relativedelta import relativedelta

from src.logging import get_logger
from src.exceptions import ValidationError

logger = get_logger(__name__)

# Common date format patterns
DATE_FORMATS = [
    '%Y-%m-%d',           # 2023-06-15
    '%Y/%m/%d',           # 2023/06/15
    '%d-%m-%Y',           # 15-06-2023
    '%d/%m/%Y',           # 15/06/2023
    '%m-%d-%Y',           # 06-15-2023
    '%m/%d/%Y',           # 06/15/2023
    '%B %Y',              # June 2023
    '%b %Y',              # Jun 2023
    '%Y-%m',              # 2023-06
    '%m/%Y',              # 06/2023
    '%Y',                 # 2023
]

# Keywords indicating current/ongoing position
CURRENT_KEYWORDS = ['present', 'current', 'now', 'ongoing', 'till date', 'today']


def parse_date(date_str: str, fuzzy: bool = True) -> Optional[datetime]:
    """
    Parse date string into datetime object.
    Handles multiple formats: "Jun 2020", "06/2020", "2020-06", etc.
    
    Args:
        date_str: Date string to parse
        fuzzy: Allow fuzzy parsing (ignore extra text)
        
    Returns:
        datetime object or None if parsing fails
    """
    try:
        if not date_str or not isinstance(date_str, str):
            logger.warning(f"Invalid date string provided: {date_str}")
            return None
        
        # Clean the input
        date_str = date_str.strip()
        
        # Check for current/present keywords
        if any(keyword in date_str.lower() for keyword in CURRENT_KEYWORDS):
            logger.debug(f"Date indicates current position: {date_str}")
            return datetime.now()
        
        # Try standard formats first
        for fmt in DATE_FORMATS:
            try:
                parsed = datetime.strptime(date_str, fmt)
                logger.debug(f"Date parsed with format '{fmt}': {date_str} -> {parsed}")
                return parsed
            except ValueError:
                continue
        
        # Try dateutil parser (more flexible)
        try:
            parsed = parser.parse(date_str, fuzzy=fuzzy)
            logger.debug(f"Date parsed with dateutil: {date_str} -> {parsed}")
            return parsed
        except (ValueError, parser.ParserError):
            pass
        
        logger.warning(f"Could not parse date: {date_str}")
        return None
    
    except Exception as e:
        logger.error(f"Date parsing failed for '{date_str}': {str(e)}")
        return None


def calculate_experience_duration(
    start_date: str, 
    end_date: str = "Present"
) -> Tuple[float, int, int]:
    """
    Calculate duration between two dates.
    Returns years (float), total months, and total days.
    
    Args:
        start_date: Start date string
        end_date: End date string (default: "Present")
        
    Returns:
        Tuple of (years_float, total_months, total_days)
        
    Raises:
        ValidationError: If date parsing fails
    """
    try:
        logger.debug(f"Calculating experience: {start_date} to {end_date}")
        
        # Parse start date
        start = parse_date(start_date)
        if start is None:
            raise ValidationError(
                Exception(f"Could not parse start date: {start_date}"),
                sys
            )
        
        # Parse end date
        end = parse_date(end_date)
        if end is None:
            raise ValidationError(
                Exception(f"Could not parse end date: {end_date}"),
                sys
            )
        
        # Ensure start is before end
        if start > end:
            logger.warning(f"Start date after end date, swapping: {start_date} <-> {end_date}")
            start, end = end, start
        
        # Calculate difference
        delta = relativedelta(end, start)
        
        # Calculate total values
        total_years = delta.years + (delta.months / 12.0) + (delta.days / 365.25)
        total_months = delta.years * 12 + delta.months
        total_days = (end - start).days
        
        logger.info(
            f"Experience calculated: {total_years:.2f} years "
            f"({total_months} months, {total_days} days)"
        )
        
        return round(total_years, 2), total_months, total_days
    
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Experience calculation failed: {str(e)}")
        raise ValidationError(e, sys)


def calculate_total_experience(experience_list: list) -> float:
    """
    Calculate total experience from list of job periods.
    
    Args:
        experience_list: List of dicts with 'start_date' and 'end_date'
        
    Returns:
        Total experience in years
        
    Example:
        experience_list = [
            {'start_date': 'Jan 2020', 'end_date': 'Dec 2021'},
            {'start_date': 'Jan 2022', 'end_date': 'Present'}
        ]
    """
    try:
        total_years = 0.0
        
        for i, job in enumerate(experience_list):
            if 'start_date' not in job or 'end_date' not in job:
                logger.warning(f"Skipping job {i}: missing date fields")
                continue
            
            try:
                years, _, _ = calculate_experience_duration(
                    job['start_date'], 
                    job['end_date']
                )
                total_years += years
                logger.debug(f"Job {i}: {years} years")
            except Exception as e:
                logger.warning(f"Could not calculate experience for job {i}: {str(e)}")
                continue
        
        logger.info(f"Total experience calculated: {total_years:.2f} years")
        return round(total_years, 2)
    
    except Exception as e:
        logger.error(f"Total experience calculation failed: {str(e)}")
        return 0.0


def format_date(date_obj: datetime, format_str: str = '%Y-%m-%d') -> str:
    """
    Format datetime object to string.
    
    Args:
        date_obj: datetime object
        format_str: Output format string
        
    Returns:
        Formatted date string
    """
    try:
        if not isinstance(date_obj, datetime):
            logger.warning(f"Invalid datetime object: {type(date_obj)}")
            return str(date_obj)
        
        formatted = date_obj.strftime(format_str)
        logger.debug(f"Date formatted: {date_obj} -> {formatted}")
        return formatted
    
    except Exception as e:
        logger.error(f"Date formatting failed: {str(e)}")
        return str(date_obj)


def is_date_valid(date_str: str) -> bool:
    """
    Check if date string is valid and can be parsed.
    
    Args:
        date_str: Date string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        parsed = parse_date(date_str)
        is_valid = parsed is not None
        
        if is_valid:
            logger.debug(f"Date is valid: {date_str}")
        else:
            logger.debug(f"Date is invalid: {date_str}")
        
        return is_valid
    
    except Exception as e:
        logger.error(f"Date validation failed: {str(e)}")
        return False


def is_future_date(date_str: str) -> bool:
    """
    Check if date is in the future.
    
    Args:
        date_str: Date string to check
        
    Returns:
        True if date is in future, False otherwise
    """
    try:
        parsed = parse_date(date_str)
        if parsed is None:
            return False
        
        is_future = parsed > datetime.now()
        logger.debug(f"Date {'is' if is_future else 'is not'} in future: {date_str}")
        return is_future
    
    except Exception as e:
        logger.error(f"Future date check failed: {str(e)}")
        return False


def extract_year(date_str: str) -> Optional[int]:
    """
    Extract year from date string.
    
    Args:
        date_str: Date string
        
    Returns:
        Year as integer or None
    """
    try:
        parsed = parse_date(date_str)
        if parsed is None:
            # Try to extract 4-digit year using regex
            match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if match:
                year = int(match.group())
                logger.debug(f"Year extracted via regex: {year}")
                return year
            return None
        
        year = parsed.year
        logger.debug(f"Year extracted: {year} from {date_str}")
        return year
    
    except Exception as e:
        logger.error(f"Year extraction failed: {str(e)}")
        return None


def extract_month_year(date_str: str) -> Optional[Tuple[int, int]]:
    """
    Extract month and year from date string.
    
    Args:
        date_str: Date string
        
    Returns:
        Tuple of (month, year) or None
    """
    try:
        parsed = parse_date(date_str)
        if parsed is None:
            return None
        
        month = parsed.month
        year = parsed.year
        logger.debug(f"Month-Year extracted: {month}/{year} from {date_str}")
        return (month, year)
    
    except Exception as e:
        logger.error(f"Month-Year extraction failed: {str(e)}")
        return None


def standardize_date_format(date_str: str, output_format: str = '%Y-%m-%d') -> Optional[str]:
    """
    Standardize date string to consistent format.
    
    Args:
        date_str: Input date string (any format)
        output_format: Desired output format
        
    Returns:
        Standardized date string or None
    """
    try:
        parsed = parse_date(date_str)
        if parsed is None:
            return None
        
        standardized = format_date(parsed, output_format)
        logger.debug(f"Date standardized: {date_str} -> {standardized}")
        return standardized
    
    except Exception as e:
        logger.error(f"Date standardization failed: {str(e)}")
        return None


def calculate_age(birth_date: str) -> Optional[int]:
    """
    Calculate age from birth date.
    
    Args:
        birth_date: Birth date string
        
    Returns:
        Age in years or None
    """
    try:
        parsed = parse_date(birth_date)
        if parsed is None:
            return None
        
        today = datetime.now()
        age = today.year - parsed.year
        
        # Adjust if birthday hasn't occurred this year
        if (today.month, today.day) < (parsed.month, parsed.day):
            age -= 1
        
        logger.debug(f"Age calculated: {age} years from {birth_date}")
        return age
    
    except Exception as e:
        logger.error(f"Age calculation failed: {str(e)}")
        return None


def get_date_range_overlap(
    start1: str, end1: str,
    start2: str, end2: str
) -> Tuple[bool, float]:
    """
    Check if two date ranges overlap and calculate overlap duration.
    
    Args:
        start1: Start of first range
        end1: End of first range
        start2: Start of second range
        end2: End of second range
        
    Returns:
        Tuple of (has_overlap, overlap_years)
    """
    try:
        # Parse all dates
        s1 = parse_date(start1)
        e1 = parse_date(end1)
        s2 = parse_date(start2)
        e2 = parse_date(end2)
        
        if None in [s1, e1, s2, e2]:
            logger.warning("Could not parse all dates for overlap calculation")
            return (False, 0.0)
        
        # Check for overlap
        has_overlap = s1 <= e2 and s2 <= e1
        
        if not has_overlap:
            logger.debug("No overlap found between date ranges")
            return (False, 0.0)
        
        # Calculate overlap duration
        overlap_start = max(s1, s2)
        overlap_end = min(e1, e2)
        
        years, _, _ = calculate_experience_duration(
            format_date(overlap_start),
            format_date(overlap_end)
        )
        
        logger.info(f"Overlap found: {years} years")
        return (True, years)
    
    except Exception as e:
        logger.error(f"Overlap calculation failed: {str(e)}")
        return (False, 0.0)


def get_current_timestamp() -> str:
    """
    Get current timestamp in standard format.
    
    Returns:
        Current timestamp string (YYYY-MM-DD HH:MM:SS)
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def get_current_date() -> str:
    """
    Get current date in standard format.
    
    Returns:
        Current date string (YYYY-MM-DD)
    """
    return datetime.now().strftime('%Y-%m-%d')