"""
Date and Time Utilities for Resume AI Platform
Production-safe, exception-consistent, pytest-ready
"""

import re
import sys
from datetime import datetime
from typing import Optional, Tuple, List

from dateutil import parser
from dateutil.relativedelta import relativedelta

from src.logging import get_logger
from src.exceptions import DataValidationError

logger = get_logger(__name__)

# Accepted date formats (fast path)
DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d",
    "%d-%m-%Y", "%d/%m/%Y",
    "%m-%d-%Y", "%m/%d/%Y",
    "%B %Y", "%b %Y",
    "%Y-%m", "%m/%Y",
    "%Y",
]

CURRENT_KEYWORDS = {
    "present", "current", "now", "ongoing", "till date", "today"
}


# ------------------------------------------------------------------
# Core Parsing
# ------------------------------------------------------------------

def parse_date(date_str: str, fuzzy: bool = True) -> Optional[datetime]:
    """
    Parse date string into datetime.
    Non-throwing helper by design.
    """
    if not date_str or not isinstance(date_str, str):
        return None

    date_str = date_str.strip()

    if any(k in date_str.lower() for k in CURRENT_KEYWORDS):
        return datetime.now()

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    try:
        return parser.parse(date_str, fuzzy=fuzzy)
    except Exception:
        return None


# ------------------------------------------------------------------
# Experience Calculations (STRICT)
# ------------------------------------------------------------------

def calculate_experience_duration(
    start_date: str,
    end_date: str = "Present"
) -> Tuple[float, int, int]:
    """
    Calculate experience duration.
    STRICT: raises DataValidationError on invalid dates.
    """
    try:
        start = parse_date(start_date)
        if start is None:
            raise DataValidationError(
                message=f"Invalid start date: {start_date}",
                error_detail=sys.exc_info()
            )

        end = parse_date(end_date)
        if end is None:
            raise DataValidationError(
                message=f"Invalid end date: {end_date}",
                error_detail=sys.exc_info()
            )

        if start > end:
            start, end = end, start

        delta = relativedelta(end, start)

        total_years = (
            delta.years +
            delta.months / 12.0 +
            delta.days / 365.25
        )

        total_months = delta.years * 12 + delta.months
        total_days = (end - start).days

        return round(total_years, 2), total_months, total_days

    except DataValidationError:
        raise
    except Exception as e:
        raise DataValidationError(
            message="Experience duration calculation failed",
            error_detail=sys.exc_info()
        )


def calculate_total_experience(experience_list: List[dict]) -> float:
    """
    Calculate total experience in years.
    Non-fatal: skips invalid entries.
    """
    total_years = 0.0

    for idx, job in enumerate(experience_list):
        start = job.get("start_date")
        end = job.get("end_date")

        if not start or not end:
            continue

        try:
            years, _, _ = calculate_experience_duration(start, end)
            total_years += years
        except DataValidationError:
            logger.warning(f"Skipping invalid job entry at index {idx}")

    return round(total_years, 2)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def format_date(date_obj: datetime, format_str: str = "%Y-%m-%d") -> str:
    if not isinstance(date_obj, datetime):
        return str(date_obj)
    return date_obj.strftime(format_str)


def is_date_valid(date_str: str) -> bool:
    return parse_date(date_str) is not None


def is_future_date(date_str: str) -> bool:
    parsed = parse_date(date_str)
    return parsed is not None and parsed > datetime.now()


def extract_year(date_str: str) -> Optional[int]:
    parsed = parse_date(date_str)
    if parsed:
        return parsed.year

    match = re.search(r"\b(19|20)\d{2}\b", date_str)
    return int(match.group()) if match else None


def extract_month_year(date_str: str) -> Optional[Tuple[int, int]]:
    parsed = parse_date(date_str)
    if not parsed:
        return None
    return parsed.month, parsed.year


def standardize_date_format(
    date_str: str,
    output_format: str = "%Y-%m-%d"
) -> Optional[str]:
    parsed = parse_date(date_str)
    if not parsed:
        return None
    return format_date(parsed, output_format)


def calculate_age(birth_date: str) -> Optional[int]:
    parsed = parse_date(birth_date)
    if not parsed:
        return None

    today = datetime.now()
    age = today.year - parsed.year
    if (today.month, today.day) < (parsed.month, parsed.day):
        age -= 1
    return age


def get_date_range_overlap(
    start1: str, end1: str,
    start2: str, end2: str
) -> Tuple[bool, float]:
    s1, e1 = parse_date(start1), parse_date(end1)
    s2, e2 = parse_date(start2), parse_date(end2)

    if None in (s1, e1, s2, e2):
        return False, 0.0

    if not (s1 <= e2 and s2 <= e1):
        return False, 0.0

    overlap_start = max(s1, s2)
    overlap_end = min(e1, e2)

    years, _, _ = calculate_experience_duration(
        format_date(overlap_start),
        format_date(overlap_end)
    )

    return True, years


# ------------------------------------------------------------------
# Time Helpers
# ------------------------------------------------------------------

def get_current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")
