"""
Experience Features Calculator
Calculates experience-related features and gap analysis
"""

import logging
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperienceCalculator:
    """
    Calculates experience features comparing resume and JD requirements
    """
    
    # Experience level classifications
    EXPERIENCE_LEVELS = {
        "entry": (0, 2),      # 0-2 years
        "junior": (2, 4),     # 2-4 years
        "mid": (4, 7),        # 4-7 years
        "senior": (7, 10),    # 7-10 years
        "lead": (10, 15),     # 10-15 years
        "executive": (15, 100) # 15+ years
    }
    
    def __init__(self):
        """Initialize experience calculator"""
        logger.info("ExperienceCalculator initialized")
    
    def calculate_total_experience(self, experience_blocks: List[Dict]) -> float:
        """
        Calculate total years of experience from experience blocks
        
        Args:
            experience_blocks: List of dicts with 'start_date', 'end_date' or 'duration'
                Example: [
                    {"start_date": "Jan 2020", "end_date": "Dec 2023"},
                    {"duration": "2 years"}
                ]
                
        Returns:
            Total years of experience
        """
        total_years = 0.0
        
        for block in experience_blocks:
            # Method 1: Parse duration directly
            if "duration" in block and block["duration"]:
                years = self._parse_duration(block["duration"])
                total_years += years
            
            # Method 2: Calculate from start/end dates
            elif "start_date" in block and "end_date" in block:
                start = block["start_date"]
                end = block["end_date"]
                years = self._calculate_duration(start, end)
                total_years += years
        
        logger.debug(f"Total experience calculated: {total_years:.1f} years")
        return total_years
    
    def _parse_duration(self, duration_str: str) -> float:
        """
        Parse duration string to years
        
        Args:
            duration_str: Duration string like "3 years", "6 months", "2.5 years"
            
        Returns:
            Duration in years
        """
        if not duration_str:
            return 0.0
        
        duration_lower = duration_str.lower()
        years = 0.0
        
        # Extract years
        year_match = re.search(r'(\d+\.?\d*)\s*(?:year|yr)', duration_lower)
        if year_match:
            years += float(year_match.group(1))
        
        # Extract months
        month_match = re.search(r'(\d+\.?\d*)\s*(?:month|mon|mo)', duration_lower)
        if month_match:
            years += float(month_match.group(1)) / 12
        
        return years
    
    def _calculate_duration(self, start_date: str, end_date: str) -> float:
        """
        Calculate duration between two dates
        
        Args:
            start_date: Start date string (various formats)
            end_date: End date string or "Present"
            
        Returns:
            Duration in years
        """
        # Handle "Present" or "Current"
        if end_date.lower() in ["present", "current", "now"]:
            end_date = datetime.now().strftime("%b %Y")
        
        # Parse dates (simplified - handles common formats)
        start_year, start_month = self._parse_date(start_date)
        end_year, end_month = self._parse_date(end_date)
        
        if start_year and end_year:
            years = end_year - start_year
            months = end_month - start_month
            return years + (months / 12)
        
        return 0.0
    
    def _parse_date(self, date_str: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Parse date string to (year, month)
        
        Args:
            date_str: Date string like "Jan 2020", "2020", "January 2020"
            
        Returns:
            Tuple of (year, month) or (None, None)
        """
        if not date_str:
            return None, None
        
        date_str = date_str.strip()
        
        # Extract year (4 digits)
        year_match = re.search(r'(20\d{2}|19\d{2})', date_str)
        if not year_match:
            return None, None
        
        year = int(year_match.group(1))
        
        # Extract month (default to January if not found)
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        month = 1  # Default
        for month_name, month_num in month_map.items():
            if month_name in date_str.lower():
                month = month_num
                break
        
        return year, month
    
    def extract_required_experience(self, jd_text: str) -> Dict[str, float]:
        """
        Extract required experience from JD text
        
        Args:
            jd_text: Job description text
            
        Returns:
            Dict with min/max required years
        """
        jd_lower = jd_text.lower()
        
        # Patterns for experience requirements
        patterns = [
        r'(\d+)\s*-\s*(\d+)\s*years?',   # RANGE FIRST
        r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience',
        r'experience[:\s]+(\d+)\s*\+?\s*years?',
        r'minimum\s+(\d+)\s*years?',
        r'at\s+least\s+(\d+)\s*years?'
        ]
        
        min_years = None
        max_years = None
        
        for pattern in patterns:
            matches = re.findall(pattern, jd_lower)
            if matches:
                if isinstance(matches[0], tuple):
                    # Range pattern (e.g., "5-7 years")
                    min_years = float(matches[0][0])
                    max_years = float(matches[0][1])
                else:
                    # Single value pattern
                    years = float(matches[0])
                    if min_years is None:
                        min_years = years
                    else:
                        max_years = years
                break
        
        # Default values if not found
        if min_years is None:
            min_years = 0.0
        
        result = {
            "min_years": min_years,
            "max_years": max_years if max_years else min_years,
            "preferred_years": max_years if max_years else min_years
        }
        
        logger.debug(f"Required experience: {result}")
        return result
    
    def calculate_experience_gap(
        self,
        candidate_years: float,
        required_years: float
    ) -> float:
        """
        Calculate experience gap (positive = exceeds, negative = falls short)
        
        Args:
            candidate_years: Candidate's total experience
            required_years: Required years from JD
            
        Returns:
            Gap in years (can be negative)
        """
        gap = candidate_years - required_years
        logger.debug(f"Experience gap: {gap:.1f} years")
        return gap
    
    def calculate_experience_match_score(
        self,
        candidate_years: float,
        required_min: float,
        required_max: Optional[float] = None
    ) -> float:
        """
        Calculate experience match score (0-1)
        
        Args:
            candidate_years: Candidate's total experience
            required_min: Minimum required years
            required_max: Maximum required years (optional)
            
        Returns:
            Match score between 0 and 1
        """
        # Perfect match range: ±1 year from requirement
        perfect_range = 1.0
        
        # If in required range, score is 1.0
        if required_max:
            if required_min <= candidate_years <= required_max:
                return 1.0
        else:
            # Single requirement - check if within perfect range
            if abs(candidate_years - required_min) <= perfect_range:
                return 1.0
        
        # Calculate penalty for being outside range
        if required_max:
            # Has range
            if candidate_years < required_min:
                # Under-experienced
                gap = required_min - candidate_years
                penalty = min(0.5, gap * 0.1)  # Max 50% penalty
                return max(0.0, 1.0 - penalty)
            else:
                # Over-experienced
                gap = candidate_years - required_max
                penalty = min(0.3, gap * 0.05)  # Max 30% penalty (less harsh)
                return max(0.0, 1.0 - penalty)
        else:
            # Single requirement
            gap = abs(candidate_years - required_min)
            penalty = min(0.5, gap * 0.1)
            return max(0.0, 1.0 - penalty)
    
    def classify_experience_level(self, years: float) -> str:
        """
        Classify experience into levels
        
        Args:
            years: Years of experience
            
        Returns:
            Experience level (entry/junior/mid/senior/lead/executive)
        """
        for level, (min_years, max_years) in self.EXPERIENCE_LEVELS.items():
            if min_years <= years < max_years:
                return level
        
        return "executive"  # Default for 15+ years
    
    def calculate_relevant_experience(
        self,
        experience_blocks: List[Dict],
        relevant_keywords: List[str]
    ) -> float:
        """
        Calculate years of relevant experience based on keywords
        
        Args:
            experience_blocks: List of experience dicts with 'description' field
            relevant_keywords: List of relevant keywords (skills, domains)
            
        Returns:
            Years of relevant experience
        """
        relevant_years = 0.0
        
        for block in experience_blocks:
            # Check if block description contains relevant keywords
            description = block.get("description", "").lower()
            
            is_relevant = any(
                keyword.lower() in description 
                for keyword in relevant_keywords
            )
            
            if is_relevant:
                # Calculate duration for this block
                if "duration" in block:
                    years = self._parse_duration(block["duration"])
                elif "start_date" in block and "end_date" in block:
                    years = self._calculate_duration(
                        block["start_date"], 
                        block["end_date"]
                    )
                else:
                    years = 0.0
                
                relevant_years += years
        
        logger.debug(f"Relevant experience: {relevant_years:.1f} years")
        return relevant_years
    
    def calculate_experience_features(
        self,
        resume_experience_blocks: List[Dict],
        jd_text: str,
        relevant_keywords: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive experience features
        
        Args:
            resume_experience_blocks: List of experience dicts from resume
            jd_text: Job description text
            relevant_keywords: Optional list of relevant keywords
            
        Returns:
            Dict with all experience features
        """
        features = {}
        
        # Total experience
        total_years = self.calculate_total_experience(resume_experience_blocks)
        features["total_experience_years"] = total_years
        
        # Experience level
        exp_level = self.classify_experience_level(total_years)
        features["experience_level"] = self._encode_experience_level(exp_level)
        
        # Required experience from JD
        required = self.extract_required_experience(jd_text)
        features["required_min_years"] = required["min_years"]
        features["required_max_years"] = required["max_years"]
        
        # Experience gap
        gap = self.calculate_experience_gap(
            total_years, 
            required["min_years"]
        )
        features["experience_gap"] = gap
        features["experience_gap_abs"] = abs(gap)
        
        # Match score
        match_score = self.calculate_experience_match_score(
            total_years,
            required["min_years"],
            required["max_years"]
        )
        features["experience_match_score"] = match_score
        
        # Relevant experience
        if relevant_keywords:
            relevant_years = self.calculate_relevant_experience(
                resume_experience_blocks,
                relevant_keywords
            )
            features["relevant_experience_years"] = relevant_years
            features["relevant_experience_ratio"] = (
                relevant_years / total_years if total_years > 0 else 0.0
            )
        else:
            features["relevant_experience_years"] = 0.0
            features["relevant_experience_ratio"] = 0.0
        
        # Number of positions
        features["num_positions"] = len(resume_experience_blocks)
        
        # Average tenure (years per position)
        if len(resume_experience_blocks) > 0:
            features["avg_tenure_years"] = total_years / len(resume_experience_blocks)
        else:
            features["avg_tenure_years"] = 0.0
        
        # Boolean flags
        features["meets_min_experience"] = 1.0 if total_years >= required["min_years"] else 0.0
        features["overqualified"] = 1.0 if total_years > required["max_years"] + 3 else 0.0
        features["underqualified"] = 1.0 if total_years < required["min_years"] - 1 else 0.0
        
        logger.info(f"Experience features calculated: {len(features)} features")
        return features
    
    def _encode_experience_level(self, level: str) -> float:
        """
        Encode experience level as numeric value
        
        Args:
            level: Experience level string
            
        Returns:
            Numeric encoding (0-5)
        """
        encoding = {
            "entry": 0.0,
            "junior": 1.0,
            "mid": 2.0,
            "senior": 3.0,
            "lead": 4.0,
            "executive": 5.0
        }
        return encoding.get(level, 0.0)
    
    def get_experience_summary(
        self,
        resume_experience_blocks: List[Dict],
        jd_text: str
    ) -> Dict[str, any]:
        """
        Get human-readable experience summary
        
        Args:
            resume_experience_blocks: List of experience dicts
            jd_text: Job description text
            
        Returns:
            Dict with summary info
        """
        total_years = self.calculate_total_experience(resume_experience_blocks)
        required = self.extract_required_experience(jd_text)
        gap = self.calculate_experience_gap(total_years, required["min_years"])
        match_score = self.calculate_experience_match_score(
            total_years,
            required["min_years"],
            required["max_years"]
        )
        
        # Generate recommendation
        if match_score >= 0.9:
            recommendation = "Excellent Experience Match"
        elif match_score >= 0.7:
            recommendation = "Good Experience Match"
        elif match_score >= 0.5:
            recommendation = "Acceptable Experience"
        else:
            recommendation = "Experience Mismatch"
        
        summary = {
            "total_years": total_years,
            "required_years": required["min_years"],
            "experience_gap": gap,
            "match_score": match_score,
            "level": self.classify_experience_level(total_years),
            "num_positions": len(resume_experience_blocks),
            "recommendation": recommendation
        }
        
        return summary


# Convenience function
def calculate_experience_match(
    resume_experience_blocks: List[Dict],
    jd_text: str
) -> float:
    """
    Quick utility to calculate experience match score
    
    Args:
        resume_experience_blocks: List of experience dicts
        jd_text: Job description text
        
    Returns:
        Match score between 0 and 1
    """
    calc = ExperienceCalculator()
    total_years = calc.calculate_total_experience(resume_experience_blocks)
    required = calc.extract_required_experience(jd_text)
    return calc.calculate_experience_match_score(
        total_years,
        required["min_years"],
        required["max_years"]
    )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example experience blocks from resume
    experience_blocks = [
        {
            "title": "Senior ML Engineer",
            "company": "Tech Corp",
            "start_date": "Jan 2020",
            "end_date": "Present",
            "description": "Building ML models for NLP using Python and TensorFlow"
        },
        {
            "title": "ML Engineer",
            "company": "AI Startup",
            "duration": "2 years 6 months",
            "description": "Developed deep learning models"
        },
        {
            "title": "Software Engineer",
            "company": "Big Company",
            "start_date": "Jun 2017",
            "end_date": "Dec 2019",
            "description": "Backend development with Python"
        }
    ]
    
    # Example JD
    jd_text = """
    We're seeking a Senior Machine Learning Engineer with 5-7 years of experience.
    Strong background in Python, NLP, and deep learning required.
    """
    
    calc = ExperienceCalculator()
    
    # Calculate features
    features = calc.calculate_experience_features(
        experience_blocks,
        jd_text,
        relevant_keywords=["ML", "Machine Learning", "NLP", "Python"]
    )
    
    print("\n=== Experience Features ===")
    for key, value in features.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Get summary
    summary = calc.get_experience_summary(experience_blocks, jd_text)
    
    print("\n=== Experience Summary ===")
    print(f"Total Experience: {summary['total_years']:.1f} years")
    print(f"Required: {summary['required_years']:.1f} years")
    print(f"Gap: {summary['experience_gap']:.1f} years")
    print(f"Match Score: {summary['match_score']:.2f}")
    print(f"Level: {summary['level']}")
    print(f"Recommendation: {summary['recommendation']}")