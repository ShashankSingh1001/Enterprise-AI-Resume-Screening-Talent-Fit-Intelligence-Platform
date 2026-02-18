"""
Education Features Calculator
Calculates education-related features and match scoring
"""

import logging
from typing import Dict, List, Optional, Set
import re

logger = logging.getLogger(__name__)


class EducationCalculator:
    """
    Calculates education features comparing resume and JD requirements
    """
    
    # Education level hierarchy (higher = more advanced)
    EDUCATION_LEVELS = {
        "high school": 1,
        "diploma": 2,
        "associate": 2,
        "bachelor": 3,
        "btech": 3,
        "be": 3,
        "bs": 3,
        "ba": 3,
        "bsc": 3,
        "masters": 4,
        "mtech": 4,
        "me": 4,
        "ms": 4,
        "ma": 4,
        "msc": 4,
        "mba": 4,
        "phd": 5,
        "doctorate": 5
    }
    
    # Tier 1 colleges/universities (expand as needed)
    TIER_1_COLLEGES = {
        # India
        "iit", "iiit", "nit", "bits", "iim",
        # US
        "mit", "stanford", "harvard", "berkeley", "cmu", "caltech",
        # UK
        "oxford", "cambridge", "imperial",
        # Others can be added
    }
    
    # STEM fields
    STEM_FIELDS = {
        "computer science", "cs", "cse", "it", "information technology",
        "software engineering", "artificial intelligence", "ai",
        "machine learning", "data science", "mathematics", "statistics",
        "engineering", "physics", "electronics", "electrical"
    }
    
    # Business/Management fields
    BUSINESS_FIELDS = {
        "business", "mba", "management", "finance", "marketing",
        "economics", "commerce"
    }
    
    def __init__(self):
        """Initialize education calculator"""
        logger.info("EducationCalculator initialized")
    
    def parse_education_level(self, education_text: str) -> int:
        """
        Parse education level from text
        
        Args:
            education_text: Education description text
            
        Returns:
            Education level (1-5)
        """
        if not education_text:
            return 0
        
        text_lower = education_text.lower()
        text_lower = re.sub(r'[^\w\s]', '', text_lower)

        
        # Check for each education level (from highest to lowest)
        for level in ["phd", "doctorate", "masters", "mtech", "mba", "ms", 
                      "bachelor", "btech", "be", "bs", "diploma", "high school"]:
            if level in text_lower:
                return self.EDUCATION_LEVELS.get(level, 0)
        
        return 0
    
    def extract_degree(self, education_text: str) -> str:
        """
        Extract degree name from education text
        
        Args:
            education_text: Education description
            
        Returns:
            Degree name (normalized)
        """
        if not education_text:
            return "unknown"
        
        text_lower = education_text.lower()
        
        # Common degree patterns
        degree_patterns = [
        r'(phd|ph\.d\.?|doctorate)',
        r'(m\.?tech|mtech)',
        r'(master of science|m\.?s\.?|masters?)',
        r'(mba)',
        r'(b\.?tech|btech)',
        r'(b\.?e\.?|be)',
        r'(bachelor of science|b\.?s\.?|bs|bachelor)',
        r'(diploma)',
        r'(high school)'
        ]

        
        for pattern in degree_patterns:
            match = re.search(pattern, text_lower)
            if match:
                degree = match.group(1).replace('.', '').lower()

                # ---- Normalization mapping ----
                if degree in ["phd", "phd", "doctorate"]:
                    return "phd"

                if degree in ["master of science", "ms"]:
                    return "ms"

                if degree.startswith("master"):
                    return "masters"

                if degree in ["mtech"]:
                    return "mtech"

                if degree in ["mba"]:
                    return "mba"

                if degree in ["btech"]:
                    return "btech"

                if degree in ["be"]:
                    return "be"

                if degree in ["bachelor of science", "bs"]:
                    return "bs"

                if degree.startswith("bachelor"):
                    return "bachelors"

                if degree == "diploma":
                    return "diploma"

                if degree == "high school":
                    return "high school"

                return degree

    
    def extract_field_of_study(self, education_text: str) -> str:
        """
        Extract field of study from education text
        
        Args:
            education_text: Education description
            
        Returns:
            Field of study (normalized)
        """
        if not education_text:
            return "unknown"
        
        text_lower = education_text.lower()
        
        # Check STEM fields
        for field in self.STEM_FIELDS:
            if field in text_lower:
                return "stem"
        
        # Check business fields
        for field in self.BUSINESS_FIELDS:
            if field in text_lower:
                return "business"
        
        # General patterns
        if any(word in text_lower for word in ["science", "engineering", "technology"]):
            return "stem"
        
        return "other"
    
    def is_tier1_college(self, college_name: str) -> bool:
        """
        Check if college is tier 1
        
        Args:
            college_name: College/university name
            
        Returns:
            True if tier 1, False otherwise
        """
        if not college_name:
            return False
        
        name_lower = college_name.lower()
        
        # Check against tier 1 list
        for tier1 in self.TIER_1_COLLEGES:
            if tier1 in name_lower:
                return True
        
        return False
    
    def extract_required_education(self, jd_text: str) -> Dict[str, any]:
        """
        Extract required education from JD
        
        Args:
            jd_text: Job description text
            
        Returns:
            Dict with required education info
        """
        jd_lower = jd_text.lower()
        jd_lower = re.sub(r"[’']", "", jd_lower)  # remove apostrophes

        
        result = {
            "min_level": 0,
            "preferred_level": 0,
            "required_field": "any",
            "degree_required": True
        }
        
        # Check if degree is required
        if any(phrase in jd_lower for phrase in ["degree required", "must have", "required:"]):
            result["degree_required"] = True
        elif any(phrase in jd_lower for phrase in ["preferred", "plus", "nice to have"]):
            result["degree_required"] = False
        
        # Extract minimum degree
        if "phd" in jd_lower or "doctorate" in jd_lower:
            result["min_level"] = 5
        elif any(word in jd_lower for word in ["masters", "mtech", "ms", "mba"]):
            result["min_level"] = 4
            if "preferred" in jd_lower:
                result["preferred_level"] = 4
                result["min_level"] = 3  # Bachelor minimum, Masters preferred
        elif any(word in jd_lower for word in ["bachelor", "btech", "be", "bs"]):
            result["min_level"] = 3
        
        # Extract field requirement
        if any(field in jd_lower for field in self.STEM_FIELDS):
            result["required_field"] = "stem"
        elif any(field in jd_lower for field in self.BUSINESS_FIELDS):
            result["required_field"] = "business"
        
        logger.debug(f"Required education: {result}")
        return result
    
    def calculate_education_match_score(
        self,
        candidate_level: int,
        required_min: int,
        required_preferred: int = 0
    ) -> float:
        """
        Calculate education match score
        
        Args:
            candidate_level: Candidate's education level (1-5)
            required_min: Minimum required level
            required_preferred: Preferred level (optional)
            
        Returns:
            Match score between 0 and 1
        """
        if required_min == 0:
            # No requirement specified
            return 1.0
        
        if candidate_level >= required_min:
            # Meets minimum requirement
            if required_preferred > 0 and candidate_level >= required_preferred:
                # Meets preferred level
                return 1.0
            elif required_preferred > 0:
                # Between min and preferred
                return 0.8
            else:
                # Meets minimum (no preferred specified)
                return 1.0
        else:
            # Below minimum requirement
            gap = required_min - candidate_level
            penalty = min(0.5, gap * 0.2)  # 20% penalty per level
            return max(0.0, 1.0 - penalty)
    
    def calculate_field_match_score(
        self,
        candidate_field: str,
        required_field: str
    ) -> float:
        """
        Calculate field of study match score
        
        Args:
            candidate_field: Candidate's field (stem/business/other)
            required_field: Required field
            
        Returns:
            Match score between 0 and 1
        """
        if required_field == "any":
            return 1.0
        
        if candidate_field == required_field:
            return 1.0
        
        # Partial match for related fields
        if candidate_field == "stem" and required_field == "business":
            return 0.5  # Some overlap
        elif candidate_field == "business" and required_field == "stem":
            return 0.5
        
        return 0.3  # Different field but has degree
    
    def calculate_education_features(
        self,
        resume_education: List[Dict],
        jd_text: str
    ) -> Dict[str, float]:
        """
        Calculate comprehensive education features
        
        Args:
            resume_education: List of education dicts from resume
                Example: [
                    {
                        "degree": "B.Tech in Computer Science",
                        "college": "IIT Delhi",
                        "year": "2020"
                    }
                ]
            jd_text: Job description text
            
        Returns:
            Dict with all education features
        """
        features = {}
        
        # Parse candidate's highest education
        highest_level = 0
        highest_degree = "unknown"
        field_of_study = "unknown"
        tier1_college = False
        
        for edu in resume_education:
            edu_text = edu.get("degree", "")
            college = edu.get("college", "")
            
            level = self.parse_education_level(edu_text)
            if level > highest_level:
                highest_level = level
                highest_degree = self.extract_degree(edu_text)
                field_of_study = self.extract_field_of_study(edu_text)
            
            if self.is_tier1_college(college):
                tier1_college = True
        
        features["education_level"] = float(highest_level)
        features["tier1_college"] = 1.0 if tier1_college else 0.0
        features["stem_degree"] = 1.0 if field_of_study == "stem" else 0.0
        features["business_degree"] = 1.0 if field_of_study == "business" else 0.0
        
        # Extract requirements from JD
        required = self.extract_required_education(jd_text)
        features["required_education_level"] = float(required["min_level"])
        features["preferred_education_level"] = float(required["preferred_level"])
        features["degree_required"] = 1.0 if required["degree_required"] else 0.0
        
        # Education match score
        education_match = self.calculate_education_match_score(
            highest_level,
            required["min_level"],
            required["preferred_level"]
        )
        features["education_match_score"] = education_match
        
        # Field match score
        field_match = self.calculate_field_match_score(
            field_of_study,
            required["required_field"]
        )
        features["field_match_score"] = field_match
        
        # Combined education score
        features["overall_education_score"] = (education_match + field_match) / 2
        
        # Boolean flags
        features["meets_education_requirement"] = (
            1.0 if highest_level >= required["min_level"] else 0.0
        )
        features["exceeds_education_requirement"] = (
            1.0 if highest_level > required["min_level"] else 0.0
        )
        features["has_advanced_degree"] = (
            1.0 if highest_level >= 4 else 0.0
        )
        
        # Number of degrees
        features["num_degrees"] = float(len(resume_education))
        
        # Education gap (positive = exceeds, negative = falls short)
        features["education_gap"] = float(highest_level - required["min_level"])
        
        logger.info(f"Education features calculated: {len(features)} features")
        return features
    
    def get_education_summary(
        self,
        resume_education: List[Dict],
        jd_text: str
    ) -> Dict[str, any]:
        """
        Get human-readable education summary
        
        Args:
            resume_education: List of education dicts
            jd_text: Job description text
            
        Returns:
            Dict with summary info
        """
        features = self.calculate_education_features(resume_education, jd_text)
        
        # Get degree names
        degrees = []
        colleges = []
        for edu in resume_education:
            if "degree" in edu:
                degrees.append(self.extract_degree(edu["degree"]))
            if "college" in edu:
                colleges.append(edu["college"])
        
        # Required education
        required = self.extract_required_education(jd_text)
        
        # Generate recommendation
        match_score = features["overall_education_score"]
        if match_score >= 0.9:
            recommendation = "Excellent Education Match"
        elif match_score >= 0.7:
            recommendation = "Good Education Match"
        elif match_score >= 0.5:
            recommendation = "Acceptable Education"
        else:
            recommendation = "Education Below Requirements"
        
        summary = {
            "highest_level": int(features["education_level"]),
            "degrees": degrees,
            "tier1_college": bool(features["tier1_college"]),
            "colleges": colleges,
            "required_level": int(features["required_education_level"]),
            "meets_requirement": bool(features["meets_education_requirement"]),
            "match_score": match_score,
            "recommendation": recommendation
        }
        
        return summary


# Convenience function
def calculate_education_match(
    resume_education: List[Dict],
    jd_text: str
) -> float:
    """
    Quick utility to calculate education match score
    
    Args:
        resume_education: List of education dicts
        jd_text: Job description text
        
    Returns:
        Match score between 0 and 1
    """
    calc = EducationCalculator()
    features = calc.calculate_education_features(resume_education, jd_text)
    return features["overall_education_score"]


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example education from resume
    resume_education = [
        {
            "degree": "M.Tech in Computer Science",
            "college": "IIT Delhi",
            "year": "2022",
            "gpa": "8.5/10"
        },
        {
            "degree": "B.Tech in Information Technology",
            "college": "NIT Trichy",
            "year": "2020",
            "gpa": "9.0/10"
        }
    ]
    
    # Example JD
    jd_text = """
    We're seeking a Machine Learning Engineer with a Bachelor's degree in Computer Science
    or related field. Master's degree preferred. Strong foundation in algorithms and data structures.
    """
    
    calc = EducationCalculator()
    
    # Calculate features
    features = calc.calculate_education_features(resume_education, jd_text)
    
    print("\n=== Education Features ===")
    for key, value in features.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Get summary
    summary = calc.get_education_summary(resume_education, jd_text)
    
    print("\n=== Education Summary ===")
    print(f"Highest Level: {summary['highest_level']}")
    print(f"Degrees: {summary['degrees']}")
    print(f"Tier 1 College: {summary['tier1_college']}")
    print(f"Colleges: {summary['colleges']}")
    print(f"Required Level: {summary['required_level']}")
    print(f"Meets Requirement: {summary['meets_requirement']}")
    print(f"Match Score: {summary['match_score']:.2f}")
    print(f"Recommendation: {summary['recommendation']}")