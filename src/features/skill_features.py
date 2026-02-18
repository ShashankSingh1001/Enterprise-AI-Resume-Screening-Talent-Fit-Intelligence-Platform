import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter
import re

logger = logging.getLogger(__name__)


class SkillMatcher:
    """
    Calculates skill matching features using various matching strategies
    Integrates with Phase 4 proficiency detection for weighted scoring
    """
    
    # Proficiency weights from Phase 4 (expert gets higher weight)
    PROFICIENCY_WEIGHTS = {
        "expert": 1.5,
        "advanced": 1.3,
        "proficient": 1.0,
        "intermediate": 0.9,
        "basic": 0.8,
        "familiar": 0.6,
        "unknown": 1.0  # Default if no proficiency detected
    }
    
    # Skill tier weights (some skills are more valuable)
    SKILL_TIER_WEIGHTS = {
        "rare": 2.0,      # Rare/specialized skills (e.g., CUDA, Kubernetes)
        "high_demand": 1.5,  # High-demand skills (e.g., Python, AWS)
        "common": 1.0,     # Common skills (e.g., Excel, Communication)
        "basic": 0.8       # Basic skills (e.g., MS Office)
    }
    
    # Skill synonyms for fuzzy matching
    SKILL_SYNONYMS = {
        "ml": ["machine learning", "ml", "machine-learning"],
        "ai": ["artificial intelligence", "ai"],
        "nlp": ["natural language processing", "nlp"],
        "dl": ["deep learning", "dl"],
        "tensorflow": ["tensorflow", "tf"],
        "pytorch": ["pytorch", "torch"],
        "javascript": ["javascript", "js"],
        "typescript": ["typescript", "ts"],
        "python": ["python", "python3"],
        "aws": ["amazon web services", "aws"],
        "gcp": ["google cloud platform", "gcp"],
        "azure": ["microsoft azure", "azure"],
    }
    
    def __init__(self):
        """Initialize skill matcher"""
        logger.info("SkillMatcher initialized")
    
    @staticmethod
    def normalize_skill(skill: str) -> str:
        """
        Normalize skill name for matching
        
        Args:
            skill: Raw skill name
            
        Returns:
            Normalized skill name (lowercase, stripped)
        """
        if not skill:
            return ""
        
        # Lowercase and strip whitespace
        normalized = skill.lower().strip()
        
        # Remove special characters except hyphens and plus
        normalized = re.sub(r'[^\w\s\-]', '', normalized)
        
        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def get_skill_synonyms(self, skill: str) -> Set[str]:
        """
        Get all synonyms for a skill
        
        Args:
            skill: Skill name
            
        Returns:
            Set of synonyms including the skill itself
        """
        normalized = self.normalize_skill(skill)
        
        # Check if skill has known synonyms
        for canonical, synonyms in self.SKILL_SYNONYMS.items():
            if normalized in synonyms or normalized == canonical:
                return set(synonyms)
        
        # Return skill itself if no synonyms found
        return {normalized}
    
    def exact_match(self, resume_skills: List[str], jd_skills: List[str]) -> Set[str]:
        """
        Find exact skill matches
        
        Args:
            resume_skills: List of skills from resume
            jd_skills: List of required skills from JD
            
        Returns:
            Set of matched skills
        """
        resume_normalized = {self.normalize_skill(s) for s in resume_skills}
        jd_normalized = {self.normalize_skill(s) for s in jd_skills}
        
        matches = resume_normalized & jd_normalized
        logger.debug(f"Exact matches: {matches}")
        return matches
    
    def fuzzy_match(self, resume_skills: List[str], jd_skills: List[str]) -> Set[str]:
        """
        Find fuzzy matches using synonyms
        
        Args:
            resume_skills: List of skills from resume
            jd_skills: List of required skills from JD
            
        Returns:
            Set of matched skills (canonical names)
        """
        matches = set()
        
        for resume_skill in resume_skills:
            resume_synonyms = self.get_skill_synonyms(resume_skill)
            
            for jd_skill in jd_skills:
                jd_synonyms = self.get_skill_synonyms(jd_skill)
                
                # Check for synonym overlap
                if resume_synonyms & jd_synonyms:
                    matches.add(self.normalize_skill(jd_skill))
        
        logger.debug(f"Fuzzy matches: {matches}")
        return matches
    
    def partial_match(self, resume_skills: List[str], jd_skills: List[str]) -> Set[str]:
        """
        Find partial matches (substring matching)
        Example: "machine learning" matches "machine learning engineer"
        
        Args:
            resume_skills: List of skills from resume
            jd_skills: List of required skills from JD
            
        Returns:
            Set of partially matched skills
        """
        matches = set()
        
        resume_normalized = [self.normalize_skill(s) for s in resume_skills]
        jd_normalized = [self.normalize_skill(s) for s in jd_skills]
        
        for resume_skill in resume_normalized:
            for jd_skill in jd_normalized:
                # Check if either skill contains the other
                if (resume_skill in jd_skill or jd_skill in resume_skill) and resume_skill and jd_skill:
                    matches.add(jd_skill)
        
        logger.debug(f"Partial matches: {matches}")
        return matches
    
    def calculate_skill_match_ratio(
        self, 
        resume_skills: List[str], 
        jd_skills: List[str],
        match_type: str = "all"
    ) -> float:
        """
        Calculate skill match ratio
        
        Args:
            resume_skills: List of skills from resume
            jd_skills: List of required skills from JD
            match_type: "exact", "fuzzy", "partial", or "all" (combines all)
            
        Returns:
            Match ratio between 0 and 1
        """
        if not jd_skills:
            return 0.0
        
        if match_type == "exact":
            matches = self.exact_match(resume_skills, jd_skills)
        elif match_type == "fuzzy":
            matches = self.fuzzy_match(resume_skills, jd_skills)
        elif match_type == "partial":
            matches = self.partial_match(resume_skills, jd_skills)
        else:  # "all" - combine all match types
            exact = self.exact_match(resume_skills, jd_skills)
            fuzzy = self.fuzzy_match(resume_skills, jd_skills)
            partial = self.partial_match(resume_skills, jd_skills)
            matches = exact | fuzzy | partial
        
        ratio = len(matches) / len(jd_skills)
        logger.debug(f"Skill match ratio ({match_type}): {ratio:.3f}")
        return ratio
    
    def calculate_weighted_skill_score(
        self,
        resume_skills: List[str],
        jd_skills: List[str],
        skill_proficiencies: Optional[Dict[str, str]] = None,
        skill_tiers: Optional[Dict[str, str]] = None
    ) -> float:
        """
        Calculate weighted skill score using proficiency and tier weights
        
        Args:
            resume_skills: List of skills from resume
            jd_skills: List of required skills from JD
            skill_proficiencies: Dict mapping skill -> proficiency level
                Example: {"python": "expert", "java": "basic"}
            skill_tiers: Dict mapping skill -> tier level
                Example: {"python": "high_demand", "excel": "common"}
                
        Returns:
            Weighted score between 0 and 1
        """
        if not jd_skills:
            return 0.0
        
        # Find all matches (exact, fuzzy, partial)
        matches = (
            self.exact_match(resume_skills, jd_skills) |
            self.fuzzy_match(resume_skills, jd_skills) |
            self.partial_match(resume_skills, jd_skills)
        )
        
        # Calculate weighted score
        total_weight = 0.0
        matched_weight = 0.0
        
        for jd_skill in jd_skills:
            normalized_jd = self.normalize_skill(jd_skill)
            
            # Get tier weight (default to common)
            tier = "common"
            if skill_tiers and normalized_jd in skill_tiers:
                tier = skill_tiers[normalized_jd]
            tier_weight = self.SKILL_TIER_WEIGHTS.get(tier, 1.0)
            
            total_weight += tier_weight
            
            # Check if skill is matched
            if normalized_jd in matches:
                # Get proficiency weight (default to unknown)
                proficiency = "unknown"
                if skill_proficiencies and normalized_jd in skill_proficiencies:
                    proficiency = skill_proficiencies[normalized_jd]
                proficiency_weight = self.PROFICIENCY_WEIGHTS.get(proficiency, 1.0)
                
                # Combined weight = tier × proficiency
                matched_weight += tier_weight * proficiency_weight
        
        # Calculate final score
        if total_weight == 0:
            return 0.0
        
        score = min(1.0, matched_weight / total_weight)
        logger.debug(f"Weighted skill score: {score:.3f}")
        return score
    
    def calculate_skill_features(
        self,
        resume_skills: List[str],
        jd_skills: List[str],
        skill_proficiencies: Optional[Dict[str, str]] = None,
        skill_tiers: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive skill matching features
        
        Args:
            resume_skills: List of skills from resume
            jd_skills: List of required skills from JD
            skill_proficiencies: Optional proficiency mapping
            skill_tiers: Optional tier mapping
            
        Returns:
            Dict with all skill features
        """
        features = {}
        
        # Basic counts
        features["total_resume_skills"] = len(resume_skills)
        features["total_jd_skills"] = len(jd_skills)
        
        # Match counts
        exact_matches = self.exact_match(resume_skills, jd_skills)
        fuzzy_matches = self.fuzzy_match(resume_skills, jd_skills)
        partial_matches = self.partial_match(resume_skills, jd_skills)
        all_matches = exact_matches | fuzzy_matches | partial_matches
        
        features["exact_matches_count"] = len(exact_matches)
        features["fuzzy_matches_count"] = len(fuzzy_matches)
        features["partial_matches_count"] = len(partial_matches)
        features["total_matches_count"] = len(all_matches)
        
        # Match ratios
        features["exact_match_ratio"] = self.calculate_skill_match_ratio(
            resume_skills, jd_skills, "exact"
        )
        features["fuzzy_match_ratio"] = self.calculate_skill_match_ratio(
            resume_skills, jd_skills, "fuzzy"
        )
        features["partial_match_ratio"] = self.calculate_skill_match_ratio(
            resume_skills, jd_skills, "partial"
        )
        features["overall_match_ratio"] = self.calculate_skill_match_ratio(
            resume_skills, jd_skills, "all"
        )
        
        # Weighted score
        features["weighted_skill_score"] = self.calculate_weighted_skill_score(
            resume_skills, jd_skills, skill_proficiencies, skill_tiers
        )
        
        # Proficiency-based features
        if skill_proficiencies:
            expert_skills = [s for s, p in skill_proficiencies.items() if p == "expert"]
            features["expert_skills_count"] = len(expert_skills)
            
            # Count expert skills that match JD
            expert_matches = set(self.normalize_skill(s) for s in expert_skills) & all_matches
            features["expert_matches_count"] = len(expert_matches)
        else:
            features["expert_skills_count"] = 0
            features["expert_matches_count"] = 0
        
        # Tier-based features
        if skill_tiers:
            rare_skills = [s for s, t in skill_tiers.items() if t == "rare"]
            features["rare_skills_count"] = len(rare_skills)
            
            # Count rare skills that match JD
            rare_matches = set(rare_skills) & all_matches
            features["rare_matches_count"] = len(rare_matches)
        else:
            features["rare_skills_count"] = 0
            features["rare_matches_count"] = 0
        
        # Unmatched skills (in JD but not in resume)
        features["missing_skills_count"] = len(jd_skills) - len(all_matches)
        
        logger.info(f"Skill features calculated: {len(features)} features")
        return features
    
    def get_missing_skills(
        self,
        resume_skills: List[str],
        jd_skills: List[str]
    ) -> List[str]:
        """
        Get list of skills required by JD but missing from resume
        
        Args:
            resume_skills: List of skills from resume
            jd_skills: List of required skills from JD
            
        Returns:
            List of missing skills
        """
        # Find all matches
        matches = (
            self.exact_match(resume_skills, jd_skills) |
            self.fuzzy_match(resume_skills, jd_skills) |
            self.partial_match(resume_skills, jd_skills)
        )
        
        # Find missing skills
        jd_normalized = {self.normalize_skill(s) for s in jd_skills}
        missing = jd_normalized - matches
        
        return sorted(list(missing))
    
    def get_skill_gap_summary(
        self,
        resume_skills: List[str],
        jd_skills: List[str],
        skill_proficiencies: Optional[Dict[str, str]] = None
    ) -> Dict[str, any]:
        """
        Get comprehensive skill gap analysis
        
        Args:
            resume_skills: List of skills from resume
            jd_skills: List of required skills from JD
            skill_proficiencies: Optional proficiency mapping
            
        Returns:
            Dict with gap analysis
        """
        # Calculate features
        features = self.calculate_skill_features(
            resume_skills, jd_skills, skill_proficiencies
        )
        
        # Get missing skills
        missing = self.get_missing_skills(resume_skills, jd_skills)
        
        # Find matched skills
        all_matches = (
            self.exact_match(resume_skills, jd_skills) |
            self.fuzzy_match(resume_skills, jd_skills) |
            self.partial_match(resume_skills, jd_skills)
        )
        
        summary = {
            "matched_skills": sorted(list(all_matches)),
            "missing_skills": missing,
            "match_ratio": features["overall_match_ratio"],
            "weighted_score": features["weighted_skill_score"],
            "total_resume_skills": features["total_resume_skills"],
            "total_jd_skills": features["total_jd_skills"],
            "recommendation": self._get_recommendation(features["overall_match_ratio"])
        }
        
        return summary
    
    @staticmethod
    def _get_recommendation(match_ratio: float) -> str:
        """Get hiring recommendation based on match ratio"""
        if match_ratio >= 0.8:
            return "Strong Match - Recommend Interview"
        elif match_ratio >= 0.6:
            return "Good Match - Consider for Interview"
        elif match_ratio >= 0.4:
            return "Moderate Match - Review Manually"
        else:
            return "Weak Match - Likely Not Suitable"


# Convenience function
def calculate_skill_match(
    resume_skills: List[str],
    jd_skills: List[str],
    skill_proficiencies: Optional[Dict[str, str]] = None
) -> float:
    """
    Quick utility to calculate skill match ratio
    
    Args:
        resume_skills: List of skills from resume
        jd_skills: List of required skills from JD
        skill_proficiencies: Optional proficiency mapping
        
    Returns:
        Match ratio between 0 and 1
    """
    matcher = SkillMatcher()
    return matcher.calculate_skill_match_ratio(resume_skills, jd_skills)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example skills
    resume_skills = [
        "Python", "Machine Learning", "TensorFlow", "PyTorch",
        "NLP", "AWS", "Docker", "SQL", "Git"
    ]
    
    jd_skills = [
        "Python", "ML", "Deep Learning", "TensorFlow",
        "Natural Language Processing", "Cloud (AWS/GCP)", "Docker"
    ]
    
    # Proficiency levels (from Phase 4 parsing)
    proficiencies = {
        "python": "expert",
        "machine learning": "expert",
        "tensorflow": "advanced",
        "pytorch": "proficient",
        "nlp": "expert",
        "aws": "intermediate",
        "docker": "basic"
    }
    
    # Skill tiers
    tiers = {
        "python": "high_demand",
        "ml": "high_demand",
        "deep learning": "high_demand",
        "tensorflow": "high_demand",
        "nlp": "rare",
        "aws": "common",
        "docker": "common"
    }
    
    matcher = SkillMatcher()
    
    # Calculate all features
    features = matcher.calculate_skill_features(
        resume_skills, jd_skills, proficiencies, tiers
    )
    
    print("\n=== Skill Matching Features ===")
    for key, value in features.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Get skill gap summary
    summary = matcher.get_skill_gap_summary(
        resume_skills, jd_skills, proficiencies
    )
    
    print("\n=== Skill Gap Summary ===")
    print(f"Matched Skills: {summary['matched_skills']}")
    print(f"Missing Skills: {summary['missing_skills']}")
    print(f"Match Ratio: {summary['match_ratio']:.3f}")
    print(f"Weighted Score: {summary['weighted_score']:.3f}")
    print(f"Recommendation: {summary['recommendation']}")