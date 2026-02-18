"""
Feature Generator
Combines all feature modules to generate complete feature vectors
Integrates: Semantic Similarity, Skills, Experience, Education
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np

from .semantic_similarity import SemanticSimilarityCalculator
from .skill_features import SkillMatcher
from .experience_features import ExperienceCalculator
from .education_features import EducationCalculator
from .config import config

logger = logging.getLogger(__name__)


class FeatureGenerator:
    """
    Combines all feature calculators to generate complete feature vectors
    for resume-JD pairs
    """
    
    def __init__(self):
        """Initialize all feature calculators"""
        self.semantic_calc = SemanticSimilarityCalculator()
        self.skill_matcher = SkillMatcher()
        self.experience_calc = ExperienceCalculator()
        self.education_calc = EducationCalculator()
        
        logger.info("FeatureGenerator initialized with all calculators")
    
    def generate_features(
        self,
        resume_data: Dict[str, Any],
        jd_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Generate complete feature vector from resume and JD data
        
        Args:
            resume_data: Parsed resume data with keys:
                - text: Full resume text
                - sections: Dict of sections (skills, experience, education, etc.)
                - skills: List of skills
                - skill_proficiencies: Dict mapping skill -> proficiency level
                - experience: List of experience blocks
                - education: List of education dicts
                
            jd_data: Parsed JD data with keys:
                - text: Full JD text
                - sections: Dict of sections
                - skills: List of required skills
                - skill_tiers: Optional dict mapping skill -> tier
                
        Returns:
            Dict with all features (30+ features)
        """
        features = {}
        
        # 1. SEMANTIC SIMILARITY FEATURES (5 features)
        logger.info("Calculating semantic similarity features...")
        semantic_features = self._calculate_semantic_features(resume_data, jd_data)
        features.update(semantic_features)
        
        # 2. SKILL MATCHING FEATURES (15 features)
        logger.info("Calculating skill matching features...")
        skill_features = self._calculate_skill_features(resume_data, jd_data)
        features.update(skill_features)
        
        # 3. EXPERIENCE FEATURES (15 features)
        logger.info("Calculating experience features...")
        experience_features = self._calculate_experience_features(resume_data, jd_data)
        features.update(experience_features)
        
        # 4. EDUCATION FEATURES (13 features)
        logger.info("Calculating education features...")
        education_features = self._calculate_education_features(resume_data, jd_data)
        features.update(education_features)
        
        # 5. ADDITIONAL FEATURES (5 features)
        logger.info("Calculating additional features...")
        additional_features = self._calculate_additional_features(resume_data, jd_data)
        features.update(additional_features)
        
        logger.info(f"Total features generated: {len(features)}")
        return features
    
    def _calculate_semantic_features(
        self,
        resume_data: Dict,
        jd_data: Dict
    ) -> Dict[str, float]:
        """Calculate semantic similarity features"""
        resume_text = resume_data.get("text", "")
        jd_text = jd_data.get("text", "")
        
        resume_sections = resume_data.get("sections", {})
        jd_sections = jd_data.get("sections", {})
        
        # Calculate weighted similarity
        results = self.semantic_calc.calculate_weighted_similarity(
            resume_text,
            jd_text,
            resume_sections,
            jd_sections
        )
        
        # Extract key features
        features = {
            "overall_similarity": results.get("overall_similarity", 0.0),
            "weighted_similarity": results.get("weighted_similarity", 0.0),
            "skills_similarity": results.get("skills_vs_required_skills", 0.0),
            "experience_similarity": results.get("experience_vs_responsibilities", 0.0),
            "education_similarity": results.get("education_vs_education", 0.0)
        }
        
        return features
    
    def _calculate_skill_features(
        self,
        resume_data: Dict,
        jd_data: Dict
    ) -> Dict[str, float]:
        """Calculate skill matching features"""
        resume_skills = resume_data.get("skills", [])
        jd_skills = jd_data.get("skills", [])
        skill_proficiencies = resume_data.get("skill_proficiencies", {})
        skill_tiers = jd_data.get("skill_tiers", {})
        
        # Calculate comprehensive skill features
        features = self.skill_matcher.calculate_skill_features(
            resume_skills,
            jd_skills,
            skill_proficiencies,
            skill_tiers
        )
        
        # Rename features to avoid conflicts and select key ones
        skill_features = {
            "skill_total_resume": features.get("total_resume_skills", 0.0),
            "skill_total_jd": features.get("total_jd_skills", 0.0),
            "skill_exact_matches": features.get("exact_matches_count", 0.0),
            "skill_fuzzy_matches": features.get("fuzzy_matches_count", 0.0),
            "skill_total_matches": features.get("total_matches_count", 0.0),
            "skill_exact_ratio": features.get("exact_match_ratio", 0.0),
            "skill_overall_ratio": features.get("overall_match_ratio", 0.0),
            "skill_weighted_score": features.get("weighted_skill_score", 0.0),
            "skill_expert_count": features.get("expert_skills_count", 0.0),
            "skill_expert_matches": features.get("expert_matches_count", 0.0),
            "skill_rare_count": features.get("rare_skills_count", 0.0),
            "skill_rare_matches": features.get("rare_matches_count", 0.0),
            "skill_missing_count": features.get("missing_skills_count", 0.0)
        }
        
        return skill_features
    
    def _calculate_experience_features(
        self,
        resume_data: Dict,
        jd_data: Dict
    ) -> Dict[str, float]:
        """Calculate experience features"""
        experience_blocks = resume_data.get("experience", [])
        jd_text = jd_data.get("text", "")
        
        # Get relevant keywords for relevant experience calculation
        relevant_keywords = resume_data.get("skills", [])
        
        # Calculate comprehensive experience features
        features = self.experience_calc.calculate_experience_features(
            experience_blocks,
            jd_text,
            relevant_keywords
        )
        
        # Rename features to avoid conflicts
        exp_features = {
            "exp_total_years": features.get("total_experience_years", 0.0),
            "exp_level": features.get("experience_level", 0.0),
            "exp_required_min": features.get("required_min_years", 0.0),
            "exp_required_max": features.get("required_max_years", 0.0),
            "exp_gap": features.get("experience_gap", 0.0),
            "exp_gap_abs": features.get("experience_gap_abs", 0.0),
            "exp_match_score": features.get("experience_match_score", 0.0),
            "exp_relevant_years": features.get("relevant_experience_years", 0.0),
            "exp_relevant_ratio": features.get("relevant_experience_ratio", 0.0),
            "exp_num_positions": features.get("num_positions", 0.0),
            "exp_avg_tenure": features.get("avg_tenure_years", 0.0),
            "exp_meets_min": features.get("meets_min_experience", 0.0),
            "exp_overqualified": features.get("overqualified", 0.0),
            "exp_underqualified": features.get("underqualified", 0.0)
        }
        
        return exp_features
    
    def _calculate_education_features(
        self,
        resume_data: Dict,
        jd_data: Dict
    ) -> Dict[str, float]:
        """Calculate education features"""
        education = resume_data.get("education", [])
        jd_text = jd_data.get("text", "")
        
        # Calculate comprehensive education features
        features = self.education_calc.calculate_education_features(
            education,
            jd_text
        )
        
        # Rename features to avoid conflicts
        edu_features = {
            "edu_level": features.get("education_level", 0.0),
            "edu_tier1": features.get("tier1_college", 0.0),
            "edu_stem": features.get("stem_degree", 0.0),
            "edu_business": features.get("business_degree", 0.0),
            "edu_required_level": features.get("required_education_level", 0.0),
            "edu_preferred_level": features.get("preferred_education_level", 0.0),
            "edu_degree_required": features.get("degree_required", 0.0),
            "edu_match_score": features.get("education_match_score", 0.0),
            "edu_field_match": features.get("field_match_score", 0.0),
            "edu_overall_score": features.get("overall_education_score", 0.0),
            "edu_meets_requirement": features.get("meets_education_requirement", 0.0),
            "edu_exceeds": features.get("exceeds_education_requirement", 0.0),
            "edu_advanced_degree": features.get("has_advanced_degree", 0.0)
        }
        
        return edu_features
    
    def _calculate_additional_features(
        self,
        resume_data: Dict,
        jd_data: Dict
    ) -> Dict[str, float]:
        """Calculate additional features (projects, certifications, etc.)"""
        features = {}
        
        # Projects
        projects = resume_data.get("projects", [])
        features["num_projects"] = float(len(projects))
        
        # Certifications
        certifications = resume_data.get("certifications", [])
        features["num_certifications"] = float(len(certifications))
        
        # Links (GitHub, LinkedIn)
        github_url = resume_data.get("github", "")
        linkedin_url = resume_data.get("linkedin", "")
        features["has_github"] = 1.0 if github_url else 0.0
        features["has_linkedin"] = 1.0 if linkedin_url else 0.0
        
        # Publications (if any)
        publications = resume_data.get("publications", [])
        features["num_publications"] = float(len(publications))
        
        return features
    
    def generate_feature_vector(
        self,
        resume_data: Dict[str, Any],
        jd_data: Dict[str, Any],
        feature_order: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Generate feature vector as numpy array (for ML model input)
        
        Args:
            resume_data: Parsed resume data
            jd_data: Parsed JD data
            feature_order: Optional list specifying feature order
                If None, uses alphabetical order
                
        Returns:
            Numpy array of features
        """
        # Generate features dict
        features_dict = self.generate_features(resume_data, jd_data)
        
        # Determine feature order
        if feature_order is None:
            feature_order = sorted(features_dict.keys())
        
        # Create vector
        vector = np.array([features_dict.get(f, 0.0) for f in feature_order])
        
        logger.debug(f"Feature vector shape: {vector.shape}")
        return vector
    
    def batch_generate_features(
        self,
        resume_data_list: List[Dict[str, Any]],
        jd_data_list: List[Dict[str, Any]]
    ) -> List[Dict[str, float]]:
        """
        Generate features for multiple resume-JD pairs
        
        Args:
            resume_data_list: List of parsed resume data
            jd_data_list: List of parsed JD data (must be same length)
            
        Returns:
            List of feature dicts
        """
        if len(resume_data_list) != len(jd_data_list):
            raise ValueError("resume_data_list and jd_data_list must have same length")
        
        logger.info(f"Batch generating features for {len(resume_data_list)} pairs")
        
        all_features = []
        for i, (resume_data, jd_data) in enumerate(zip(resume_data_list, jd_data_list)):
            logger.debug(f"Processing pair {i+1}/{len(resume_data_list)}")
            features = self.generate_features(resume_data, jd_data)
            all_features.append(features)
        
        logger.info(f"Batch generation complete: {len(all_features)} feature sets")
        return all_features
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names in order
        
        Returns:
            List of feature names
        """
        # Generate dummy data to get feature names
        dummy_resume = {
            "text": "",
            "sections": {},
            "skills": [],
            "skill_proficiencies": {},
            "experience": [],
            "education": [],
            "projects": [],
            "certifications": [],
            "github": "",
            "linkedin": "",
            "publications": []
        }
        dummy_jd = {
            "text": "",
            "sections": {},
            "skills": [],
            "skill_tiers": {}
        }
        
        features = self.generate_features(dummy_resume, dummy_jd)
        return sorted(features.keys())
    
    def get_feature_summary(
        self,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Get human-readable summary of features
        
        Args:
            features: Feature dict
            
        Returns:
            Dict with summary info
        """
        summary = {
            "total_features": len(features),
            "semantic_similarity": {
                "overall": features.get("overall_similarity", 0.0),
                "weighted": features.get("weighted_similarity", 0.0)
            },
            "skill_match": {
                "match_ratio": features.get("skill_overall_ratio", 0.0),
                "weighted_score": features.get("skill_weighted_score", 0.0),
                "missing_skills": int(features.get("skill_missing_count", 0))
            },
            "experience": {
                "total_years": features.get("exp_total_years", 0.0),
                "match_score": features.get("exp_match_score", 0.0),
                "gap_years": features.get("exp_gap", 0.0)
            },
            "education": {
                "level": int(features.get("edu_level", 0)),
                "match_score": features.get("edu_overall_score", 0.0),
                "tier1": bool(features.get("edu_tier1", 0.0))
            },
            "overall_recommendation": self._get_recommendation(features)
        }
        
        return summary
    
    def _get_recommendation(self, features: Dict[str, float]) -> str:
        """Generate overall hiring recommendation"""
        # Calculate weighted overall score
        weights = {
            "weighted_similarity": 0.15,
            "skill_weighted_score": 0.40,
            "exp_match_score": 0.30,
            "edu_overall_score": 0.15
        }
        
        overall_score = sum(
            features.get(key, 0.0) * weight
            for key, weight in weights.items()
        )
        
        if overall_score >= 0.8:
            return "Strong Recommend - Immediate Interview"
        elif overall_score >= 0.65:
            return "Recommend - Schedule Interview"
        elif overall_score >= 0.5:
            return "Consider - Review Manually"
        elif overall_score >= 0.35:
            return "Weak Match - Likely Reject"
        else:
            return "Poor Match - Reject"


# Convenience function
def generate_features(resume_data: Dict, jd_data: Dict) -> Dict[str, float]:
    """
    Quick utility to generate features
    
    Args:
        resume_data: Parsed resume data
        jd_data: Parsed JD data
        
    Returns:
        Feature dict
    """
    generator = FeatureGenerator()
    return generator.generate_features(resume_data, jd_data)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example resume data (would come from Phase 4 parsers)
    resume_data = {
        "text": "Senior ML Engineer with 5 years experience in Python and NLP...",
        "sections": {
            "skills": "Python, Machine Learning, NLP, TensorFlow",
            "experience": "5 years building ML models",
            "education": "M.Tech in CS from IIT Delhi"
        },
        "skills": ["Python", "Machine Learning", "NLP", "TensorFlow", "AWS"],
        "skill_proficiencies": {
            "python": "expert",
            "machine learning": "expert",
            "nlp": "advanced",
            "tensorflow": "proficient"
        },
        "experience": [
            {
                "title": "Senior ML Engineer",
                "start_date": "Jan 2020",
                "end_date": "Present",
                "description": "Building NLP models"
            },
            {
                "title": "ML Engineer",
                "duration": "3 years",
                "description": "ML development"
            }
        ],
        "education": [
            {
                "degree": "M.Tech in Computer Science",
                "college": "IIT Delhi",
                "year": "2020"
            }
        ],
        "projects": ["NLP Chatbot", "Image Classification"],
        "certifications": ["TensorFlow Developer"],
        "github": "github.com/user",
        "linkedin": "linkedin.com/in/user",
        "publications": []
    }
    
    # Example JD data
    jd_data = {
        "text": "Seeking ML Engineer with 5+ years experience. Python, NLP required. Masters preferred.",
        "sections": {
            "required_skills": "Python, Machine Learning, NLP",
            "responsibilities": "Build ML models",
            "education": "Masters in CS"
        },
        "skills": ["Python", "Machine Learning", "NLP", "Deep Learning"],
        "skill_tiers": {
            "python": "high_demand",
            "nlp": "rare",
            "machine learning": "high_demand"
        }
    }
    
    # Generate features
    generator = FeatureGenerator()
    
    print("\n=== Generating Features ===")
    features = generator.generate_features(resume_data, jd_data)
    
    print(f"\nTotal Features: {len(features)}")
    print("\nSample Features:")
    for i, (key, value) in enumerate(sorted(features.items())[:10]):
        print(f"  {key}: {value:.3f}")
    print("  ...")
    
    # Get feature summary
    summary = generator.get_feature_summary(features)
    
    print("\n=== Feature Summary ===")
    print(f"Semantic Similarity: {summary['semantic_similarity']['weighted']:.3f}")
    print(f"Skill Match: {summary['skill_match']['match_ratio']:.3f}")
    print(f"Experience Match: {summary['experience']['match_score']:.3f}")
    print(f"Education Match: {summary['education']['match_score']:.3f}")
    print(f"\nRecommendation: {summary['overall_recommendation']}")
    
    # Get feature vector
    print("\n=== Feature Vector ===")
    feature_vector = generator.generate_feature_vector(resume_data, jd_data)
    print(f"Shape: {feature_vector.shape}")
    print(f"Sample values: {feature_vector[:5]}")
    
    # Get feature names
    print("\n=== Feature Names ===")
    feature_names = generator.get_feature_names()
    print(f"Total feature names: {len(feature_names)}")
    print(f"First 10: {feature_names[:10]}")