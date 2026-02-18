from typing import Dict
import os
from dotenv import load_dotenv
load_dotenv()

class FeatureConfig:
    """Configuration for feature engineering"""
    
    # SBERT Model Configuration
    SBERT_MODEL = "all-MiniLM-L6-v2"  # Lightweight, fast (22MB, 384 dims)
    # Alternative models:
    # - "paraphrase-MiniLM-L3-v2" (Faster, 61MB, 384 dims)
    # - "all-mpnet-base-v2" (Better quality, 420MB, 768 dims)
    
    SBERT_BATCH_SIZE = 32
    SBERT_EMBEDDING_DIM = 384
    
    # Section Weights for Weighted Similarity
    DEFAULT_SECTION_WEIGHTS: Dict[str, float] = {
        "skills": 0.4,        # Skills are most important
        "experience": 0.3,    # Experience is second priority
        "education": 0.15,    # Education is important but less than skills
        "overall": 0.15       # Overall semantic match
    }
    
    # Skill Matching Configuration
    SKILL_EXACT_MATCH_WEIGHT = 1.0
    SKILL_PARTIAL_MATCH_WEIGHT = 0.5
    SKILL_SYNONYM_MATCH_WEIGHT = 0.7
    
    # Context-aware skill proficiency weights (from Phase 4)
    SKILL_PROFICIENCY_WEIGHTS: Dict[str, float] = {
        "expert": 1.5,
        "advanced": 1.3,
        "proficient": 1.0,
        "intermediate": 0.9,
        "basic": 0.8,
        "familiar": 0.6
    }
    
    # Experience Configuration
    EXPERIENCE_PERFECT_MATCH_RANGE = 1.0  # ±1 year is perfect match
    EXPERIENCE_PENALTY_PER_YEAR = 0.1     # Penalty for each year gap
    EXPERIENCE_MAX_PENALTY = 0.5          # Maximum penalty
    
    # Education Level Scoring
    EDUCATION_LEVELS: Dict[str, int] = {
        "phd": 5,
        "doctorate": 5,
        "masters": 4,
        "mba": 4,
        "bachelor": 3,
        "btech": 3,
        "diploma": 2,
        "high school": 1,
        "unknown": 0
    }
    
    # Feature Vector Configuration
    TOTAL_FEATURES = 30  # Target number of features
    
    # Feature Groups
    FEATURE_GROUPS = {
        "similarity": [
            "overall_similarity",
            "skills_similarity",
            "experience_similarity",
            "education_similarity",
            "weighted_similarity"
        ],
        "skill_match": [
            "total_skills_count",
            "matched_skills_count",
            "skill_match_ratio",
            "weighted_skill_score",
            "expert_skills_count",
            "rare_skills_count"
        ],
        "experience": [
            "total_experience_years",
            "required_experience_years",
            "experience_gap",
            "experience_match_score",
            "relevant_experience_years"
        ],
        "education": [
            "education_level",
            "required_education_level",
            "education_match",
            "top_tier_college"
        ],
        "additional": [
            "num_projects",
            "num_certifications",
            "github_present",
            "linkedin_present",
            "publications_count"
        ]
    }
    
    # PostgreSQL Feature Store Configuration
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", 5432))
    DB_NAME = os.getenv("DB_NAME", "resume_screening")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    
    # Feature Store Tables
    FEATURE_TABLE = "candidate_features"
    SIMILARITY_TABLE = "similarity_scores"
    SKILL_MATCH_TABLE = "skill_matches"
    
    # Caching Configuration
    ENABLE_CACHING = True
    CACHE_SIZE = 1000  # LRU cache size
    
    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# Singleton config instance
config = FeatureConfig()