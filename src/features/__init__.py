"""
Features Package
Feature engineering modules for resume screening system
"""

from .semantic_similarity import (
    SemanticSimilarityCalculator,
    calculate_similarity,
    get_sbert_model
)
from .skill_features import (
    SkillMatcher,
    calculate_skill_match
)
from .experience_features import (
    ExperienceCalculator,
    calculate_experience_match
)
from .education_features import (
    EducationCalculator,
    calculate_education_match
)
from .feature_generator import (
    FeatureGenerator,
    generate_features
)
from .feature_store import (
    FeatureStore,
    create_feature_store
)
from .config import FeatureConfig, config

__all__ = [
    'SemanticSimilarityCalculator',
    'calculate_similarity',
    'get_sbert_model',
    'SkillMatcher',
    'calculate_skill_match',
    'ExperienceCalculator',
    'calculate_experience_match',
    'EducationCalculator',
    'calculate_education_match',
    'FeatureGenerator',
    'generate_features',
    'FeatureStore',
    'create_feature_store',
    'FeatureConfig',
    'config'
]

__version__ = '1.0.0'