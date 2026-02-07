"""
NLP Module - Natural Language Processing for Resume Screening
Provides optimized parsers for resumes and job descriptions
"""

from src.nlp.entity_extractor import EntityExtractor, extract_contact_info
from src.nlp.skill_extractor import SkillExtractor, extract_skills_from_text
from src.nlp.resume_parser import ResumeParser, parse_resume
from src.nlp.jd_parser import JDParser, parse_jd

__all__ = [
    'EntityExtractor',
    'SkillExtractor',
    'ResumeParser',
    'JDParser',
    'extract_contact_info',
    'extract_skills_from_text',
    'parse_resume',
    'parse_jd'
]