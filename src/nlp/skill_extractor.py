"""
Skill Extractor - FlashText Version
Ultra-fast O(n) skill extraction with weighted scoring
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import sys
import re
try:
    from flashtext import KeywordProcessor
    FLASHTEXT_AVAILABLE = True
except ImportError:
    FLASHTEXT_AVAILABLE = False
    import re

from src.logging import get_logger
from src.exceptions import ResumeParsingError

logger = get_logger(__name__)


class SkillExtractor:
    """Fast skill extraction using FlashText (O(n) complexity)"""
    
    def __init__(self, skills_db_path: Optional[str] = None):
        if skills_db_path is None:
            skills_db_path = Path(__file__).parent.parent.parent / "data" / "external" / "skills_database.json"
        
        try:
            with open(skills_db_path, "r", encoding="utf-8") as f:
                self.skills_db = json.load(f)
        except Exception as e:
            raise ResumeParsingError(
                message=f"Failed loading skills database: {e}",
                error_detail=sys.exc_info()
            )
        
        # Initialize FlashText processor or fallback
        if FLASHTEXT_AVAILABLE:
            self.processor = KeywordProcessor(case_sensitive=False)
            self.use_flashtext = True
        else:
            logger.warning("FlashText not available, using regex fallback")
            self.use_flashtext = False
        
        self.skill_to_category: Dict[str, str] = {}
        self.skill_weights: Dict[str, float] = {}
        self._build_skill_index()
        
        logger.info(f"SkillExtractor initialized ({'FlashText' if self.use_flashtext else 'Regex'})")
    
    def _build_skill_index(self):
        """Build FlashText keyword processor or regex patterns"""
        for category, data in self.skills_db.items():
            skills_list = data.get("skills", [])
            
            # Handle both old (list of strings) and new (list of objects) formats
            if skills_list and isinstance(skills_list[0], str):
                # Old format: ["Python", "Java"]
                for skill in skills_list:
                    self._add_skill(skill, category, 1.0)
            else:
                # New format: [{"name": "Python", "weight": 1.2, "aliases": ["py"]}]
                for skill_obj in skills_list:
                    name = skill_obj.get("name")
                    if not name:
                        continue
                    weight = skill_obj.get("weight", 1.0)
                    aliases = skill_obj.get("aliases", [])
                    
                    self._add_skill(name, category, weight)
                    for alias in aliases:
                        self._add_skill(alias, category, weight, canonical=name)
    
    def _add_skill(self, skill: str, category: str, weight: float, canonical: Optional[str] = None):
        """Add skill to index (FlashText or regex)"""
        canonical_name = canonical if canonical else skill
        
        if self.use_flashtext:
            self.processor.add_keyword(skill.lower(), canonical_name)
        
        self.skill_to_category[canonical_name] = category
        self.skill_weights[canonical_name] = weight
    
    def extract_skills(self, text: str) -> Dict:
        """Extract skills from text using FlashText or regex"""
        try:
            if self.use_flashtext:
                found = set(self.processor.extract_keywords(text))
            else:
                # Regex fallback (slower)
                found = self._extract_skills_regex(text)
            
            # Categorize skills
            categorized = {}
            for skill in found:
                cat = self.skill_to_category.get(skill)
                if cat:
                    categorized.setdefault(cat, []).append(skill)
            
            return {
                "skills": sorted(found),
                "categorized_skills": categorized,
                "skill_count": len(found)
            }
        except Exception as e:
            raise ResumeParsingError(
                message=f"Skill extraction failed: {e}",
                error_detail=sys.exc_info()
            )
    def extract_skills_with_context(self, text: str) -> Dict:
        """Extract skills with proficiency-aware weighting"""
        base_skills = self.extract_skills(text)

        skill_contexts = {}
        for skill in base_skills['skills']:
            pattern = rf'(.{{0,50}})\b{re.escape(skill)}\b(.{{0,50}})'
            matches = re.finditer(pattern, text, re.IGNORECASE)

            max_multiplier = 1.0

            for match in matches:
                before = match.group(1).lower()
                after = match.group(2).lower()
                full_context = before + skill.lower() + after

                # ❗ Negative indicators ONLY if they appear BEFORE the skill
                if any(word in before for word in ['basic', 'familiar', 'learning', 'beginner']):
                    max_multiplier = min(max_multiplier, 0.8)
                    continue

                # Strong proficiency indicators
                if any(word in full_context for word in ['expert', 'advanced', 'proficient', 'mastery']):
                    max_multiplier = max(max_multiplier, 1.5)
                elif any(word in full_context for word in ['experienced', 'skilled', 'strong']):
                    max_multiplier = max(max_multiplier, 1.3)

                # Years of experience
                years_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)', full_context)
                if years_match:
                    years = int(years_match.group(1))
                    if years >= 5:
                        max_multiplier = max(max_multiplier, 1.4)
                    elif years >= 3:
                        max_multiplier = max(max_multiplier, 1.2)

            base_weight = self.skill_weights.get(skill, 1.0)
            skill_contexts[skill] = {
                'base_weight': base_weight,
                'context_multiplier': max_multiplier,
                'final_weight': base_weight * max_multiplier
            }

        return {
            **base_skills,
            'skill_contexts': skill_contexts
        }



    
    def _extract_skills_regex(self, text: str) -> set:
        """Fallback regex extraction (O(n*m) - slower)"""
        found = set()
        text_lower = text.lower()
        for skill in self.skill_weights.keys():
            if re.search(rf'\b{re.escape(skill.lower())}\b', text_lower):
                found.add(skill)
        return found
    
    def calculate_skill_match_score(self, resume_skills: List[str], jd_skills: List[str]) -> float:
        """Weighted skill match score (better than simple count)"""
        if not jd_skills:
            return 0.0
        
        resume_set = set(resume_skills)
        total_weight = sum(self.skill_weights.get(s, 1.0) for s in jd_skills)
        matched_weight = sum(self.skill_weights.get(s, 1.0) for s in jd_skills if s in resume_set)
        
        return round(matched_weight / total_weight, 3)
    
    def get_missing_skills(self, resume_skills: List[str], jd_skills: List[str]) -> List[str]:
        """Find skills in JD missing from resume"""
        return sorted(list(set(jd_skills) - set(resume_skills)))
    
    def categorize_skills_by_type(self, skills: List[str]) -> Dict[str, List[str]]:
        """Group skills by category"""
        categorized = {}
        for skill in skills:
            cat = self.skill_to_category.get(skill)
            if cat:
                categorized.setdefault(cat, []).append(skill)
        return categorized


# Convenience helper
_SKILL_EXTRACTOR = SkillExtractor()
def extract_skills_from_text(text: str) -> List[str]:
    """Quick skill extraction"""
    return _SKILL_EXTRACTOR.extract_skills(text)["skills"]