"""
Job Description Parser - Extract structured information from JDs
Uses optimized EntityExtractor + SkillExtractor
"""

import re
from typing import Dict, List, Optional, Any
import sys

from src.logging import get_logger
from src.exceptions import JDParsingError
from src.nlp.entity_extractor import EntityExtractor
from src.nlp.skill_extractor import SkillExtractor
from src.utils.text_utils import clean_text

logger = get_logger(__name__)


class JDParser:
    """Parse job descriptions into structured requirements"""
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.skill_extractor = SkillExtractor()
        logger.info("JDParser initialized")
    
    def parse(self, jd_text: str, jd_id: Optional[str] = None) -> Dict[str, Any]:
        """Main parsing - extracts all JD sections"""
        try:
            logger.info(f"Parsing JD: {jd_id or 'unknown'}")
            cleaned_text = clean_text(jd_text)
            
            parsed_data = {
                'jd_id': jd_id,
                'raw_text': jd_text,
                'job_info': self._extract_job_info(cleaned_text),
                'required_skills': self._extract_required_skills(cleaned_text),
                'preferred_skills': self._extract_preferred_skills(cleaned_text),
                'qualifications': self._extract_qualifications(cleaned_text),
                'responsibilities': self._extract_responsibilities(cleaned_text),
                'benefits': self._extract_benefits(cleaned_text),
                'metadata': self._extract_metadata(cleaned_text)
            }
            
            logger.info(f"JD parsed successfully: {jd_id}")
            return parsed_data
        except Exception as e:
            error_msg = f"Failed to parse JD {jd_id}: {str(e)}"
            logger.error(error_msg)
            raise JDParsingError(
                message=error_msg,
                jd_id=jd_id,
                error_detail=sys.exc_info()
            )
    
    def _extract_job_info(self, text: str) -> Dict[str, Any]:
        """Extract job title, company, location, type"""
        try:
            entities = self.entity_extractor.extract_all_entities(text)
            job_title = self._extract_job_title(text)
            company = entities['organizations'][0] if entities['organizations'] else None
            location = entities['locations'][0] if entities['locations'] else None
            emp_type = self._extract_employment_type(text)
            
            return {
                'title': job_title,
                'company': company,
                'location': location,
                'employment_type': emp_type,
                'remote': 'remote' in text.lower(),
                'salary_range': self._extract_salary(text)
            }
        except Exception as e:
            logger.warning(f"Error extracting job info: {str(e)}")
            return {}
    
    def _extract_required_skills(self, text: str) -> Dict[str, Any]:
        """Extract required/must-have skills"""
        try:
            req_section = self._extract_section(
                text, 
                ['required', 'requirements', 'must have', 'minimum qualifications']
            )
            skill_text = req_section if req_section else text
            skill_data = self.skill_extractor.extract_skills(skill_text)
            
            return {
                'all_skills': skill_data['skills'],
                'categorized_skills': skill_data['categorized_skills'],
                'skill_count': skill_data['skill_count'],
                'top_skills': skill_data['skills'][:10]
            }
        except Exception as e:
            logger.warning(f"Error extracting required skills: {str(e)}")
            return {'all_skills': [], 'skill_count': 0}
    
    def _extract_preferred_skills(self, text: str) -> Dict[str, Any]:
        """Extract preferred/nice-to-have skills"""
        try:
            pref_section = self._extract_section(
                text,
                ['preferred', 'nice to have', 'bonus', 'plus']
            )
            if not pref_section:
                return {'all_skills': [], 'skill_count': 0}
            
            skill_data = self.skill_extractor.extract_skills(pref_section)
            return {
                'all_skills': skill_data['skills'],
                'categorized_skills': skill_data['categorized_skills'],
                'skill_count': skill_data['skill_count']
            }
        except Exception as e:
            logger.warning(f"Error extracting preferred skills: {str(e)}")
            return {'all_skills': [], 'skill_count': 0}
    
    def _extract_qualifications(self, text: str) -> Dict[str, Any]:
        """Extract education and experience requirements"""
        try:
            qual_section = self._extract_section(
                text,
                ['qualifications', 'requirements', 'minimum requirements']
            )
            qual_text = qual_section if qual_section else text
            
            # Extract education requirements
            degree_keywords = self.skill_extractor.skills_db.get('education_keywords', {}).get('degrees', {})
            required_degrees = []
            for degree_level, keywords in degree_keywords.items():
                for keyword in keywords:
                    if re.search(rf'\b{re.escape(keyword)}\b', qual_text, re.IGNORECASE):
                        required_degrees.append(degree_level)
                        break
            
            # Extract experience requirement (multiple patterns)
            exp_patterns = [
                r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:professional\s+)?experience',  # 5+ years experience
                r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:professional\s+)?(?:Python|development|programming)',  # 5+ years Python
                r'minimum\s+(?:of\s+)?(\d+)\+?\s*(?:years?|yrs?)',  # minimum of 5 years
                r'at\s+least\s+(\d+)\+?\s*(?:years?|yrs?)',  # at least 5 years
            ]
            
            min_experience = 0
            for pattern in exp_patterns:
                exp_matches = re.findall(pattern, qual_text, re.IGNORECASE)
                if exp_matches:
                    min_experience = int(exp_matches[0])
                    break
            
            return {
                'required_degrees': required_degrees,
                'highest_degree_required': required_degrees[0] if required_degrees else None,
                'min_experience_years': min_experience,
                'certifications': self._extract_certifications(qual_text)
            }
        except Exception as e:
            logger.warning(f"Error extracting qualifications: {str(e)}")
            return {}
    
    def _extract_responsibilities(self, text: str) -> Dict[str, Any]:
        """Extract job responsibilities/duties"""
        try:
            resp_section = self._extract_section(
                text,
                ['responsibilities', 'duties', 'what you will do', 'role']
            )
            if not resp_section:
                return {'items': [], 'count': 0}
            
            # Split into bullet points or sentences
            resp_items = re.split(r'\n\s*[•●\*\-]|\n\d+\.|\.(?=\s+[A-Z])', resp_section)
            resp_items = [r.strip() for r in resp_items if len(r.strip()) > 20][:10]
            
            return {
                'items': resp_items,
                'count': len(resp_items)
            }
        except Exception as e:
            logger.warning(f"Error extracting responsibilities: {str(e)}")
            return {'items': [], 'count': 0}
    
    def _extract_benefits(self, text: str) -> Dict[str, Any]:
        """Extract benefits/perks"""
        try:
            benefits_section = self._extract_section(
                text,
                ['benefits', 'perks', 'what we offer']
            )
            if not benefits_section:
                return {'items': [], 'count': 0}
            
            benefit_items = re.split(r'\n\s*[•●\*\-]|\n\d+\.', benefits_section)
            benefit_items = [b.strip() for b in benefit_items if len(b.strip()) > 10][:10]
            
            return {
                'items': benefit_items,
                'count': len(benefit_items)
            }
        except Exception as e:
            logger.warning(f"Error extracting benefits: {str(e)}")
            return {'items': [], 'count': 0}
    
    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata about JD"""
        try:
            return {
                'text_length': len(text),
                'word_count': len(text.split()),
                'has_salary': bool(re.search(r'\$\d+|\d+k', text, re.IGNORECASE)),
                'has_remote_option': 'remote' in text.lower(),
                'has_benefits': 'benefits' in text.lower() or 'perks' in text.lower(),
                'sections_found': self._find_sections(text),
                'urgency_indicators': self._check_urgency(text)
            }
        except Exception as e:
            logger.warning(f"Error extracting metadata: {str(e)}")
            return {}
    
    # Helper methods
    def _extract_section(self, text: str, section_names: List[str]) -> Optional[str]:
        """Extract section with strict boundaries to prevent leakage"""
        for name in section_names:
            # Try multiple patterns for robustness
            patterns = [
                # Pattern 1: All-caps header
                rf'(?im)^{re.escape(name.upper())}:?\s*$\n(.*?)(?=^[A-Z][A-Z\s]+:?\s*$|\Z)',
                
                # Pattern 2: Title case header
                rf'(?im)^{re.escape(name.title())}:?\s*$\n(.*?)(?=^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:?\s*$|\Z)',
                
                # Pattern 3: With colon
                rf'(?im)^{re.escape(name)}:[ \t]*\n(.*?)(?=^[A-Z].*?:[ \t]*$|\Z)',
                
                # Pattern 4: Case-insensitive flexible (fallback)
                rf'(?i)^[\s]*{re.escape(name)}[\s]*:?[\s]*$\n(.*?)(?=\n\s*[A-Z][A-Za-z\s]+:|\Z)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    
                    # Post-validation: ensure content doesn't start with another header
                    lines = content.split('\n')
                    if lines and re.match(r'^[A-Z][A-Z\s]+:?\s*$', lines[0]):
                        # Skip first line if it's a header
                        content = '\n'.join(lines[1:]).strip()
                    
                    if content:  # Only return if we have actual content
                        return content
        
        return None
    
    def _extract_job_title(self, text: str) -> Optional[str]:
        """Extract job title from JD"""
        # Strategy 1: First line if short (most common in JDs)
        first_line = text.strip().split('\n')[0].strip()
        if len(first_line.split()) <= 10 and len(first_line) < 100:
            # Check if it looks like a title (not a section header or company name)
            if not any(word in first_line.lower() for word in ['about', 'description', 'posted', 'apply']):
                return first_line
        
        # Strategy 2: Match against job titles database
        job_titles_db = self.skill_extractor.skills_db.get('job_titles', {})
        for category, titles in job_titles_db.items():
            for title in titles:
                if re.search(rf'\b{re.escape(title)}\b', text[:300], re.IGNORECASE):
                    return title
        
        return None
    
    def _extract_employment_type(self, text: str) -> Optional[str]:
        """Extract employment type (full-time, part-time, etc.)"""
        patterns = {
            'full-time': r'\bfull[\s-]?time\b',
            'part-time': r'\bpart[\s-]?time\b',
            'contract': r'\bcontract\b',
            'internship': r'\bintern(?:ship)?\b'
        }
        for emp_type, pattern in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return emp_type
        return None
    
    def _extract_salary(self, text: str) -> Optional[str]:
        """Extract salary range if mentioned"""
        salary_pattern = re.compile(
            r'\$?(\d{2,3}),?(\d{3})\s*[-–]\s*\$?(\d{2,3}),?(\d{3})|'
            r'\$?(\d{2,3})k\s*[-–]\s*\$?(\d{2,3})k',
            re.IGNORECASE
        )
        match = salary_pattern.search(text)
        return match.group(0) if match else None
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract required certifications"""
        cert_keywords = self.skill_extractor.skills_db.get('certifications', {}).get('skills', [])
        
        # Handle both list of strings and list of objects
        found_certs = []
        if cert_keywords and isinstance(cert_keywords[0], dict):
            for cert_obj in cert_keywords:
                cert = cert_obj.get('name', '')
                if re.search(rf'\b{re.escape(cert)}\b', text, re.IGNORECASE):
                    found_certs.append(cert)
        else:
            for cert in cert_keywords:
                if re.search(rf'\b{re.escape(cert)}\b', text, re.IGNORECASE):
                    found_certs.append(cert)
        return found_certs
    
    def _find_sections(self, text: str) -> List[str]:
        """Find which sections are present in JD"""
        common_sections = ['responsibilities', 'requirements', 'qualifications', 'skills', 'benefits']
        found = []
        for section in common_sections:
            if re.search(rf'\b{section}\b', text, re.IGNORECASE):
                found.append(section)
        return found
    
    def _check_urgency(self, text: str) -> bool:
        """Check if JD has urgency indicators"""
        urgency_keywords = ['urgent', 'immediate', 'asap', 'hiring immediately']
        return any(keyword in text.lower() for keyword in urgency_keywords)


# Convenience function
def parse_jd(jd_text: str) -> Dict[str, Any]:
    """Quick helper to parse job description"""
    parser = JDParser()
    return parser.parse(jd_text)