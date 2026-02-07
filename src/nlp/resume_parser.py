"""
Resume Parser - Extract structured information from resumes
Uses optimized EntityExtractor + SkillExtractor
"""

import re
from typing import Dict, List, Optional, Any
import sys

from src.logging import get_logger
from src.exceptions import ResumeParsingError
from src.nlp.entity_extractor import EntityExtractor
from src.nlp.skill_extractor import SkillExtractor
from src.utils.date_utils import parse_date, calculate_total_experience
from src.utils.text_utils import clean_text

logger = get_logger(__name__)


class ResumeParser:
    """Parse resumes into structured data"""
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.skill_extractor = SkillExtractor()
        logger.info("ResumeParser initialized")
    
    def parse(self, resume_text: str, resume_id: Optional[str] = None) -> Dict[str, Any]:
        """Main parsing - extracts all resume sections"""
        try:
            logger.info(f"Parsing resume: {resume_id or 'unknown'}")
            cleaned_text = clean_text(resume_text)
            
            parsed_data = {
                'resume_id': resume_id,
                'raw_text': resume_text,
                'contact_info': self._extract_contact_info(cleaned_text),
                'skills': self._extract_skills(cleaned_text),
                'experience': self._extract_experience(resume_text),
                'education': self._extract_education(resume_text),
                'projects': self._extract_projects(resume_text),
                'summary': self._extract_summary(resume_text),
                'metadata': self._extract_metadata(cleaned_text)
            }
            
            logger.info(f"Resume parsed successfully: {resume_id}")
            return parsed_data
        except Exception as e:
            error_msg = f"Failed to parse resume {resume_id}: {str(e)}"
            logger.error(error_msg)
            raise ResumeParsingError(
                message=error_msg,
                resume_id=resume_id,
                error_detail=sys.exc_info()
            )
    
    def _extract_contact_info(self, text: str) -> Dict[str, Any]:
        """Extract name, email, phone, location, URLs"""
        try:
            entities = self.entity_extractor.extract_all_entities(text)
            name = self.entity_extractor.extract_name_from_resume(text)
            location = entities['locations'][0] if entities['locations'] else None
            
            return {
                'name': name,
                'email': entities['emails'][0] if entities['emails'] else None,
                'phone': entities['phones'][0] if entities['phones'] else None,
                'location': location,
                'linkedin': entities['urls']['linkedin'][0] if entities['urls']['linkedin'] else None,
                'github': entities['urls']['github'][0] if entities['urls']['github'] else None,
                'portfolio': entities['urls']['all'][0] if entities['urls']['all'] else None
            }
        except Exception as e:
            logger.warning(f"Error extracting contact info: {str(e)}")
            return {}
    
    def _extract_skills(self, text: str) -> Dict[str, Any]:
        """Extract technical skills using SkillExtractor"""
        try:
            skill_data = self.skill_extractor.extract_skills(text)
            return {
                'all_skills': skill_data['skills'],
                'categorized_skills': skill_data['categorized_skills'],
                'skill_count': skill_data['skill_count'],
                'top_skills': skill_data['skills'][:10]
            }
        except Exception as e:
            logger.warning(f"Error extracting skills: {str(e)}")
            return {'all_skills': [], 'skill_count': 0}
    
    def _extract_experience(self, text: str) -> Dict[str, Any]:
        """Extract work experience with companies, roles, dates"""
        try:
            experience_entries = []
            exp_section = self._extract_section(text, ['experience', 'work experience', 'employment'])
            
            if not exp_section:
                logger.warning("No experience section found")
                return {'entries': [], 'total_years': 0, 'total_companies': 0}
            
            job_titles = self._extract_job_titles(exp_section)
            companies = self.entity_extractor.extract_organizations(exp_section)
            dates = self.entity_extractor.extract_dates(exp_section)
            
            # Block-based parsing for better accuracy
            experience_entries = self._match_dates_to_companies(exp_section, companies, dates, job_titles)
            
            total_years = self._calculate_total_experience(experience_entries)
            
            return {
                'entries': experience_entries,
                'total_years': total_years,
                'total_companies': len(experience_entries)
            }
        except Exception as e:
            logger.warning(f"Error extracting experience: {str(e)}")
            return {'entries': [], 'total_years': 0, 'total_companies': 0}
    
    def _match_dates_to_companies(self, exp_section: str, companies: List[str], 
                                   dates: List[Dict], job_titles: List[str]) -> List[Dict]:
        """Match dates to companies using block-based parsing"""
        from datetime import datetime
        
        entries = []
        used_companies = set()
        # Split section into blocks (separated by blank lines or bullets)
        blocks = re.split(r'\n\s*\n|(?=^[•●\*\-]\s)', exp_section, flags=re.MULTILINE)
        
        title_idx = 0
        for block in blocks:
            if len(block.strip()) < 20:
                continue
            
            # Find company in this block
            company = None
            for comp in companies:
                if comp.lower() in block.lower() and comp not in used_companies:
                    company = comp
                    used_companies.add(comp)
                    break
            
            if not company:
                continue
            
            # Find dates in THIS block only
            block_dates = []
            for date in dates:
                date_str = date['date_string']
                if date_str in block:
                    block_dates.append(date)
            
            # Sort dates chronologically
            block_dates.sort(key=lambda d: d.get('parsed_date') or datetime.min)
            
            # Assign dates
            start_date = block_dates[0]['date_string'] if len(block_dates) > 0 else None
            end_date = block_dates[-1]['date_string'] if len(block_dates) > 1 else 'Present'
            
            # Handle case where only one date (assume it's start, end is Present)
            if len(block_dates) == 1:
                end_date = 'Present'
            
            entry = {
                'company': company,
                'title': job_titles[title_idx] if title_idx < len(job_titles) else 'Not specified',
                'start_date': start_date,
                'end_date': end_date,
                'duration': None
            }
            
            entries.append(entry)
            title_idx += 1
            
            if len(entries) >= 5:  # Limit to 5 most recent
                break
        
        return entries
    
    def _extract_education(self, text: str) -> Dict[str, Any]:
        """Extract degrees, institutions, years"""
        try:
            education_entries = []
            edu_section = self._extract_section(text, ['education', 'academic', 'qualification'])
            
            if not edu_section:
                logger.warning("No education section found")
                return {'entries': [], 'highest_degree': None, 'total_degrees': 0}
            
            # Extract degrees using keywords from skills DB (with safety check)
            degree_keywords = self.skill_extractor.skills_db.get('education_keywords', {}).get('degrees', {})
            found_degrees = []
            
            if degree_keywords:  # Safety check
                for degree_level, keywords in degree_keywords.items():
                    for keyword in keywords:
                        if re.search(rf'\b{re.escape(keyword)}\b', edu_section, re.IGNORECASE):
                            found_degrees.append({'level': degree_level, 'degree': keyword})
                            break
            else:
                logger.warning("Education keywords not found in skills database")
            
            institutions = self.entity_extractor.extract_organizations(edu_section)
            years = re.findall(r'\b(?:19|20)\d{2}\b', edu_section)  # Fixed: non-capturing group
            
            # Build education entries
            for i, degree in enumerate(found_degrees):
                entry = {
                    'degree': degree['degree'],
                    'level': degree['level'],
                    'institution': institutions[i] if i < len(institutions) else None,
                    'year': years[i] if i < len(years) else None,
                    'field': self._extract_field_of_study(edu_section)
                }
                education_entries.append(entry)
            
            # Highest degree
            degree_hierarchy = ['phd', 'master', 'bachelor', 'associate', 'diploma', 'certification', 'high_school']
            highest_degree = None
            for level in degree_hierarchy:
                if any(e['level'] == level for e in education_entries):
                    highest_degree = level
                    break
            
            return {
                'entries': education_entries,
                'highest_degree': highest_degree,
                'total_degrees': len(education_entries)
            }
        except Exception as e:
            logger.warning(f"Error extracting education: {str(e)}")
            return {'entries': [], 'highest_degree': None, 'total_degrees': 0}
    
    def _extract_projects(self, text: str) -> Dict[str, Any]:
        """Extract project names, descriptions, technologies"""
        try:
            proj_section = self._extract_section(text, ['projects', 'personal projects'])
            if not proj_section:
                return {'entries': [], 'total_projects': 0}
            
            project_blocks = re.split(r'\n\s*\n|•|●|\*', proj_section)
            projects = []
            
            for block in project_blocks[:5]:
                if len(block.strip()) < 20:
                    continue
                
                project_skills = self.skill_extractor.extract_skills(block)
                lines = block.strip().split('\n')
                project_name = lines[0].strip() if lines else 'Unnamed Project'
                
                projects.append({
                    'name': project_name,
                    'description': block.strip(),
                    'technologies': project_skills['skills']
                })
            
            return {
                'entries': projects,
                'total_projects': len(projects)
            }
        except Exception as e:
            logger.warning(f"Error extracting projects: {str(e)}")
            return {'entries': [], 'total_projects': 0}
    
    def _extract_summary(self, text: str) -> Optional[str]:
        """Extract professional summary/objective"""
        try:
            summary = self._extract_section(text, ['summary', 'objective', 'profile', 'about'])
            return summary[:500].strip() if summary else None
        except Exception as e:
            logger.warning(f"Error extracting summary: {str(e)}")
            return None
    
    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata about resume"""
        try:
            return {
                'text_length': len(text),
                'word_count': len(text.split()),
                'has_email': '@' in text,
                'has_phone': bool(re.search(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', text)),
                'has_linkedin': 'linkedin' in text.lower(),
                'has_github': 'github' in text.lower(),
                'sections_found': self._find_sections(text)
            }
        except Exception as e:
            logger.warning(f"Error extracting metadata: {str(e)}")
            return {}
    
    # Helper methods
    def _extract_section(self, text: str, section_names: List[str]) -> Optional[str]:
        """
        Extract section content robustly without mistaking job titles
        for section headers.
        """
        lines = text.splitlines()
        section_names = [name.lower() for name in section_names]

        start_idx = None

        # 1. Find section header (prefix match, not equality)
        for i, line in enumerate(lines):
            normalized = line.strip().lower().rstrip(":")
            for name in section_names:
                if normalized.startswith(name):
                    start_idx = i + 1
                    break
            if start_idx is not None:
                break

        if start_idx is None:
            return None

        # 2. Skip blank lines after header
        while start_idx < len(lines) and not lines[start_idx].strip():
            start_idx += 1

        if start_idx >= len(lines):
            return None

        # 3. Collect content until next REAL section header
        section_lines = []
        for line in lines[start_idx:]:
            stripped = line.strip()

            # Stop ONLY at real section headers
            if stripped.isupper() and len(stripped.split()) <= 4:
                break

            if re.match(r'^[A-Za-z\s]+:$', stripped):
                break

            section_lines.append(line)

        content = "\n".join(section_lines).strip()
        return content if content else None




    
    def _extract_job_titles(self, text: str) -> List[str]:
        """Extract job titles from text"""
        job_titles_db = self.skill_extractor.skills_db.get('job_titles', {})
        found_titles = []
        for category, titles in job_titles_db.items():
            for title in titles:
                if re.search(rf'\b{re.escape(title)}\b', text, re.IGNORECASE):
                    found_titles.append(title)
        return found_titles[:5]
    
    def _extract_field_of_study(self, text: str) -> Optional[str]:
        """Extract field of study"""
        fields = self.skill_extractor.skills_db.get('education_keywords', {}).get('fields', [])
        for field in fields:
            if re.search(rf'\b{re.escape(field)}\b', text, re.IGNORECASE):
                return field
        return None
    
    def _calculate_total_experience(self, experience_entries: List[Dict]) -> float:
        """Calculate total years from entries"""
        total_months = 0
        for entry in experience_entries:
            start = entry.get('start_date')
            end = entry.get('end_date')
            if start and end:
                try:
                    start_parsed = parse_date(start)
                    end_parsed = parse_date(end)
                    if start_parsed and end_parsed:
                        duration = calculate_total_experience([{
                            'start_date': start_parsed,
                            'end_date': end_parsed
                        }])
                        total_months += duration.get('total_months', 0)
                except:
                    pass
        return round(total_months / 12, 1)
    
    def _find_sections(self, text: str) -> List[str]:
        """Find which sections exist in resume"""
        common_sections = ['experience', 'education', 'skills', 'projects', 'summary', 'certifications']
        found = []
        for section in common_sections:
            if re.search(rf'\b{section}\b', text, re.IGNORECASE):
                found.append(section)
        return found


# Convenience function
def parse_resume(resume_text: str) -> Dict[str, Any]:
    """Quick helper to parse resume"""
    parser = ResumeParser()
    return parser.parse(resume_text)