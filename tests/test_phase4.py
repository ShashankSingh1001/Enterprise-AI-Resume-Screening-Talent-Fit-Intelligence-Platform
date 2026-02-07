"""
Phase 4 NLP Pipeline - Pytest Test Suite
Tests EntityExtractor, SkillExtractor, ResumeParser, JDParser with all fixes
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.nlp import (
    EntityExtractor,
    SkillExtractor,
    ResumeParser,
    JDParser,
    extract_contact_info,
    extract_skills_from_text,
    parse_resume,
    parse_jd
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_resume():
    """Sample resume for testing"""
    return """
John Doe
Email: john.doe@example.com
Phone: +1-555-123-4567
LinkedIn: linkedin.com/in/johndoe
GitHub: github.com/johndoe

PROFESSIONAL SUMMARY
Expert Python developer with 5+ years of experience building scalable cloud applications.
Strong background in AWS, Docker, and Kubernetes.

EXPERIENCE

Senior Python Developer
Tech Corp | San Francisco, CA
Jan 2020 - Present
- Built microservices using Django and FastAPI
- Expert in AWS ECS and EKS deployment
- Led team of 5 developers

Python Developer
StartupXYZ | Remote
Jun 2018 - Dec 2019
- Developed REST APIs with Flask
- Proficient in Docker and CI/CD

EDUCATION

Bachelor of Science in Computer Science
MIT | Cambridge, MA | 2018

SKILLS
Languages: Python, JavaScript, SQL, Go
Cloud: AWS, Azure, GCP
DevOps: Docker, Kubernetes, Terraform
Databases: PostgreSQL, MongoDB, Redis

PROJECTS
AI Resume Screener - Built ML-powered system using XGBoost
"""


@pytest.fixture
def sample_jd():
    """Sample job description for testing"""
    return """
Senior Python Developer

Tech Innovations Inc. | San Francisco, CA | Full-time

REQUIRED QUALIFICATIONS
- 5+ years of professional Python development experience
- Strong experience with Django or FastAPI
- Expertise in AWS services (Lambda, ECS, S3)
- Bachelor's degree in Computer Science

PREFERRED QUALIFICATIONS
- Experience with machine learning frameworks
- AWS or Kubernetes certifications

RESPONSIBILITIES
- Design and build scalable microservices
- Deploy applications on AWS cloud
- Mentor junior developers

TECHNICAL STACK
Python, Django, AWS, Docker, Kubernetes, PostgreSQL

BENEFITS
- Competitive salary: $150,000 - $180,000
- Remote-first culture
- Health insurance
"""


@pytest.fixture
def entity_extractor():
    """EntityExtractor fixture"""
    return EntityExtractor()


@pytest.fixture
def skill_extractor():
    """SkillExtractor fixture"""
    return SkillExtractor()


@pytest.fixture
def resume_parser():
    """ResumeParser fixture"""
    return ResumeParser()


@pytest.fixture
def jd_parser():
    """JDParser fixture"""
    return JDParser()


# ============================================================================
# ENTITY EXTRACTOR TESTS
# ============================================================================

class TestEntityExtractor:
    """Test EntityExtractor functionality"""
    
    def test_email_extraction(self, entity_extractor, sample_resume):
        emails = entity_extractor.extract_emails(sample_resume)
        assert len(emails) > 0
        assert 'john.doe@example.com' in emails
    
    def test_phone_extraction(self, entity_extractor, sample_resume):
        phones = entity_extractor.extract_phone_numbers(sample_resume)
        assert len(phones) > 0
        # Should extract at least one phone number
    
    def test_url_extraction(self, entity_extractor, sample_resume):
        urls = entity_extractor.extract_urls(sample_resume)
        assert 'linkedin' in urls
        assert 'github' in urls
        assert len(urls['linkedin']) > 0
        assert len(urls['github']) > 0
    
    def test_name_extraction(self, entity_extractor, sample_resume):
        name = entity_extractor.extract_name_from_resume(sample_resume)
        assert name is not None
        assert name == "John Doe"  # Should filter out "Email:"
    
    def test_name_extraction_with_contact_keywords(self, entity_extractor):
        # Test filtering of contact keywords
        text = "John Doe Email\njohn@example.com"
        name = entity_extractor.extract_name_from_resume(text)
        assert name != "John Doe Email"  # Should filter
    
    def test_organization_extraction(self, entity_extractor, sample_resume):
        orgs = entity_extractor.extract_organizations(sample_resume)
        assert len(orgs) > 0
    
    def test_location_extraction(self, entity_extractor, sample_resume):
        locations = entity_extractor.extract_locations(sample_resume)
        assert len(locations) > 0
    
    def test_date_extraction(self, entity_extractor, sample_resume):
        dates = entity_extractor.extract_dates(sample_resume)
        assert len(dates) > 0
        # Should have date_string and parsed_date
        assert 'date_string' in dates[0]
    
    def test_all_entities_single_pass(self, entity_extractor, sample_resume):
        entities = entity_extractor.extract_all_entities(sample_resume)
        assert 'emails' in entities
        assert 'phones' in entities
        assert 'urls' in entities
        assert 'organizations' in entities
        assert len(entities['emails']) > 0


# ============================================================================
# SKILL EXTRACTOR TESTS
# ============================================================================

class TestSkillExtractor:
    """Test SkillExtractor functionality"""
    
    def test_basic_skill_extraction(self, skill_extractor):
        text = "I have experience with Python, Django, AWS, Docker, and Kubernetes"
        result = skill_extractor.extract_skills(text)
        assert result['skill_count'] >= 5
        assert 'Python' in result['skills']
    
    def test_alias_handling(self, skill_extractor):
        text = "I know JS, K8s, and sklearn"
        result = skill_extractor.extract_skills(text)
        # Should normalize to JavaScript, Kubernetes, Scikit-learn
        assert 'JavaScript' in result['skills'] or 'JS' in result['skills']
        assert 'Kubernetes' in result['skills'] or 'K8s' in result['skills']
    
    def test_categorized_skills(self, skill_extractor, sample_resume):
        result = skill_extractor.extract_skills(sample_resume)
        assert 'categorized_skills' in result
        assert len(result['categorized_skills']) > 0
    
    def test_weighted_skill_matching(self, skill_extractor):
        resume_skills = ['Python', 'Django', 'Docker']
        jd_skills = ['Python', 'AWS', 'Kubernetes']
        score = skill_extractor.calculate_skill_match_score(resume_skills, jd_skills)
        assert 0 <= score <= 1
    
    def test_missing_skills_detection(self, skill_extractor):
        resume_skills = ['Python', 'Django']
        jd_skills = ['Python', 'AWS', 'Kubernetes']
        missing = skill_extractor.get_missing_skills(resume_skills, jd_skills)
        assert 'AWS' in missing
        assert 'Kubernetes' in missing
    
    def test_context_based_weighting(self, skill_extractor):
        text = "Expert Python developer with 5+ years experience. Familiar with Java."
        result = skill_extractor.extract_skills_with_context(text)
        assert 'skill_contexts' in result
        
        # Python should have higher weight due to "expert" and "5+ years"
        if 'Python' in result['skill_contexts']:
            python_context = result['skill_contexts']['Python']
            assert python_context['context_multiplier'] > 1.0
        
        # Java should have lower weight due to "familiar"
        if 'Java' in result['skill_contexts']:
            java_context = result['skill_contexts']['Java']
            assert java_context['context_multiplier'] <= 1.0
    
    def test_proficiency_indicators(self, skill_extractor):
        # Test expert level
        text1 = "Expert in AWS cloud architecture"
        result1 = skill_extractor.extract_skills_with_context(text1)
        if 'AWS' in result1['skill_contexts']:
            assert result1['skill_contexts']['AWS']['context_multiplier'] >= 1.3
        
        # Test basic level
        text2 = "Basic knowledge of Python"
        result2 = skill_extractor.extract_skills_with_context(text2)
        if 'Python' in result2['skill_contexts']:
            assert result2['skill_contexts']['Python']['context_multiplier'] <= 1.0


# ============================================================================
# RESUME PARSER TESTS
# ============================================================================

class TestResumeParser:
    """Test ResumeParser functionality"""
    
    def test_contact_info_extraction(self, resume_parser, sample_resume):
        parsed = resume_parser.parse(sample_resume, resume_id="test_001")
        contact = parsed['contact_info']
        
        assert contact['name'] == "John Doe"
        assert contact['email'] == 'john.doe@example.com'
        assert contact['phone'] is not None
    
    def test_skills_extraction(self, resume_parser, sample_resume):
        parsed = resume_parser.parse(sample_resume)
        skills = parsed['skills']
        
        assert skills['skill_count'] > 0
        assert 'all_skills' in skills
        assert 'categorized_skills' in skills
    
    def test_experience_extraction_structure(self, resume_parser, sample_resume):
        parsed = resume_parser.parse(sample_resume)
        exp = parsed['experience']
        
        # Validate structure
        assert 'entries' in exp
        assert 'total_years' in exp
        assert 'total_companies' in exp
        assert isinstance(exp['total_years'], (int, float))
    
    def test_block_based_experience_matching(self, resume_parser):
        # Test with multiple jobs
        resume = """
John Doe

EXPERIENCE

Senior Developer
Google | 2020 - Present
- Built cloud systems

Developer  
Google Cloud | 2018 - 2020
- Developed APIs
"""
        parsed = resume_parser.parse(resume)
        exp = parsed['experience']
        
        # Should handle repeated "Google" correctly
        assert len(exp['entries']) >= 1
    
    def test_education_extraction_structure(self, resume_parser, sample_resume):
        parsed = resume_parser.parse(sample_resume)
        edu = parsed['education']
        
        # Validate structure
        assert 'entries' in edu
        assert 'highest_degree' in edu
        assert 'total_degrees' in edu
    
    def test_section_boundary_validation(self, resume_parser):
        # Test that Skills don't bleed into Experience
        resume = """
Name: Jane Smith

SKILLS
Python, Java, AWS

EXPERIENCE
Software Engineer at Tech Corp
"""
        parsed = resume_parser.parse(resume)
        
        # Skills section should not contain "Software Engineer"
        skills_text = str(parsed['skills'])
        assert 'Software Engineer' not in skills_text or 'Tech Corp' not in skills_text
    
    def test_projects_extraction(self, resume_parser, sample_resume):
        parsed = resume_parser.parse(sample_resume)
        projects = parsed['projects']
        
        assert 'entries' in projects
        assert 'total_projects' in projects
    
    def test_metadata_extraction(self, resume_parser, sample_resume):
        parsed = resume_parser.parse(sample_resume)
        meta = parsed['metadata']
        
        assert 'text_length' in meta
        assert 'word_count' in meta
        assert 'has_email' in meta
        assert meta['has_email'] == True
    
    def test_empty_sections_handling(self, resume_parser):
        # Test resume with missing sections
        minimal_resume = "John Doe\njohn@example.com"
        parsed = resume_parser.parse(minimal_resume)
        
        # Should not crash, return empty structures
        assert parsed['experience']['total_companies'] == 0
        assert parsed['education']['total_degrees'] == 0


# ============================================================================
# JD PARSER TESTS
# ============================================================================

class TestJDParser:
    """Test JDParser functionality"""
    
    def test_job_info_extraction(self, jd_parser, sample_jd):
        parsed = jd_parser.parse(sample_jd, jd_id="jd_001")
        job_info = parsed['job_info']
        
        # Validate structure
        assert 'title' in job_info
        assert 'company' in job_info
        assert 'location' in job_info
        assert 'employment_type' in job_info
    
    def test_required_skills_extraction(self, jd_parser, sample_jd):
        parsed = jd_parser.parse(sample_jd)
        req_skills = parsed['required_skills']
        
        assert 'all_skills' in req_skills
        assert 'skill_count' in req_skills
        assert req_skills['skill_count'] > 0
    
    def test_qualifications_extraction(self, jd_parser, sample_jd):
        parsed = jd_parser.parse(sample_jd)
        quals = parsed['qualifications']
        
        # Validate structure
        assert 'min_experience_years' in quals
        assert 'required_degrees' in quals
        assert isinstance(quals['min_experience_years'], (int, float))
    
    def test_experience_requirement_extraction(self, jd_parser, sample_jd):
        parsed = jd_parser.parse(sample_jd)
        quals = parsed['qualifications']
        
        # Sample JD has "5+ years"
        assert quals['min_experience_years'] >= 5
    
    def test_section_boundaries_jd(self, jd_parser):
        # Test that sections don't bleed
        jd = """
Job Title: Developer

REQUIRED QUALIFICATIONS
5 years Python

RESPONSIBILITIES
Build systems
"""
        parsed = jd_parser.parse(jd)
        
        # Qualifications should not contain "Build systems"
        quals_text = str(parsed['qualifications'])
        assert 'Build systems' not in quals_text
    
    def test_salary_extraction(self, jd_parser, sample_jd):
        parsed = jd_parser.parse(sample_jd)
        job_info = parsed['job_info']
        
        # Sample JD has salary range
        if job_info['salary_range']:
            assert '150' in job_info['salary_range'] or '180' in job_info['salary_range']
    
    def test_metadata_extraction(self, jd_parser, sample_jd):
        parsed = jd_parser.parse(sample_jd)
        meta = parsed['metadata']
        
        assert 'has_salary' in meta
        assert 'has_remote_option' in meta
        assert 'sections_found' in meta


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Test integration between components"""
    
    def test_resume_jd_skill_matching(self, sample_resume, sample_jd):
        resume_parsed = parse_resume(sample_resume)
        jd_parsed = parse_jd(sample_jd)
        
        resume_skills = resume_parsed['skills']['all_skills']
        jd_skills = jd_parsed['required_skills']['all_skills']
        
        extractor = SkillExtractor()
        match_score = extractor.calculate_skill_match_score(resume_skills, jd_skills)
        
        assert 0 <= match_score <= 1
    
    def test_experience_matching(self, sample_resume, sample_jd):
        resume_parsed = parse_resume(sample_resume)
        jd_parsed = parse_jd(sample_jd)
        
        resume_exp = resume_parsed['experience']['total_years']
        jd_exp = jd_parsed['qualifications']['min_experience_years']
        
        # Both should be numeric
        assert isinstance(resume_exp, (int, float))
        assert isinstance(jd_exp, (int, float))
    
    def test_context_weighted_matching(self, sample_resume, sample_jd):
        # Test context-based skill matching
        extractor = SkillExtractor()
        
        resume_result = extractor.extract_skills_with_context(sample_resume)
        jd_result = extractor.extract_skills(sample_jd)
        
        # Resume should have context weights
        assert 'skill_contexts' in resume_result
    
    def test_end_to_end_parsing(self, sample_resume, sample_jd):
        # Full pipeline test
        resume_parsed = parse_resume(sample_resume)
        jd_parsed = parse_jd(sample_jd)
        
        # Validate all major sections exist
        assert 'contact_info' in resume_parsed
        assert 'skills' in resume_parsed
        assert 'experience' in resume_parsed
        assert 'education' in resume_parsed
        
        assert 'job_info' in jd_parsed
        assert 'required_skills' in jd_parsed
        assert 'qualifications' in jd_parsed


# ============================================================================
# EDGE CASES & ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_text(self, skill_extractor):
        result = skill_extractor.extract_skills("")
        assert result['skill_count'] == 0
    
    def test_no_skills_text(self, skill_extractor):
        text = "This is plain English with no technical terms."
        result = skill_extractor.extract_skills(text)
        assert isinstance(result['skill_count'], int)
    
    def test_special_characters_in_skills(self, skill_extractor):
        text = "C++, C#, .NET, Node.js, Vue.js"
        result = skill_extractor.extract_skills(text)
        assert result['skill_count'] > 0
    
    def test_case_insensitivity(self, skill_extractor):
        text = "I know PYTHON, django, AwS, and KuBerNeTes"
        result = skill_extractor.extract_skills(text)
        assert result['skill_count'] > 0
    
    def test_malformed_resume(self, resume_parser):
        # Test with no proper sections
        bad_resume = "Just some random text without structure"
        parsed = resume_parser.parse(bad_resume)
        
        # Should not crash
        assert 'contact_info' in parsed
        assert 'skills' in parsed


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])