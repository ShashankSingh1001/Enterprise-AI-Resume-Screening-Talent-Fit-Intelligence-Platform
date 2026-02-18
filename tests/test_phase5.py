"""
Comprehensive Phase 5 Feature Engineering Tests
Tests all 6 modules: Semantic Similarity, Skills, Experience, Education, Generator, Store
"""

import pytest
import numpy as np
import logging
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import all Phase 5 modules
from src.features.semantic_similarity import (
    SemanticSimilarityCalculator,
    SBERTSingleton,
    get_sbert_model,
    calculate_similarity
)
from src.features.skill_features import (
    SkillMatcher,
    calculate_skill_match
)
from src.features.experience_features import (
    ExperienceCalculator,
    calculate_experience_match
)
from src.features.education_features import (
    EducationCalculator,
    calculate_education_match
)
from src.features.feature_generator import (
    FeatureGenerator,
    generate_features
)
from src.features.feature_store import (
    FeatureStore,
    create_feature_store
)
from src.features.config import config

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


# ============================================================================
# MODULE 1: SEMANTIC SIMILARITY TESTS
# ============================================================================

class TestSemanticSimilarity:
    """Test Semantic Similarity Calculator"""
    
    def setup_method(self):
        """Initialize calculator before each test"""
        self.calc = SemanticSimilarityCalculator()
    
    # Singleton Tests
    def test_singleton_pattern(self):
        """Verify singleton returns same instance"""
        instance1 = SBERTSingleton()
        instance2 = SBERTSingleton()
        assert instance1 is instance2
    
    def test_model_loaded_once(self):
        """Verify model is loaded only once"""
        singleton = SBERTSingleton()
        model1 = singleton.get_model()
        model2 = singleton.get_model()
        assert model1 is model2
    
    # Cosine Similarity Tests
    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0"""
        vec = np.array([1.0, 2.0, 3.0])
        sim = self.calc.cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.001
    
    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0"""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        sim = self.calc.cosine_similarity(vec1, vec2)
        assert abs(sim - 0.0) < 0.001
    
    def test_empty_vector(self):
        """Empty vectors should return 0.0"""
        vec1 = np.array([])
        vec2 = np.array([1.0, 2.0])
        sim = self.calc.cosine_similarity(vec1, vec2)
        assert sim == 0.0
    
    # Text Encoding Tests
    def test_encode_normal_text(self):
        """Test encoding normal text"""
        text = "Python machine learning engineer"
        embedding = self.calc.encode_text(text)
        assert embedding.shape == (384,)
        assert not np.all(embedding == 0)
    
    def test_encode_empty_text(self):
        """Empty text should return zero vector"""
        embedding = self.calc.encode_text("")
        assert embedding.shape == (384,)
        assert np.all(embedding == 0)
    
    def test_encode_batch(self):
        """Test batch encoding"""
        texts = ["Python developer", "Java engineer", "ML specialist"]
        embeddings = self.calc.encode_batch(texts)
        assert len(embeddings) == 3
        assert all(emb.shape == (384,) for emb in embeddings)
    
    # Basic Similarity Tests
    def test_identical_text_similarity(self):
        """Identical texts should have high similarity"""
        text = "Python machine learning engineer"
        sim = self.calc.calculate_similarity(text, text)
        assert sim > 0.99
    
    def test_similar_text_similarity(self):
        """Similar texts should have high similarity"""
        text1 = "Python developer with ML experience"
        text2 = "ML engineer proficient in Python"
        sim = self.calc.calculate_similarity(text1, text2)
        assert sim > 0.6
    
    def test_different_text_similarity(self):
        """Different texts should have low similarity"""
        text1 = "Python machine learning engineer"
        text2 = "Sales and marketing manager"
        sim = self.calc.calculate_similarity(text1, text2)
        assert sim < 0.5
    
    # Section Similarity Tests
    def test_section_similarity(self):
        """Test section-wise similarity"""
        resume_sections = {"skills": "Python, ML, NLP"}
        jd_sections = {"required_skills": "Python, Machine Learning"}
        
        sims = self.calc.calculate_section_similarity(resume_sections, jd_sections)
        assert "skills_vs_required_skills" in sims
        assert sims["skills_vs_required_skills"] > 0.5
    
    # Weighted Similarity Tests
    def test_weighted_similarity(self):
        """Test weighted similarity calculation"""
        resume = "Python ML engineer"
        jd = "ML engineer needed"
        
        results = self.calc.calculate_weighted_similarity(resume, jd)
        assert "overall_similarity" in results
        assert "weighted_similarity" in results
        assert 0 <= results["weighted_similarity"] <= 1
    
    # Batch Processing Tests
    def test_batch_similarity(self):
        """Test batch similarity calculation"""
        resumes = ["Python engineer", "Java developer", "ML specialist"]
        jds = ["Python developer", "Java engineer", "Machine learning expert"]
        
        sims = self.calc.batch_calculate_similarity(resumes, jds)
        assert len(sims) == 3
        assert all(0 <= sim <= 1 for sim in sims)
    
    def test_batch_mismatched_lengths(self):
        """Test batch with mismatched lengths"""
        with pytest.raises(ValueError):
            self.calc.batch_calculate_similarity(["A", "B"], ["C"])
    
    # Convenience Function Test
    def test_calculate_similarity_function(self):
        """Test quick similarity function"""
        sim = calculate_similarity("Python engineer", "Python developer")
        assert 0 <= sim <= 1
        assert sim > 0.7


# ============================================================================
# MODULE 2: SKILL FEATURES TESTS
# ============================================================================

class TestSkillFeatures:
    """Test Skill Matcher"""
    
    def setup_method(self):
        """Initialize matcher before each test"""
        self.matcher = SkillMatcher()
    
    # Normalization Tests
    def test_normalize_skill(self):
        """Test skill normalization"""
        assert self.matcher.normalize_skill("  Python  ") == "python"
        assert self.matcher.normalize_skill("C++") == "c"
        assert self.matcher.normalize_skill("Node.js") == "nodejs"
    
    # Exact Match Tests
    def test_exact_match(self):
        """Test exact skill matching"""
        resume_skills = ["Python", "Java", "SQL"]
        jd_skills = ["Python", "SQL", "Docker"]
        
        matches = self.matcher.exact_match(resume_skills, jd_skills)
        assert "python" in matches
        assert "sql" in matches
        assert len(matches) == 2
    
    # Fuzzy Match Tests
    def test_fuzzy_match(self):
        """Test fuzzy matching with synonyms"""
        resume_skills = ["ML", "JavaScript"]
        jd_skills = ["Machine Learning", "JS"]
        
        matches = self.matcher.fuzzy_match(resume_skills, jd_skills)
        assert len(matches) >= 1  # Should match ML/Machine Learning
    
    # Partial Match Tests
    def test_partial_match(self):
        """Test partial substring matching"""
        resume_skills = ["Machine Learning"]
        jd_skills = ["Machine Learning Engineer"]
        
        matches = self.matcher.partial_match(resume_skills, jd_skills)
        assert len(matches) >= 1
    
    # Match Ratio Tests
    def test_skill_match_ratio_exact(self):
        """Test exact match ratio calculation"""
        resume_skills = ["Python", "Java", "SQL"]
        jd_skills = ["Python", "Java"]
        
        ratio = self.matcher.calculate_skill_match_ratio(
            resume_skills, jd_skills, match_type="exact"
        )
        assert ratio == 1.0  # Both JD skills matched
    
    def test_skill_match_ratio_all(self):
        """Test combined match ratio"""
        resume_skills = ["Python", "ML", "JavaScript"]
        jd_skills = ["Python", "Machine Learning", "JS"]
        
        ratio = self.matcher.calculate_skill_match_ratio(
            resume_skills, jd_skills, match_type="all"
        )
        assert ratio > 0.5  # Should match most
    
    # Weighted Score Tests
    def test_weighted_skill_score(self):
        """Test weighted skill scoring"""
        resume_skills = ["Python", "ML"]
        jd_skills = ["Python", "Machine Learning"]
        
        proficiencies = {"python": "expert", "ml": "basic"}
        tiers = {"python": "high_demand", "machine learning": "rare"}
        
        score = self.matcher.calculate_weighted_skill_score(
            resume_skills, jd_skills, proficiencies, tiers
        )
        assert 0 <= score <= 1
    
    # Feature Calculation Tests
    def test_calculate_skill_features(self):
        """Test comprehensive feature calculation"""
        resume_skills = ["Python", "Java", "SQL"]
        jd_skills = ["Python", "SQL", "Docker"]
        
        features = self.matcher.calculate_skill_features(
            resume_skills, jd_skills
        )
        
        assert "total_resume_skills" in features
        assert "total_jd_skills" in features
        assert "exact_matches_count" in features
        assert "overall_match_ratio" in features
        assert features["total_resume_skills"] == 3
        assert features["total_jd_skills"] == 3
    
    # Missing Skills Tests
    def test_get_missing_skills(self):
        """Test missing skills detection"""
        resume_skills = ["Python", "Java"]
        jd_skills = ["Python", "SQL", "Docker"]
        
        missing = self.matcher.get_missing_skills(resume_skills, jd_skills)
        assert "sql" in missing
        assert "docker" in missing
        assert len(missing) == 2
    
    # Gap Summary Tests
    def test_skill_gap_summary(self):
        """Test skill gap analysis"""
        resume_skills = ["Python", "Java", "SQL"]
        jd_skills = ["Python", "SQL"]
        
        summary = self.matcher.get_skill_gap_summary(
            resume_skills, jd_skills
        )
        
        assert "matched_skills" in summary
        assert "missing_skills" in summary
        assert "match_ratio" in summary
        assert "recommendation" in summary
        assert summary["match_ratio"] == 1.0  # All JD skills matched
    
    # Convenience Function Test
    def test_calculate_skill_match_function(self):
        """Test quick skill match function"""
        ratio = calculate_skill_match(["Python"], ["Python", "Java"])
        assert 0 <= ratio <= 1


# ============================================================================
# MODULE 3: EXPERIENCE FEATURES TESTS
# ============================================================================

class TestExperienceFeatures:
    """Test Experience Calculator"""
    
    def setup_method(self):
        """Initialize calculator before each test"""
        self.calc = ExperienceCalculator()
    
    # Duration Parsing Tests
    def test_parse_duration_years(self):
        """Test parsing years from duration string"""
        assert self.calc._parse_duration("3 years") == 3.0
        assert self.calc._parse_duration("2.5 years") == 2.5
    
    def test_parse_duration_months(self):
        """Test parsing months from duration string"""
        assert self.calc._parse_duration("6 months") == 0.5
        assert self.calc._parse_duration("18 months") == 1.5
    
    def test_parse_duration_combined(self):
        """Test parsing combined years and months"""
        years = self.calc._parse_duration("2 years 6 months")
        assert abs(years - 2.5) < 0.01
    
    # Date Parsing Tests
    def test_parse_date(self):
        """Test date parsing"""
        year, month = self.calc._parse_date("Jan 2020")
        assert year == 2020
        assert month == 1
        
        year, month = self.calc._parse_date("2020")
        assert year == 2020
    
    # Total Experience Tests
    def test_calculate_total_experience(self):
        """Test total experience calculation"""
        blocks = [
            {"duration": "3 years"},
            {"start_date": "Jan 2020", "end_date": "Jan 2023"}
        ]
        
        total = self.calc.calculate_total_experience(blocks)
        assert total >= 5.0  # At least 3 + 3 years
    
    # Required Experience Extraction Tests
    def test_extract_required_experience(self):
        """Test extracting required years from JD"""
        jd = "We need 5+ years of experience in Python development"
        
        required = self.calc.extract_required_experience(jd)
        assert required["min_years"] == 5.0
    
    def test_extract_required_experience_range(self):
        """Test extracting experience range"""
        jd = "Looking for 3-5 years of experience"
        
        required = self.calc.extract_required_experience(jd)
        assert required["min_years"] == 3.0
        assert required["max_years"] == 5.0
    
    # Gap Calculation Tests
    def test_calculate_experience_gap_positive(self):
        """Test positive experience gap (exceeds requirement)"""
        gap = self.calc.calculate_experience_gap(7.0, 5.0)
        assert gap == 2.0
    
    def test_calculate_experience_gap_negative(self):
        """Test negative experience gap (falls short)"""
        gap = self.calc.calculate_experience_gap(3.0, 5.0)
        assert gap == -2.0
    
    # Match Score Tests
    def test_experience_match_score_perfect(self):
        """Test perfect match score"""
        score = self.calc.calculate_experience_match_score(5.0, 5.0)
        assert score == 1.0
    
    def test_experience_match_score_in_range(self):
        """Test match within range"""
        score = self.calc.calculate_experience_match_score(6.0, 5.0, 8.0)
        assert score == 1.0
    
    def test_experience_match_score_underqualified(self):
        """Test underqualified score"""
        score = self.calc.calculate_experience_match_score(2.0, 5.0)
        assert score < 1.0
        assert score >= 0.0
    
    # Level Classification Tests
    def test_classify_experience_level(self):
        """Test experience level classification"""
        assert self.calc.classify_experience_level(1.5) == "entry"
        assert self.calc.classify_experience_level(5.0) == "mid"
        assert self.calc.classify_experience_level(8.0) == "senior"
        assert self.calc.classify_experience_level(12.0) == "lead"
    
    # Relevant Experience Tests
    def test_calculate_relevant_experience(self):
        """Test relevant experience calculation"""
        blocks = [
            {
                "duration": "3 years",
                "description": "Python development and ML"
            },
            {
                "duration": "2 years",
                "description": "Sales and marketing"
            }
        ]
        
        relevant = self.calc.calculate_relevant_experience(
            blocks, ["Python", "ML"]
        )
        assert relevant >= 3.0  # Only first block matches
    
    # Feature Calculation Tests
    def test_calculate_experience_features(self):
        """Test comprehensive feature calculation"""
        blocks = [{"duration": "5 years"}]
        jd = "5+ years of experience required"
        
        features = self.calc.calculate_experience_features(blocks, jd)
        
        assert "total_experience_years" in features
        assert "experience_gap" in features
        assert "experience_match_score" in features
        assert "meets_min_experience" in features
        assert features["total_experience_years"] == 5.0
        assert features["meets_min_experience"] == 1.0
    
    # Summary Tests
    def test_get_experience_summary(self):
        """Test experience summary generation"""
        blocks = [{"duration": "5 years"}]
        jd = "5 years required"
        
        summary = self.calc.get_experience_summary(blocks, jd)
        
        assert "total_years" in summary
        assert "required_years" in summary
        assert "match_score" in summary
        assert "recommendation" in summary
    
    # Convenience Function Test
    def test_calculate_experience_match_function(self):
        """Test quick experience match function"""
        blocks = [{"duration": "5 years"}]
        jd = "5 years required"
        
        score = calculate_experience_match(blocks, jd)
        assert 0 <= score <= 1


# ============================================================================
# MODULE 4: EDUCATION FEATURES TESTS
# ============================================================================

class TestEducationFeatures:
    """Test Education Calculator"""
    
    def setup_method(self):
        """Initialize calculator before each test"""
        self.calc = EducationCalculator()
    
    # Level Parsing Tests
    def test_parse_education_level_phd(self):
        """Test PhD level parsing"""
        level = self.calc.parse_education_level("PhD in Computer Science")
        assert level == 5
    
    def test_parse_education_level_masters(self):
        """Test Masters level parsing"""
        level = self.calc.parse_education_level("M.Tech in CS")
        assert level == 4
    
    def test_parse_education_level_bachelor(self):
        """Test Bachelor level parsing"""
        level = self.calc.parse_education_level("B.Tech in IT")
        assert level == 3
    
    # Degree Extraction Tests
    def test_extract_degree(self):
        """Test degree extraction"""
        assert self.calc.extract_degree("B.Tech in CS") in ["btech", "btech"]
        assert self.calc.extract_degree("Master of Science") in ["ms", "masters"]
    
    # Field Extraction Tests
    def test_extract_field_of_study_stem(self):
        """Test STEM field detection"""
        field = self.calc.extract_field_of_study("Computer Science")
        assert field == "stem"
        
        field = self.calc.extract_field_of_study("Engineering")
        assert field == "stem"
    
    def test_extract_field_of_study_business(self):
        """Test business field detection"""
        field = self.calc.extract_field_of_study("MBA in Marketing")
        assert field == "business"
    
    # Tier 1 College Tests
    def test_is_tier1_college(self):
        """Test tier 1 college detection"""
        assert self.calc.is_tier1_college("IIT Delhi") == True
        assert self.calc.is_tier1_college("MIT") == True
        assert self.calc.is_tier1_college("Stanford University") == True
        assert self.calc.is_tier1_college("Unknown College") == False
    
    # Required Education Extraction Tests
    def test_extract_required_education(self):
        """Test extracting required education from JD"""
        jd = "Bachelor's degree in Computer Science required"
        
        required = self.calc.extract_required_education(jd)
        assert required["min_level"] == 3
        assert required["required_field"] == "stem"
    
    def test_extract_required_education_masters_preferred(self):
        """Test extracting with preferred Masters"""
        jd = "Bachelor's required, Master's preferred"
        
        required = self.calc.extract_required_education(jd)
        assert required["min_level"] == 3
        assert required["preferred_level"] == 4
    
    # Match Score Tests
    def test_education_match_score_perfect(self):
        """Test perfect education match"""
        score = self.calc.calculate_education_match_score(3, 3)
        assert score == 1.0
    
    def test_education_match_score_exceeds(self):
        """Test exceeding minimum"""
        score = self.calc.calculate_education_match_score(4, 3)
        assert score == 1.0
    
    def test_education_match_score_below(self):
        """Test below minimum"""
        score = self.calc.calculate_education_match_score(2, 3)
        assert score < 1.0
    
    # Field Match Tests
    def test_field_match_score_exact(self):
        """Test exact field match"""
        score = self.calc.calculate_field_match_score("stem", "stem")
        assert score == 1.0
    
    def test_field_match_score_different(self):
        """Test different field"""
        score = self.calc.calculate_field_match_score("stem", "business")
        assert score < 1.0
    
    # Feature Calculation Tests
    def test_calculate_education_features(self):
        """Test comprehensive feature calculation"""
        education = [
            {
                "degree": "M.Tech in Computer Science",
                "college": "IIT Delhi",
                "year": "2020"
            }
        ]
        jd = "Bachelor's in CS required"
        
        features = self.calc.calculate_education_features(education, jd)
        
        assert "education_level" in features
        assert "tier1_college" in features
        assert "stem_degree" in features
        assert "education_match_score" in features
        assert features["education_level"] == 4.0
        assert features["tier1_college"] == 1.0
    
    # Summary Tests
    def test_get_education_summary(self):
        """Test education summary generation"""
        education = [{"degree": "B.Tech in CS", "college": "IIT Delhi"}]
        jd = "Bachelor's required"
        
        summary = self.calc.get_education_summary(education, jd)
        
        assert "highest_level" in summary
        assert "tier1_college" in summary
        assert "meets_requirement" in summary
        assert "recommendation" in summary
    
    # Convenience Function Test
    def test_calculate_education_match_function(self):
        """Test quick education match function"""
        education = [{"degree": "B.Tech in CS"}]
        jd = "Bachelor's required"
        
        score = calculate_education_match(education, jd)
        assert 0 <= score <= 1


# ============================================================================
# MODULE 5: FEATURE GENERATOR TESTS
# ============================================================================

class TestFeatureGenerator:
    """Test Feature Generator"""
    
    def setup_method(self):
        """Initialize generator before each test"""
        self.generator = FeatureGenerator()
    
    def get_sample_resume_data(self):
        """Get sample resume data for testing"""
        return {
            "text": "Senior ML Engineer with Python",
            "sections": {"skills": "Python, ML"},
            "skills": ["Python", "ML"],
            "skill_proficiencies": {"python": "expert"},
            "experience": [{"duration": "5 years"}],
            "education": [{"degree": "B.Tech in CS"}],
            "projects": ["Project 1"],
            "certifications": ["Cert 1"],
            "github": "github.com/user",
            "linkedin": "linkedin.com/user",
            "publications": []
        }
    
    def get_sample_jd_data(self):
        """Get sample JD data for testing"""
        return {
            "text": "ML Engineer with 5 years Python",
            "sections": {"required_skills": "Python, ML"},
            "skills": ["Python", "ML"],
            "skill_tiers": {"python": "high_demand"}
        }
    
    # Feature Generation Tests
    def test_generate_features(self):
        """Test complete feature generation"""
        resume_data = self.get_sample_resume_data()
        jd_data = self.get_sample_jd_data()
        
        features = self.generator.generate_features(resume_data, jd_data)
        
        # Check feature groups exist
        assert "overall_similarity" in features
        assert "skill_overall_ratio" in features
        assert "exp_total_years" in features
        assert "edu_level" in features
        assert "num_projects" in features
        
        # Check all values are numeric
        assert all(isinstance(v, (int, float)) for v in features.values())
    
    def test_generate_feature_vector(self):
        """Test feature vector generation"""
        resume_data = self.get_sample_resume_data()
        jd_data = self.get_sample_jd_data()
        
        vector = self.generator.generate_feature_vector(resume_data, jd_data)
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) > 30  # Should have 30+ features
        assert all(isinstance(v, (int, float)) for v in vector)
    
    def test_batch_generate_features(self):
        """Test batch feature generation"""
        resume_data_list = [self.get_sample_resume_data()] * 3
        jd_data_list = [self.get_sample_jd_data()] * 3
        
        all_features = self.generator.batch_generate_features(
            resume_data_list, jd_data_list
        )
        
        assert len(all_features) == 3
        assert all(isinstance(f, dict) for f in all_features)
    
    def test_batch_mismatched_lengths(self):
        """Test batch with mismatched lengths"""
        with pytest.raises(ValueError):
            self.generator.batch_generate_features([{}], [{}, {}])
    
    def test_get_feature_names(self):
        """Test getting feature names"""
        feature_names = self.generator.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 30
        assert all(isinstance(name, str) for name in feature_names)
    
    def test_get_feature_summary(self):
        """Test feature summary generation"""
        resume_data = self.get_sample_resume_data()
        jd_data = self.get_sample_jd_data()
        
        features = self.generator.generate_features(resume_data, jd_data)
        summary = self.generator.get_feature_summary(features)
        
        assert "total_features" in summary
        assert "semantic_similarity" in summary
        assert "skill_match" in summary
        assert "experience" in summary
        assert "education" in summary
        assert "overall_recommendation" in summary
    
    # Convenience Function Test
    def test_generate_features_function(self):
        """Test quick feature generation function"""
        resume_data = self.get_sample_resume_data()
        jd_data = self.get_sample_jd_data()
        
        features = generate_features(resume_data, jd_data)
        assert isinstance(features, dict)
        assert len(features) > 30


# ============================================================================
# MODULE 6: FEATURE STORE TESTS
# ============================================================================

class TestFeatureStore:
    """Test Feature Store (PostgreSQL)"""
    
    @pytest.fixture
    def mock_connection(self):
        """Mock PostgreSQL connection"""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connect.return_value = mock_conn
            yield mock_conn, mock_cursor
    
    def test_feature_store_initialization(self):
        """Test feature store initialization"""
        store = FeatureStore(
            host="localhost",
            port=5432,
            database="test_db",
            user="test_user",
            password="test_pass"
        )
        
        assert store.host == "localhost"
        assert store.port == 5432
        assert store.database == "test_db"
    
    def test_feature_store_context_manager(self, mock_connection):
        """Test context manager usage"""
        mock_conn, _ = mock_connection
        
        with patch('psycopg2.connect', return_value=mock_conn):
            with FeatureStore() as store:
                assert store.conn is not None
    
    def test_store_candidate(self, mock_connection):
        """Test storing candidate"""
        mock_conn, mock_cursor = mock_connection
        mock_cursor.fetchone.return_value = [123]
        
        with patch('psycopg2.connect', return_value=mock_conn):
            store = FeatureStore()
            store.connect()
            
            candidate_id = store.store_candidate(
                resume_hash="abc123",
                resume_text="Test resume",
                candidate_name="John Doe",
                email="john@example.com"
            )
            
            assert candidate_id == 123
    
    def test_store_job_description(self, mock_connection):
        """Test storing job description"""
        mock_conn, mock_cursor = mock_connection
        mock_cursor.fetchone.return_value = [456]
        
        with patch('psycopg2.connect', return_value=mock_conn):
            store = FeatureStore()
            store.connect()
            
            jd_id = store.store_job_description(
                jd_hash="def456",
                jd_text="Test JD",
                job_title="ML Engineer"
            )
            
            assert jd_id == 456
    
    def test_store_features(self, mock_connection):
        """Test storing features"""
        mock_conn, mock_cursor = mock_connection
        
        with patch('psycopg2.connect', return_value=mock_conn):
            store = FeatureStore()
            store.connect()
            
            features = {"skill_match": 0.85}
            feature_vector = np.array([0.85, 0.90])
            
            # Should not raise exception
            store.store_features(123, 456, features, feature_vector)
    
    def test_get_features(self, mock_connection):
        """Test retrieving features"""
        mock_conn, mock_cursor = mock_connection
        mock_cursor.fetchone.return_value = (
            '{"skill_match": 0.85}',
            [0.85, 0.90]
        )
        
        with patch('psycopg2.connect', return_value=mock_conn):
            store = FeatureStore()
            store.connect()
            
            result = store.get_features(123, 456)
            
            assert result is not None
            features_dict, feature_vector = result
            assert features_dict["skill_match"] == 0.85
            assert isinstance(feature_vector, np.ndarray)
    
    def test_get_top_candidates(self, mock_connection):
        """Test getting top candidates"""
        mock_conn, mock_cursor = mock_connection
        mock_cursor.fetchall.return_value = []
        
        with patch('psycopg2.connect', return_value=mock_conn):
            store = FeatureStore()
            store.connect()
            
            candidates = store.get_top_candidates(456, limit=5)
            
            assert isinstance(candidates, list)
    
    def test_get_statistics(self, mock_connection):
        """Test getting statistics"""
        mock_conn, mock_cursor = mock_connection
        mock_cursor.fetchone.return_value = {
            "total_candidates": 100,
            "total_jobs": 20,
            "avg_score": 0.75
        }
        
        with patch('psycopg2.connect', return_value=mock_conn):
            store = FeatureStore()
            store.connect()
            
            stats = store.get_statistics()
            
            assert isinstance(stats, dict)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestPhase5Integration:
    """Integration tests for Phase 5"""
    
    def test_end_to_end_feature_generation(self):
        """Test complete pipeline from resume to features"""
        # Sample data
        resume_data = {
            "text": "Senior ML Engineer with 5 years Python experience",
            "sections": {"skills": "Python, ML, TensorFlow"},
            "skills": ["Python", "ML", "TensorFlow"],
            "skill_proficiencies": {"python": "expert", "ml": "advanced"},
            "experience": [{"duration": "5 years"}],
            "education": [{"degree": "M.Tech in CS", "college": "IIT Delhi"}],
            "projects": ["NLP Chatbot", "Image Classifier"],
            "certifications": ["TensorFlow Cert"],
            "github": "github.com/user",
            "linkedin": "",
            "publications": []
        }
        
        jd_data = {
            "text": "Looking for ML Engineer with 5+ years Python",
            "sections": {"required_skills": "Python, ML"},
            "skills": ["Python", "ML", "Deep Learning"],
            "skill_tiers": {"python": "high_demand", "ml": "rare"}
        }
        
        # Generate features
        generator = FeatureGenerator()
        features = generator.generate_features(resume_data, jd_data)
        
        # Validate output
        assert len(features) >= 40  # Should have 40+ features
        assert all(isinstance(v, (int, float)) for v in features.values())
        
        # Check key features
        assert features["overall_similarity"] > 0.5
        assert features["skill_overall_ratio"] > 0.5
        assert features["exp_total_years"] == 5.0
        assert features["edu_level"] == 4.0
        assert features["num_projects"] == 2
    
    def test_feature_consistency(self):
        """Test that features are consistent across multiple runs"""
        resume_data = {
            "text": "Python developer",
            "sections": {},
            "skills": ["Python"],
            "skill_proficiencies": {},
            "experience": [{"duration": "3 years"}],
            "education": [{"degree": "B.Tech"}],
            "projects": [],
            "certifications": [],
            "github": "",
            "linkedin": "",
            "publications": []
        }
        
        jd_data = {
            "text": "Python developer needed",
            "sections": {},
            "skills": ["Python"],
            "skill_tiers": {}
        }
        
        generator = FeatureGenerator()
        
        # Generate twice
        features1 = generator.generate_features(resume_data, jd_data)
        features2 = generator.generate_features(resume_data, jd_data)
        
        # Should be identical
        assert features1.keys() == features2.keys()
        for key in features1.keys():
            assert abs(features1[key] - features2[key]) < 0.001


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PHASE 5: FEATURE ENGINEERING - COMPREHENSIVE TEST SUITE")
    print("="*80 + "\n")
    
    # Run with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes",
        "-x"  # Stop on first failure
    ])