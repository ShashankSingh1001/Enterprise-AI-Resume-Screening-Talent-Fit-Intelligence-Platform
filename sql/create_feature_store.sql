-- Feature Store Database Schema
-- PostgreSQL 12+
-- Database: resume_screening

-- Create database (run as postgres user)
-- CREATE DATABASE resume_screening;
-- \c resume_screening;

-- ========================================
-- CANDIDATES TABLE
-- ========================================
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id SERIAL PRIMARY KEY,
    resume_hash VARCHAR(64) UNIQUE NOT NULL,
    resume_text TEXT,
    candidate_name VARCHAR(255),
    email VARCHAR(255),
    phone VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE candidates IS 'Stores candidate information and resumes';
COMMENT ON COLUMN candidates.resume_hash IS 'MD5/SHA256 hash of resume for deduplication';

-- ========================================
-- JOB DESCRIPTIONS TABLE
-- ========================================
CREATE TABLE IF NOT EXISTS job_descriptions (
    jd_id SERIAL PRIMARY KEY,
    jd_hash VARCHAR(64) UNIQUE NOT NULL,
    jd_text TEXT,
    job_title VARCHAR(255),
    company VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE job_descriptions IS 'Stores job descriptions';
COMMENT ON COLUMN job_descriptions.jd_hash IS 'Hash of JD for deduplication';

-- ========================================
-- FEATURE VECTORS TABLE
-- ========================================
CREATE TABLE IF NOT EXISTS feature_vectors (
    feature_id SERIAL PRIMARY KEY,
    candidate_id INTEGER REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    jd_id INTEGER REFERENCES job_descriptions(jd_id) ON DELETE CASCADE,
    features JSONB NOT NULL,
    feature_vector FLOAT[] NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(candidate_id, jd_id)
);

COMMENT ON TABLE feature_vectors IS 'Stores computed feature vectors (50+ features)';
COMMENT ON COLUMN feature_vectors.features IS 'JSON with named features';
COMMENT ON COLUMN feature_vectors.feature_vector IS 'Numpy array as PostgreSQL array';

-- ========================================
-- SIMILARITY SCORES TABLE
-- ========================================
CREATE TABLE IF NOT EXISTS similarity_scores (
    score_id SERIAL PRIMARY KEY,
    candidate_id INTEGER REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    jd_id INTEGER REFERENCES job_descriptions(jd_id) ON DELETE CASCADE,
    overall_similarity FLOAT,
    weighted_similarity FLOAT,
    skill_match_score FLOAT,
    experience_match_score FLOAT,
    education_match_score FLOAT,
    final_score FLOAT,
    recommendation VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(candidate_id, jd_id)
);

COMMENT ON TABLE similarity_scores IS 'Denormalized scores for quick ranking';
COMMENT ON COLUMN similarity_scores.final_score IS 'Weighted overall score for ranking';

-- ========================================
-- SCREENING RESULTS TABLE
-- ========================================
CREATE TABLE IF NOT EXISTS screening_results (
    result_id SERIAL PRIMARY KEY,
    candidate_id INTEGER REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    jd_id INTEGER REFERENCES job_descriptions(jd_id) ON DELETE CASCADE,
    status VARCHAR(50) CHECK (status IN ('shortlisted', 'rejected', 'pending', 'interviewed')),
    final_score FLOAT,
    recommendation TEXT,
    screened_by VARCHAR(100),
    screened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

COMMENT ON TABLE screening_results IS 'Stores HR screening decisions';
COMMENT ON COLUMN screening_results.status IS 'Screening outcome';

-- ========================================
-- INDEXES FOR PERFORMANCE
-- ========================================

-- Candidates indexes
CREATE INDEX IF NOT EXISTS idx_candidates_hash ON candidates(resume_hash);
CREATE INDEX IF NOT EXISTS idx_candidates_email ON candidates(email);
CREATE INDEX IF NOT EXISTS idx_candidates_name ON candidates(candidate_name);

-- Job descriptions indexes
CREATE INDEX IF NOT EXISTS idx_jd_hash ON job_descriptions(jd_hash);
CREATE INDEX IF NOT EXISTS idx_jd_title ON job_descriptions(job_title);

-- Feature vectors indexes
CREATE INDEX IF NOT EXISTS idx_features_candidate ON feature_vectors(candidate_id);
CREATE INDEX IF NOT EXISTS idx_features_jd ON feature_vectors(jd_id);

-- Similarity scores indexes (critical for ranking)
CREATE INDEX IF NOT EXISTS idx_similarity_score ON similarity_scores(final_score DESC);
CREATE INDEX IF NOT EXISTS idx_similarity_jd ON similarity_scores(jd_id, final_score DESC);
CREATE INDEX IF NOT EXISTS idx_similarity_skill ON similarity_scores(skill_match_score DESC);

-- Screening results indexes
CREATE INDEX IF NOT EXISTS idx_screening_status ON screening_results(status);
CREATE INDEX IF NOT EXISTS idx_screening_jd ON screening_results(jd_id, status);

-- ========================================
-- VIEWS FOR COMMON QUERIES
-- ========================================

-- Top candidates view
CREATE OR REPLACE VIEW v_top_candidates AS
SELECT 
    c.candidate_id,
    c.candidate_name,
    c.email,
    c.phone,
    j.job_title,
    j.company,
    s.final_score,
    s.skill_match_score,
    s.experience_match_score,
    s.education_match_score,
    s.recommendation,
    s.created_at as scored_at
FROM similarity_scores s
JOIN candidates c ON s.candidate_id = c.candidate_id
JOIN job_descriptions j ON s.jd_id = j.jd_id
ORDER BY s.final_score DESC;

COMMENT ON VIEW v_top_candidates IS 'Quick view of all scored candidates';

-- Shortlisted candidates view
CREATE OR REPLACE VIEW v_shortlisted_candidates AS
SELECT 
    c.candidate_id,
    c.candidate_name,
    c.email,
    j.job_title,
    j.company,
    sr.final_score,
    sr.screened_by,
    sr.screened_at,
    sr.notes
FROM screening_results sr
JOIN candidates c ON sr.candidate_id = c.candidate_id
JOIN job_descriptions j ON sr.jd_id = j.jd_id
WHERE sr.status = 'shortlisted'
ORDER BY sr.screened_at DESC;

COMMENT ON VIEW v_shortlisted_candidates IS 'Shortlisted candidates only';

-- ========================================
-- FUNCTIONS FOR CONVENIENCE
-- ========================================

-- Function to get candidate rank for a job
CREATE OR REPLACE FUNCTION get_candidate_rank(p_candidate_id INTEGER, p_jd_id INTEGER)
RETURNS INTEGER AS $$
DECLARE
    v_rank INTEGER;
BEGIN
    SELECT rank INTO v_rank
    FROM (
        SELECT 
            candidate_id,
            RANK() OVER (ORDER BY final_score DESC) as rank
        FROM similarity_scores
        WHERE jd_id = p_jd_id
    ) ranked
    WHERE candidate_id = p_candidate_id;
    
    RETURN v_rank;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_candidate_rank IS 'Get candidate ranking for a specific job';

-- ========================================
-- SAMPLE QUERIES
-- ========================================

-- Get top 10 candidates for a job
-- SELECT * FROM v_top_candidates WHERE job_title = 'ML Engineer' LIMIT 10;

-- Get candidate rank
-- SELECT get_candidate_rank(123, 456);

-- Get screening statistics
-- SELECT 
--     status,
--     COUNT(*) as count,
--     AVG(final_score) as avg_score
-- FROM screening_results
-- GROUP BY status;

-- Find candidates with high skill match but low overall score (potential to improve)
-- SELECT * FROM similarity_scores
-- WHERE skill_match_score > 0.8 AND final_score < 0.6;

-- ========================================
-- GRANTS (adjust user as needed)
-- ========================================

-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO resume_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO resume_user;