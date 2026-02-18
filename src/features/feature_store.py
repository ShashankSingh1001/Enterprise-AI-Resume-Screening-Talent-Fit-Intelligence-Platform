"""
Feature Store
PostgreSQL-based storage for feature vectors and screening results
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
import numpy as np

from .config import config

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    PostgreSQL-based feature store for resume screening features
    Stores feature vectors, similarity scores, and screening results
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None
    ):
        """
        Initialize feature store connection
        
        Args:
            host: PostgreSQL host (default from config)
            port: PostgreSQL port (default from config)
            database: Database name (default from config)
            user: Database user (default from config)
            password: Database password (default from config)
        """
        self.host = host or config.DB_HOST
        self.port = port or config.DB_PORT
        self.database = database or config.DB_NAME
        self.user = user or config.DB_USER
        self.password = password or config.DB_PASSWORD
        
        self.conn = None
        logger.info("FeatureStore initialized")
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from database")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
    
    def create_tables(self):
        """Create feature store tables if they don't exist"""
        create_tables_sql = """
        -- Candidates table
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
        
        -- Job descriptions table
        CREATE TABLE IF NOT EXISTS job_descriptions (
            jd_id SERIAL PRIMARY KEY,
            jd_hash VARCHAR(64) UNIQUE NOT NULL,
            jd_text TEXT,
            job_title VARCHAR(255),
            company VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Feature vectors table
        CREATE TABLE IF NOT EXISTS feature_vectors (
            feature_id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(candidate_id),
            jd_id INTEGER REFERENCES job_descriptions(jd_id),
            features JSONB NOT NULL,
            feature_vector FLOAT[] NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(candidate_id, jd_id)
        );
        
        -- Similarity scores table (for quick lookups)
        CREATE TABLE IF NOT EXISTS similarity_scores (
            score_id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(candidate_id),
            jd_id INTEGER REFERENCES job_descriptions(jd_id),
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
        
        -- Screening results table
        CREATE TABLE IF NOT EXISTS screening_results (
            result_id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(candidate_id),
            jd_id INTEGER REFERENCES job_descriptions(jd_id),
            status VARCHAR(50),  -- shortlisted, rejected, pending
            final_score FLOAT,
            recommendation TEXT,
            screened_by VARCHAR(100),
            screened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_candidates_hash ON candidates(resume_hash);
        CREATE INDEX IF NOT EXISTS idx_jd_hash ON job_descriptions(jd_hash);
        CREATE INDEX IF NOT EXISTS idx_features_candidate ON feature_vectors(candidate_id);
        CREATE INDEX IF NOT EXISTS idx_features_jd ON feature_vectors(jd_id);
        CREATE INDEX IF NOT EXISTS idx_similarity_score ON similarity_scores(final_score DESC);
        CREATE INDEX IF NOT EXISTS idx_screening_status ON screening_results(status);
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(create_tables_sql)
                self.conn.commit()
            logger.info("Feature store tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            self.conn.rollback()
            raise
    
    def store_candidate(
        self,
        resume_hash: str,
        resume_text: str,
        candidate_name: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None
    ) -> int:
        """
        Store or update candidate information
        
        Args:
            resume_hash: Unique hash of resume
            resume_text: Full resume text
            candidate_name: Candidate name
            email: Email address
            phone: Phone number
            
        Returns:
            candidate_id
        """
        sql = """
        INSERT INTO candidates (resume_hash, resume_text, candidate_name, email, phone)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (resume_hash) 
        DO UPDATE SET 
            resume_text = EXCLUDED.resume_text,
            candidate_name = EXCLUDED.candidate_name,
            email = EXCLUDED.email,
            phone = EXCLUDED.phone,
            updated_at = CURRENT_TIMESTAMP
        RETURNING candidate_id;
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (resume_hash, resume_text, candidate_name, email, phone))
                candidate_id = cur.fetchone()[0]
                self.conn.commit()
            logger.debug(f"Stored candidate with ID: {candidate_id}")
            return candidate_id
        except Exception as e:
            logger.error(f"Failed to store candidate: {e}")
            self.conn.rollback()
            raise
    
    def store_job_description(
        self,
        jd_hash: str,
        jd_text: str,
        job_title: Optional[str] = None,
        company: Optional[str] = None
    ) -> int:
        """
        Store or update job description
        
        Args:
            jd_hash: Unique hash of JD
            jd_text: Full JD text
            job_title: Job title
            company: Company name
            
        Returns:
            jd_id
        """
        sql = """
        INSERT INTO job_descriptions (jd_hash, jd_text, job_title, company)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (jd_hash)
        DO UPDATE SET
            jd_text = EXCLUDED.jd_text,
            job_title = EXCLUDED.job_title,
            company = EXCLUDED.company,
            updated_at = CURRENT_TIMESTAMP
        RETURNING jd_id;
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (jd_hash, jd_text, job_title, company))
                jd_id = cur.fetchone()[0]
                self.conn.commit()
            logger.debug(f"Stored JD with ID: {jd_id}")
            return jd_id
        except Exception as e:
            logger.error(f"Failed to store JD: {e}")
            self.conn.rollback()
            raise
    
    def store_features(
        self,
        candidate_id: int,
        jd_id: int,
        features: Dict[str, float],
        feature_vector: np.ndarray
    ):
        """
        Store feature vector
        
        Args:
            candidate_id: Candidate ID
            jd_id: Job description ID
            features: Feature dict
            feature_vector: Numpy feature vector
        """
        sql = """
        INSERT INTO feature_vectors (candidate_id, jd_id, features, feature_vector)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (candidate_id, jd_id)
        DO UPDATE SET
            features = EXCLUDED.features,
            feature_vector = EXCLUDED.feature_vector,
            created_at = CURRENT_TIMESTAMP;
        """
        
        try:
            # Convert numpy array to list for PostgreSQL
            feature_list = feature_vector.tolist()
            features_json = json.dumps(features)
            
            with self.conn.cursor() as cur:
                cur.execute(sql, (candidate_id, jd_id, features_json, feature_list))
                self.conn.commit()
            logger.debug(f"Stored features for candidate {candidate_id}, JD {jd_id}")
        except Exception as e:
            logger.error(f"Failed to store features: {e}")
            self.conn.rollback()
            raise
    
    def store_similarity_scores(
        self,
        candidate_id: int,
        jd_id: int,
        scores: Dict[str, float],
        recommendation: str
    ):
        """
        Store similarity scores for quick retrieval
        
        Args:
            candidate_id: Candidate ID
            jd_id: Job description ID
            scores: Dict with similarity scores
            recommendation: Recommendation text
        """
        sql = """
        INSERT INTO similarity_scores (
            candidate_id, jd_id, 
            overall_similarity, weighted_similarity,
            skill_match_score, experience_match_score, education_match_score,
            final_score, recommendation
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (candidate_id, jd_id)
        DO UPDATE SET
            overall_similarity = EXCLUDED.overall_similarity,
            weighted_similarity = EXCLUDED.weighted_similarity,
            skill_match_score = EXCLUDED.skill_match_score,
            experience_match_score = EXCLUDED.experience_match_score,
            education_match_score = EXCLUDED.education_match_score,
            final_score = EXCLUDED.final_score,
            recommendation = EXCLUDED.recommendation,
            created_at = CURRENT_TIMESTAMP;
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (
                    candidate_id, jd_id,
                    scores.get("overall_similarity", 0.0),
                    scores.get("weighted_similarity", 0.0),
                    scores.get("skill_weighted_score", 0.0),
                    scores.get("exp_match_score", 0.0),
                    scores.get("edu_overall_score", 0.0),
                    scores.get("final_score", 0.0),
                    recommendation
                ))
                self.conn.commit()
            logger.debug(f"Stored similarity scores for candidate {candidate_id}, JD {jd_id}")
        except Exception as e:
            logger.error(f"Failed to store similarity scores: {e}")
            self.conn.rollback()
            raise
    
    def get_features(
        self,
        candidate_id: int,
        jd_id: int
    ) -> Optional[Tuple[Dict[str, float], np.ndarray]]:
        """
        Retrieve features for a candidate-JD pair
        
        Args:
            candidate_id: Candidate ID
            jd_id: Job description ID
            
        Returns:
            Tuple of (features_dict, feature_vector) or None if not found
        """
        sql = """
        SELECT features, feature_vector
        FROM feature_vectors
        WHERE candidate_id = %s AND jd_id = %s;
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (candidate_id, jd_id))
                result = cur.fetchone()
                
                if result:
                    features_dict = json.loads(result[0])
                    feature_vector = np.array(result[1])
                    return features_dict, feature_vector
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve features: {e}")
            raise
    
    def get_top_candidates(
        self,
        jd_id: int,
        limit: int = 10,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Get top candidates for a job description
        
        Args:
            jd_id: Job description ID
            limit: Maximum number of candidates to return
            min_score: Minimum score threshold
            
        Returns:
            List of candidate dicts with scores
        """
        sql = """
        SELECT 
            c.candidate_id,
            c.candidate_name,
            c.email,
            c.phone,
            s.final_score,
            s.skill_match_score,
            s.experience_match_score,
            s.education_match_score,
            s.recommendation
        FROM similarity_scores s
        JOIN candidates c ON s.candidate_id = c.candidate_id
        WHERE s.jd_id = %s AND s.final_score >= %s
        ORDER BY s.final_score DESC
        LIMIT %s;
        """
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (jd_id, min_score, limit))
                results = cur.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to get top candidates: {e}")
            raise
    
    def batch_store_features(
        self,
        candidate_ids: List[int],
        jd_ids: List[int],
        features_list: List[Dict[str, float]],
        feature_vectors: List[np.ndarray]
    ):
        """
        Batch store multiple feature vectors
        
        Args:
            candidate_ids: List of candidate IDs
            jd_ids: List of JD IDs
            features_list: List of feature dicts
            feature_vectors: List of feature vectors
        """
        if not (len(candidate_ids) == len(jd_ids) == len(features_list) == len(feature_vectors)):
            raise ValueError("All input lists must have the same length")
        
        sql = """
        INSERT INTO feature_vectors (candidate_id, jd_id, features, feature_vector)
        VALUES %s
        ON CONFLICT (candidate_id, jd_id)
        DO UPDATE SET
            features = EXCLUDED.features,
            feature_vector = EXCLUDED.feature_vector,
            created_at = CURRENT_TIMESTAMP;
        """
        
        try:
            # Prepare data
            values = [
                (
                    candidate_ids[i],
                    jd_ids[i],
                    json.dumps(features_list[i]),
                    feature_vectors[i].tolist()
                )
                for i in range(len(candidate_ids))
            ]
            
            with self.conn.cursor() as cur:
                execute_values(cur, sql, values)
                self.conn.commit()
            logger.info(f"Batch stored {len(values)} feature vectors")
        except Exception as e:
            logger.error(f"Failed to batch store features: {e}")
            self.conn.rollback()
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get feature store statistics
        
        Returns:
            Dict with statistics
        """
        sql = """
        SELECT 
            (SELECT COUNT(*) FROM candidates) as total_candidates,
            (SELECT COUNT(*) FROM job_descriptions) as total_jobs,
            (SELECT COUNT(*) FROM feature_vectors) as total_features,
            (SELECT COUNT(*) FROM screening_results WHERE status = 'shortlisted') as shortlisted,
            (SELECT COUNT(*) FROM screening_results WHERE status = 'rejected') as rejected,
            (SELECT AVG(final_score) FROM similarity_scores) as avg_score;
        """
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql)
                stats = dict(cur.fetchone())
                return stats
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            raise


# Convenience functions
def create_feature_store(
    host: str = None,
    port: int = None,
    database: str = None,
    user: str = None,
    password: str = None
) -> FeatureStore:
    """
    Create and initialize feature store
    
    Args:
        host, port, database, user, password: PostgreSQL connection params
        
    Returns:
        FeatureStore instance
    """
    store = FeatureStore(host, port, database, user, password)
    store.connect()
    store.create_tables()
    return store


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example: Create and use feature store
    print("\n=== Feature Store Example ===")
    print("\nNote: This example requires PostgreSQL to be running")
    print("Connection: localhost:5432/resume_screening")
    print("\nTo run this example:")
    print("1. Install PostgreSQL")
    print("2. Create database: CREATE DATABASE resume_screening;")
    print("3. Update config.py with your credentials")
    print("\nExample code:")
    
    example_code = """
    # Initialize feature store
    with FeatureStore() as store:
        # Create tables
        store.create_tables()
        
        # Store candidate
        candidate_id = store.store_candidate(
            resume_hash="abc123",
            resume_text="Senior ML Engineer...",
            candidate_name="John Doe",
            email="john@example.com"
        )
        
        # Store JD
        jd_id = store.store_job_description(
            jd_hash="def456",
            jd_text="Looking for ML Engineer...",
            job_title="ML Engineer",
            company="Tech Corp"
        )
        
        # Store features
        features = {"skill_match": 0.85, "exp_match": 0.90}
        feature_vector = np.array([0.85, 0.90, 0.78])
        
        store.store_features(candidate_id, jd_id, features, feature_vector)
        
        # Get top candidates
        top_candidates = store.get_top_candidates(jd_id, limit=5)
        print(f"Top candidates: {top_candidates}")
        
        # Get statistics
        stats = store.get_statistics()
        print(f"Statistics: {stats}")
    """
    
    print(example_code)