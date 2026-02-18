"""
Semantic Similarity Calculator using Sentence-BERT
Calculates similarity between resume and job description using sentence embeddings
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from functools import lru_cache

# Lazy import to avoid loading model unnecessarily
_sentence_transformer = None

logger = logging.getLogger(__name__)


class SBERTSingleton:
    """
    Singleton pattern for SentenceTransformer model
    Ensures model is loaded only once across the application
    """
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self):
        """Load model lazily on first use"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading SBERT model: all-MiniLM-L6-v2")
                # Using lightweight model for speed (22MB, 384 dimensions)
                # Alternative: paraphrase-MiniLM-L3-v2 (even faster, 61MB, 384 dimensions)
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("SBERT model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load SBERT model: {e}")
                raise
        return self._model


def get_sbert_model():
    """Get singleton SBERT model instance"""
    return SBERTSingleton().get_model()


class SemanticSimilarityCalculator:
    """
    Calculates semantic similarity between resume and job description
    Uses SBERT embeddings and cosine similarity
    """
    
    def __init__(self):
        """Initialize calculator with singleton SBERT model"""
        self.model = get_sbert_model()
        logger.info("SemanticSimilarityCalculator initialized")
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        if vec1.size == 0 or vec2.size == 0:
            return 0.0
        
        # Cosine similarity: (A · B) / (||A|| × ||B||)
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product == 0:
            return 0.0
        
        similarity = dot_product / norm_product
        # Clip to [0, 1] range (cosine can be [-1, 1])
        return float(max(0.0, min(1.0, similarity)))
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text into embedding vector
        
        Args:
            text: Input text to encode
            
        Returns:
            Embedding vector (384 dimensions)
        """
        if not text or not text.strip():
            return np.zeros(384)  # Return zero vector for empty text
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            return np.zeros(384)
    
    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode multiple texts in batch (faster than one-by-one)
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter empty texts
        valid_texts = [t if t and t.strip() else " " for t in texts]
        
        try:
            embeddings = self.model.encode(valid_texts, convert_to_numpy=True, batch_size=32)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode batch: {e}")
            return [np.zeros(384) for _ in texts]
    
    def calculate_similarity(
        self, 
        resume_text: str, 
        jd_text: str
    ) -> float:
        """
        Calculate overall similarity between resume and JD
        
        Args:
            resume_text: Full resume text
            jd_text: Full job description text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Encode both texts
        resume_embedding = self.encode_text(resume_text)
        jd_embedding = self.encode_text(jd_text)
        
        # Calculate cosine similarity
        similarity = self.cosine_similarity(resume_embedding, jd_embedding)
        
        logger.debug(f"Overall similarity: {similarity:.3f}")
        return similarity
    
    def calculate_section_similarity(
        self,
        resume_sections: Dict[str, str],
        jd_sections: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate similarity for specific sections (skills, experience, etc.)
        
        Args:
            resume_sections: Dict with section names as keys, text as values
                Example: {"skills": "Python, ML, NLP", "experience": "..."}
            jd_sections: Dict with JD section names as keys, text as values
                Example: {"required_skills": "Python, ML", "responsibilities": "..."}
                
        Returns:
            Dict with section-wise similarity scores
        """
        similarities = {}
        
        # Common section mappings
        section_pairs = [
            ("skills", "required_skills"),
            ("skills", "preferred_skills"),
            ("experience", "responsibilities"),
            ("experience", "requirements"),
            ("education", "education"),
            ("projects", "responsibilities")
        ]
        
        for resume_key, jd_key in section_pairs:
            if resume_key in resume_sections and jd_key in jd_sections:
                resume_text = resume_sections[resume_key]
                jd_text = jd_sections[jd_key]
                
                if resume_text and jd_text:
                    resume_emb = self.encode_text(resume_text)
                    jd_emb = self.encode_text(jd_text)
                    similarity = self.cosine_similarity(resume_emb, jd_emb)
                    
                    pair_name = f"{resume_key}_vs_{jd_key}"
                    similarities[pair_name] = similarity
                    logger.debug(f"{pair_name}: {similarity:.3f}")
        
        return similarities
    
    def calculate_weighted_similarity(
        self,
        resume_text: str,
        jd_text: str,
        resume_sections: Optional[Dict[str, str]] = None,
        jd_sections: Optional[Dict[str, str]] = None,
        section_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate weighted similarity combining overall and section-wise scores
        
        Args:
            resume_text: Full resume text
            jd_text: Full job description text
            resume_sections: Optional resume sections
            jd_sections: Optional JD sections
            section_weights: Optional weights for each section
                Default: {"skills": 0.4, "experience": 0.3, "education": 0.15, "overall": 0.15}
                
        Returns:
            Dict with individual and weighted similarity scores
        """
        # Default weights (skills matter most)
        if section_weights is None:
            section_weights = {
                "skills": 0.4,
                "experience": 0.3,
                "education": 0.15,
                "overall": 0.15
            }
        
        results = {}
        
        # Overall similarity
        overall_sim = self.calculate_similarity(resume_text, jd_text)
        results["overall_similarity"] = overall_sim
        
        # Section-wise similarities
        section_sims = {}
        if resume_sections and jd_sections:
            section_sims = self.calculate_section_similarity(resume_sections, jd_sections)
            results.update(section_sims)
        
        # Calculate weighted score
        weighted_score = 0.0
        total_weight = 0.0
        
        # Add overall similarity contribution
        if "overall" in section_weights:
            weighted_score += overall_sim * section_weights["overall"]
            total_weight += section_weights["overall"]
        
        # Add section similarity contributions
        for section, weight in section_weights.items():
            if section == "overall":
                continue
            
            # Find matching section similarity
            matching_keys = [k for k in section_sims.keys() if section in k]
            if matching_keys:
                # Average if multiple matches
                avg_sim = np.mean([section_sims[k] for k in matching_keys])
                weighted_score += avg_sim * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            results["weighted_similarity"] = weighted_score / total_weight
        else:
            results["weighted_similarity"] = overall_sim
        
        logger.info(f"Weighted similarity: {results['weighted_similarity']:.3f}")
        return results
    
    def batch_calculate_similarity(
        self,
        resume_texts: List[str],
        jd_texts: List[str]
    ) -> List[float]:
        """
        Calculate similarities for multiple resume-JD pairs in batch
        More efficient than calling calculate_similarity repeatedly
        
        Args:
            resume_texts: List of resume texts
            jd_texts: List of JD texts (must be same length as resume_texts)
            
        Returns:
            List of similarity scores
        """
        if len(resume_texts) != len(jd_texts):
            raise ValueError("resume_texts and jd_texts must have same length")
        
        if not resume_texts:
            return []
        
        logger.info(f"Batch calculating similarity for {len(resume_texts)} pairs")
        
        # Encode all texts in batch
        resume_embeddings = self.encode_batch(resume_texts)
        jd_embeddings = self.encode_batch(jd_texts)
        
        # Calculate similarities
        similarities = [
            self.cosine_similarity(resume_emb, jd_emb)
            for resume_emb, jd_emb in zip(resume_embeddings, jd_embeddings)
        ]
        
        logger.info(f"Batch similarity calculation complete. Avg: {np.mean(similarities):.3f}")
        return similarities


# Convenience function for quick similarity calculation
def calculate_similarity(resume_text: str, jd_text: str) -> float:
    """
    Quick utility function to calculate similarity between resume and JD
    
    Args:
        resume_text: Resume text
        jd_text: Job description text
        
    Returns:
        Similarity score between 0 and 1
    """
    calculator = SemanticSimilarityCalculator()
    return calculator.calculate_similarity(resume_text, jd_text)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example 1: Basic similarity
    resume = """
    Senior Software Engineer with 5+ years of experience in Python, Machine Learning, and NLP.
    Built production ML models using TensorFlow and PyTorch. Expert in cloud deployment (AWS, Docker).
    """
    
    jd = """
    We're seeking a Machine Learning Engineer with strong Python skills.
    Experience with NLP, TensorFlow, and cloud platforms (AWS/GCP) required.
    """
    
    calc = SemanticSimilarityCalculator()
    
    # Basic similarity
    sim = calc.calculate_similarity(resume, jd)
    print(f"\nBasic Similarity: {sim:.3f}")
    
    # Section-wise similarity
    resume_sections = {
        "skills": "Python, Machine Learning, NLP, TensorFlow, PyTorch, AWS, Docker",
        "experience": "5+ years building production ML models"
    }
    
    jd_sections = {
        "required_skills": "Python, NLP, TensorFlow, AWS",
        "responsibilities": "Build ML models, deploy to cloud"
    }
    
    section_sims = calc.calculate_section_similarity(resume_sections, jd_sections)
    print(f"\nSection-wise Similarities:")
    for section, score in section_sims.items():
        print(f"  {section}: {score:.3f}")
    
    # Weighted similarity
    weighted_results = calc.calculate_weighted_similarity(
        resume, jd, resume_sections, jd_sections
    )
    print(f"\nWeighted Results:")
    for key, value in weighted_results.items():
        print(f"  {key}: {value:.3f}")
    
    # Batch processing
    print("\nBatch Processing Demo:")
    resumes = [resume] * 3
    jds = [jd] * 3
    batch_sims = calc.batch_calculate_similarity(resumes, jds)
    print(f"Batch similarities: {[f'{s:.3f}' for s in batch_sims]}")