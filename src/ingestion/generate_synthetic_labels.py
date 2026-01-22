"""
Synthetic Hiring Label Generator
Creates realistic hiring decisions for training data
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
import re

from src.logging import get_logger

logger = get_logger(__name__)


class SyntheticLabelGenerator:
    """
    Generate synthetic hiring labels based on heuristics
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize label generator
        
        Args:
            random_seed: For reproducibility
        """
        np.random.seed(random_seed)
        self.random_seed = random_seed
        
        self.role_skills = {
            'data_scientist': ['python', 'machine learning', 'sql', 'statistics', 
                              'deep learning', 'tensorflow', 'pytorch', 'pandas'],
            'software_engineer': ['java', 'python', 'javascript', 'react', 'node.js',
                                 'sql', 'git', 'docker', 'kubernetes'],
            'data_analyst': ['sql', 'excel', 'tableau', 'python', 'r', 
                            'power bi', 'statistics', 'data visualization'],
            'devops': ['docker', 'kubernetes', 'jenkins', 'aws', 'terraform',
                      'ansible', 'ci/cd', 'linux'],
            'web_developer': ['html', 'css', 'javascript', 'react', 'vue',
                             'node.js', 'mongodb', 'rest api']
        }
        
        self.education_scores = {
            'phd': 1.0,
            'master': 0.9,
            'bachelor': 0.7,
            'associate': 0.5,
            'diploma': 0.4,
            'certification': 0.3,
            'high_school': 0.2
        }
    
    def calculate_skill_match_score(
        self, 
        resume_text: str, 
        jd_text: str,
        role_type: str = 'software_engineer'
    ) -> float:
        """
        Calculate skill match score between resume and JD
        
        Args:
            resume_text: Resume content
            jd_text: Job description content
            role_type: Type of role
            
        Returns:
            Skill match score (0-1)
        """
        resume_lower = resume_text.lower()
        jd_lower = jd_text.lower()
        
        relevant_skills = self.role_skills.get(role_type, [])
        
        if not relevant_skills:
            resume_words = set(resume_lower.split())
            jd_words = set(jd_lower.split())
            overlap = len(resume_words & jd_words)
            return min(overlap / 100, 1.0)
        
        matches = sum(1 for skill in relevant_skills 
                     if skill in resume_lower and skill in jd_lower)
        
        score = matches / len(relevant_skills)
        score += np.random.normal(0, 0.1)
        
        return max(0, min(score, 1.0))
    
    def extract_experience_years(self, resume_text: str) -> float:
        """
        Extract years of experience from resume
        
        Args:
            resume_text: Resume content
            
        Returns:
            Years of experience
        """
        patterns = [
            r'(\d+)\+?\s*years?\s*of\s*experience',
            r'experience:\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yrs?\s*exp'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, resume_text.lower())
            if match:
                return float(match.group(1))
        
        return np.random.randint(0, 15)
    
    def extract_education_level(self, resume_text: str) -> str:
        """
        Extract education level from resume
        
        Args:
            resume_text: Resume content
            
        Returns:
            Education level
        """
        resume_lower = resume_text.lower()
        
        if 'phd' in resume_lower or 'ph.d' in resume_lower or 'doctorate' in resume_lower:
            return 'phd'
        elif "master's" in resume_lower or 'msc' in resume_lower or 'm.s.' in resume_lower:
            return 'master'
        elif "bachelor's" in resume_lower or 'bsc' in resume_lower or 'b.s.' in resume_lower or 'b.tech' in resume_lower:
            return 'bachelor'
        elif 'associate' in resume_lower:
            return 'associate'
        elif 'diploma' in resume_lower:
            return 'diploma'
        elif 'certification' in resume_lower or 'certified' in resume_lower:
            return 'certification'
        else:
            return 'high_school'
    
    def generate_hiring_decision(
        self,
        resume_text: str,
        jd_text: str,
        role_type: str = 'software_engineer',
        min_experience: float = 2.0,
        min_education: str = 'bachelor'
    ) -> Dict:
        """
        Generate hiring decision based on multiple factors
        
        Args:
            resume_text: Resume content
            jd_text: Job description
            role_type: Type of role
            min_experience: Minimum years required
            min_education: Minimum education level
            
        Returns:
            Dictionary with decision and scores
        """
        skill_score = self.calculate_skill_match_score(resume_text, jd_text, role_type)
        experience_years = self.extract_experience_years(resume_text)
        education_level = self.extract_education_level(resume_text)
        
        exp_score = min(experience_years / max(min_experience, 1), 1.0)
        
        edu_score = self.education_scores.get(education_level, 0.2)
        min_edu_score = self.education_scores.get(min_education, 0.5)
        edu_normalized = edu_score / min_edu_score if min_edu_score > 0 else 0.5
        
        cultural_fit = np.random.beta(5, 2)
        
        final_score = (
            0.40 * skill_score +
            0.30 * exp_score +
            0.20 * edu_normalized +
            0.10 * cultural_fit
        )
        
        final_score += np.random.normal(0, 0.05)
        final_score = max(0, min(final_score, 1.0))
        
        threshold = 0.50 + np.random.normal(0, 0.08)
        threshold = max(0.35, min(threshold, 0.65))
        selected = 1 if final_score >= threshold else 0
        
        return {
            'skill_match_score': round(skill_score, 3),
            'experience_years': experience_years,
            'experience_score': round(exp_score, 3),
            'education_level': education_level,
            'education_score': round(edu_score, 3),
            'cultural_fit_score': round(cultural_fit, 3),
            'final_score': round(final_score, 3),
            'selected': selected,
            'decision_threshold': round(threshold, 3),
            'role_type': role_type
        }
    
    def generate_labels_for_dataset(
        self,
        resumes_df: pd.DataFrame,
        jds_df: pd.DataFrame,
        n_samples: int = 1000,
        resume_col: str = 'resume_text',
        jd_col: str = 'jd_text'
    ) -> pd.DataFrame:
        """
        Generate synthetic labels for entire dataset
        
        Args:
            resumes_df: DataFrame with resumes
            jds_df: DataFrame with JDs
            n_samples: Number of resume-JD pairs to create
            resume_col: Column name for resume text
            jd_col: Column name for JD text
            
        Returns:
            DataFrame with labeled data
        """
        logger.info(f"Generating {n_samples} synthetic hiring labels...")
        
        results = []
        
        resume_sample = resumes_df.sample(n=min(n_samples, len(resumes_df)), 
                                          replace=True, random_state=self.random_seed)
        jd_sample = jds_df.sample(n=min(n_samples, len(jds_df)), 
                                  replace=True, random_state=self.random_seed)
        
        for idx in range(n_samples):
            resume_idx = idx % len(resume_sample)
            jd_idx = idx % len(jd_sample)
            
            resume_text = str(resume_sample.iloc[resume_idx][resume_col])
            jd_text = str(jd_sample.iloc[jd_idx][jd_col])
            
            role_type = np.random.choice(list(self.role_skills.keys()))
            
            decision = self.generate_hiring_decision(
                resume_text=resume_text,
                jd_text=jd_text,
                role_type=role_type,
                min_experience=np.random.choice([0, 2, 5, 7, 10]),
                min_education=np.random.choice(['bachelor', 'master', 'phd'])
            )
            
            result = {
                'resume_id': f"resume_{idx}",
                'jd_id': f"jd_{idx}",
                'resume_text': resume_text[:500],
                'jd_text': jd_text[:500],
                **decision,
                'created_at': datetime.now().isoformat()
            }
            
            results.append(result)
        
        labeled_df = pd.DataFrame(results)
        
        selection_rate = labeled_df['selected'].mean()
        avg_score = labeled_df['final_score'].mean()
        
        logger.info(f"Generated {len(labeled_df)} labeled samples")
        logger.info(f"Selection rate: {selection_rate:.1%}")
        logger.info(f"Average score: {avg_score:.3f}")
        logger.info(f"Selected: {labeled_df['selected'].sum()}")
        logger.info(f"Rejected: {(~labeled_df['selected'].astype(bool)).sum()}")
        
        return labeled_df


def main():
    """Generate synthetic labels for training"""
    from src.ingestion.data_loader import DataLoader
    
    loader = DataLoader()
    resumes_df, jds_df = loader.load_and_merge_datasets()
    
    generator = SyntheticLabelGenerator(random_seed=42)
    
    labeled_df = generator.generate_labels_for_dataset(
        resumes_df=resumes_df,
        jds_df=jds_df,
        n_samples=2000,
        resume_col='resume_text',
        jd_col='jd_text'
    )
    
    output_path = 'data/processed/training_data.csv'
    labeled_df.to_csv(output_path, index=False)
    logger.info(f"Saved labeled data to {output_path}")
    
    print("\nSample of labeled data:")
    print(labeled_df[['resume_id', 'jd_id', 'skill_match_score', 
                      'experience_years', 'final_score', 'selected']].head(10))
    
    print("\nClass Distribution:")
    print(labeled_df['selected'].value_counts())
    
    print("\nScore Statistics:")
    print(labeled_df['final_score'].describe())


if __name__ == "__main__":
    main()