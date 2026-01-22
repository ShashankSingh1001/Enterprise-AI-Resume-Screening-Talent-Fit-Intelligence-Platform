"""
Text Processing Utilities for Resume AI Platform
Provides text cleaning, extraction, and NLP preprocessing functions
"""

import re
import sys
from typing import List, Optional
from collections import Counter
import string

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

from src.logging import get_logger
from src.exceptions import FileProcessingError

logger = get_logger(__name__)

# Download required NLTK data (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
    logger.info("spaCy model loaded successfully")
except OSError:
    logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

# Constants
STOPWORDS = set(stopwords.words('english'))


def clean_text(text: str, remove_numbers: bool = False, 
               remove_punctuation: bool = False) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text to clean
        remove_numbers: Whether to remove numeric digits
        remove_punctuation: Whether to remove punctuation
        
    Returns:
        Cleaned text
    """
    try:
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided for cleaning")
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s@.+\-(),]', ' ', text)
        
        # Remove numbers if requested
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation if requested
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize unicode characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Remove extra spaces again after all operations
        text = ' '.join(text.split())
        
        logger.debug(f"Text cleaned: {len(text)} characters")
        return text
    
    except Exception as e:
        logger.error(f"Text cleaning failed: {str(e)}")
        raise FileProcessingError(e, sys)


def extract_email(text: str) -> Optional[str]:
    """
    Extract email address from text using regex.
    
    Args:
        text: Text containing email
        
    Returns:
        First valid email found, or None
    """
    try:
        # Regex pattern for email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        emails = re.findall(email_pattern, text)
        
        if emails:
            email = emails[0]
            logger.debug(f"Email extracted: {email}")
            return email
        
        logger.debug("No email found in text")
        return None
    
    except Exception as e:
        logger.error(f"Email extraction failed: {str(e)}")
        return None


def extract_phone(text: str) -> Optional[str]:
    """
    Extract phone number from text.
    Handles multiple formats: (123) 456-7890, 123-456-7890, 1234567890
    
    Args:
        text: Text containing phone number
        
    Returns:
        Standardized phone number, or None
    """
    try:
        # Pattern for various phone formats
        phone_patterns = [
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # (123) 456-7890 or 123-456-7890
            r'\d{10}',  # 1234567890
            r'\+\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'  # +1-123-456-7890
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            if matches:
                phone = matches[0]
                # Standardize format: remove all non-digits
                phone_digits = re.sub(r'\D', '', phone)
                
                # Format as XXX-XXX-XXXX (for 10 digits)
                if len(phone_digits) == 10:
                    formatted = f"{phone_digits[:3]}-{phone_digits[3:6]}-{phone_digits[6:]}"
                    logger.debug(f"Phone extracted: {formatted}")
                    return formatted
                elif len(phone_digits) > 10:
                    # International format
                    logger.debug(f"Phone extracted: {phone_digits}")
                    return phone_digits
        
        logger.debug("No phone number found in text")
        return None
    
    except Exception as e:
        logger.error(f"Phone extraction failed: {str(e)}")
        return None


def extract_urls(text: str) -> List[str]:
    """
    Extract all URLs from text.
    
    Args:
        text: Text containing URLs
        
    Returns:
        List of URLs found
    """
    try:
        # Regex pattern for URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        
        urls = re.findall(url_pattern, text)
        
        logger.debug(f"Extracted {len(urls)} URLs")
        return urls
    
    except Exception as e:
        logger.error(f"URL extraction failed: {str(e)}")
        return []


def remove_stopwords(text: str, language: str = 'english') -> str:
    """
    Remove common stopwords from text.
    
    Args:
        text: Input text
        language: Language for stopwords (default: english)
        
    Returns:
        Text with stopwords removed
    """
    try:
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in STOPWORDS]
        
        result = ' '.join(filtered_words)
        logger.debug(f"Removed stopwords: {len(words)} -> {len(filtered_words)} words")
        return result
    
    except Exception as e:
        logger.error(f"Stopword removal failed: {str(e)}")
        return text


def lemmatize_text(text: str) -> str:
    """
    Lemmatize text (convert words to base form).
    Example: "running" -> "run", "better" -> "good"
    
    Args:
        text: Input text
        
    Returns:
        Lemmatized text
    """
    try:
        if nlp is None:
            logger.warning("spaCy not loaded, returning original text")
            return text
        
        doc = nlp(text)
        lemmatized = ' '.join([token.lemma_ for token in doc])
        
        logger.debug(f"Text lemmatized: {len(text)} -> {len(lemmatized)} characters")
        return lemmatized
    
    except Exception as e:
        logger.error(f"Lemmatization failed: {str(e)}")
        return text


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Input text
        lowercase: Convert to lowercase
        
    Returns:
        List of tokens
    """
    try:
        if nlp is None:
            # Fallback to simple split
            logger.warning("spaCy not loaded, using simple tokenization")
            tokens = text.lower().split() if lowercase else text.split()
        else:
            doc = nlp(text)
            tokens = [token.text.lower() if lowercase else token.text 
                     for token in doc if not token.is_space]
        
        logger.debug(f"Tokenized into {len(tokens)} tokens")
        return tokens
    
    except Exception as e:
        logger.error(f"Tokenization failed: {str(e)}")
        return text.split()


def extract_keywords(text: str, n: int = 10, method: str = 'frequency') -> List[str]:
    """
    Extract top N keywords from text.
    
    Args:
        text: Input text
        n: Number of keywords to extract
        method: 'frequency' or 'tfidf'
        
    Returns:
        List of top keywords
    """
    try:
        # Clean and tokenize
        clean = clean_text(text)
        tokens = tokenize(clean)
        
        # Remove stopwords
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
        
        if method == 'frequency':
            # Simple frequency count
            counter = Counter(tokens)
            keywords = [word for word, _ in counter.most_common(n)]
        
        elif method == 'tfidf':
            # TF-IDF based (needs corpus, using single document here)
            if len(tokens) < 5:
                logger.warning("Text too short for TF-IDF, using frequency")
                counter = Counter(tokens)
                keywords = [word for word, _ in counter.most_common(n)]
            else:
                vectorizer = TfidfVectorizer(max_features=n)
                try:
                    vectorizer.fit_transform([' '.join(tokens)])
                    keywords = vectorizer.get_feature_names_out().tolist()
                except:
                    # Fallback to frequency
                    counter = Counter(tokens)
                    keywords = [word for word, _ in counter.most_common(n)]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.debug(f"Extracted {len(keywords)} keywords using {method}")
        return keywords
    
    except Exception as e:
        logger.error(f"Keyword extraction failed: {str(e)}")
        return []


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple word-based similarity between two texts.
    (For semantic similarity, use SBERT in features module)
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0-1)
    """
    try:
        # Tokenize and clean
        tokens1 = set(tokenize(clean_text(text1)))
        tokens2 = set(tokenize(clean_text(text2)))
        
        # Remove stopwords
        tokens1 = {t for t in tokens1 if t not in STOPWORDS}
        tokens2 = {t for t in tokens2 if t not in STOPWORDS}
        
        # Calculate Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        if union == 0:
            return 0.0
        
        similarity = intersection / union
        
        logger.debug(f"Text similarity calculated: {similarity:.3f}")
        return similarity
    
    except Exception as e:
        logger.error(f"Similarity calculation failed: {str(e)}")
        return 0.0


def extract_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    try:
        if nlp is None:
            # Simple split on punctuation
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        else:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        
        logger.debug(f"Extracted {len(sentences)} sentences")
        return sentences
    
    except Exception as e:
        logger.error(f"Sentence extraction failed: {str(e)}")
        return [text]


def remove_special_characters(text: str, keep_chars: str = '') -> str:
    """
    Remove special characters from text.
    
    Args:
        text: Input text
        keep_chars: Characters to keep (e.g., '@.-')
        
    Returns:
        Cleaned text
    """
    try:
        # Build pattern: keep alphanumeric, spaces, and specified chars
        pattern = f'[^a-zA-Z0-9\s{re.escape(keep_chars)}]'
        cleaned = re.sub(pattern, ' ', text)
        cleaned = ' '.join(cleaned.split())  # Remove extra spaces
        
        logger.debug(f"Removed special characters, kept: '{keep_chars}'")
        return cleaned
    
    except Exception as e:
        logger.error(f"Special character removal failed: {str(e)}")
        return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize all whitespace to single spaces.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    return ' '.join(text.split())


def truncate_text(text: str, max_length: int = 500, suffix: str = '...') -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length - len(suffix)] + suffix
    logger.debug(f"Text truncated: {len(text)} -> {len(truncated)} characters")
    return truncated