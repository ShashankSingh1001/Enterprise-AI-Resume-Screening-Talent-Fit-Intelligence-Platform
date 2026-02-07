"""
Text Processing Utilities for Resume AI Platform
Provides text cleaning, extraction, and NLP preprocessing functions
"""

import re
import sys
import string
from typing import List, Optional
from collections import Counter

import nltk
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from src.logging import get_logger
from src.exceptions import FileProcessingError

logger = get_logger(__name__)

# =====================================================
# NLTK SETUP (SAFE)
# =====================================================

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    logger.info("Downloading NLTK stopwords...")
    nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))

# =====================================================
# spaCy SETUP (SAFE + OPTIONAL)
# =====================================================

_NLP = None

def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded")
        except OSError:
            logger.warning(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            )
            _NLP = None
    return _NLP


# =====================================================
# CORE TEXT CLEANING
# =====================================================

def clean_text(
    text: str,
    remove_numbers: bool = False,
    remove_punctuation: bool = False
) -> str:
    """
    Clean and normalize text.
    """
    try:
        if not isinstance(text, str) or not text.strip():
            return ""

        text = text.lower()
        text = " ".join(text.split())

        # Keep common resume symbols
        text = re.sub(r"[^\w\s@.+\-(),]", " ", text)

        if remove_numbers:
            text = re.sub(r"\d+", "", text)

        if remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # ASCII normalize
        text = text.encode("ascii", "ignore").decode("ascii")

        return " ".join(text.split())

    except Exception as e:
        logger.error(f"Text cleaning failed: {e}")
        raise FileProcessingError(
            message="Text cleaning failed",
            error_detail=sys.exc_info()
        )


# =====================================================
# EXTRACTION HELPERS
# =====================================================

def extract_email(text: str) -> Optional[str]:
    try:
        pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        matches = re.findall(pattern, text)
        return matches[0] if matches else None
    except Exception:
        return None


def extract_phone(text: str) -> Optional[str]:
    try:
        patterns = [
            r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            r"\d{10}",
        ]

        for p in patterns:
            match = re.search(p, text)
            if match:
                digits = re.sub(r"\D", "", match.group())
                if len(digits) == 10:
                    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
                return digits

        return None

    except Exception:
        return None


def extract_urls(text: str) -> List[str]:
    try:
        pattern = r"(https?://\S+|www\.\S+)"
        return re.findall(pattern, text)
    except Exception:
        return []


# =====================================================
# NLP UTILITIES
# =====================================================

def remove_stopwords(text: str, language: str = "english") -> str:
    try:
        tokens = text.split()
        return " ".join(t for t in tokens if t.lower() not in STOPWORDS)
    except Exception:
        return text


def lemmatize_text(text: str) -> str:
    nlp = _get_nlp()
    if nlp is None:
        return text

    try:
        doc = nlp(text)
        return " ".join(token.lemma_ for token in doc)
    except Exception:
        return text


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    nlp = _get_nlp()

    try:
        if nlp is None:
            tokens = text.split()
        else:
            tokens = [t.text for t in nlp(text) if not t.is_space]

        return [t.lower() for t in tokens] if lowercase else tokens

    except Exception:
        return text.split()


# =====================================================
# KEYWORD & SIMILARITY
# =====================================================

def extract_keywords(
    text: str,
    n: int = 10,
    method: str = "frequency"
) -> List[str]:
    try:
        cleaned = clean_text(text)
        tokens = tokenize(cleaned)
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]

        if not tokens:
            return []

        if method == "frequency":
            return [w for w, _ in Counter(tokens).most_common(n)]

        if method == "tfidf":
            if len(tokens) < 5:
                return [w for w, _ in Counter(tokens).most_common(n)]

            vectorizer = TfidfVectorizer(max_features=n)
            vectorizer.fit([" ".join(tokens)])
            return vectorizer.get_feature_names_out().tolist()

        raise ValueError(f"Unknown method: {method}")

    except Exception:
        return []


def calculate_text_similarity(text1: str, text2: str) -> float:
    try:
        t1 = set(tokenize(clean_text(text1)))
        t2 = set(tokenize(clean_text(text2)))

        t1 -= STOPWORDS
        t2 -= STOPWORDS

        union = t1 | t2
        return len(t1 & t2) / len(union) if union else 0.0

    except Exception:
        return 0.0


# =====================================================
# SENTENCE & FORMAT HELPERS
# =====================================================

def extract_sentences(text: str) -> List[str]:
    nlp = _get_nlp()

    try:
        if nlp is None:
            return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        return [s.text.strip() for s in nlp(text).sents]

    except Exception:
        return [text]


def remove_special_characters(text: str, keep_chars: str = "") -> str:
    try:
        pattern = rf"[^a-zA-Z0-9\s{re.escape(keep_chars)}]"
        return " ".join(re.sub(pattern, " ", text).split())
    except Exception:
        return text


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix
