"""
Entity Extractor - Production Optimized
Singleton spaCy model + single NLP pass + libphonenumber
"""

import re
import spacy
import sys
from datetime import datetime
from typing import List, Dict, Optional
from dateutil import parser as date_parser

try:
    import phonenumbers
    from phonenumbers import PhoneNumberMatcher
    PHONENUMBERS_AVAILABLE = True
except ImportError:
    PHONENUMBERS_AVAILABLE = False

from src.logging import get_logger
from src.exceptions import ResumeParsingError

logger = get_logger(__name__)

# Singleton spaCy model (loaded once globally)
_NLP = None

def _get_nlp(model="en_core_web_sm"):
    global _NLP
    if _NLP is None:
        try:
            logger.info(f"Loading spaCy model once: {model}")
            _NLP = spacy.load(model)
        except OSError:
            logger.warning(f"Downloading spaCy model: {model}")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model])
            _NLP = spacy.load(model)
    return _NLP

# Compiled regex patterns (faster)
EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
URL_RE = re.compile(r'(https?://\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+)', re.IGNORECASE)
LINKEDIN_RE = re.compile(r'linkedin\.com/in/[\w-]+', re.IGNORECASE)
GITHUB_RE = re.compile(r'github\.com/[\w-]+', re.IGNORECASE)


class EntityExtractor:
    """High-performance entity extractor with singleton spaCy model"""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = _get_nlp(spacy_model)
    
    # Main extraction (single spaCy pass)
    def extract_all_entities(self, text: str) -> Dict[str, any]:
        """Single-pass extraction for best performance"""
        try:
            doc = self.nlp(text)
            return {
                "emails": self._extract_emails(text),
                "phones": self._extract_phones(text),
                "urls": self._extract_urls(text),
                "dates": self._extract_dates(text),
                "organizations": self._extract_orgs(doc),
                "locations": self._extract_locations(doc),
                "names": self._extract_persons(doc)
            }
        except Exception as e:
            error_msg = f"Error extracting entities: {str(e)}"
            logger.error(error_msg)
            raise ResumeParsingError(
                message=error_msg,
                error_detail=sys.exc_info()
            )
    
    # Private fast helpers
    def _extract_emails(self, text: str) -> List[str]:
        return list(set(EMAIL_RE.findall(text)))
    
    def _extract_phones(self, text: str) -> List[str]:
        """Use libphonenumber if available, else regex"""
        if PHONENUMBERS_AVAILABLE:
            numbers = []
            try:
                # Try without region first (better for international numbers)
                for match in PhoneNumberMatcher(text, None):
                    num = match.number
                    formatted = phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.E164)
                    numbers.append(formatted)
                if numbers:
                    return list(set(numbers))
            except:
                pass
        
        # Fallback regex (more flexible patterns)
        patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # +1-555-123-4567
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # (555) 123-4567
            r'\d{3}[-.\s]\d{3}[-.\s]\d{4}',  # 555-123-4567
        ]
        
        phones = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            phones.extend(matches)
        
        # Clean and validate
        cleaned = []
        for phone in phones:
            clean = re.sub(r'[^\d+]', '', phone)
            if len(clean) >= 10:
                cleaned.append(phone)
        
        return list(set(cleaned)) if cleaned else []
    
    def _extract_urls(self, text: str) -> Dict[str, List[str]]:
        urls = list(set(URL_RE.findall(text)))
        return {
            "all": urls,
            "linkedin": list(set(LINKEDIN_RE.findall(text))),
            "github": list(set(GITHUB_RE.findall(text)))
        }
    
    def _extract_dates(self, text: str) -> List[Dict]:
        """Extract dates with multiple patterns"""
        patterns = [
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b\d{4}\b',
            r'\b(?:Present|Current|Ongoing|Till date)\b'
        ]
        results = []
        for p in patterns:
            for m in re.finditer(p, text, re.IGNORECASE):
                date_str = m.group()
                try:
                    parsed = (datetime.now() if date_str.lower() in ["present", "current", "ongoing", "till date"]
                             else date_parser.parse(date_str, fuzzy=True))
                except:
                    parsed = None
                results.append({
                    "date_string": date_str,
                    "parsed_date": parsed,
                    "position": m.start()
                })
        return results
    
    def _extract_orgs(self, doc) -> List[str]:
        return list({e.text for e in doc.ents if e.label_ == "ORG"})
    
    def _extract_locations(self, doc) -> List[str]:
        return list({e.text for e in doc.ents if e.label_ in ("GPE", "LOC")})
    
    def _extract_persons(self, doc) -> List[str]:
        return list({e.text for e in doc.ents if e.label_ == "PERSON"})
    
    # Public API wrappers (for compatibility)
    def extract_emails(self, text: str) -> List[str]:
        return self._extract_emails(text)
    
    def extract_phone_numbers(self, text: str) -> List[str]:
        return self._extract_phones(text)
    
    def extract_urls(self, text: str) -> Dict[str, List[str]]:
        return self._extract_urls(text)
    
    def extract_dates(self, text: str) -> List[Dict]:
        return self._extract_dates(text)
    
    def extract_organizations(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return self._extract_orgs(doc)
    
    def extract_locations(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return self._extract_locations(doc)
    
    def extract_person_names(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return self._extract_persons(doc)
    
    def extract_name_from_resume(self, text: str) -> Optional[str]:
        """Extract candidate name with contact keyword filtering"""

        CONTACT_KEYWORDS = [
            'email', 'phone', 'tel', 'mobile',
            'linkedin', 'github', 'contact',
            '@', '+', 'http', 'www'
        ]

        def clean_name(name: str) -> Optional[str]:
            name = name.strip()

            # Remove contact keywords inside the name
            tokens = [
                t for t in name.split()
                if t.lower() not in CONTACT_KEYWORDS
            ]
            if len(tokens) < 2:
                return None

            cleaned = " ".join(tokens)

            if len(cleaned) < 3 or len(cleaned) > 50:
                return None

            return cleaned.title()

        # Strategy 1: Explicit "Name:" pattern
        name_match = re.search(r'Name\s*:\s*([^\n]+)', text, re.IGNORECASE)
        if name_match:
            cleaned = clean_name(name_match.group(1))
            if cleaned:
                return cleaned

        # Strategy 2: First non-empty line
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            cleaned = clean_name(line)
            if cleaned:
                return cleaned
            break  # only check first meaningful line

        # Strategy 3: spaCy PERSON entities
        doc = self.nlp(text[:500])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                cleaned = clean_name(ent.text)
                if cleaned:
                    return cleaned

        return None



# Convenience singleton
_EXTRACTOR = EntityExtractor()

def extract_contact_info(text: str) -> Dict[str, any]:
    """Quick helper to extract contact info"""
    entities = _EXTRACTOR.extract_all_entities(text)
    return {
        "name": _EXTRACTOR.extract_name_from_resume(text),
        "emails": entities["emails"],
        "phones": entities["phones"],
        "urls": entities["urls"]
    }