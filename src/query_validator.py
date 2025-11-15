"""Query validation and cleaning for questions_clean.csv."""
import re
from typing import Tuple, Optional


def is_valid_query(query: str, min_words: int = 1, min_chars: int = 2) -> bool:
    """
    Check if query is valid and meaningful.
    
    Args:
        query: Input query string
        min_words: Minimum number of meaningful words required
        min_chars: Minimum number of characters required (excluding punctuation)
        
    Returns:
        True if query is valid, False otherwise
    """
    if not query or not isinstance(query, str):
        return False
    
    query = query.strip()
    if not query:
        return False
    
    # Remove all punctuation and whitespace, count remaining characters
    chars_only = re.sub(r'[^\w\s]', '', query)
    chars_only = re.sub(r'\s+', '', chars_only)
    
    if len(chars_only) < min_chars:
        return False
    
    # Count meaningful words (at least 2 characters each)
    words = [w for w in query.split() if len(re.sub(r'[^\w]', '', w)) >= 2]
    
    if len(words) < min_words:
        return False
    
    # Check for patterns that indicate noise
    noise_patterns = [
        r'^[.,;:!?\-_\s]+$',  # Only punctuation
        r'^\.\s*\.\s*\.\s*$',  # Only dots
        r'^0+\s*0+.*$',  # Only zeros
        r'^\?+\s*$',  # Only question marks
        r'^[^\w\s]+$',  # Only special characters
    ]
    
    for pattern in noise_patterns:
        if re.match(pattern, query):
            return False
    
    # Check if query is too repetitive (e.g., "? ? ? ?")
    if len(set(query.split())) == 1 and len(query.split()) > 3:
        return False
    
    return True


def clean_query(query: str) -> Optional[str]:
    """
    Clean and normalize query, removing noise.
    
    Args:
        query: Input query string
        
    Returns:
        Cleaned query or None if query is invalid
    """
    if not query or not isinstance(query, str):
        return None
    
    # Strip whitespace
    cleaned = query.strip()
    
    if not cleaned:
        return None
    
    # Remove excessive punctuation at start/end
    cleaned = re.sub(r'^[.,;:!?\-_\s]+', '', cleaned)
    cleaned = re.sub(r'[.,;:!?\-_\s]+$', '', cleaned)
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Remove zero-width characters
    cleaned = re.sub(r'[\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff]', '', cleaned)
    
    cleaned = cleaned.strip()
    
    # Validate after cleaning
    if not is_valid_query(cleaned):
        return None
    
    return cleaned


def validate_and_clean_questions(questions: list[Tuple[int, str]]) -> Tuple[list[Tuple[int, str]], dict]:
    """
    Validate and clean a list of questions.
    
    Args:
        questions: List of (q_id, query) tuples
        
    Returns:
        (cleaned_questions, stats) - cleaned questions and statistics
    """
    cleaned = []
    stats = {
        'total': len(questions),
        'valid': 0,
        'invalid': 0,
        'cleaned': 0,
        'removed': 0,
    }
    
    for q_id, query in questions:
        original = query
        
        # Try to clean
        cleaned_query = clean_query(query)
        
        if cleaned_query is None:
            stats['removed'] += 1
            continue
        
        if cleaned_query != original:
            stats['cleaned'] += 1
        
        if is_valid_query(cleaned_query):
            cleaned.append((q_id, cleaned_query))
            stats['valid'] += 1
        else:
            stats['invalid'] += 1
            stats['removed'] += 1
    
    return cleaned, stats

