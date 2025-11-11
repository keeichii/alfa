"""Text processing and normalization utilities for LLM-ready data preparation."""
import re
import unicodedata
from typing import Optional


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces, preserve newlines."""
    # replace tabs with spaces
    text = text.replace("\t", " ")
    # collapse multiple spaces but preserve newlines
    lines = text.split("\n")
    normalized_lines = []
    for line in lines:
        # collapse multiple spaces within line
        line = re.sub(r" +", " ", line)
        normalized_lines.append(line.strip())
    return "\n".join(normalized_lines)


def remove_control_characters(text: str) -> str:
    """Remove control characters except newlines, tabs, and carriage returns."""
    # keep newlines (\n), tabs (\t), carriage returns (\r)
    allowed = {"\n", "\t", "\r"}
    result = []
    for char in text:
        if unicodedata.category(char)[0] == "C":  # control character
            if char not in allowed:
                continue
        result.append(char)
    return "".join(result)


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters (NFKC normalization)."""
    return unicodedata.normalize("NFKC", text)


def clean_text(text: str, preserve_structure: bool = True) -> str:
    """
    Comprehensive text cleaning for LLM preparation.
    
    Args:
        text: Input text
        preserve_structure: If True, preserve paragraph structure (newlines)
        
    Returns:
        Cleaned text ready for LLM processing
    """
    if not text:
        return ""
    
    # normalize unicode first
    text = normalize_unicode(text)
    
    # remove control characters (except newlines/tabs if preserving structure)
    text = remove_control_characters(text)
    
    # normalize whitespace
    text = normalize_whitespace(text)
    
    # remove zero-width characters
    text = re.sub(r"[\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff]", "", text)
    
    # remove excessive newlines (more than 2 consecutive)
    if preserve_structure:
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = re.sub(r"\n+", " ", text)
    
    return text.strip()


def extract_and_clean_title(title: str) -> str:
    """Extract and clean title, handling edge cases."""
    if not title:
        return ""
    
    title = clean_text(title, preserve_structure=False)
    # remove excessive punctuation at end
    title = re.sub(r"[.!?]{2,}$", ".", title)
    return title.strip()


def extract_and_clean_text(text: str) -> str:
    """Extract and clean main text, preserving structure."""
    if not text:
        return ""
    
    text = clean_text(text, preserve_structure=True)
    return text


def merge_short_paragraphs(text: str, min_length: int = 50) -> str:
    """
    Merge short paragraphs to improve chunking quality.
    
    Args:
        text: Text with paragraphs separated by newlines
        min_length: Minimum character length for standalone paragraph
        
    Returns:
        Text with short paragraphs merged
    """
    if not text:
        return ""
    
    paragraphs = text.split("\n")
    merged = []
    current = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            if current:
                merged.append(current)
                current = ""
            continue
        
        # if current paragraph is short, try to merge with next
        if len(current) < min_length and current:
            current += " " + para
        else:
            if current:
                merged.append(current)
            current = para
    
    if current:
        merged.append(current)
    
    return "\n".join(merged)


def validate_document(title: str, text: str) -> tuple:
    """
    Validate document quality.
    
    Returns:
        (is_valid, error_message)
    """
    if not title and not text:
        return False, "Both title and text are empty"
    
    if text and len(text.strip()) < 10:
        return False, f"Text too short: {len(text.strip())} characters"
    
    # check for excessive repetition (potential data corruption)
    if text:
        words = text.split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.1 and len(words) > 100:
                return False, f"Excessive repetition detected: {unique_ratio:.2f} unique ratio"
    
    return True, None

