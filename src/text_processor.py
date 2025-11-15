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


def extract_and_clean_title(title: str, normalize_for_search: bool = True, normalization_mode: str = "smart") -> str:
    """
    Extract and clean title, handling edge cases.
    
    Args:
        title: Input title
        normalize_for_search: Apply retrieval normalization
        normalization_mode: "smart", "letters_numbers", or "aggressive"
    """
    if not title:
        return ""
    
    title = clean_text(title, preserve_structure=False)
    # remove excessive punctuation at end
    title = re.sub(r"[.!?]{2,}$", ".", title)
    
    # Apply retrieval normalization if requested (to match text normalization)
    if normalize_for_search:
        title = normalize_for_retrieval(title, mode=normalization_mode)
    
    return title.strip()


def remove_common_footers(text: str) -> str:
    """
    Remove common footer patterns that appear in multiple documents.
    
    This helps reduce noise and improve retrieval quality by removing
    repetitive legal text, copyright notices, etc.
    """
    if not text:
        return text
    
    # Common footer patterns (banking domain) - expanded list
    footer_patterns = [
        # Copyright and legal info
        r"©\s*\d{4}[-\d]*\s*АО\s*«Альфа-Банк».*?$",
        r"АО\s*«Альфа-Банк»\s*является\s*оператором\s*по\s*обработке\s*персональных\s*данных.*?$",
        r"Генеральная\s*лицензия\s*Банка\s*России.*?$",
        r"Центр\s*раскрытия\s*корпоративной\s*информации.*?$",
        r"Информация\s*профессионального\s*участника\s*рынка\s*ценных\s*бумаг.*?$",
        r"Ул\.\s*Каланчевская.*?Москва.*?$",
        r"АО\s*«Альфа-Банк»\s*использует\s*файлы\s*«cookie».*?$",
        r"Политика\s*в\s*отношении\s*обработки\s*персональных\s*данных.*?$",
        r"Информация\s*о\s*лицах.*?находится\s*Банк.*?$",
        r"Участник\s*системы\s*обязательного\s*страхования\s*вкладов.*?$",
        r"Информация\s*о\s*процентных\s*ставках.*?вклада.*?$",
        # Generic patterns
        r"©\s*\d{4}[-\d]*.*?$",  # Generic copyright
        r"Все\s*права\s*защищены.*?$",
        r"Политика\s*конфиденциальности.*?$",
        r"Условия\s*использования.*?$",
        # Navigation and menu items
        r"^(Главная|О\s*банке|Услуги|Контакты|Карта\s*сайта).*?$",
        r"^(Вернуться\s*наверх|Наверх|Вверх).*?$",
        # Contact info patterns
        r"\+?\d{1,3}[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}.*?$",  # Phone numbers
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}.*?$",  # Email addresses (if standalone)
    ]
    
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            cleaned_lines.append(line)
            continue
        
        # Check if line matches any footer pattern
        is_footer = False
        for pattern in footer_patterns:
            if re.search(pattern, line_stripped, re.IGNORECASE | re.MULTILINE):
                is_footer = True
                break
        
        if not is_footer:
            cleaned_lines.append(line)
    
    result = "\n".join(cleaned_lines)
    # Remove trailing empty lines
    result = result.rstrip()
    
    return result


def normalize_for_retrieval(text: str, mode: str = "letters_numbers") -> str:
    """
    Normalize text for better retrieval: lowercase + optional symbol removal.
    
    Args:
        text: Input text
        mode: 
            - "letters_numbers" (default): lowercase + letters + digits + spaces
            - "smart": lowercase + letters + digits + important symbols (|, %, ., ,, -)
            - "aggressive": only letters + spaces (loses numbers - NOT recommended!)
        
    Returns:
        Normalized text optimized for retrieval
    """
    if not text:
        return ""
    
    text = text.lower()
    
    if mode == "aggressive":
        # Only letters and spaces (loses numbers - NOT recommended for banking!)
        text = re.sub(r'[^а-яёa-z\s]', ' ', text)
    elif mode == "smart":
        # Preserve digits and important symbols for banking domain
        # Important symbols: | (tables), % (percentages), . , (decimals), - (ranges)
        text = re.sub(r'[^а-яёa-z0-9\s|%.,\-]', ' ', text)
    else:  # letters_numbers mode (default)
        # Lowercase + letters + digits + spaces only
        text = re.sub(r'[^а-яёa-z0-9\s]', ' ', text)
    
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def extract_and_clean_text(text: str, normalize_for_search: bool = True, normalization_mode: str = "letters_numbers") -> str:
    """
    Extract and clean main text, preserving structure.
    
    Args:
        text: Input text
        normalize_for_search: If True, apply retrieval normalization
        normalization_mode: "letters_numbers" (default), "smart", or "aggressive"
    """
    if not text:
        return ""
    
    text = clean_text(text, preserve_structure=True)
    # Remove common footers to reduce noise
    text = remove_common_footers(text)
    
    # Apply retrieval normalization if requested
    if normalize_for_search:
        text = normalize_for_retrieval(text, mode=normalization_mode)
    
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


def validate_document(title: str, text: str, strict: bool = False) -> tuple:
    """
    Validate document quality with soft filtering for banking domain.
    
    Args:
        title: Document title
        text: Document text
        strict: If True, use strict filtering. If False, preserve short but meaningful docs.
    
    Returns:
        (is_valid, error_message_or_warning)
    """
    if not title and not text:
        return False, "Both title and text are empty"
    
    text_stripped = text.strip() if text else ""
    text_len = len(text_stripped)
    
    # Banking domain keywords that indicate important short documents
    banking_keywords = [
        "кредит", "карта", "счет", "счёт", "расчетный", "расчётный", "бик", "реквизиты",
        "вклад", "депозит", "перевод", "платеж", "платёж", "комиссия", "процент", "ставка",
        "лимит", "баланс", "выписка", "отделение", "офис", "филиал"
    ]
    
    # Soft check: if text is short but contains banking keywords, keep it
    if text_len < 10:
        # Check for banking keywords even in short text
        text_lower = text_stripped.lower()
        has_keywords = any(keyword in text_lower for keyword in banking_keywords)
        if has_keywords and not strict:
            return True, None  # Keep short but meaningful banking docs
        return False, f"Text too short: {text_len} characters"
    
    # For very short texts (10-50 chars), be more lenient if they have keywords
    if 10 <= text_len < 50:
        text_lower = text_stripped.lower()
        has_keywords = any(keyword in text_lower for keyword in banking_keywords)
        if has_keywords:
            return True, None  # Keep short meaningful docs
    
    # Check for excessive repetition (potential data corruption)
    # Made softer: only reject if very repetitive AND long
    if text:
        words = text.split()
        if len(words) > 100:  # Only check repetition for longer texts
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.05:  # Very strict threshold (was 0.1)
                return False, f"Excessive repetition detected: {unique_ratio:.2f} unique ratio"
            elif unique_ratio < 0.15 and strict:
                # Warning but don't reject unless strict mode
                return True, f"Low uniqueness: {unique_ratio:.2f} (kept due to soft mode)"
    
    return True, None

