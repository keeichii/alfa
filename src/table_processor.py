"""Table processing and structure detection for banking domain."""
import re
from typing import List, Tuple, Dict


def detect_table(text: str) -> bool:
    """Detect if text contains a table structure."""
    # Check for markdown table format
    if re.search(r'\|.*\|', text, re.MULTILINE):
        return True
    
    # Check for tab-separated values
    lines = text.split('\n')
    if len(lines) >= 2:
        tab_count = sum(1 for line in lines[:5] if '\t' in line)
        if tab_count >= 2:
            return True
    
    # Check for multiple spaces (aligned columns)
    space_aligned = sum(1 for line in lines[:5] if re.search(r'  +', line))
    if space_aligned >= 2:
        return True
    
    return False


def extract_table_structure(text: str) -> Tuple[str, bool]:
    """
    Extract and normalize table structure, converting to readable text format.
    
    IMPORTANT: This function is called AFTER normalize_for_retrieval, which may
    remove table markers (|). So we need to handle cases where tables are
    already partially processed.
    
    Returns:
        (normalized_text, is_table) - normalized text with preserved structure
    """
    if not text or not text.strip():
        return text, False
    
    # Check if text contains table markers BEFORE normalization
    # But since this is called after normalization, we need to detect tables
    # by other means (multiple columns, aligned data, etc.)
    
    # If text is very short after normalization, it might be a table that was
    # over-normalized. In this case, preserve the original structure.
    if len(text.strip()) < 50:
        # Check if it looks like table data (numbers, short words, separators)
        # If so, don't process further - return as is
        lines = text.split('\n')
        if len(lines) >= 2:
            # Check if lines have similar structure (potential table)
            line_lengths = [len(line.strip()) for line in lines if line.strip()]
            if line_lengths and max(line_lengths) - min(line_lengths) < 20:
                # Similar line lengths - might be table, preserve as is
                return text, False
    
    if not detect_table(text):
        return text, False
    
    lines = text.split('\n')
    normalized_lines = []
    is_table = False
    table_rows = []
    has_regular_text = False
    
    for line in lines:
        line = line.strip()
        if not line:
            # If we were collecting a table, flush it
            if table_rows:
                formatted = _format_table_as_text(table_rows)
                if formatted:  # Only add if formatting succeeded
                    normalized_lines.extend(formatted)
                else:
                    # Formatting failed, preserve original table rows as text
                    for row in table_rows:
                        normalized_lines.append(' '.join(row))
                table_rows = []
            normalized_lines.append("")
            continue
        
        # Markdown table format
        if '|' in line:
            is_table = True
            # Clean up markdown table separators
            if re.match(r'^[\|\s\-:]+$', line):
                # Skip separator rows like |---|---|
                continue
            # Extract cells
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if len(cells) >= 2:  # At least 2 columns
                table_rows.append(cells)
            else:
                # Not a table row, flush and add as regular text
                if table_rows:
                    formatted = _format_table_as_text(table_rows)
                    if formatted:
                        normalized_lines.extend(formatted)
                    else:
                        # Formatting failed, preserve original
                        for row in table_rows:
                            normalized_lines.append(' '.join(row))
                    table_rows = []
                normalized_lines.append(line)
                has_regular_text = True
        # Tab-separated or space-aligned
        elif '\t' in line or re.search(r'  +', line):
            is_table = True
            # Split by tabs or multiple spaces
            if '\t' in line:
                cells = [c.strip() for c in line.split('\t') if c.strip()]
            else:
                cells = [c.strip() for c in re.split(r'  +', line) if c.strip()]
            if len(cells) >= 2:  # At least 2 columns
                table_rows.append(cells)
            else:
                # Not a table row, flush and add as regular text
                if table_rows:
                    formatted = _format_table_as_text(table_rows)
                    if formatted:
                        normalized_lines.extend(formatted)
                    else:
                        # Formatting failed, preserve original
                        for row in table_rows:
                            normalized_lines.append(' '.join(row))
                    table_rows = []
                normalized_lines.append(re.sub(r'[\t ]+', ' ', line).strip())
                has_regular_text = True
        else:
            # Regular text line - flush table if collecting
            if table_rows:
                formatted = _format_table_as_text(table_rows)
                if formatted:
                    normalized_lines.extend(formatted)
                else:
                    # Formatting failed, preserve original
                    for row in table_rows:
                        normalized_lines.append(' '.join(row))
                table_rows = []
            normalized_lines.append(line)
            has_regular_text = True
    
    # Flush any remaining table
    if table_rows:
        formatted = _format_table_as_text(table_rows)
        if formatted:
            normalized_lines.extend(formatted)
        else:
            # Formatting failed, preserve original
            for row in table_rows:
                normalized_lines.append(' '.join(row))
    
    normalized_text = '\n'.join(normalized_lines)
    
    # CRITICAL: If normalized text is empty or too short, return original text
    # This prevents losing content when table processing fails
    if not normalized_text.strip() or len(normalized_text.strip()) < len(text.strip()) * 0.1:
        # Table processing removed too much content, return original
        return text, False
    
    return normalized_text, is_table


def _format_table_as_text(rows: List[List[str]]) -> List[str]:
    """
    Format table rows as readable text.
    
    Example:
        [["Валюта", "Покупка", "Продажа"], ["USD", "77", "80.4"]]
        -> ["Валюта: Покупка 77, Продажа 80.4", "USD: Покупка 77, Продажа 80.4"]
    """
    if not rows or len(rows) < 2:
        return []
    
    formatted = []
    headers = rows[0] if len(rows[0]) > 1 else None
    
    # If we have headers, format as "Header: Value1, Value2"
    if headers and len(headers) >= 2:
        for row in rows[1:]:
            if len(row) >= 2:
                # First column is usually the key
                key = row[0]
                values = []
                for i, header in enumerate(headers[1:], 1):
                    if i < len(row):
                        values.append(f"{header} {row[i]}")
                if values:
                    formatted.append(f"{key}: {', '.join(values)}")
    else:
        # No headers, format as "Column1: Column2, Column3"
        for row in rows:
            if len(row) >= 2:
                key = row[0]
                values = ', '.join(row[1:])
                formatted.append(f"{key}: {values}")
    
    return formatted if formatted else []


def preserve_table_in_chunk(table_text: str, chunk_text: str) -> str:
    """
    Ensure table structure is preserved when chunking.
    
    Args:
        table_text: Original table text
        chunk_text: Chunk text that may contain part of table
        
    Returns:
        Chunk text with preserved table structure
    """
    # If chunk contains table markers, ensure proper formatting
    if '|' in chunk_text:
        lines = chunk_text.split('\n')
        formatted_lines = []
        for line in lines:
            if '|' in line:
                # Ensure proper table formatting
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                formatted_line = " | ".join(cells)
                formatted_lines.append(formatted_line)
            else:
                formatted_lines.append(line)
        return '\n'.join(formatted_lines)
    
    return chunk_text


def extract_numeric_values(text: str) -> List[Dict[str, str]]:
    """
    Extract numeric values (amounts, rates, percentages) for enhanced retrieval.
    
    Returns:
        List of dicts with 'value', 'type', 'context'
    """
    numeric_patterns = [
        # Currency amounts: 1000 руб., $100, 100 EUR
        (r'(\d+[.,]?\d*)\s*(руб|RUB|USD|EUR|CNY|долл|евро|юань)', 'currency'),
        # Percentages: 5%, 10.5%
        (r'(\d+[.,]?\d*)%', 'percentage'),
        # Rates: 10.5%, курс 77.5
        (r'(?:курс|ставка|процент|rate)\s*[:\-]?\s*(\d+[.,]?\d*)', 'rate'),
        # Large numbers: 1000000, 1 000 000
        (r'(\d{1,3}(?:\s+\d{3})+)', 'large_number'),
        # Dates with numbers: 01.01.2024
        (r'(\d{1,2}[./]\d{1,2}[./]\d{2,4})', 'date'),
    ]
    
    extracted = []
    for pattern, num_type in numeric_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Get context (20 chars before and after)
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end].strip()
            
            extracted.append({
                'value': match.group(1),
                'type': num_type,
                'context': context
            })
    
    return extracted


def enhance_query_for_numerics(query: str) -> str:
    """
    Enhance query to better match numeric values and banking terms.
    
    Adds variations of numeric patterns, banking synonyms, and query expansion
    to improve BM25 and dense retrieval matching.
    """
    enhanced = query
    
    # Extended banking domain synonyms and expansions
    banking_synonyms = {
        "счет": ["счет", "счёт", "аккаунт", "account", "банковский счет"],
        "карта": ["карта", "card", "кредитная карта", "дебетовая карта", "пластиковая карта"],
        "кредит": ["кредит", "займ", "loan", "кредитная линия", "кредитование"],
        "вклад": ["вклад", "депозит", "deposit", "сберегательный счет"],
        "платеж": ["платеж", "платёж", "оплата", "payment", "транзакция", "перевод средств"],
        "перевод": ["перевод", "transfer", "трансфер", "перечисление", "перевод денег"],
        "бик": ["бик", "bik", "банковский идентификационный код", "банковский код"],
        "реквизиты": ["реквизиты", "details", "банковские реквизиты", "платежные реквизиты"],
        "отделение": ["отделение", "офис", "банк", "branch", "office", "филиал", "банковское отделение"],
        "смс": ["смс", "sms", "сообщение", "код подтверждения", "смс-уведомление"],
        "онлайн": ["онлайн", "online", "интернет-банк", "мобильный банк", "интернет банкинг"],
        "номер": ["номер", "number", "num", "номер счета", "номер карты"],
        "узнать": ["узнать", "найти", "посмотреть", "проверить", "find", "check", "уточнить"],
        "получить": ["получить", "get", "заказать", "оформить", "выпустить"],
        "процент": ["процент", "ставка", "rate", "процентная ставка", "годовая ставка"],
        "комиссия": ["комиссия", "fee", "плата", "стоимость", "тариф"],
        "лимит": ["лимит", "limit", "ограничение", "максимум"],
        "баланс": ["баланс", "balance", "остаток", "остаток средств"],
        "выписка": ["выписка", "statement", "отчет", "история операций"],
        "кэшбэк": ["кэшбэк", "cashback", "возврат", "бонус"],
    }
    
    # Query rewriting patterns (common question reformulations)
    query_rewrites = {
        "как": ["как", "способ", "метод", "инструкция"],
        "где": ["где", "место", "адрес", "локация"],
        "когда": ["когда", "время", "срок", "дата"],
        "сколько": ["сколько", "сумма", "размер", "объем"],
        "можно": ["можно", "возможно", "разрешено", "доступно"],
        "нужно": ["нужно", "необходимо", "требуется", "надо"],
    }
    
    # Add synonyms for key banking terms
    query_lower = query.lower()
    added_synonyms = set()
    
    for key, synonyms in banking_synonyms.items():
        if key in query_lower:
            for synonym in synonyms[:3]:  # Add first 3 synonyms
                if synonym not in query_lower and synonym not in added_synonyms:
                    enhanced += f" {synonym}"
                    added_synonyms.add(synonym)
    
    # Add query rewriting patterns
    for pattern, rewrites in query_rewrites.items():
        if pattern in query_lower:
            for rewrite in rewrites[:2]:  # Add first 2 rewrites
                if rewrite not in query_lower and rewrite not in added_synonyms:
                    enhanced += f" {rewrite}"
                    added_synonyms.add(rewrite)
    
    # Extract numbers from query and add variations
    numbers = re.findall(r'\d+[.,]?\d*', query)
    for num in numbers:
        # Add variations: with/without spaces, with/without decimal
        if '.' in num or ',' in num:
            # Add integer version
            int_version = re.sub(r'[.,]\d+', '', num)
            if int_version and int_version not in enhanced:
                enhanced += f" {int_version}"
        else:
            # Add decimal versions
            enhanced += f" {num}.0 {num},0"
    
    # Add common banking query patterns
    if any(word in query_lower for word in ["где", "как", "какой", "что"]):
        # Add "информация" and "помощь" for informational queries
        if "информация" not in enhanced.lower():
            enhanced += " информация"
        if "условия" not in enhanced.lower() and any(w in query_lower for w in ["кредит", "вклад", "карта"]):
            enhanced += " условия"
    
    # Remove extra spaces and limit length
    enhanced = re.sub(r'\s+', ' ', enhanced).strip()
    # Limit to reasonable length (keep original + ~100 chars expansion)
    if len(enhanced) > len(query) + 150:
        # Keep original query + first 30 words of expansion
        words = enhanced.split()
        original_words = query.split()
        max_expansion_words = 30
        enhanced = " ".join(original_words + words[len(original_words):len(original_words) + max_expansion_words])
    
    return enhanced

