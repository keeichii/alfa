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
    Extract and normalize table structure.
    
    Returns:
        (normalized_text, is_table) - normalized text with preserved structure
    """
    if not detect_table(text):
        return text, False
    
    lines = text.split('\n')
    normalized_lines = []
    is_table = False
    
    for line in lines:
        line = line.strip()
        if not line:
            normalized_lines.append("")
            continue
        
        # Markdown table format
        if '|' in line:
            is_table = True
            # Clean up markdown table separators
            if re.match(r'^[\|\s\-:]+$', line):
                # Skip separator rows like |---|---|
                continue
            # Normalize table row
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            normalized_line = " | ".join(cells)
            normalized_lines.append(normalized_line)
        # Tab-separated or space-aligned
        elif '\t' in line or re.search(r'  +', line):
            is_table = True
            # Replace multiple spaces/tabs with single space, preserve structure
            normalized_line = re.sub(r'[\t ]+', ' ', line).strip()
            normalized_lines.append(normalized_line)
        else:
            normalized_lines.append(line)
    
    normalized_text = '\n'.join(normalized_lines)
    return normalized_text, is_table


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
    Enhance query to better match numeric values.
    
    Adds variations of numeric patterns to improve BM25 matching.
    """
    enhanced = query
    
    # Extract numbers from query
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
    
    return enhanced

