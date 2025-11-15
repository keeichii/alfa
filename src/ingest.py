"""Ingest raw CSV data and convert to corpus JSONL format with advanced text processing."""
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

from src.utils import logger, save_jsonl
from src.text_processor import (
    extract_and_clean_title,
    extract_and_clean_text,
    validate_document
)
from src.table_processor import extract_table_structure

# increase csv field size limit for large text fields
limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(limit)
        break
    except OverflowError:
        limit = limit // 10


def read_websites_csv(
    csv_path: str, 
    validate: bool = True, 
    normalize_for_search: bool = True, 
    normalization_mode: str = "smart",
    log_rejected: bool = True,
    rejected_log_path: Optional[str] = None
) -> list[Dict[str, str]]:
    """
    Read websites CSV and convert to corpus format with advanced processing.
    
    Args:
        csv_path: Path to websites_updated.csv
        validate: Whether to validate document quality
        
    Returns:
        List of documents in format:
        {
            "id": web_id,
            "title": cleaned_title,
            "text": cleaned_text,
            "contents": title + "\\n" + text  # for compatibility
        }
        
    Time Complexity: O(n) where n is number of rows
    Space Complexity: O(n) for storing all documents
    """
    corpus = []
    skipped = 0
    skipped_reasons = {}
    rejected_docs = []  # For logging rejected documents
    
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        total_rows = 0
        for row_num, row in enumerate(reader, start=2):  # start=2 because header is row 1
            total_rows += 1
            try:
                web_id = str(row.get("web_id", "")).strip()
                raw_title = (row.get("title") or "").strip()
                raw_text = (row.get("text") or "").strip()
                
                if not web_id:
                    skipped += 1
                    reason = "missing_web_id"
                    skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                    if log_rejected:
                        rejected_docs.append({
                            "web_id": web_id or "N/A",
                            "row": row_num,
                            "reason": reason,
                            "text_length": 0
                        })
                    continue
                
                # IMPORTANT: Process tables BEFORE normalization
                # Normalization removes table markers (|), so we need to process tables first
                from src.table_processor import extract_table_structure
                text, has_table = extract_table_structure(raw_text)
                
                # clean and normalize text (AFTER table processing)
                title = extract_and_clean_title(raw_title, normalize_for_search=normalize_for_search, normalization_mode=normalization_mode)
                text = extract_and_clean_text(text, normalize_for_search=normalize_for_search, normalization_mode=normalization_mode)
                
                # validate document quality (soft mode by default)
                if validate:
                    is_valid, error_msg = validate_document(title, text, strict=False)
                    if not is_valid:
                        skipped += 1
                        reason = error_msg or "validation_failed"
                        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                        if log_rejected:
                            rejected_docs.append({
                                "web_id": web_id,
                                "row": row_num,
                                "reason": reason,
                                "text_length": len(text) if text else 0
                            })
                        if row_num <= 10:  # log first few for debugging
                            logger.debug(f"Row {row_num} (web_id={web_id}): {error_msg}")
                        continue
                
                # if no title but has text, use first sentence as title
                if not title and text:
                    first_sentence = text.split(".", 1)[0].strip()
                    if len(first_sentence) < 100:
                        title = first_sentence
                        text = text[len(first_sentence):].lstrip(". ")
                
                # ensure we have content
                if not title and not text:
                    skipped += 1
                    reason = "empty_content"
                    skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                    if log_rejected:
                        rejected_docs.append({
                            "web_id": web_id,
                            "row": row_num,
                            "reason": reason,
                            "text_length": 0
                        })
                    continue
                
                # combine for compatibility (FlashRAG format)
                # Standard format: "title\ntext" - title and text separated by newline
                # Title and text will be scored separately in retriever with title having higher weight
                if title:
                    if text:
                        contents = f"{title}\n{text}".strip()
                    else:
                        # If no text, just use title (already validated that title exists)
                        contents = title
                else:
                    contents = text
                
                corpus.append({
                    "id": web_id,
                    "title": title,
                    "text": text,
                    "contents": contents  # for FlashRAG compatibility
                })
                
            except Exception as e:
                skipped += 1
                reason = f"error_{type(e).__name__}"
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                if log_rejected:
                    rejected_docs.append({
                        "web_id": row.get("web_id", "N/A") if 'row' in locals() else "N/A",
                        "row": row_num,
                        "reason": reason,
                        "text_length": 0
                    })
                logger.warning(f"Error processing row {row_num}: {e}")
                continue
    
    # Log detailed statistics
    if skipped > 0:
        logger.info(f"Skipped {skipped} rows out of {total_rows} total ({skipped*100/total_rows:.1f}%)")
        logger.info("Skipped by reason:")
        for reason, count in sorted(skipped_reasons.items(), key=lambda x: x[1], reverse=True):
            pct = count * 100 / total_rows
            logger.info(f"  {reason}: {count} ({pct:.1f}%)")
    
    # Save rejected documents log if requested
    if log_rejected and rejected_docs and rejected_log_path:
        import json
        log_path = Path(rejected_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(rejected_docs, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved rejected documents log to {rejected_log_path}")
    
    logger.info(f"Loaded {len(corpus)} documents from {csv_path} (retention: {len(corpus)*100/total_rows:.1f}%)")
    return corpus


def build_corpus(
    input_csv: str, 
    output_jsonl: str, 
    normalize_for_search: bool = True, 
    normalization_mode: str = "smart",
    log_rejected: bool = True
) -> None:
    """
    Build corpus JSONL from websites CSV.
    
    Args:
        input_csv: Path to websites_updated.csv
        output_jsonl: Path to output corpus.jsonl
        normalize_for_search: Apply retrieval normalization
        normalization_mode: "letters_numbers" (default), "smart", or "aggressive"
    """
    logger.info(f"Building corpus from {input_csv}")
    # Set rejected log path
    rejected_log_path = None
    if log_rejected:
        output_path = Path(output_jsonl)
        rejected_log_path = str(output_path.parent / f"{output_path.stem}_rejected.json")
    
    corpus = read_websites_csv(
        input_csv, 
        normalize_for_search=normalize_for_search, 
        normalization_mode=normalization_mode,
        log_rejected=log_rejected,
        rejected_log_path=rejected_log_path
    )
    save_jsonl(corpus, output_jsonl)
    logger.info(f"Saved {len(corpus)} documents to {output_jsonl}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build corpus JSONL from websites CSV")
    parser.add_argument("--input", required=True, help="Path to websites_updated.csv")
    parser.add_argument("--output", required=True, help="Path to output corpus.jsonl")
    args = parser.parse_args()
    
    build_corpus(args.input, args.output)

