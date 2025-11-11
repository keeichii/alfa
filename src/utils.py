"""Utility functions for the RAG pipeline."""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line in {file_path}: {e}")
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save list of dictionaries to JSONL file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_timestamp() -> str:
    """Get current timestamp in format YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_corpus_format(doc: Dict[str, Any]) -> bool:
    """Validate that document has required fields."""
    return "id" in doc and "contents" in doc and isinstance(doc["contents"], str)


def validate_chunk_format(chunk: Dict[str, Any]) -> bool:
    """Validate that chunk has required fields."""
    required = ["id", "contents"]
    return all(field in chunk for field in required) and isinstance(chunk["contents"], str)

