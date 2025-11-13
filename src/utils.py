"""Utility functions for the RAG pipeline."""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True  # force reconfiguration
)
logger = logging.getLogger(__name__)

# ensure immediate flushing
for handler in logger.handlers:
    handler.flush()
sys.stdout.flush()


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


def resolve_device(preferred: Optional[str] = None, fallback: str = "cpu") -> str:
    """
    Resolve the best available device given a user preference.

    - "auto": pick CUDA if available, else CPU.
    - "cuda" or "cuda:{id}": verify CUDA availability, otherwise fall back.
    - "cpu": always CPU.
    """
    pref = (preferred or "auto").lower()
    if pref in {"cpu", "cpu:0"}:
        return "cpu"

    if pref == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return fallback

    if pref.startswith("cuda"):
        if torch is not None and torch.cuda.is_available():
            device_parts = pref.split(":", 1)
            if len(device_parts) == 2:
                try:
                    idx = int(device_parts[1])
                    if idx < torch.cuda.device_count():
                        return f"cuda:{idx}"
                except ValueError:
                    logger.warning("Invalid CUDA device index %s, defaulting to cuda", device_parts[1])
            return "cuda"
        logger.warning("CUDA requested but not available; falling back to %s", fallback)
        return fallback

    # default fallback
    return fallback


def supports_fp16(device: str) -> bool:
    """Return True if the runtime supports fp16 on the specified device."""
    return torch is not None and device.startswith("cuda")

