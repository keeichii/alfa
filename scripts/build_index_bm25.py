#!/usr/bin/env python3
"""
Build BM25 index using FlashRAG.

Usage:
    python scripts/build_index_bm25.py --config configs/base.yaml
"""
import argparse
import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "FlashRAG"))

import yaml
from flashrag.retriever.index_builder import Index_Builder


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index using FlashRAG")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config file")
    parser.add_argument("--corpus_path", help="Path to chunks.jsonl (overrides config)")
    parser.add_argument("--save_dir", help="Path to save index (overrides config)")
    parser.add_argument("--backend", choices=["bm25s", "pyserini"], help="BM25 backend (overrides config)")
    args = parser.parse_args()
    
    # load configs
    config = load_config(args.config)
    models_config = load_config(project_root / "configs" / "models.yaml")
    
    # get paths
    corpus_path = args.corpus_path or config["data"]["chunks_jsonl"]
    save_dir = args.save_dir or config["indexes"]["bm25_dir"]
    backend = args.backend or models_config["bm25"]["backend"]
    
    print(f"Building BM25 index...")
    print(f"  Corpus: {corpus_path}")
    print(f"  Save dir: {save_dir}")
    print(f"  Backend: {backend}")
    
    # create index builder
    index_builder = Index_Builder(
        retrieval_method="bm25",
        model_path=None,  # not needed for BM25
        corpus_path=corpus_path,
        save_dir=save_dir,
        max_length=512,
        batch_size=256,
        use_fp16=False,
        bm25_backend=backend
    )
    
    # build index
    index_builder.build_index()
    
    # copy chunks.jsonl to BM25 index directory for retrieval
    from shutil import copy2
    from pathlib import Path
    chunks_source = config.get("data", {}).get("chunks_jsonl", "data/processed/chunks.jsonl")
    chunks_dest = Path(save_dir) / "chunks.jsonl"
    if Path(chunks_source).exists():
        copy2(chunks_source, chunks_dest)
        print(f"Copied chunks.jsonl to {chunks_dest}")
    
    print(f"BM25 index built successfully in {save_dir}")


if __name__ == "__main__":
    main()

