#!/usr/bin/env python3
"""
Check corpus statistics: compare input CSV with processed corpus.

Usage:
    python scripts/check_corpus_stats.py --config configs/base.yaml
    python scripts/check_corpus_stats.py --input data/raw/websites_updated.csv --corpus data/processed/corpus.jsonl
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Check corpus statistics")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--input", help="Path to websites_updated.csv (overrides config)")
    parser.add_argument("--corpus", help="Path to corpus.jsonl (overrides config)")
    args = parser.parse_args()
    
    if args.config:
        config = load_config(args.config)
        input_csv = args.input or config["data"]["raw_websites"]
        corpus_jsonl = args.corpus or config["data"]["corpus_jsonl"]
    else:
        if not args.input or not args.corpus:
            parser.error("Either --config or both --input and --corpus must be provided")
        input_csv = args.input
        corpus_jsonl = args.corpus
    
    # Read input CSV
    print("Reading input CSV...")
    input_web_ids = set()
    input_text_lengths = []
    
    with open(input_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            web_id = row.get("web_id", "").strip()
            if web_id:
                input_web_ids.add(web_id)
                text = row.get("text", "").strip()
                input_text_lengths.append(len(text))
    
    print(f"Input CSV: {len(input_web_ids)} unique web_ids")
    print(f"  Text length: min={min(input_text_lengths)}, max={max(input_text_lengths)}, "
          f"mean={sum(input_text_lengths)/len(input_text_lengths):.0f}")
    
    # Read processed corpus
    print("\nReading processed corpus...")
    corpus_web_ids = set()
    corpus_text_lengths = []
    
    if Path(corpus_jsonl).exists():
        with open(corpus_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                web_id = doc.get("id", "")
                if web_id:
                    corpus_web_ids.add(web_id)
                    text = doc.get("text", "").strip()
                    corpus_text_lengths.append(len(text))
        
        print(f"Corpus JSONL: {len(corpus_web_ids)} unique web_ids")
        if corpus_text_lengths:
            print(f"  Text length: min={min(corpus_text_lengths)}, max={max(corpus_text_lengths)}, "
                  f"mean={sum(corpus_text_lengths)/len(corpus_text_lengths):.0f}")
    else:
        print(f"Corpus file not found: {corpus_jsonl}")
        return
    
    # Compare
    print("\n=== Comparison ===")
    missing = input_web_ids - corpus_web_ids
    extra = corpus_web_ids - input_web_ids
    
    print(f"Missing in corpus: {len(missing)} web_ids ({len(missing)*100/len(input_web_ids):.1f}%)")
    if missing:
        print(f"  Examples: {sorted(list(missing))[:10]}")
    
    print(f"Extra in corpus: {len(extra)} web_ids")
    if extra:
        print(f"  Examples: {sorted(list(extra))[:10]}")
    
    retention = len(corpus_web_ids) * 100 / len(input_web_ids) if input_web_ids else 0
    print(f"\nRetention rate: {retention:.1f}%")
    
    # Text length distribution
    print("\n=== Text Length Distribution ===")
    bins = [0, 50, 100, 200, 500, 1000, 2000, float('inf')]
    bin_labels = ["0-50", "50-100", "100-200", "200-500", "500-1000", "1000-2000", "2000+"]
    
    input_dist = Counter()
    corpus_dist = Counter()
    
    for length in input_text_lengths:
        for i, bin_max in enumerate(bins[1:], 0):
            if length <= bin_max:
                input_dist[bin_labels[i]] += 1
                break
    
    for length in corpus_text_lengths:
        for i, bin_max in enumerate(bins[1:], 0):
            if length <= bin_max:
                corpus_dist[bin_labels[i]] += 1
                break
    
    print("Input CSV distribution:")
    for label in bin_labels:
        count = input_dist[label]
        pct = count * 100 / len(input_text_lengths) if input_text_lengths else 0
        print(f"  {label:10s}: {count:5d} ({pct:5.1f}%)")
    
    print("\nCorpus JSONL distribution:")
    for label in bin_labels:
        count = corpus_dist[label]
        pct = count * 100 / len(corpus_text_lengths) if corpus_text_lengths else 0
        print(f"  {label:10s}: {count:5d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()

