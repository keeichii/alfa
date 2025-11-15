#!/usr/bin/env python3
"""
Check consistency between questions and ground truth.

Usage:
    python scripts/check_ground_truth.py --questions data/raw/questions_clean.csv --ground_truth path/to/ground_truth.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def read_questions(csv_path: str) -> set[int]:
    """Read question IDs from CSV."""
    q_ids = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_id = row.get("q_id", "").strip()
            if q_id:
                try:
                    q_ids.add(int(q_id))
                except ValueError:
                    pass
    return q_ids


def load_ground_truth(json_path: str) -> dict:
    """Load ground truth from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Convert to dict[int, set[int]]
    gt = {}
    for q_id_str, web_ids in data.items():
        try:
            q_id = int(q_id_str)
            if isinstance(web_ids, list):
                gt[q_id] = set(int(wid) for wid in web_ids)
            else:
                gt[q_id] = {int(web_ids)}
        except (ValueError, TypeError):
            continue
    return gt


def main():
    parser = argparse.ArgumentParser(description="Check consistency between questions and ground truth")
    parser.add_argument("--questions", required=True, help="Path to questions_clean.csv")
    parser.add_argument("--ground_truth", required=True, help="Path to ground truth JSON file")
    args = parser.parse_args()
    
    print("Reading questions...")
    question_q_ids = read_questions(args.questions)
    print(f"Found {len(question_q_ids)} questions")
    
    print("\nReading ground truth...")
    ground_truth = load_ground_truth(args.ground_truth)
    gt_q_ids = set(ground_truth.keys())
    print(f"Found {len(gt_q_ids)} ground truth entries")
    
    print("\n=== Comparison ===")
    missing_in_gt = question_q_ids - gt_q_ids
    missing_in_questions = gt_q_ids - question_q_ids
    common = question_q_ids & gt_q_ids
    
    print(f"Questions without ground truth: {len(missing_in_gt)}")
    if missing_in_gt:
        print(f"  Examples: {sorted(list(missing_in_gt))[:10]}")
    
    print(f"\nGround truth without questions: {len(missing_in_questions)}")
    if missing_in_questions:
        print(f"  Examples: {sorted(list(missing_in_questions))[:10]}")
    
    print(f"\nCommon questions: {len(common)} ({len(common)*100/len(question_q_ids):.1f}% coverage)")
    
    # Statistics about ground truth
    if ground_truth:
        web_id_counts = [len(web_ids) for web_ids in ground_truth.values()]
        print(f"\n=== Ground Truth Statistics ===")
        print(f"Average relevant docs per query: {sum(web_id_counts)/len(web_id_counts):.1f}")
        print(f"Min relevant docs: {min(web_id_counts)}")
        print(f"Max relevant docs: {max(web_id_counts)}")
        
        # Distribution
        from collections import Counter
        dist = Counter(web_id_counts)
        print("\nDistribution of relevant docs per query:")
        for count in sorted(dist.keys()):
            print(f"  {count} docs: {dist[count]} queries")


if __name__ == "__main__":
    main()

