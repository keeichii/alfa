#!/usr/bin/env python3
"""
Simple direct test of BM25 retriever to verify it returns non-zero scores.
Bypasses FlashRAG wrapper to test bm25s directly.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "FlashRAG"))

import bm25s
import json

def test_bm25_direct():
    """Test BM25 retriever directly."""
    print("=" * 80)
    print("DIRECT BM25 TEST")
    print("=" * 80)
    
    # Load BM25 index
    index_path = Path("indexes/bm25/bm25")
    if not index_path.exists():
        print(f"ERROR: BM25 index not found at {index_path}")
        return False
    
    print(f"\n[1] Loading BM25 index from {index_path}...")
    try:
        searcher = bm25s.BM25.load(str(index_path))
        print("✓ BM25 index loaded")
    except Exception as e:
        print(f"✗ Failed to load BM25 index: {e}")
        return False
    
    # Load tokenizer
    print("\n[2] Loading tokenizer...")
    try:
        tokenizer = bm25s.tokenization.Tokenizer(stopwords=[])
        tokenizer.load_stopwords(str(index_path))
        tokenizer.load_vocab(str(index_path))
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return False
    
    # Test queries
    test_queries = [
        "Номер счета",
        "Где узнать бик и счёт",
        "Кэшбэк",
    ]
    
    print(f"\n[3] Testing {len(test_queries)} queries...")
    all_passed = True
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: '{query}'")
        print(f"{'='*80}")
        
        # Tokenize query
        try:
            query_tokens = tokenizer.tokenize([query], return_as='tuple')
            print(f"  Tokenized: {query_tokens}")
            
            if not query_tokens or (isinstance(query_tokens, list) and len(query_tokens) > 0 and 
                                   isinstance(query_tokens[0], (list, tuple)) and len(query_tokens[0]) == 0):
                print(f"  ✗ ERROR: Empty tokens after tokenization!")
                all_passed = False
                continue
            
            if isinstance(query_tokens, list) and len(query_tokens) > 0:
                if isinstance(query_tokens[0], (list, tuple)):
                    token_count = len(query_tokens[0])
                    print(f"  Token count: {token_count}")
                    if token_count == 0:
                        print(f"  ✗ ERROR: Zero tokens!")
                        all_passed = False
                        continue
        except Exception as e:
            print(f"  ✗ Tokenization failed: {e}")
            all_passed = False
            continue
        
            # Retrieve
            try:
                k = 10
                raw_results, raw_scores = searcher.retrieve(query_tokens, k=k)
                # Handle numpy arrays
                import numpy as np
                if isinstance(raw_results, np.ndarray):
                    result_count = raw_results.size if raw_results.size > 0 else 0
                else:
                    result_count = len(raw_results[0]) if raw_results and len(raw_results) > 0 else 0
                print(f"  Retrieved {result_count} results")
                
                if result_count == 0:
                    print(f"  ✗ ERROR: No results returned!")
                    all_passed = False
                    continue
            
            # Check scores
            scores = list(raw_scores[0]) if raw_scores and len(raw_scores) > 0 else []
            print(f"  Scores: {len(scores)} values")
            
            if not scores:
                print(f"  ✗ ERROR: No scores returned!")
                all_passed = False
                continue
            
            # Analyze scores
            scores_float = [float(s) for s in scores]
            non_zero = sum(1 for s in scores_float if s != 0.0)
            min_score = min(scores_float)
            max_score = max(scores_float)
            mean_score = sum(scores_float) / len(scores_float)
            
            print(f"  Score statistics:")
            print(f"    min={min_score:.6f}, max={max_score:.6f}, mean={mean_score:.6f}")
            print(f"    non-zero={non_zero}/{len(scores_float)}")
            
            # Show top 5 scores
            print(f"  Top 5 scores:")
            for i, score in enumerate(scores_float[:5], 1):
                print(f"    {i}. {score:.6f}")
            
            if non_zero == 0:
                print(f"  ✗ ERROR: All scores are 0.0!")
                all_passed = False
            else:
                print(f"  ✓ Scores look good (non-zero: {non_zero}/{len(scores_float)})")
                
        except Exception as e:
            print(f"  ✗ Retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            continue
    
    print(f"\n{'='*80}")
    if all_passed:
        print("✓ ALL TESTS PASSED - BM25 returns non-zero scores")
    else:
        print("✗ SOME TESTS FAILED - Check logs above")
    print(f"{'='*80}")
    
    return all_passed

if __name__ == "__main__":
    success = test_bm25_direct()
    sys.exit(0 if success else 1)

