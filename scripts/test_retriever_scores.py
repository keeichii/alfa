#!/usr/bin/env python3
"""
Test script to verify that retrievers return non-zero scores for relevant documents.

This script:
1. Loads BM25 and Dense retrievers
2. Performs searches on test queries
3. Checks that returned scores are non-zero
4. Outputs detailed score information for analysis
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "FlashRAG"))

import yaml
from src.retriever import HybridRetriever
from src.utils import logger, resolve_device

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def test_retriever_scores(
    config_path: str = "configs/base.yaml",
    test_queries: List[str] = None,
    k: int = 10
):
    """Test that retrievers return non-zero scores."""
    
    if test_queries is None:
        test_queries = [
            "Номер счета",
            "Где узнать бик и счёт",
            "Мне не приходят коды для подтверждения операции",
            "Кэшбэк",
            "Оформила рассрочку",
        ]
    
    logger.info("=" * 80)
    logger.info("TESTING RETRIEVER SCORES")
    logger.info("=" * 80)
    
    # Load configs
    config = load_config(config_path)
    models_config = load_config(project_root / "configs" / "models.yaml")
    
    # Get paths
    faiss_path = Path(config["indexes"]["faiss_dir"])
    faiss_index_files = list(faiss_path.glob("*.index"))
    if not faiss_index_files:
        logger.error(f"ERROR: No FAISS index found in {faiss_path}")
        return False
    faiss_index_path = faiss_index_files[0]
    faiss_meta_path = faiss_path / "faiss_meta.json"
    if not faiss_meta_path.exists():
        faiss_meta_path = None
    
    bm25_dir = config["indexes"]["bm25_dir"]
    bm25_corpus_path = str(Path(bm25_dir) / "chunks.jsonl")
    
    # Get parameters
    retrieval_config = config["retrieval"]
    embeddings_config = models_config["embeddings"]
    faiss_config = models_config.get("faiss", {})
    
    device = resolve_device()
    
    # Initialize retriever
    logger.info("\n[1] Initializing HybridRetriever...")
    retriever = HybridRetriever(
        faiss_index_path=str(faiss_index_path),
        faiss_meta_path=str(faiss_meta_path) if faiss_meta_path else None,
        bm25_index_dir=str(bm25_dir),
        bm25_corpus_path=bm25_corpus_path,
        embedding_model_name=embeddings_config["model_name"],
        device=device,
        query_batch_size=retrieval_config.get("retrieval_batch_size", 32),
        weight_dense=retrieval_config.get("weight_dense", 0.6),
        weight_bm25=retrieval_config.get("weight_bm25", 0.4),
        fusion_method=retrieval_config.get("fusion_method", "rrf"),
        rrf_k=retrieval_config.get("rrf_k", 60),
        enhance_numerics=retrieval_config.get("enhance_numerics", True),
        normalization_mode=config["text_processing"].get("normalization_mode", "smart"),
        min_score_threshold=retrieval_config.get("min_score_threshold", 0.0),
        filter_by_document_type=retrieval_config.get("filter_by_document_type", False),
        prefer_table_chunks=retrieval_config.get("prefer_table_chunks", False),
        embedding_fp16=embeddings_config.get("use_fp16", True),
        faiss_use_gpu=faiss_config.get("faiss_gpu", True),
        faiss_use_fp16=faiss_config.get("use_fp16", True),
    )
    logger.info("✓ Retriever initialized.")
    
    # Test each query
    logger.info(f"\n[2] Testing {len(test_queries)} queries with k={k}...")
    all_passed = True
    
    for query_idx, query in enumerate(test_queries, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Query {query_idx}/{len(test_queries)}: '{query}'")
        logger.info(f"{'='*80}")
        
        # Retrieve candidates
        try:
            candidates, score_lists = retriever.retrieve_batch(
                [query],
                k=k,
                return_scores=True,
                fusion_method="rrf"
            )
            candidates = candidates[0] if candidates else []
            score_lists = score_lists[0] if score_lists else {}
        except Exception as e:
            logger.error(f"✗ Retrieval failed: {e}")
            all_passed = False
            continue
        
        if not candidates:
            logger.warning(f"✗ No candidates returned for query: '{query}'")
            all_passed = False
            continue
        
        logger.info(f"✓ Retrieved {len(candidates)} candidates")
        
        # Analyze scores
        bm25_scores = []
        dense_scores = []
        rrf_scores = []
        source_scores_dicts = []
        
        for i, candidate in enumerate(candidates[:k], 1):
            bm25_score = candidate.get("bm25_score")
            dense_score = candidate.get("dense_score")
            rrf_score = candidate.get("rrf_score")
            source_scores = candidate.get("source_scores", {})
            
            # Collect scores
            if bm25_score is not None:
                bm25_scores.append(float(bm25_score))
            if dense_score is not None:
                dense_scores.append(float(dense_score))
            if rrf_score is not None:
                rrf_scores.append(float(rrf_score))
            if source_scores:
                source_scores_dicts.append(source_scores)
            
            # Log top candidates
            if i <= 5:
                logger.info(f"\n  Candidate {i}:")
                logger.info(f"    chunk_id: {candidate.get('chunk_id', 'N/A')}")
                logger.info(f"    doc_id: {candidate.get('doc_id', 'N/A')}")
                logger.info(f"    score: {candidate.get('score', 'N/A')}")
                logger.info(f"    dense_score: {dense_score}")
                logger.info(f"    bm25_score: {bm25_score}")
                logger.info(f"    rrf_score: {rrf_score}")
                logger.info(f"    source_scores: {source_scores}")
                logger.info(f"    contents_preview: {candidate.get('contents', '')[:100]}...")
        
        # Statistics
        logger.info(f"\n  Score Statistics:")
        if bm25_scores:
            logger.info(f"    BM25 scores: min={min(bm25_scores):.6f}, max={max(bm25_scores):.6f}, "
                       f"mean={sum(bm25_scores)/len(bm25_scores):.6f}, "
                       f"non-zero={sum(1 for s in bm25_scores if s != 0.0)}/{len(bm25_scores)}")
        else:
            logger.warning(f"    BM25 scores: NONE FOUND (all None or missing)")
        
        if dense_scores:
            logger.info(f"    Dense scores: min={min(dense_scores):.6f}, max={max(dense_scores):.6f}, "
                       f"mean={sum(dense_scores)/len(dense_scores):.6f}, "
                       f"non-zero={sum(1 for s in dense_scores if s != 0.0)}/{len(dense_scores)}")
        else:
            logger.warning(f"    Dense scores: NONE FOUND (all None or missing)")
        
        if rrf_scores:
            logger.info(f"    RRF scores: min={min(rrf_scores):.6f}, max={max(rrf_scores):.6f}, "
                       f"mean={sum(rrf_scores)/len(rrf_scores):.6f}")
        
        # Check for issues
        issues = []
        if not bm25_scores or all(s == 0.0 for s in bm25_scores):
            issues.append("All BM25 scores are 0.0 or missing")
        if not dense_scores or all(s == 0.0 for s in dense_scores):
            issues.append("All Dense scores are 0.0 or missing")
        if issues:
            logger.error(f"  ✗ ISSUES DETECTED:")
            for issue in issues:
                logger.error(f"      - {issue}")
            all_passed = False
        else:
            logger.info(f"  ✓ Scores look good!")
        
        # Check raw score_lists
        if score_lists:
            logger.info(f"\n  Raw score_lists from retriever:")
            for source, scores in score_lists.items():
                if isinstance(scores, list) and len(scores) > 0:
                    non_zero = sum(1 for s in scores if s != 0.0)
                    logger.info(f"    {source}: {len(scores)} scores, "
                               f"non-zero={non_zero}/{len(scores)}, "
                               f"min={min(scores):.6f}, max={max(scores):.6f}")
                else:
                    logger.warning(f"    {source}: empty or invalid scores")
    
    # Final summary
    logger.info(f"\n{'='*80}")
    if all_passed:
        logger.info("✓ ALL TESTS PASSED - Retrievers return non-zero scores")
    else:
        logger.error("✗ SOME TESTS FAILED - Check logs above for details")
    logger.info(f"{'='*80}")
    
    return all_passed

def main():
    parser = argparse.ArgumentParser(description="Test retriever scores")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--k", type=int, default=10, help="Number of candidates to retrieve")
    parser.add_argument("--queries", nargs="+", help="Test queries (default: predefined set)")
    args = parser.parse_args()
    
    success = test_retriever_scores(
        config_path=args.config,
        test_queries=args.queries,
        k=args.k
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

