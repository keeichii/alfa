#!/usr/bin/env python3
"""
Quick test script for first 10 questions and first 10 documents.
Tests full pipeline: retrieval + reranking with actual functions from other scripts.

Usage:
    python scripts/test_small.py --config configs/base.yaml
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import torch
except Exception:
    torch = None  # type: ignore

# Increase CSV field size limit
csv.field_size_limit(min(2**31 - 1, 10 * 1024 * 1024))

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "FlashRAG"))

import yaml
from src.retriever import HybridRetriever
from src.reranker import Reranker
from src.reranker_ensemble import RerankerEnsemble
from src.utils import logger, resolve_device, load_jsonl
from src.query_validator import validate_and_clean_questions


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_questions(csv_path: str, limit: int = 10) -> list[tuple[int, str]]:
    """Read first N questions from CSV file."""
    questions = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(questions) >= limit:
                break
            q_id = int(row.get("q_id", 0))
            query = row.get("query", "").strip()
            if q_id and query:
                questions.append((q_id, query))
    return questions


def load_corpus_sample(corpus_path: str, limit: int = 10) -> List[Dict]:
    """Load first N documents from corpus."""
    corpus = []
    if not Path(corpus_path).exists():
        logger.warning(f"Corpus file not found: {corpus_path}")
        return corpus
    
    for doc in load_jsonl(corpus_path):
        if len(corpus) >= limit:
            break
        corpus.append(doc)
    return corpus


def print_candidate(candidate: Dict, idx: int, show_details: bool = True):
    """Print candidate information."""
    print(f"\n  [{idx+1}] chunk_id={candidate.get('chunk_id', 'N/A')}")
    print(f"      doc_id={candidate.get('doc_id', 'N/A')}")
    print(f"      score={candidate.get('score', 0.0):.6f}")
    if show_details:
        print(f"      dense_score={candidate.get('dense_score', 0.0):.6f}")
        print(f"      bm25_score={candidate.get('bm25_score', 0.0):.6f}")
        print(f"      title_score={candidate.get('title_score', 0.0):.6f}")
        print(f"      text_score={candidate.get('text_score', 0.0):.6f}")
    contents = candidate.get('contents', '')[:100]
    if len(candidate.get('contents', '')) > 100:
        contents += "..."
    print(f"      contents_preview: {contents}")


def main():
    parser = argparse.ArgumentParser(description="Test pipeline on first 10 questions")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--n-questions", type=int, default=10, help="Number of questions to test")
    parser.add_argument("--n-docs", type=int, default=10, help="Number of corpus docs to show")
    parser.add_argument("--skip-reranker", action="store_true", help="Skip reranker (test retrieval only)")
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("SMALL TEST: Testing pipeline on first N questions")
    logger.info("=" * 80)
    
    # Load configs
    config = load_config(args.config)
    models_config = load_config(project_root / "configs" / "models.yaml")
    
    # Get paths
    questions_path = config["data"]["raw_questions"]
    corpus_path = config["data"]["corpus_jsonl"]
    bm25_dir = config["indexes"]["bm25_dir"]
    
    # Get parameters
    retrieval_config = config["retrieval"]
    k_retrieve = retrieval_config["k_retrieve"]
    k_final = retrieval_config["k_final"]
    k_rerank = retrieval_config.get("k_rerank", min(k_retrieve, 20))
    use_reranker = retrieval_config.get("use_reranker", True) and not args.skip_reranker
    
    reranker_config = models_config["reranker"]
    faiss_config = models_config.get("faiss", {})
    bm25_corpus_override = retrieval_config.get("bm25_corpus_path")
    
    # Read questions (first N)
    logger.info(f"\n[1] Reading first {args.n_questions} questions from {questions_path}")
    all_questions = read_questions(questions_path, limit=args.n_questions)
    logger.info(f"    Loaded {len(all_questions)} questions")
    
    # Validate and clean questions (using actual function)
    logger.info("\n[2] Validating and cleaning questions...")
    questions, validation_stats = validate_and_clean_questions(all_questions)
    logger.info(
        f"    Validation: total={validation_stats['total']}, "
        f"valid={validation_stats['valid']}, "
        f"cleaned={validation_stats['cleaned']}, "
        f"removed={validation_stats['removed']}"
    )
    
    if not questions:
        logger.error("No valid questions after validation!")
        return
    
    # Load corpus sample (for reference)
    logger.info(f"\n[3] Loading first {args.n_docs} documents from corpus...")
    corpus_sample = load_corpus_sample(corpus_path, limit=args.n_docs)
    logger.info(f"    Loaded {len(corpus_sample)} documents")
    if corpus_sample:
        logger.info(f"    Sample doc IDs: {[doc.get('id', 'N/A') for doc in corpus_sample[:5]]}")
    
    # Initialize retriever (using actual initialization)
    logger.info("\n[4] Initializing hybrid retriever...")
    faiss_path = Path(config["indexes"]["faiss_dir"])
    faiss_index_files = list(faiss_path.glob("*.index"))
    if not faiss_index_files:
        logger.error(f"\n✗ ERROR: No FAISS index found in {faiss_path}")
        logger.error("   Please build indexes first:")
        logger.error("   make build_index")
        logger.error("   or")
        logger.error("   make chunk_corpus && make build_index")
        return
    faiss_index_path = faiss_index_files[0]
    faiss_meta_path = faiss_path / "faiss_meta.json"
    if not faiss_meta_path.exists():
        faiss_meta_path = None
    
    # Check BM25 index
    bm25_path = Path(bm25_dir)
    bm25_index_files = list(bm25_path.rglob("*.json"))  # bm25s uses .json files
    if not bm25_index_files and not (bm25_path / "bm25").exists():
        logger.error(f"\n✗ ERROR: No BM25 index found in {bm25_path}")
        logger.error("   Please build indexes first:")
        logger.error("   make build_index")
        logger.error("   or")
        logger.error("   make chunk_corpus && make build_index")
        return
    
    bm25_corpus_path = bm25_corpus_override or str(Path(bm25_dir) / "chunks.jsonl")
    embeddings_config = models_config["embeddings"]
    
    device = resolve_device()
    logger.info(f"    Using device: {device}")
    
    try:
        retriever = HybridRetriever(
            faiss_index_path=str(faiss_index_path),
            faiss_meta_path=str(faiss_meta_path) if faiss_meta_path else None,
            bm25_index_dir=Path(bm25_dir),
            bm25_corpus_path=bm25_corpus_path,
            embedding_model_name=embeddings_config["model_name"],
            device=device,
            query_batch_size=retrieval_config.get("batch_size", 32),
            weight_dense=retrieval_config.get("weight_dense", 0.6),
            weight_bm25=retrieval_config.get("weight_bm25", 0.4),
            fusion_method=retrieval_config.get("fusion_method", "rrf"),
            rrf_k=retrieval_config.get("rrf_k", 60),
            enhance_numerics=retrieval_config.get("enhance_numerics", True),
            normalization_mode=retrieval_config.get("normalization_mode", "smart"),
            min_score_threshold=retrieval_config.get("min_score_threshold", 0.0),
            filter_by_document_type=retrieval_config.get("filter_by_document_type", False),
            prefer_table_chunks=retrieval_config.get("prefer_table_chunks", False),
            embedding_fp16=embeddings_config.get("use_fp16", True),
            faiss_use_gpu=faiss_config.get("use_gpu", True),
            faiss_use_fp16=faiss_config.get("use_fp16", True),
        )
        logger.info("    ✓ Retriever initialized successfully")
    except Exception as e:
        logger.error(f"    ✗ Failed to initialize retriever: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize reranker (if enabled)
    reranker = None
    if use_reranker:
        logger.info("\n[5] Initializing reranker...")
        try:
            reranker_ensemble_config = models_config.get("reranker_ensemble", {})
            if reranker_ensemble_config.get("enabled", False):
                reranker = RerankerEnsemble(
                    model_names=reranker_ensemble_config["models"],
                    weights=reranker_ensemble_config.get("weights"),
                    device=device,
                    batch_size=retrieval_config.get("reranker_batch_size", 24),
                    max_length=reranker_config.get("max_length", 512),
                    use_fp16=reranker_config.get("use_fp16", True),
                    second_pass=reranker_ensemble_config.get("second_pass", False),
                    second_pass_topk=reranker_ensemble_config.get("second_pass_topk", 20),
                )
                logger.info("    ✓ RerankerEnsemble initialized")
            else:
                reranker = Reranker(
                    model_name=reranker_config["model_name"],
                    device=device,
                    batch_size=retrieval_config.get("reranker_batch_size", 24),
                    max_length=reranker_config.get("max_length", 512),
                    use_fp16=reranker_config.get("use_fp16", True),
                )
                logger.info("    ✓ Reranker initialized")
        except Exception as e:
            logger.error(f"    ✗ Failed to initialize reranker: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Test retrieval
    logger.info(f"\n[6] Testing retrieval on {len(questions)} questions...")
    logger.info(f"    k_retrieve={k_retrieve}, k_rerank={k_rerank}, k_final={k_final}")
    
    queries = [q[1] for q in questions]
    q_ids = [q[0] for q in questions]
    
    try:
        start_time = time.time()
        batch_candidates, batch_scores = retriever.retrieve_batch(
            queries=queries,
            k=k_retrieve,
            return_scores=True,
        )
        retrieval_time = time.time() - start_time
        logger.info(f"    ✓ Retrieval completed in {retrieval_time:.2f}s")
        logger.info(f"    Retrieved {sum(len(cands) for cands in batch_candidates)} total candidates")
    except Exception as e:
        logger.error(f"    ✗ Retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test reranking (if enabled)
    if use_reranker and reranker:
        logger.info(f"\n[7] Testing reranking (top {k_rerank} candidates)...")
        try:
            start_time = time.time()
            reranked_results = []
            for query, candidates in zip(queries, batch_candidates):
                if not candidates:
                    reranked_results.append([])
                    continue
                # Limit to k_rerank for reranking
                top_candidates = candidates[:k_rerank]
                reranked = reranker.batch_rerank(
                    queries=[query],
                    candidates_list=[top_candidates],
                    top_k=k_final,
                    return_scores=False,
                )
                reranked_results.append(reranked[0] if reranked else [])
            rerank_time = time.time() - start_time
            logger.info(f"    ✓ Reranking completed in {rerank_time:.2f}s")
            final_candidates = reranked_results
        except Exception as e:
            logger.error(f"    ✗ Reranking failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to retrieval results
            final_candidates = [cands[:k_final] for cands in batch_candidates]
    else:
        logger.info("\n[7] Skipping reranking (disabled or not available)")
        final_candidates = [cands[:k_final] for cands in batch_candidates]
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    
    for idx, (q_id, query, candidates) in enumerate(zip(q_ids, queries, final_candidates)):
        print(f"\n{'='*80}")
        print(f"Question {idx+1}/{len(questions)}")
        print(f"  q_id: {q_id}")
        print(f"  query: {query}")
        print(f"  candidates: {len(candidates)}")
        
        if candidates:
            print("\n  Top candidates:")
            for cand_idx, candidate in enumerate(candidates[:5]):  # Show top 5
                print_candidate(candidate, cand_idx, show_details=True)
        else:
            print("  ⚠ No candidates retrieved!")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    total_candidates = sum(len(cands) for cands in final_candidates)
    avg_candidates = total_candidates / len(questions) if questions else 0
    logger.info(f"Total questions tested: {len(questions)}")
    logger.info(f"Total candidates retrieved: {total_candidates}")
    logger.info(f"Average candidates per question: {avg_candidates:.1f}")
    logger.info(f"Retrieval time: {retrieval_time:.2f}s")
    if use_reranker and reranker:
        logger.info(f"Reranking time: {rerank_time:.2f}s")
        logger.info(f"Total time: {retrieval_time + rerank_time:.2f}s")
    
    # Check for common issues
    logger.info("\n" + "=" * 80)
    logger.info("CHECKS")
    logger.info("=" * 80)
    
    issues = []
    for idx, (q_id, candidates) in enumerate(zip(q_ids, final_candidates)):
        if not candidates:
            issues.append(f"Question {q_id}: No candidates retrieved")
        elif len(candidates) < k_final:
            issues.append(f"Question {q_id}: Only {len(candidates)} candidates (expected {k_final})")
        
        # Check scores
        for cand in candidates:
            if cand.get("score", 0) == 0 and cand.get("dense_score", 0) == 0 and cand.get("bm25_score", 0) == 0:
                issues.append(f"Question {q_id}: Candidate {cand.get('chunk_id')} has all zero scores")
                break
    
    if issues:
        logger.warning(f"Found {len(issues)} potential issues:")
        for issue in issues[:10]:  # Show first 10
            logger.warning(f"  - {issue}")
        if len(issues) > 10:
            logger.warning(f"  ... and {len(issues) - 10} more")
    else:
        logger.info("✓ No obvious issues detected")
    
    logger.info("\n" + "=" * 80)
    logger.info("Test completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

