#!/usr/bin/env python3
"""
Evaluate retrieval pipeline: hybrid retrieval + reranking + metrics.

Usage:
    python scripts/eval.py --config configs/base.yaml
"""
import argparse
import json
import sys
import time
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "FlashRAG"))

import yaml
from src.retriever import HybridRetriever
from src.reranker import Reranker
from src.evaluator import RetrievalEvaluator
from src.failure_logger import FailureLogger
from src.utils import logger, get_timestamp


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_questions(csv_path: str) -> list[tuple[int, str]]:
    """Read questions from CSV file."""
    import csv
    questions = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_id = int(row.get("q_id", 0))
            query = row.get("query", "").strip()
            if q_id and query:
                questions.append((q_id, query))
    return questions


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval pipeline")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--ground_truth", help="Path to ground truth JSON file (optional)")
    args = parser.parse_args()
    
    # load configs
    config = load_config(args.config)
    models_config = load_config(project_root / "configs" / "models.yaml")
    
    # get paths
    questions_path = config["data"]["raw_questions"]
    faiss_dir = config["indexes"]["faiss_dir"]
    bm25_dir = config["indexes"]["bm25_dir"]
    reports_dir = config["outputs"]["reports_dir"]
    logs_dir = config["outputs"]["logs_dir"]
    
    # get parameters
    retrieval_config = config["retrieval"]
    k_retrieve = retrieval_config["k_retrieve"]
    k_final = retrieval_config["k_final"]
    use_reranker = retrieval_config.get("use_reranker", True)
    weight_dense = retrieval_config["hybrid_weight_dense"]
    weight_bm25 = retrieval_config["hybrid_weight_bm25"]
    
    embeddings_config = models_config["embeddings"]
    reranker_config = models_config["reranker"]
    
    # read questions
    logger.info(f"Reading questions from {questions_path}")
    questions = read_questions(questions_path)
    logger.info(f"Loaded {len(questions)} questions")
    
    # find FAISS index file
    faiss_path = Path(faiss_dir)
    faiss_index_files = list(faiss_path.glob("*.index"))
    if not faiss_index_files:
        raise FileNotFoundError(f"No FAISS index found in {faiss_dir}")
    faiss_index_path = faiss_index_files[0]
    faiss_meta_path = faiss_path / "faiss_meta.json"
    
    if not faiss_meta_path.exists():
        logger.info(f"Using FAISS metadata from {faiss_meta_path}")
    else:
        logger.warning(f"FAISS metadata not found at {faiss_meta_path}, will try to infer from chunks")
        faiss_meta_path = None
    
    # initialize retriever
    logger.info("Initializing hybrid retriever...")
    retriever = HybridRetriever(
        faiss_index_path=str(faiss_index_path),
        faiss_meta_path=str(faiss_meta_path) if faiss_meta_path else None,
        bm25_index_path=bm25_dir,
        embedding_model_name=embeddings_config["model_name"],
        weight_dense=weight_dense,
        weight_bm25=weight_bm25,
        device=embeddings_config["device"],
        normalize_embeddings=embeddings_config.get("normalize_embeddings", True)
    )
    
    # initialize reranker if enabled
    reranker = None
    if use_reranker:
        logger.info("Initializing reranker...")
        reranker = Reranker(
            model_name=reranker_config["model_name"],
            device=reranker_config["device"],
            batch_size=reranker_config["batch_size"]
        )
    
    # initialize evaluator
    evaluator = RetrievalEvaluator()
    if args.ground_truth:
        evaluator.load_ground_truth(args.ground_truth)
    
    # initialize failure logger
    failure_logger = FailureLogger(evaluator.ground_truth)
    
    # process questions
    logger.info("Processing questions...")
    results = {}
    candidate_logs = []
    
    start_time = time.time()
    for q_id, query in questions:
        # retrieve with fusion method from config
        fusion_method = retrieval_config.get("fusion_method", "weighted")
        candidates, scores = retriever.retrieve(
            query, 
            k=k_retrieve, 
            return_scores=True,
            fusion_method=fusion_method
        )
        
        # log candidates before reranking
        log_entry_before = {
            "q_id": q_id,
            "query": query,
            "stage": "before_rerank",
            "candidates": [
                {
                    "chunk_id": c["chunk_id"],
                    "doc_id": int(c["doc_id"]),
                    "score": c["score"],
                    "dense_score": c.get("dense_score", 0.0),
                    "bm25_score": c.get("bm25_score", 0.0)
                }
                for c in candidates[:k_retrieve]
            ]
        }
        candidate_logs.append(log_entry_before)
        
        # rerank if enabled
        candidates_after_rerank = None
        if reranker and candidates:
            candidates_before_rerank_list = candidates.copy()
            candidates = reranker.rerank(query, candidates, top_k=k_final, return_scores=False)
            candidates_after_rerank = candidates
            
            # log candidates after reranking
            log_entry_after = {
                "q_id": q_id,
                "query": query,
                "stage": "after_rerank",
                "candidates": [
                    {
                        "chunk_id": c["chunk_id"],
                        "doc_id": int(c["doc_id"]),
                        "score": c.get("rerank_score", c.get("score", 0.0))
                    }
                    for c in candidates[:k_final]
                ]
            }
            candidate_logs.append(log_entry_after)
        else:
            candidates_before_rerank_list = candidates
        
        # extract web_ids (doc_ids)
        web_ids = [int(c["doc_id"]) for c in candidates[:k_final]]
        results[q_id] = web_ids
        
        # log failures
        failure_logger.log_retrieval_failure(
            q_id=q_id,
            query=query,
            retrieved_web_ids=web_ids,
            candidates_before_rerank=candidates_before_rerank_list,
            candidates_after_rerank=candidates_after_rerank,
            k=k_final
        )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Processed {len(questions)} questions in {elapsed_time:.2f} seconds")
    
    # evaluate if ground truth available
    metrics = {}
    if evaluator.ground_truth:
        logger.info("Computing metrics...")
        eval_metrics = config["evaluation"]["metrics"]
        metrics = evaluator.evaluate(results, metrics=eval_metrics)
        logger.info(f"Hit@5: {metrics.get('hit@5', 0):.4f}")
        logger.info(f"Recall@5: {metrics.get('recall@5', 0):.4f}")
        logger.info(f"MRR: {metrics.get('mrr', 0):.4f}")
    else:
        logger.warning("No ground truth provided, skipping evaluation")
    
    # save report
    timestamp = get_timestamp()
    report_path = Path(reports_dir) / f"report_{timestamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": timestamp,
        "config": {
            "k_retrieve": k_retrieve,
            "k_final": k_final,
            "use_reranker": use_reranker,
            "hybrid_weight_dense": weight_dense,
            "hybrid_weight_bm25": weight_bm25,
            "embedding_model": embeddings_config["model_name"],
            "reranker_model": reranker_config["model_name"] if use_reranker else None
        },
        "metrics": metrics,
        "num_queries": len(questions),
        "processing_time_seconds": elapsed_time
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Report saved to {report_path}")
    
    # save candidate logs
    if config["evaluation"].get("save_candidate_logs", True):
        log_path = Path(logs_dir) / f"candidates_{timestamp}.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            for log_entry in candidate_logs:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        logger.info(f"Candidate logs saved to {log_path}")
    
    # save failure logs
    failure_summary = failure_logger.get_failure_summary()
    if failure_summary["total"] > 0:
        failure_path = Path(logs_dir) / f"failures_{timestamp}.json"
        failure_logger.save_failures(str(failure_path))
        logger.warning(f"Found {failure_summary['total']} retrieval failures")
        logger.info(f"Failure summary: {failure_summary}")
        
        # add failure summary to report
        report["failure_summary"] = failure_summary
    
    return results, metrics


if __name__ == "__main__":
    main()
