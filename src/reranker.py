"""Cross-encoder reranker for reordering retrieval candidates."""
import logging
from typing import List, Dict, Optional, Tuple

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    raise ImportError("sentence-transformers is required for reranking")

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from src.utils import logger, supports_fp16


class Reranker:
    """
    Cross-encoder reranker for reordering retrieval candidates.
    
    Algorithm: Uses cross-encoder to score query-document pairs
    Time Complexity: O(n * m) where n is candidates, m is model forward pass
    Space Complexity: O(n) for storing scores
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
        use_fp16: bool = False,
    ):
        """
        Initialize reranker.
        
        Args:
            model_name: Name of cross-encoder model
            device: Device for model ("cpu" or "cuda")
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        logger.info(f"Loading reranker model: {model_name}")
        self.device = device
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and supports_fp16(device)
        if self.use_fp16 and torch is not None:
            try:
                self.model.model.half()  # type: ignore[attr-defined]
                logger.info("Reranker weights converted to fp16")
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to convert reranker to fp16: %s", exc)
                self.use_fp16 = False
        logger.info(f"Reranker initialized on {device} (fp16={self.use_fp16})")
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
        return_scores: bool = False
    ) -> List[Dict] | Tuple[List[Dict], List[float]]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: Query string
            candidates: List of candidate chunks (must have "contents" or "text" field)
            top_k: Number of top candidates to return
            return_scores: Whether to return reranking scores
            
        Returns:
            Reranked candidates, or (candidates, scores) if return_scores
        """
        if not candidates:
            return [] if not return_scores else ([], [])
        
        # prepare pairs for cross-encoder
        pairs = []
        for candidate in candidates:
            # get text content
            text = candidate.get("contents", candidate.get("text", ""))
            if not text:
                # fallback: use chunk_id
                text = str(candidate.get("chunk_id", ""))
            pairs.append([query, text])
        
        # score pairs in batches
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            batch_scores = self.model.predict(batch, show_progress_bar=False)
            scores.extend(batch_scores.tolist())
        
        # combine candidates with scores
        scored_candidates = list(zip(candidates, scores))
        
        # sort by score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # get top-k
        top_candidates = [cand for cand, _ in scored_candidates[:top_k]]
        
        # add rerank scores to candidates
        for i, (cand, score) in enumerate(scored_candidates[:top_k]):
            top_candidates[i]["rerank_score"] = float(score)
            if "score" not in top_candidates[i]:
                top_candidates[i]["score"] = float(score)
        
        if return_scores:
            top_scores = [float(score) for _, score in scored_candidates[:top_k]]
            return top_candidates, top_scores
        return top_candidates
    
    def batch_rerank(
        self,
        queries: List[str],
        candidates_list: List[List[Dict]],
        top_k: int = 5,
        return_scores: bool = False
    ) -> List[List[Dict]] | Tuple[List[List[Dict]], List[List[float]]]:
        """
        Batch rerank for multiple queries.
        
        Args:
            queries: List of query strings
            candidates_list: List of candidate lists (one per query)
            top_k: Number of top candidates per query
            return_scores: Whether to return scores
            
        Returns:
            List of reranked candidates per query, or (results, scores_list) if return_scores
        """
        if not queries:
            return ([], []) if return_scores else []
        
        pair_inputs: List[List[str]] = []
        candidate_counts: List[int] = []
        
        for query, candidates in zip(queries, candidates_list):
            count = 0
            for candidate in candidates:
                text = candidate.get("contents", candidate.get("text", ""))
                if not text:
                    text = str(candidate.get("chunk_id", ""))
                pair_inputs.append([query, text])
                count += 1
            candidate_counts.append(count)
        
        if not pair_inputs:
            empty_results = [[] for _ in queries]
            return (empty_results, [[] for _ in queries]) if return_scores else empty_results
        
        scores: List[float] = []
        for i in range(0, len(pair_inputs), self.batch_size):
            batch_pairs = pair_inputs[i:i + self.batch_size]
            batch_scores = self.model.predict(batch_pairs, show_progress_bar=False)
            scores.extend(batch_scores.tolist())
        
        results: List[List[Dict]] = []
        scores_list: List[List[float]] = []
        score_idx = 0
        
        for candidates, count in zip(candidates_list, candidate_counts):
            candidate_scores = scores[score_idx:score_idx + count]
            score_idx += count
            
            scored_candidates = []
            for candidate, score in zip(candidates, candidate_scores):
                candidate_copy = dict(candidate)
                candidate_copy["rerank_score"] = float(score)
                if "score" not in candidate_copy:
                    candidate_copy["score"] = float(score)
                scored_candidates.append((candidate_copy, float(score)))
            
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [cand for cand, _ in scored_candidates[:top_k]]
            results.append(top_candidates)
            if return_scores:
                top_scores = [score for _, score in scored_candidates[:top_k]]
                scores_list.append(top_scores)
        
        if return_scores:
            return results, scores_list
        return results

