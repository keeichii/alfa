"""FlashRAG-backed hybrid retriever."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLASHRAG_PATH = PROJECT_ROOT / "FlashRAG"
if str(FLASHRAG_PATH) not in sys.path:
    sys.path.insert(0, str(FLASHRAG_PATH))

from flashrag.retriever.retriever import MultiRetrieverRouter  # type: ignore

from src.table_processor import enhance_query_for_numerics
from src.utils import logger, resolve_device


class HybridRetriever:
    """Wrapper around FlashRAG's `MultiRetrieverRouter` tuned for bm25s + dense fusion."""

    def __init__(
        self,
        faiss_index_path: str,
        faiss_meta_path: Optional[str],  # kept for backwards compatibility; unused
        bm25_index_dir: str,
        bm25_corpus_path: str,
        embedding_model_name: str,
        device: str = "cuda",
        normalize_embeddings: bool = True,  # kept for API compatibility
        query_batch_size: int = 32,
        weight_dense: float = 0.6,
        weight_bm25: float = 0.4,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        enhance_numerics: bool = True,
        normalization_mode: str = "smart",  # normalization mode for queries
        min_score_threshold: float = 0.0,  # minimum score threshold before reranking
        filter_by_document_type: bool = False,  # filter candidates by document type
        prefer_table_chunks: bool = False,  # prefer chunks with tables for numeric queries
        embedding_fp16: bool = True,
        faiss_use_gpu: bool = True,
        faiss_gpu_device: Optional[int] = None,  # kept for API compatibility
        faiss_use_fp16: bool = True,
    ) -> None:
        del faiss_meta_path  # flashrag handles corpus metadata internally
        del normalize_embeddings
        del faiss_gpu_device

        self.device = resolve_device(device)
        self.embedding_model_name = embedding_model_name
        self.query_batch_size = max(1, int(query_batch_size))
        self.fusion_method = fusion_method
        self.rrf_k = int(rrf_k)
        self.weight_dense = float(weight_dense)
        self.weight_bm25 = float(weight_bm25)
        self.enhance_numerics = enhance_numerics
        self.normalization_mode = normalization_mode
        self.min_score_threshold = float(min_score_threshold)
        self.filter_by_document_type = bool(filter_by_document_type)
        self.prefer_table_chunks = bool(prefer_table_chunks)
        self.embedding_fp16 = bool(embedding_fp16)
        self.faiss_use_gpu = bool(faiss_use_gpu)
        self.faiss_use_fp16 = bool(faiss_use_fp16)
        self.default_topk = 200  # Increased to support k_retrieve=180

        self.faiss_index_path = Path(faiss_index_path)
        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.faiss_index_path}")
        
        # Load FAISS metadata to get pooling_method and other settings
        self.faiss_pooling_method = None
        faiss_meta_path = self.faiss_index_path.parent / "faiss_meta.json"
        if faiss_meta_path.exists():
            try:
                import json
                with open(faiss_meta_path, "r", encoding="utf-8") as f:
                    faiss_meta = json.load(f)
                    self.faiss_pooling_method = faiss_meta.get("pooling_method", "mean")
            except Exception as e:
                logger.warning(f"Could not load FAISS metadata: {e}, using default pooling_method='mean'")
                self.faiss_pooling_method = "mean"
        else:
            logger.warning(f"FAISS metadata not found at {faiss_meta_path}, using default pooling_method='mean'")
            self.faiss_pooling_method = "mean"

        self.bm25_index_dir = Path(bm25_index_dir)
        if not self.bm25_index_dir.exists():
            raise FileNotFoundError(f"BM25 index directory not found: {self.bm25_index_dir}")

        self.bm25_corpus_path = Path(bm25_corpus_path)
        if not self.bm25_corpus_path.exists():
            raise FileNotFoundError(f"BM25 corpus JSONL not found: {self.bm25_corpus_path}")

        router_config = self._build_router_config()
        self.router = MultiRetrieverRouter(router_config)
        # Set RRF k parameter for the router
        self.router.rrf_k = self.rrf_k
        self._retriever_map = {getattr(r, "source_name", r.retrieval_method): r for r in self.router.retriever_list}
        logger.info(
            "Hybrid retriever initialised via FlashRAG: dense=%s, bm25=%s, fusion=%s",
            embedding_model_name,
            self._resolve_bm25_index_path(),
            self.fusion_method,
        )

    def retrieve(
        self,
        query: str,
        k: int = 50,
        return_scores: bool = False,
        fusion_method: Optional[str] = None,
    ):
        results, scores = self.retrieve_batch([query], k=k, return_scores=True, fusion_method=fusion_method)
        if return_scores:
            return (results[0] if results else [], scores[0] if scores else {})
        return results[0] if results else []

    def retrieve_batch(
        self,
        queries: List[str],
        k: int = 50,
        return_scores: bool = False,
        fusion_method: Optional[str] = None,
    ):
        if not queries:
            return ([], []) if return_scores else []

        topk = max(1, int(k))
        method = fusion_method or self.fusion_method
        prepared_queries = self._prepare_queries(queries)

        self.router.final_topk = topk
        self.router.merge_method = method
        self._update_retriever_topk(topk)

        if method == "weighted":
            self.router.weights = self._normalized_weights()

        results, score_lists = self.router.batch_search(prepared_queries, num=topk, return_score=True)

        processed_results: List[List[Dict[str, Any]]] = []
        processed_scores: List[Dict[str, Dict[str, float]]] = []

        for idx, docs in enumerate(results):
            raw_scores: List[float] = []
            # CRITICAL: score_lists from router.batch_search after RRF merge contains RRF scores, not source scores
            # Source scores should already be in documents (bm25_score, dense_score fields)
            # But we can use score_lists as fallback for the primary score
            if idx < len(score_lists):
                candidate_scores = score_lists[idx]
                if isinstance(candidate_scores, list):
                    raw_scores = candidate_scores
            candidates: List[Dict[str, Any]] = []
            # Get original query (before normalization) for numeric detection
            original_query = queries[idx] if idx < len(queries) else ""
            is_numeric_query = self._is_numeric_query(original_query)
            
            # DEBUG: Check if documents have source scores before processing
            if docs and len(docs) > 0:
                sample_doc = docs[0]
                has_bm25 = "bm25_score" in sample_doc
                has_dense = "dense_score" in sample_doc
                has_source_scores = "source_scores" in sample_doc
                logger.debug(
                    f"Query {idx}: Sample doc has bm25_score={has_bm25}, "
                    f"dense_score={has_dense}, source_scores={has_source_scores}"
                )
                if has_bm25:
                    logger.debug(f"  Sample bm25_score value: {sample_doc.get('bm25_score')}")
                if has_dense:
                    logger.debug(f"  Sample dense_score value: {sample_doc.get('dense_score')}")
            
            for doc_idx, doc in enumerate(docs[:topk]):
                # CRITICAL: Use RRF score from score_lists as fallback, but prefer source scores from doc
                fallback_score = raw_scores[doc_idx] if doc_idx < len(raw_scores) else None
                # Use original query (before normalization) for title/text scoring
                candidate = self._to_candidate(doc, fallback_score, query=original_query)
                
                # Filter by minimum score threshold if enabled
                if self.min_score_threshold > 0.0 and candidate["score"] < self.min_score_threshold:
                    continue
                
                # Filter/prefer by document type if enabled
                if self.filter_by_document_type or self.prefer_table_chunks:
                    has_table = candidate.get("has_table", False) or self._detect_table_in_candidate(candidate)
                    candidate["has_table"] = has_table
                    
                    # Prefer table chunks for numeric queries
                    if self.prefer_table_chunks and is_numeric_query and has_table:
                        candidate["score"] = candidate["score"] * 1.2  # Boost score by 20%
                
                candidates.append(candidate)
            processed_results.append(candidates)
            if return_scores:
                processed_scores.append(
                    {
                        "hybrid": {cand["chunk_id"]: cand["score"] for cand in candidates},
                        "dense": {cand["chunk_id"]: cand["dense_score"] for cand in candidates},
                        "bm25": {cand["chunk_id"]: cand["bm25_score"] for cand in candidates},
                    }
                )

        if return_scores:
            return processed_results, processed_scores
        return processed_results

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _prepare_queries(self, queries: List[str]) -> List[str]:
        from src.text_processor import normalize_for_retrieval
        from src.query_validator import clean_query, is_valid_query
        
        prepared = []
        for query in queries:
            if not query or not query.strip():
                # Empty query - use placeholder to avoid BM25 errors
                prepared.append(" ")
                continue
            
            original_query = query.strip()
            
            # Clean query first (remove noise, excessive punctuation)
            cleaned_query = clean_query(original_query)
            if cleaned_query is None or not is_valid_query(cleaned_query):
                # Invalid query after cleaning - use placeholder
                logger.debug(f"Skipping invalid query: {original_query[:50]}...")
                prepared.append(" ")
                continue
            
            query = cleaned_query
            
            # Apply numeric enhancement if enabled
            if self.enhance_numerics:
                try:
                    query = enhance_query_for_numerics(query)
                except Exception:
                    pass  # Keep cleaned query if enhancement fails
            
            # Normalize query to match corpus normalization
            try:
                normalized = normalize_for_retrieval(query, mode=self.normalization_mode)
            except Exception:
                normalized = query  # Fallback to query if normalization fails
            
            # Ensure normalized query is not empty and has at least one character
            if not normalized or not normalized.strip() or len(normalized.strip()) == 0:
                # Fallback to cleaned query if normalization removed everything
                if cleaned_query and cleaned_query.strip():
                    prepared.append(cleaned_query.strip())
                else:
                    # CRITICAL FIX: Use a non-empty, non-whitespace placeholder if all else fails
                    # A single space " " can be tokenized into empty tokens by some tokenizers,
                    # leading to bm25_score = 0.0. Use a known valid token or a short, generic word.
                    prepared.append("запрос")  # Use a generic Russian word as ultimate fallback
            else:
                prepared.append(normalized.strip())
        return prepared

    def _normalized_weights(self) -> Dict[str, float]:
        raw = {
            "dense": max(0.0, self.weight_dense),
            "bm25": max(0.0, self.weight_bm25),
        }
        total = sum(raw.values())
        if total <= 0:
            return {name: 0.5 for name in raw}
        return {name: value / total for name, value in raw.items()}

    def _update_retriever_topk(self, topk: int) -> None:
        for retriever in self.router.retriever_list:
            retriever.topk = topk
            if hasattr(retriever, "batch_size"):
                retriever.batch_size = self.query_batch_size

    def _build_router_config(self) -> Dict[str, Any]:
        merge_method = self.fusion_method if self.fusion_method != "weighted" else "weighted"
        weights = self._normalized_weights() if merge_method == "weighted" else {}

        bm25_config = {
            "name": "bm25",
            "retrieval_method": "bm25",
            "retrieval_model_path": "",  # Not used for BM25, but required by FlashRAG
            "retrieval_topk": self.default_topk,
            "retrieval_batch_size": self.query_batch_size,
            "retrieval_query_max_length": 256,
            "retrieval_use_fp16": False,
            "retrieval_pooling_method": "mean",
            "instruction": None,
            "index_path": str(self._resolve_bm25_index_path()),
            "corpus_path": str(self.bm25_corpus_path),
            "bm25_backend": "bm25s",
            "save_retrieval_cache": False,
            "use_retrieval_cache": False,
            "retrieval_cache_path": None,
            "use_reranker": False,
            "silent_retrieval": True,
        }

        dense_config = {
            "name": "dense",
            "retrieval_method": self.embedding_model_name,
            "retrieval_model_path": self.embedding_model_name,
            "retrieval_topk": self.default_topk,
            "retrieval_batch_size": self.query_batch_size,
            "retrieval_query_max_length": 256,  # Queries are shorter than documents
            "retrieval_use_fp16": self.embedding_fp16 or self.faiss_use_fp16,
            "retrieval_pooling_method": self.faiss_pooling_method,  # Use pooling_method from index metadata
            "instruction": None,  # Auto-detect (will use "query: " for E5, "passage: " was used during indexing)
            "index_path": str(self.faiss_index_path),
            "corpus_path": str(self.bm25_corpus_path),
            "save_retrieval_cache": False,
            "use_retrieval_cache": False,
            "retrieval_cache_path": None,
            "use_reranker": False,
            "silent_retrieval": True,
            "use_sentence_transformer": True,
            "faiss_gpu": self.faiss_use_gpu,
        }

        multi_retriever_setting = {
            "merge_method": merge_method,
            "topk": self.default_topk,
            "retriever_list": [bm25_config, dense_config],
        }
        if merge_method == "weighted":
            multi_retriever_setting["weights"] = weights
        elif merge_method == "rrf":
            # FlashRAG uses k=60 by default, but we allow customization
            # The rrf_k is passed via the router's internal rrf_merge method
            pass  # RRF k is handled internally by FlashRAG
        
        return {
            "device": self.device,
            "silent_retrieval": True,
            "multi_retriever_setting": multi_retriever_setting,
        }

    def _resolve_bm25_index_path(self) -> Path:
        candidate = self.bm25_index_dir / "bm25"
        if candidate.exists():
            return candidate
        return self.bm25_index_dir

    def _compute_title_text_scores(self, doc: Dict[str, Any], query: str) -> tuple[float, float]:
        """
        Compute separate scores for title and text portions of the document.
        
        Returns:
            (title_score, text_score) - scores for title and text portions
        """
        title = doc.get("title", "").strip()
        text = doc.get("text", "").strip()
        contents = doc.get("contents", "").strip()
        
        # If contents is in format "title\ntext", split it
        if not title and contents:
            parts = contents.split("\n", 1)
            if len(parts) == 2:
                title = parts[0].strip()
                text = parts[1].strip()
            else:
                text = contents
        
        # Get base scores
        source_scores = doc.get("source_scores", {})
        dense_score = float(doc.get("dense_score", source_scores.get("dense", 0.0)))
        bm25_score = float(doc.get("bm25_score", source_scores.get("bm25", 0.0)))
        base_score = max(dense_score, bm25_score) if (dense_score > 0 or bm25_score > 0) else 0.0
        
        if not query or base_score == 0.0:
            return (0.0, 0.0)
        
        # Simple heuristic: check query term overlap with title vs text
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        title_lower = title.lower() if title else ""
        text_lower = text.lower() if text else ""
        
        # Count matching terms
        title_matches = sum(1 for term in query_terms if term in title_lower) if title_lower else 0
        text_matches = sum(1 for term in query_terms if term in text_lower) if text_lower else 0
        
        total_terms = len(query_terms)
        if total_terms == 0:
            return (base_score * 0.5, base_score * 0.5)
        
        # Compute title and text scores based on term overlap
        title_ratio = title_matches / total_terms if total_terms > 0 else 0.0
        text_ratio = text_matches / total_terms if total_terms > 0 else 0.0
        
        # Normalize ratios (if both are 0, split evenly)
        total_ratio = title_ratio + text_ratio
        if total_ratio == 0:
            title_ratio = 0.5
            text_ratio = 0.5
        else:
            title_ratio = title_ratio / total_ratio
            text_ratio = text_ratio / total_ratio
        
        # Apply base score with ratios
        title_score = base_score * title_ratio
        text_score = base_score * text_ratio
        
        return (title_score, text_score)

    def _to_candidate(self, doc: Dict[str, Any], fallback_score: Optional[float] = None, query: Optional[str] = None) -> Dict[str, Any]:
        chunk_id = str(
            doc.get(
                "chunk_id",
                doc.get("id", doc.get("doc_id", doc.get("document_id", "unknown"))),
            )
        )
        doc_id = doc.get("doc_id")
        if doc_id is None:
            # Extract doc_id from chunk_id format: {doc_id}_{seg_idx}_{chunk_idx} or {doc_id}_{idx}
            if "_" in chunk_id:
                doc_id = chunk_id.split("_")[0]
            else:
                doc_id = chunk_id
        # Ensure doc_id is a string (FlashRAG normalizes to string)
        doc_id = str(doc_id) if doc_id is not None else "0"

        source_scores = doc.get("source_scores") or {}
        if not isinstance(source_scores, dict):
            source_scores = {}

        # Get source scores - prefer from document fields, then from source_scores dict
        # CRITICAL: Handle sentinel values (e.g., -3.4028234663852886e+38 for missing dense_score)
        # If dense_score is the minimum float32 value, it means the document wasn't found by dense retriever
        dense_score_raw = doc.get("dense_score", source_scores.get("dense", None))
        if dense_score_raw is not None:
            dense_score_float = float(dense_score_raw)
            # Check if it's the sentinel value (minimum float32)
            if dense_score_float <= -3.4e38:
                dense_score = None  # Document not found by dense retriever
            else:
                dense_score = dense_score_float
        else:
            dense_score = None
        
        bm25_score_raw = doc.get("bm25_score", source_scores.get("bm25", None))
        # CRITICAL: Handle 0.0 as a valid score (means no match, but still a score)
        # Only treat None/absent as "not found"
        if bm25_score_raw is not None:
            bm25_score_float = float(bm25_score_raw)
            # Check for sentinel values
            if bm25_score_float <= -3.4e38:
                bm25_score = None  # Sentinel value - document not found
            else:
                bm25_score = bm25_score_float  # 0.0 is valid (no match, but still a score)
        else:
            bm25_score = None
        
        # Compute separate title and text scores if query is available
        title_score = 0.0
        text_score = 0.0
        if query:
            title_score, text_score = self._compute_title_text_scores(doc, query)
        
        # For display/logging: prioritize original source scores over RRF score
        # RRF scores look like 1/(k+rank) and are typically 0.01-0.02 range
        # Real BM25/dense scores are usually different (can be larger or have different distribution)
        rrf_score = doc.get("rrf_score") or doc.get("score")
        
        # CRITICAL: If we have real source scores, use them for display
        # RRF scores are rank-based and don't reflect actual relevance
        # Handle None values properly (documents not found by a specific retriever)
        if dense_score is not None and dense_score > 0:
            base_score = dense_score
        elif bm25_score is not None and bm25_score > 0:
            base_score = bm25_score
        elif dense_score is not None or bm25_score is not None:
            # At least one score exists, but might be 0.0
            base_score = max(
                dense_score if dense_score is not None else 0.0,
                bm25_score if bm25_score is not None else 0.0
            )
            if base_score <= 0:
                base_score = None
        else:
            base_score = None
        
        # If no source scores available, try RRF score
        if base_score is None and rrf_score is not None:
            rrf_score = float(rrf_score)
            # Check if RRF score looks like rank-based (very small, around 0.01-0.02)
            if rrf_score < 0.1:
                # This is likely RRF score, but we have no source scores, so use it
                base_score = rrf_score
            else:
                # Might be a real score, use it
                base_score = rrf_score
        
        # Final fallback
        if base_score is None:
            base_score = fallback_score or 0.0
        base_score = float(base_score or 0.0)
        
        # IMPORTANT: Combine title and text scores with title having higher weight (2:1 ratio)
        # If we computed separate scores, use weighted combination
        if query and (title_score > 0 or text_score > 0):
            # Title weight: 2.0, Text weight: 1.0
            # Normalize to ensure title has priority
            total_weight = 2.0 + 1.0
            score = (title_score * 2.0 + text_score * 1.0) / total_weight
            # If combined score is too low, fall back to base_score
            if score < base_score * 0.1:
                score = base_score
        else:
            # No query or no separate scores, use base score
            score = base_score
        
        score = float(score or 0.0)

        # Build candidate dict - include rrf_score for debugging
        # CRITICAL: Try to get scores from doc directly if they're not in the processed fields
        # This handles cases where scores were set but not properly extracted
        if dense_score is None:
            # Try to get from doc directly
            dense_score_raw = doc.get("dense_score")
            if dense_score_raw is not None:
                dense_score_float = float(dense_score_raw)
                if dense_score_float > -3.4e38:  # Not a sentinel
                    dense_score = dense_score_float
            # If still None, check source_scores
            if dense_score is None and "dense" in source_scores:
                dense_score = float(source_scores["dense"])
        
        if bm25_score is None:
            # Try to get from doc directly
            bm25_score_raw = doc.get("bm25_score")
            if bm25_score_raw is not None:
                bm25_score_float = float(bm25_score_raw)
                if bm25_score_float > -3.4e38:  # Not a sentinel
                    bm25_score = bm25_score_float
            # If still None, check source_scores
            if bm25_score is None and "bm25" in source_scores:
                bm25_score = float(source_scores["bm25"])
        
        # CRITICAL: Convert None to 0.0 for JSON serialization, but preserve None semantics
        # Use 0.0 for "no match" scores, None only for "not found by retriever"
        candidate = {
            "chunk_id": chunk_id,
            "doc_id": str(doc_id),
            "score": score,
            "dense_score": dense_score if dense_score is not None else 0.0,  # 0.0 if not found by dense retriever
            "bm25_score": bm25_score if bm25_score is not None else 0.0,  # 0.0 if not found by BM25 retriever
            "rrf_score": float(rrf_score) if rrf_score is not None else None,  # RRF fusion score for debugging
            "title_score": title_score,
            "text_score": text_score,
            "contents": doc.get("contents") or doc.get("text") or "",
            "source_scores": {key: float(value) for key, value in source_scores.items() if value is not None},
            "sources": doc.get("sources", []),
            "has_table": doc.get("has_table", False),
        }
        return candidate
    
    @staticmethod
    def _is_numeric_query(query: str) -> bool:
        """Check if query contains numeric patterns (numbers, rates, amounts, etc.)."""
        import re
        # Check for numbers
        if re.search(r'\d+', query):
            return True
        # Check for numeric-related terms
        numeric_terms = ["процент", "ставка", "сумма", "лимит", "комиссия", "курс", "цена", "стоимость"]
        query_lower = query.lower()
        return any(term in query_lower for term in numeric_terms)
    
    @staticmethod
    def _detect_table_in_candidate(candidate: Dict[str, Any]) -> bool:
        """Detect if candidate contains table structure."""
        from src.table_processor import detect_table
        contents = candidate.get("contents") or candidate.get("text") or ""
        return detect_table(contents)

