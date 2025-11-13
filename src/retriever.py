"""Hybrid retriever: FAISS dense + FlashRAG bm25s with fusion."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLASHRAG_PATH = PROJECT_ROOT / "FlashRAG"
if str(FLASHRAG_PATH) not in sys.path:
    sys.path.insert(0, str(FLASHRAG_PATH))

from flashrag.config import Config as FlashRAGConfig  # type: ignore
from flashrag.retriever.retriever import BM25Retriever as FlashRAGBM25  # type: ignore

import json
try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # lazy-require if dense enabled
from typing import TypedDict
from sentence_transformers import SentenceTransformer

try:  # optional dependency used for convenience conversions
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy should be available but guard just in case
    np = None  # type: ignore

from src.table_processor import enhance_query_for_numerics
from src.utils import logger, load_jsonl


class FlashRAGBM25Retriever:
    """
    Thin wrapper around FlashRAG's BM25 retriever.

    Only the bm25s backend is supported, dense retrieval has been removed entirely.
    The adapter keeps the previous interface expected by the rest of the project:
    ``retrieve`` for a single query and ``retrieve_batch`` for batched processing.
    """

    def __init__(
        self,
        index_dir: str,
        corpus_path: str,
        default_k: int = 30,
        enhance_numerics: bool = True,
    ) -> None:
        self.index_root = Path(index_dir)
        if not self.index_root.exists():
            raise FileNotFoundError(f"BM25 index directory not found: {self.index_root}")

        self.index_path = self._resolve_index_path(self.index_root)
        self.corpus_path = self._resolve_corpus_path(corpus_path, self.index_root)

        self.default_k = max(1, int(default_k))
        self.enhance_numerics = enhance_numerics

        self._flashrag_config = self._build_flashrag_config()
        self._retriever = FlashRAGBM25(self._flashrag_config.final_config)

        logger.info(
            "Initialized FlashRAG BM25 retriever (bm25s backend): index=%s corpus=%s",
            self.index_path,
            self.corpus_path,
        )

        self._corpus = self._load_corpus_metadata()
        self._chunk_lookup = self._build_chunk_lookup(self._corpus)

    @staticmethod
    def _resolve_index_path(root: Path) -> Path:
        bm25_dir = root / "bm25"
        return bm25_dir if bm25_dir.exists() else root

    @staticmethod
    def _resolve_corpus_path(config_corpus: str, index_root: Path) -> Path:
        if config_corpus:
            explicit = Path(config_corpus)
            if explicit.exists():
                return explicit
        candidate = index_root / "chunks.jsonl"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            "Could not locate corpus JSONL for BM25. Expected either the configured path "
            f"({config_corpus}) or {candidate}"
        )

    def _build_flashrag_config(self) -> FlashRAGConfig:
        cfg_dict: Dict[str, Any] = {
            "disable_save": True,
            "retrieval_method": "bm25",
            "bm25_backend": "bm25s",
            "retrieval_topk": self.default_k,
            "index_path": str(self.index_path),
            "corpus_path": str(self.corpus_path),
            "save_retrieval_cache": False,
            "use_retrieval_cache": False,
            "use_reranker": False,
            "silent_retrieval": True,
            "data_dir": str(self.index_root),
            "save_dir": str(self.index_root),
        }
        return FlashRAGConfig(config_dict=cfg_dict)

    def _load_corpus_metadata(self) -> List[Dict[str, Any]]:
        corpus = getattr(self._retriever, "corpus", None)
        if corpus:
            return corpus  # type: ignore[return-value]
        logger.debug("FlashRAG retriever corpus missing, loading from %s", self.corpus_path)
        return load_jsonl(str(self.corpus_path))

    @staticmethod
    def _build_chunk_lookup(corpus: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        lookup: Dict[str, Dict[str, Any]] = {}
        for idx, chunk in enumerate(corpus):
            chunk_id = str(chunk.get("id", idx))
            lookup[chunk_id] = chunk
        return lookup

    def _prepare_queries(self, queries: List[str]) -> List[str]:
        if not self.enhance_numerics:
            return queries
        return [enhance_query_for_numerics(q) for q in queries]

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        return_scores: bool = False,
    ):
        batch_results = self.retrieve_batch([query], k=k, return_scores=return_scores)
        if return_scores:
            results, scores = batch_results
            result = results[0] if results else []
            score = scores[0] if scores else {"bm25": {}}
            return result, score
        else:
            return batch_results[0] if batch_results else []

    def retrieve_batch(
        self,
        queries: List[str],
        k: Optional[int] = None,
        return_scores: bool = False,
    ):
        if not queries:
            return ([], []) if return_scores else []

        topk = max(1, k or self.default_k)
        prepared_queries = self._prepare_queries(queries)

        results_raw, scores_raw = self._retriever.batch_search(
            prepared_queries,
            num=topk,
            return_score=True,
        )

        results_raw = self._ensure_list(results_raw)
        scores_raw = self._ensure_list(scores_raw)

        if len(results_raw) < len(queries):
            results_raw += [[] for _ in range(len(queries) - len(results_raw))]
        if len(scores_raw) < len(queries):
            scores_raw += [[] for _ in range(len(queries) - len(scores_raw))]

        converted_batches: List[List[Dict[str, Any]]] = []
        score_batches: List[Dict[str, float]] = []

        for docs, scores in zip(results_raw, scores_raw):
            converted = self._convert_results(docs, scores)
            converted_batches.append(converted)
            score_batches.append(
                {"bm25": {cand["chunk_id"]: cand["bm25_score"] for cand in converted}}
            )

        if return_scores:
            return converted_batches, score_batches
        return converted_batches

    def _convert_results(self, docs: Any, scores: Any) -> List[Dict[str, Any]]:
        docs_list = self._ensure_list(docs)
        scores_list = self._ensure_list(scores)

        if len(scores_list) < len(docs_list):
            scores_list += [0.0] * (len(docs_list) - len(scores_list))

        candidates: List[Dict[str, Any]] = []
        for doc, score in zip(docs_list, scores_list):
            candidate = self._to_candidate(doc, score)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _to_candidate(self, item: Any, score: float) -> Optional[Dict[str, Any]]:
        chunk_meta: Optional[Dict[str, Any]] = None
        chunk_id: Optional[str] = None

        if isinstance(item, dict):
            chunk_id = str(item.get("id") or item.get("chunk_id"))
            chunk_meta = item
        elif isinstance(item, (int, float)):
            idx = int(item)
            try:
                chunk_meta = self._corpus[idx]
                chunk_id = str(chunk_meta.get("id", idx))
            except IndexError:
                logger.debug("BM25 result index %s out of range", idx)
                chunk_meta = None
        elif isinstance(item, str):
            chunk_id = item
            chunk_meta = self._chunk_lookup.get(chunk_id)

        if chunk_meta is None and chunk_id is not None:
            chunk_meta = self._chunk_lookup.get(chunk_id, {})

        if chunk_meta is None:
            return None

        if chunk_id is None:
            chunk_id = str(chunk_meta.get("id"))

        doc_id = chunk_meta.get("doc_id")
        if doc_id is None:
            doc_id = chunk_id.split("_")[0] if "_" in chunk_id else chunk_id

        contents = (
            chunk_meta.get("contents")
            or chunk_meta.get("text")
            or ""
        )

        return {
            "chunk_id": chunk_id,
            "doc_id": str(doc_id),
            "score": float(score),
            "bm25_score": float(score),
            "dense_score": 0.0,
            "contents": contents,
        }

    @staticmethod
    def _ensure_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if np is not None and isinstance(value, np.ndarray):  # type: ignore[attr-defined]
            return value.tolist()
        return [value]


class HybridRetriever:
    """
    Combine FAISS dense retrieval with FlashRAG bm25s and fuse results (weighted or RRF).
    """

    def __init__(
        self,
        faiss_index_path: str,
        faiss_meta_path: Optional[str],
        bm25_index_dir: str,
        bm25_corpus_path: str,
        embedding_model_name: str,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        query_batch_size: int = 32,
        weight_dense: float = 0.6,
        weight_bm25: float = 0.4,
        fusion_method: str = "weighted",  # "weighted" or "rrf"
        enhance_numerics: bool = True,
    ) -> None:
        if faiss is None:
            raise ImportError("faiss is required for dense retrieval. Install faiss-cpu or faiss-gpu.")

        self.weight_dense = float(weight_dense)
        self.weight_bm25 = float(weight_bm25)
        self.fusion_method = fusion_method
        self.normalize_embeddings = normalize_embeddings
        self.query_batch_size = max(1, int(query_batch_size))
        self.enhance_numerics = enhance_numerics

        # Dense setup
        logger.info(f"Loading FAISS index from {faiss_index_path}")
        self.faiss_index = faiss.read_index(faiss_index_path)
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)

        self.chunk_ids: List[str] = []
        self.doc_ids: List[Optional[str]] = []
        if faiss_meta_path and Path(faiss_meta_path).exists():
            with open(faiss_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.chunk_ids = meta.get("chunk_ids", [])
            self.doc_ids = meta.get("doc_ids", [])
        else:
            logger.warning("FAISS metadata missing; dense results will use fallback ids")

        # Sparse setup (FlashRAG bm25s)
        self.bm25 = FlashRAGBM25Retriever(
            index_dir=bm25_index_dir,
            corpus_path=bm25_corpus_path,
            default_k=50,
            enhance_numerics=enhance_numerics,
        )
        self._corpus = self.bm25._corpus

    def _encode(self, queries: List[str]) -> "np.ndarray":
        emb = self.embedding_model.encode(
            queries,
            batch_size=self.query_batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return (np.asarray(emb, dtype="float32") if np is not None else emb)

    @staticmethod
    def _normalize(scores: "np.ndarray") -> "np.ndarray":
        if scores.size == 0:
            return scores
        min_s, max_s = float(scores.min()), float(scores.max())
        if max_s == min_s:
            return np.ones_like(scores)
        return (scores - min_s) / (max_s - min_s)

    def _dense_search_batch(self, queries: List[str], k: int) -> Tuple[List[List[int]], List[List[float]]]:
        if not queries:
            return [], []
        q_emb = self._encode(queries)
        k = min(k, self.faiss_index.ntotal)
        scores, idxs = self.faiss_index.search(q_emb, k)
        idx_lists = [row.tolist() for row in idxs]
        score_lists = [row.tolist() for row in scores]
        return idx_lists, score_lists

    def _fuse(
        self,
        dense_idxs: List[int],
        dense_scores: List[float],
        sparse_idxs: List[int],
        sparse_scores: List[float],
        k: int,
    ) -> List[Tuple[int, float, float, float]]:
        d_scores = np.asarray(dense_scores, dtype=float)
        s_scores = np.asarray(sparse_scores, dtype=float)
        d_norm = self._normalize(d_scores) if d_scores.size else d_scores
        s_norm = self._normalize(s_scores) if s_scores.size else s_scores

        d_map = {i: s for i, s in zip(dense_idxs, d_norm)}
        s_map = {i: s for i, s in zip(sparse_idxs, s_norm)}

        if self.fusion_method == "rrf":
            d_rank = {i: r + 1 for r, i in enumerate(dense_idxs)}
            s_rank = {i: r + 1 for r, i in enumerate(sparse_idxs)}
            all_ids = set(dense_idxs) | set(sparse_idxs)
            fused = []
            for i in all_ids:
                score = 0.0
                if i in d_rank:
                    score += 1.0 / (60 + d_rank[i])
                if i in s_rank:
                    score += 1.0 / (60 + s_rank[i])
                fused.append((i, score, float(d_map.get(i, 0.0)), float(s_map.get(i, 0.0))))
        else:
            all_ids = set(dense_idxs) | set(sparse_idxs)
            fused = []
            for i in all_ids:
                ds = float(d_map.get(i, 0.0))
                ss = float(s_map.get(i, 0.0))
                score = self.weight_dense * ds + self.weight_bm25 * ss
                fused.append((i, score, ds, ss))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:k]

    def retrieve(
        self,
        query: str,
        k: int = 50,
        return_scores: bool = False,
    ):
        results, scores = self.retrieve_batch([query], k=k, return_scores=True)
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
        if fusion_method:
            self.fusion_method = fusion_method

        dense_idxs_batch, dense_scores_batch = self._dense_search_batch(queries, k)
        bm25_results_batch, bm25_scores_batch = self.bm25.retrieve_batch(queries, k=k, return_scores=True)

        out: List[List[Dict[str, Any]]] = []
        out_scores: List[Dict[str, Any]] = []
        for qi, (d_idxs, d_scores, bm25_cands, bm25_scores_map) in enumerate(
            zip(dense_idxs_batch, dense_scores_batch, bm25_results_batch, bm25_scores_batch)
        ):
            # Map bm25 candidates back to dense indices if possible by chunk_id
            bm25_dense_idxs: List[int] = []
            bm25_dense_vals: List[float] = []
            for cand in bm25_cands:
                chunk_id = cand.get("chunk_id")
                # try to map chunk_id into dense idx via chunk_ids if present, else fallback to doc alignment
                idx = None
                if self.chunk_ids:
                    try:
                        idx = self.chunk_ids.index(chunk_id)
                    except ValueError:
                        idx = None
                if idx is None:
                    # fallback: try doc_id approximate mapping by first occurrence
                    doc_id = str(cand.get("doc_id"))
                    if self.doc_ids:
                        try:
                            idx = next(i for i, did in enumerate(self.doc_ids) if str(did) == doc_id)
                        except StopIteration:
                            idx = None
                if idx is not None:
                    bm25_dense_idxs.append(idx)
                    bm25_dense_vals.append(float(cand.get("bm25_score", 0.0)))

            fused = self._fuse(d_idxs, d_scores, bm25_dense_idxs, bm25_dense_vals, k)

            # Build candidates
            cands: List[Dict[str, Any]] = []
            for idx, fused_score, dsc, ssc in fused:
                if self._corpus and idx < len(self._corpus):
                    meta = self._corpus[idx]
                    chunk_id = str(meta.get("id", idx))
                    doc_id = str(meta.get("doc_id", chunk_id.split("_")[0] if "_" in chunk_id else chunk_id))
                    contents = meta.get("contents") or meta.get("text") or ""
                else:
                    chunk_id = self.chunk_ids[idx] if self.chunk_ids and idx < len(self.chunk_ids) else f"chunk_{idx}"
                    doc_id = chunk_id.split("_")[0] if "_" in chunk_id else chunk_id
                    contents = ""
                cands.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "score": float(fused_score),
                        "dense_score": float(dsc),
                        "bm25_score": float(ssc),
                        "contents": contents,
                    }
                )
            out.append(cands)
            if return_scores:
                out_scores.append(
                    {
                        "dense": {i: float(s) for i, s in zip(d_idxs, d_scores)},
                        "bm25": {c["chunk_id"]: float(c.get("bm25_score", 0.0)) for c in bm25_cands},
                        "hybrid": {c["chunk_id"]: float(c["score"]) for c in cands},
                    }
                )
        if return_scores:
            return out, out_scores
        return out

