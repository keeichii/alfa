# About this project

This repository implements a production-ready RAG retrieval pipeline tailored for the AlfaBank public site dataset.

- Hybrid retrieval: FAISS dense + FlashRAG bm25s (sparse) with fusion (weighted/RRF)
- Cross-encoder reranking
- Semantic-structural chunking with table/numeric handling
- Batched processing, detailed logging, robust error handling

## Architecture

Pipeline:

CSV → corpus.jsonl → chunks.jsonl → (BM25 index + FAISS index) → Hybrid retrieval → Cross-encoder rerank → Top-5 submit.csv

Key components:
- `src/chunker.py`, `src/semantic_chunker.py`, `src/table_processor.py`
- `src/retriever.py` (Hybrid: FAISS + bm25s via FlashRAG)
- `src/reranker.py`, `src/evaluator.py`, `src/failure_logger.py`, `src/utils.py`

Benchmarking and evaluation:
- `scripts/benchmark.py`, `scripts/eval.py`, `scripts/submit.py`

## Configuration
See `configs/base.yaml` and `configs/models.yaml`. Important fields:
- retrieval.k_retrieve, k_rerank, k_final, batch_size
- retrieval.hybrid_weight_dense, hybrid_weight_bm25, fusion_method
- retrieval.enhance_numerics (enable numeric-aware matching)
- models.reranker.model_name, device, batch_size

## Performance notes
- Use GPU for the reranker when available (5–8x speedup)
- Keep k_retrieve=20–30 for balance of quality/speed
- Batch size 32–64 for CPU/GPU respectively
- Build indices once; avoid recomputing

## Git hygiene
- Large outputs, indexes, caches are ignored via `.gitignore`
- Line endings normalized via `.gitattributes`


