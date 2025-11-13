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
- `retrieval.k_retrieve`, `k_rerank`, `k_final`, `batch_size`
- `retrieval.hybrid_weight_dense`, `hybrid_weight_bm25`, `fusion_method`
- `retrieval.enhance_numerics` (enable numeric-aware matching)
- `models.embeddings.device = auto` (fallback to CUDA when available)
- `models.embeddings.use_fp16` (enables half precision in `HybridRetriever`)
- `models.faiss.use_gpu`, `use_float16`, `gpu_device` (move FAISS to GPU via FlashRAG)
- `models.reranker.device`, `use_fp16`, `batch_size`

## Performance notes
- GPU usage (auto-detected):
  - SentenceTransformer embeddings run on CUDA with fp16 for higher throughput.
  - FAISS index is cloned to GPU (`StandardGpuResources`) with optional fp16 search.
  - CrossEncoder reranker converts weights to fp16 on CUDA.
- Keep `k_retrieve = 20–30` for balance of quality/speed; adjust fusion weights to tune precision/recall.
- Batch size 32–64 for CPU/GPU respectively; increase gradually while monitoring VRAM.
- Build indices once; reuse via `indexes/faiss` and `indexes/bm25`.
- Use `python scripts/benchmark.py --config configs/base.yaml --n 200` to profile retrieval/rerank timings.

## Git hygiene
- Large outputs, indexes, caches are ignored via `.gitignore`
- Line endings normalized via `.gitattributes`


