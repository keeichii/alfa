.PHONY: help build_corpus check_corpus chunk_corpus build_index_bm25 build_index_dense build_index eval_retriever eval_full eval submit clean clean_corpus clean_chunks clean_indexes

# Default config
CONFIG ?= configs/base.yaml

help:
	@echo "Available targets (step-by-step pipeline):"
	@echo ""
	@echo "Data preparation:"
	@echo "  make build_corpus      - Step 1: Build corpus.jsonl from websites CSV"
	@echo "  make check_corpus      - Step 2: Check corpus statistics (optional)"
	@echo "  make chunk_corpus      - Step 3: Chunk corpus into smaller pieces"
	@echo ""
	@echo "Index building:"
	@echo "  make build_index_bm25  - Step 4: Build BM25 index"
	@echo "  make build_index_dense - Step 5: Build FAISS dense index"
	@echo "  make build_index       - Steps 4+5: Build both indexes"
	@echo ""
	@echo "Evaluation and submission:"
	@echo "  make eval_retriever    - Step 6: Evaluate retriever only (no reranker)"
	@echo "  make eval_full         - Step 7: Evaluate full pipeline (retriever + reranker)"
	@echo "  make eval              - Step 7: Alias for eval_full"
	@echo "  make submit            - Step 8: Generate submit.csv for submission"
	@echo ""
	@echo "Utilities:"
	@echo "  make check_ground_truth - Check consistency between questions and ground truth"
	@echo "  make clean_corpus      - Clean corpus.jsonl and rejected log"
	@echo "  make clean_chunks      - Clean chunks.jsonl"
	@echo "  make clean_indexes     - Clean all indexes"
	@echo "  make clean             - Clean all processed data and indexes"
	@echo ""
	@echo "Usage: make <target> CONFIG=configs/base.yaml"
	@echo ""
	@echo "Example workflow:"
	@echo "  make build_corpus && make check_corpus && make chunk_corpus"
	@echo "  make build_index"
	@echo "  make eval_retriever && make eval_full"
	@echo "  make submit"

# Step 1: Build corpus from CSV
build_corpus:
	@echo "=== Step 1: Building corpus from CSV ==="
	python scripts/build_corpus.py --config $(CONFIG)
	@echo "Corpus built. Check data/processed/corpus.jsonl and corpus_rejected.json for details."

# Step 2: Check corpus statistics (optional diagnostic)
check_corpus:
	@echo "=== Step 2: Checking corpus statistics ==="
	python scripts/check_corpus_stats.py --config $(CONFIG)
	@echo "Corpus statistics check complete."

# Step 3: Chunk corpus
chunk_corpus: build_corpus
	@echo "=== Step 3: Chunking corpus ==="
	python scripts/chunk_corpus.py --config $(CONFIG)
	@echo "Corpus chunked. Check data/processed/chunks.jsonl"

# Step 4: Build BM25 index
build_index_bm25: chunk_corpus
	@echo "=== Step 4: Building BM25 index ==="
	python scripts/build_index_bm25.py --config $(CONFIG)
	@echo "BM25 index built. Check indexes/bm25/"

# Step 5: Build FAISS dense index
build_index_dense: chunk_corpus
	@echo "=== Step 5: Building FAISS dense index ==="
	python scripts/build_index_dense.py --config $(CONFIG)
	@echo "FAISS index built. Check indexes/faiss/"

# Build both indexes (Steps 4+5)
build_index: build_index_bm25 build_index_dense
	@echo "=== Both indexes built successfully ==="

# Step 6: Evaluate retriever only (no reranker)
eval_retriever: build_index
	@echo "=== Step 6: Evaluating retriever only (no reranker) ==="
	python scripts/eval.py --config $(CONFIG) --mode retriever
	@echo "Retriever-only evaluation complete. Check outputs/reports/"

# Step 7: Evaluate full pipeline (retriever + reranker)
eval_full: build_index
	@echo "=== Step 7: Evaluating full pipeline (retriever + reranker) ==="
	python scripts/eval.py --config $(CONFIG) --mode retriever+reranker
	@echo "Full pipeline evaluation complete. Check outputs/reports/"

# Alias for eval_full
eval: eval_full

# Step 8: Generate submission
submit: build_index
	@echo "=== Step 8: Generating submission ==="
	python scripts/submit.py --config $(CONFIG)
	@echo "Submission generated. Check outputs/submits/"

# Utility: Check ground truth consistency
check_ground_truth:
	@echo "=== Checking ground truth consistency ==="
	@if [ -z "$(GT_PATH)" ]; then \
		echo "Usage: make check_ground_truth GT_PATH=path/to/ground_truth.json"; \
		exit 1; \
	fi
	python scripts/check_ground_truth.py --questions data/raw/questions_clean.csv --ground_truth $(GT_PATH)

# Clean targets (for debugging)
clean_corpus:
	@echo "Cleaning corpus files..."
	rm -f data/processed/corpus.jsonl
	rm -f data/processed/corpus_rejected.json
	@echo "Corpus files cleaned"

clean_chunks:
	@echo "Cleaning chunks..."
	rm -f data/processed/chunks.jsonl
	@echo "Chunks cleaned"

clean_indexes:
	@echo "Cleaning indexes..."
	rm -rf indexes/faiss/*
	rm -rf indexes/bm25/*
	@echo "Indexes cleaned"

# Full clean
clean: clean_corpus clean_chunks clean_indexes
	@echo "Cleaning output files..."
	rm -rf outputs/submits/*
	rm -rf outputs/reports/*
	rm -rf outputs/logs/*
	@echo "All processed data and indexes cleaned"
