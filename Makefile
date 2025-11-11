.PHONY: help build_corpus chunk_corpus build_index eval submit clean

# Default config
CONFIG ?= configs/base.yaml

help:
	@echo "Available targets:"
	@echo "  make build_corpus    - Build corpus.jsonl from websites CSV"
	@echo "  make chunk_corpus    - Chunk corpus into smaller pieces"
	@echo "  make build_index     - Build FAISS and BM25 indexes"
	@echo "  make eval            - Evaluate retrieval and generate metrics report"
	@echo "  make submit          - Generate submit.csv for submission"
	@echo "  make clean           - Clean processed data and indexes"
	@echo ""
	@echo "Usage: make <target> CONFIG=configs/base.yaml"

build_corpus:
	python scripts/build_corpus.py --config $(CONFIG)

chunk_corpus:
	python scripts/chunk_corpus.py --config $(CONFIG)

build_index: build_corpus chunk_corpus
	@echo "Building indexes..."
	python scripts/build_index_dense.py --config $(CONFIG)
	python scripts/build_index_bm25.py --config $(CONFIG)

eval: build_index
	python scripts/eval.py --config $(CONFIG)

submit: build_index
	python scripts/submit.py --config $(CONFIG)

clean:
	rm -f data/processed/*.jsonl
	rm -rf indexes/faiss/*
	rm -rf indexes/bm25/*
	rm -rf outputs/submits/*
	rm -rf outputs/reports/*
	rm -rf outputs/logs/*
	@echo "Cleaned processed data and indexes"

