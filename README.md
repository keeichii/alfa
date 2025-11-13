# RAG Pipeline for Q&A System

Интеллектуальный pipeline RAG-системы для поиска релевантных фрагментов в корпусе данных по пользовательским запросам с максимизацией метрики Hit@5.

## Структура проекта

```
alfa/
├── configs/              # YAML конфигурации
│   ├── base.yaml         # Базовые параметры пайплайна
│   └── models.yaml       # Конфигурации моделей
├── data/
│   ├── raw/              # Исходные CSV файлы
│   └── processed/        # Обработанные данные
├── indexes/              # Векторные и BM25 индексы
│   ├── faiss/
│   └── bm25/
├── outputs/
│   ├── submits/          # Файлы для сабмита
│   ├── reports/          # Отчёты метрик
│   └── logs/             # Логи кандидатов и провалов
├── scripts/              # Скрипты для запуска пайплайна
├── src/                  # Модули пайплайна
│   ├── ingest.py         # Загрузка и обработка CSV
│   ├── chunker.py        # Разбиение на чанки
│   ├── semantic_chunker.py  # Семантико-структурное чанкование
│   ├── table_processor.py  # Обработка таблиц и числовых значений
│   ├── text_processor.py   # Нормализация текста
│   ├── retriever.py      # Гибридный ретривер (FAISS + FlashRAG bm25s)
│   ├── reranker.py       # Cross-encoder реранкер
│   ├── evaluator.py      # Метрики оценки
│   ├── failure_logger.py # Логирование провалов
│   └── utils.py          # Утилиты
├── requirements.txt      # Зависимости Python
├── Makefile             # Команды для сборки
├── README.md            # Этот файл
└── about.md             # Подробное описание системы
```

## Быстрый запуск

### Установка

```bash
# 1. Установка Python пакетов
# 1. Установка Python пакетов (faiss-cpu по умолчанию)
pip install -r requirements.txt

# 1a. Для GPU: установите faiss-gpu и torch с поддержкой CUDA
pip install faiss-gpu torch --extra-index-url https://download.pytorch.org/whl/cu121

# 2. Установка FlashRAG
cd FlashRAG
pip install -e .
pip install flashrag-dev[full]
cd ..

```

### Запуск полного пайплайна

```bash
# Предобработка данных
make build_corpus    # CSV → corpus.jsonl
make chunk_corpus     # corpus.jsonl → chunks.jsonl

# Построение индекса
make build_index      # FAISS + BM25 (bm25s) индексы через FlashRAG

# Оценка и генерация submit
make eval             # Ретрив + реранк + метрики
make submit           # Формирует submit.csv
```

Или одной командой:
```bash
make build_corpus chunk_corpus build_index eval submit
```

## Используемый стек

### Основные библиотеки

- **Python 3.10+**
- **FlashRAG** — построение BM25 индексов и инфраструктура RAG
- **sentence-transformers** — эмбеддинги и CrossEncoder для реранка
- **bm25s** — быстрый BM25-поиск
- **chonkie** — токенизация и чанкинг
- **PyYAML** — конфигурации
- **numpy, scikit-learn** — вычисления и метрики

### Модели

- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-12-v2` (настраивается)
- **Tokenizer:** `o200k_base` для чанкинга

## Конфигурация

Основные параметры в `configs/base.yaml`:

```yaml
retrieval:
  k_retrieve: 30            # кандидатов на первом этапе
  k_rerank: 20              # кандидатов, передаваемых в cross-encoder
  k_final: 5                # финальный топ-K
  batch_size: 32            # размер батча вопросов
  hybrid_weight_dense: 0.6  # вес dense поиска
  hybrid_weight_bm25: 0.4   # вес BM25 поиска
  fusion_method: "weighted" # или "rrf"
  enhance_numerics: true    # улучшение для числовых значений при BM25

chunking:
  size_tokens: 600
  overlap_tokens: 150
  use_semantic_chunking: true  # семантическое чанкование
```

## Выходные файлы

- **`outputs/submits/submit_*.csv`** — файлы для сабмита (q_id, web_list)
- **`outputs/reports/report_*.json`** — отчёты с метриками (Hit@5, Recall@K, MRR, NDCG)
- **`outputs/logs/candidates_*.jsonl`** — логи кандидатов до/после реранка
- **`outputs/logs/failures_*.json`** — логи провалов с предложениями по улучшению

## Метрики

- **Hit@5** — целевая метрика (доля вопросов с релевантным документом в топ-5)
- **Recall@K** — средняя доля найденных релевантных документов
- **MRR** — средний обратный ранг первого релевантного документа
- **NDCG@K** — нормализованный дисконтированный кумулятивный выигрыш

## Особенности

- ✅ Семантико-структурное чанкование (таблицы, заголовки)
- ✅ Обработка числовых значений (валюты, проценты, курсы)
- ✅ Гибридный ретривер: FAISS dense + FlashRAG bm25s (weighted / RRF fusion)
- ✅ Батчевая обработка запросов (retrieval + rerank)
- ✅ Cross-encoder реранкинг (fp16 на GPU)
- ✅ Детальное логирование провалов для анализа
- ✅ Полная воспроизводимость (конфиги, версии)
- ✅ Автонастройка GPU (CUDA) для эмбеддингов, ретривера и реранкера

## GPU / CUDA

- Установите `torch` и `faiss-gpu` с соответствующим CUDA билдом (пример для CUDA 12.1 выше).
- В `configs/models.yaml` оставьте `"device: auto"` — пайплайн сам выберет `cuda`, если она доступна.
- Дополнительно:
  - `embeddings.use_fp16: true` включает half precision для SentenceTransformer.
  - `faiss.use_gpu: true` переносит FAISS индекс на GPU и включает fp16 (если доступно).
  - `reranker.use_fp16: true` ускоряет cross-encoder на GPU.
- Профилируйте производительность с `python scripts/benchmark.py --config configs/base.yaml --n 100`.

## Документация

Подробное описание системы, алгоритмов и компонентов см. в **[about.md](about.md)**.

## Лицензия

Проект использует только OpenSource библиотеки и модели.
