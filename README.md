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
│   ├── retriever.py      # Гибридный ретривер (FAISS + BM25)
│   ├── reranker.py       # Cross-encoder реранкер
│   ├── evaluator.py      # Метрики оценки
│   ├── failure_logger.py # Логирование провалов
│   └── utils.py          # Утилиты
├── requirements.txt      # Зависимости Python
├── Makefile             # Команды для сборки
├── README.md            # Этот файл
└── data.md              # Подробное описание системы
```

## Быстрый запуск

### Установка

```bash
# 1. Установка Python пакетов
pip install -r requirements.txt

# 2. Установка FlashRAG
cd FlashRAG
pip install -e .
pip install flashrag-dev[full]
cd ..

# 3. Установка FAISS
conda install -c pytorch faiss-cpu=1.8.0
# или для GPU: conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

### Запуск полного пайплайна

```bash
# Предобработка данных
make build_corpus    # CSV → corpus.jsonl
make chunk_corpus     # corpus.jsonl → chunks.jsonl

# Построение индексов
make build_index      # FAISS + BM25 индексы

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
- **FlashRAG** — построение индексов и компоненты RAG
- **sentence-transformers** — эмбеддинги (E5, mpnet)
- **faiss-cpu/faiss-gpu** — векторный поиск
- **bm25s** — разреженный поиск BM25
- **chonkie** — токенизация и чанкинг
- **PyYAML** — конфигурации
- **numpy, scikit-learn** — вычисления и метрики

### Модели

- **Embeddings:** `intfloat/multilingual-e5-base` (настраивается)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-12-v2` (настраивается)
- **Tokenizer:** `o200k_base` для чанкинга

## Конфигурация

Основные параметры в `configs/base.yaml`:

```yaml
retrieval:
  k_retrieve: 50           # кандидатов до реранка
  k_final: 5               # финальный топ-K
  hybrid_weight_dense: 0.6 # вес dense поиска
  hybrid_weight_bm25: 0.4   # вес BM25 поиска
  fusion_method: "weighted" # или "rrf"
  enhance_numerics: true # улучшение для числовых значений

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
- ✅ Гибридный ретрив (FAISS + BM25) с RRF fusion
- ✅ Cross-encoder реранкинг
- ✅ Детальное логирование провалов для анализа
- ✅ Полная воспроизводимость (конфиги, версии)

## Документация

Подробное описание системы, алгоритмов и компонентов см. в **[data.md](data.md)**.

## Лицензия

Проект использует только OpenSource библиотеки и модели.
