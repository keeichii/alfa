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
│   ├── retriever.py      # FlashRAG BM25s ретривер
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

```

### Запуск полного пайплайна

```bash
# Предобработка данных
make build_corpus    # CSV → corpus.jsonl
make chunk_corpus     # corpus.jsonl → chunks.jsonl

# Построение индекса
make build_index      # BM25 (bm25s) индекс через FlashRAG

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
- **sentence-transformers** — CrossEncoder для реранка
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
- ✅ FlashRAG BM25s ретривер + CrossEncoder реранкер
- ✅ Батчевая обработка запросов (retrieval + rerank)
- ✅ Cross-encoder реранкинг
- ✅ Детальное логирование провалов для анализа
- ✅ Полная воспроизводимость (конфиги, версии)

## Документация

Подробное описание системы, алгоритмов и компонентов см. в **[data.md](data.md)**.

## Лицензия

Проект использует только OpenSource библиотеки и модели.
