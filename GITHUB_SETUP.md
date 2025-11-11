# Инструкция по загрузке на GitHub

## Что было сделано

### 1. Обновлен `.gitignore`

Игнорируются:
- ✅ Большие файлы данных (`data/processed/*.jsonl`, `data/raw/*.csv`)
- ✅ Индексы (`indexes/faiss/*`, `indexes/bm25/*`)
- ✅ Сгенерированные файлы (`outputs/*`)
- ✅ Python кэш (`__pycache__/`, `*.pyc`)
- ✅ Виртуальное окружение (`.venv/`, `venv/`)
- ✅ FlashRAG (внешняя библиотека)
- ✅ Временные файлы

### 2. Созданы `.gitkeep` файлы

Для сохранения структуры пустых папок:
- `data/processed/.gitkeep`
- `data/raw/.gitkeep`
- `indexes/faiss/.gitkeep`
- `indexes/bm25/.gitkeep`
- `outputs/submits/.gitkeep`
- `outputs/reports/.gitkeep`
- `outputs/logs/.gitkeep`

### 3. Добавлен `.gitattributes`

Для правильной обработки текстовых и бинарных файлов.

## Что будет в репозитории

### ✅ Включено:
- Исходный код (`src/`, `scripts/`)
- Конфигурации (`configs/`)
- Документация (`README.md`, `data.md`, `mb_fix.md`, `IMPLEMENTATION_STATUS.md`)
- Makefile и requirements.txt
- Структура папок (через .gitkeep)

### ❌ Исключено:
- Большие файлы данных (47MB+)
- Индексы (41MB+)
- Сгенерированные outputs
- Python кэш
- Виртуальное окружение
- FlashRAG (нужно установить отдельно)

## Команды для загрузки

```bash
# 1. Проверить статус
git status

# 2. Добавить файлы
git add .

# 3. Проверить что будет закоммичено (должны быть только нужные файлы)
git status

# 4. Закоммитить
git commit -m "Initial commit: RAG pipeline for Q&A system"

# 5. Добавить remote (если еще не добавлен)
git remote add origin https://github.com/username/alfa.git

# 6. Загрузить на GitHub
git push -u origin main
```

## После загрузки

Пользователям нужно будет:

1. **Клонировать репозиторий:**
   ```bash
   git clone https://github.com/username/alfa.git
   cd alfa
   ```

2. **Установить зависимости:**
   ```bash
   pip install -r requirements.txt
   cd FlashRAG && pip install -e . && pip install flashrag-dev[full] && cd ..
   conda install -c pytorch faiss-cpu=1.8.0
   ```

3. **Подготовить данные:**
   - Положить `questions_clean.csv` и `websites_updated.csv` в `data/raw/`
   - Запустить `make build_corpus chunk_corpus build_index`

## Размер репозитория

Без больших файлов репозиторий будет ~1-5MB вместо ~130MB+.

