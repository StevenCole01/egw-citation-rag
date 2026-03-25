# WhiteRAG

> A citation-based RAG system for querying Ellen G. White Writings.

WhiteRAG ingests EPUB books, extracts structured text, stores embeddings in a vector database, and generates answers grounded in the source material with explicit citations (book, chapter, paragraph).

---

## Quick Start

```bash
# 1. Create virtual environment (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ingest EPUBs → structured JSON
python -m src.ingestion.cli

# 4. Chunk paragraphs → embedding-ready JSON
python -m src.preprocessing.cli

# 5. Run tests
python -m pytest tests/ -v
```

---

## Usage

### Phase 1 — Ingestion

Place `.epub` files in `data/raw/`, then:

```bash
python -m src.ingestion.cli [--input-dir data/raw] [--output-dir data/processed]
```

Produces one JSON file per book in `data/processed/`:

```json
[
  {
    "book_title": "Steps to Christ",
    "chapter_title": "God's Love for Man",
    "paragraph_id": 1,
    "text": "Nature and revelation alike testify of God's love..."
  }
]
```

### Phase 2 — Chunking

```bash
python -m src.preprocessing.cli \
  [--input-dir data/processed] \
  [--output-dir data/processed] \
  [--min-words 300] \
  [--max-words 700] \
  [--overlap 1]
```

Produces `<book>_chunks.json` alongside each paragraph file:

```json
[
  {
    "book_title": "Steps to Christ",
    "chapter_title": "God's Love for Man",
    "paragraph_range": [1, 4],
    "text": "Nature and revelation alike testify...",
    "word_count": 342
  }
]
```

> **Chunking rules:** chunks never span chapter boundaries; the last `--overlap` paragraph(s) of each chunk are prepended to the next for contextual continuity.

---

## Project Structure

```
egw-citation-rag/
├── data/
│   ├── raw/              # EPUB files (local only, not committed)
│   └── processed/        # JSON output (local only, not committed)
├── src/
│   ├── ingestion/        # EPUB parsing (Phase 1)
│   ├── preprocessing/    # Chunking (Phase 2)
│   ├── embeddings/       # Embedding generation (Phase 3)
│   ├── retrieval/        # Similarity search (Phase 4)
│   ├── generation/       # LLM response generation (Phase 5)
│   └── utils/
├── app/                  # UI — Streamlit or FastAPI (Phase 6)
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

---

## Development Phases

| Phase | Description                  | Status |
|-------|------------------------------|--------|
| 1     | EPUB Ingestion               | ✅     |
| 2     | Chunking                     | ✅     |
| 3     | Embeddings + Vector Store    | ⬜     |
| 4     | Retrieval                    | ⬜     |
| 5     | Generation with Citations    | ⬜     |
| 6     | Interface (CLI / Streamlit)  | ⬜     |
