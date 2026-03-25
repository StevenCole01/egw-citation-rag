# WhiteRAG

> A citation-based RAG system for querying Ellen G. White Writings.

WhiteRAG ingests books (EPUB), extracts structured text, stores embeddings in a vector database, and generates answers grounded in the source material with explicit citations (book, chapter, paragraph).

## Quick Start

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run EPUB Ingestion (Phase 1)

Place `.epub` files in `data/raw/`, then:

```bash
python -m src.ingestion.cli
```

Structured JSON output will be written to `data/processed/`.

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
egw-citation-rag/
├── data/
│   ├── raw/              # Original EPUB/PDF files
│   └── processed/        # Cleaned structured data (JSON)
├── src/
│   ├── ingestion/        # EPUB/PDF parsing
│   ├── preprocessing/    # Cleaning + chunking
│   ├── embeddings/       # Embedding generation
│   ├── retrieval/        # Similarity search
│   ├── generation/       # LLM response generation
│   └── utils/            # Shared utilities
├── app/                  # UI (Streamlit or API)
├── notebooks/            # Experiments
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Output Format (Phase 1)

Each EPUB file produces a JSON array:

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

## Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | EPUB Ingestion | ✅ |
| 2 | Chunking | ⬜ |
| 3 | Embeddings + Vector Store | ⬜ |
| 4 | Retrieval | ⬜ |
| 5 | Generation with Citations | ⬜ |
| 6 | Interface (CLI / Streamlit) | ⬜ |
