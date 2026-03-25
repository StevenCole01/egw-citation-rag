# WhiteRAG 📖

> A citation-based Retrieval-Augmented Generation (RAG) system for querying Ellen G. White's writings with verifiable, paragraph-level citations.

WhiteRAG ingests EPUB books, builds a semantic vector index, and generates answers that are strictly grounded in the source text — every response includes the book, chapter, and paragraph range it came from.

---

## How It Works

```
EPUB books  →  Paragraphs  →  Chunks  →  ChromaDB  →  Answer + Citations
   (raw)         (JSON)        (JSON)    (vectors)       (via LLM)
```

1. **Ingest** — Parse EPUB files into structured paragraphs (book → chapter → paragraph)
2. **Chunk** — Group paragraphs into overlapping chunks for better semantic coverage
3. **Embed** — Generate vector embeddings and store them in ChromaDB
4. **Retrieve** — Embed a user query and find the top-k most similar chunks
5. **Generate** — Pass retrieved chunks to an LLM with a strict grounding prompt
6. **Cite** — Return the answer alongside the exact source location (book, chapter, ¶)

---

## Setup

### Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys) (for answer generation)
- A [HuggingFace token](https://huggingface.co/settings/tokens) (optional — suppresses rate-limit warnings)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/egw-citation-rag.git
cd egw-citation-rag

# 2. Create a virtual environment (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your keys:
#   OPENAI_API_KEY=sk-...
#   HF_TOKEN=hf_...       (optional)
```

---

## Getting Books

EGW books are freely available from [egwwritings.org](https://egwwritings.org). Use the included downloader to fetch them automatically:

```bash
# See all 56 available books with their codes
python scripts/download_books.py --list

# Download a curated starter set (recommended)
python scripts/download_books.py --books SC DA GC PP PK AA MB COL MH

# Download all 56 books (~200 MB total)
python scripts/download_books.py

# Re-download a specific book
python scripts/download_books.py --books DA --force
```

Downloaded EPUBs are saved to `data/raw/` (excluded from version control).

**Recommended starter set:**

| Code | Title |
|------|-------|
| `SC` | Steps to Christ |
| `DA` | The Desire of Ages |
| `GC` | The Great Controversy |
| `PP` | Patriarchs and Prophets |
| `PK` | Prophets and Kings |
| `AA` | The Acts of the Apostles |
| `MB` | Thoughts from the Mount of Blessing |
| `COL` | Christ's Object Lessons |
| `MH` | The Ministry of Healing |

---

## Building the Index

Run these three commands once after downloading books (or any time you add new books):

```bash
# Step 1 — Parse EPUBs → structured JSON (one file per book)
python -m src.ingestion.cli

# Step 2 — Chunk paragraphs → embedding-ready JSON
python -m src.preprocessing.cli

# Step 3 — Embed chunks → ChromaDB vector store
#           Downloads the embedding model (~90 MB) on first run
python -m src.embeddings.cli
```

---

## Using WhiteRAG

### Option A — Streamlit UI (recommended)

```bash
streamlit run app/streamlit_app.py
```

Opens at **http://localhost:8501**. Enter your OpenAI API key in the sidebar, then ask any question.

**Features:**
- Sidebar controls: model selector, number of sources, excerpt toggle, and **front-matter filtering** (excludes structural sections like Preface/Foreword)
- Gradient-styled answer card
- Citation cards with similarity scores and source previews
- Works without an API key (retrieval-only mode)

### Option B — Generation CLI

```bash
# One-shot
python -m src.generation.cli --query "What does Ellen White say about prayer?"

# Interactive REPL
python -m src.generation.cli

# Options
python -m src.generation.cli --help
```

**Example output:**
```
Q: What does Ellen White say about prayer?

A: Ellen White emphasizes continual, living prayer rather than formal repetition.
   She writes that prayer is "the breath of the soul" and the means by which we
   receive divine strength for daily life.

────────────────────────────────────────────────────────────
Sources:
  [1] Prayer — Preface [¶13–15]
  [2] Steps to Christ — The Privilege of Prayer [¶47–49]
  [3] Prayer — Information about this Book [¶8–9]
────────────────────────────────────────────────────────────
```

> The LLM is strictly constrained to the retrieved passages — it cannot use outside knowledge or fabricate citations.

### Option C — Retrieval-only CLI (no API key needed)

```bash
# Search without LLM generation — useful for exploring the index
python -m src.retrieval.cli --query "What is the nature of faith?" --top-k 5

# Interactive REPL
python -m src.retrieval.cli
```

---

## CLI Reference

### Download books
```bash
python scripts/download_books.py [--books CODE ...] [--list] [--output data/raw] [--force] [--delay 1.0]
```

### Ingestion
```bash
python -m src.ingestion.cli [--input-dir data/raw] [--output-dir data/processed]
```

### Chunking
```bash
python -m src.preprocessing.cli \
  [--input-dir data/processed] [--output-dir data/processed] \
  [--min-words 300] [--max-words 700] [--overlap 1]
```

> `--overlap N` re-includes the last N paragraphs of a chunk into the next one for contextual continuity. Chunks never cross chapter boundaries.

### Embedding
```bash
python -m src.embeddings.cli \
  [--chunks-dir data/processed] [--db-path data/chroma] \
  [--collection egw_writings] [--model all-MiniLM-L6-v2] [--batch-size 64]
```

> Re-running is always safe — chunks use deterministic IDs so duplicates are never created.

### Retrieval
```bash
python -m src.retrieval.cli [--query TEXT] [--top-k 5] [--db-path data/chroma] [--collection egw_writings]
```
> By default, structural sections like "Preface" or "Table of Contents" are filtered out to improve relevance.

### Generation
```bash
python -m src.generation.cli [--query TEXT] [--top-k 5] [--model gpt-4o-mini]
```

### Streamlit
```bash
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
egw-citation-rag/
├── app/
│   └── streamlit_app.py       # Phase 6 — Streamlit UI
├── data/
│   ├── raw/                   # EPUB input (local only, gitignored)
│   ├── processed/             # JSON paragraphs + chunks (gitignored)
│   └── chroma/                # ChromaDB vector store (gitignored)
├── scripts/
│   └── download_books.py      # EGW book downloader (56 books)
├── src/
│   ├── ingestion/
│   │   ├── epub_parser.py     # Phase 1 — EPUB → paragraphs
│   │   └── cli.py
│   ├── preprocessing/
│   │   ├── chunker.py         # Phase 2 — Paragraphs → chunks
│   │   └── cli.py
│   ├── embeddings/
│   │   ├── store.py           # Phase 3 — Embedding + ChromaDB upsert
│   │   └── cli.py
│   ├── retrieval/
│   │   ├── retriever.py       # Phase 4 — Similarity search
│   │   └── cli.py
│   ├── generation/
│   │   ├── generator.py       # Phase 5 — Grounded LLM generation
│   │   └── cli.py
│   └── utils/
│       └── text_cleaning.py
├── tests/
│   ├── test_epub_parser.py    # 20 tests
│   ├── test_chunker.py        # 13 tests
│   ├── test_store.py          # 19 tests
│   ├── test_retriever.py      # 15 tests
│   └── test_generator.py      # 17 tests
├── .env.example
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
python -m pytest tests/ -v
# 84 tests — all phases covered, fully mocked (no network or GPU needed)
```

---

## Development Phases

| Phase | Description                 | Status |
|-------|-----------------------------|--------|
| 1     | EPUB Ingestion              | ✅ |
| 2     | Text Chunking               | ✅ |
| 3     | Embeddings + Vector Store   | ✅ |
| 4     | Retrieval                   | ✅ |
| 5     | Generation with Citations   | ✅ |
| 6     | Streamlit Interface         | ✅ |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB (local persistence) |
| LLM | OpenAI (`gpt-4o-mini` default) |
| EPUB Parsing | `ebooklib` + `beautifulsoup4` |
| UI | Streamlit |
| Testing | pytest (84 tests, fully mocked) |

---

## Privacy & Copyright

Raw EPUB files, processed JSON, and vector store data are **excluded from version control** via `.gitignore`. Only directory placeholders (`.gitkeep`) are committed. Do not commit copyrighted book content to public repositories.
