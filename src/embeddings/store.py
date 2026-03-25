"""Embedding store for WhiteRAG — ChromaDB + sentence-transformers.

Provides a thin wrapper around ChromaDB that handles:
- Initializing a persistent local collection
- Generating embeddings via sentence-transformers
- Upserting chunks with full citation metadata
- Reloading an existing collection from disk
"""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # loads HF_TOKEN and other vars from .env if present

import chromadb
from chromadb import Collection
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from src.utils.front_matter import is_front_matter


# Default model — fast, high-quality, 384-dim embeddings
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "egw_writings"
DEFAULT_DB_PATH = "data/chroma"


def get_collection(
    db_path: str | Path = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
) -> Collection:
    """Load or create a persistent ChromaDB collection.

    If the collection already exists on disk it is returned as-is,
    allowing incremental upserts without re-embedding everything.

    Args:
        db_path: Directory for ChromaDB persistence.
        collection_name: Name of the ChromaDB collection.

    Returns:
        A ChromaDB Collection object.
    """
    db_path = Path(db_path)
    db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def embed_and_upsert(
    chunks: list[dict],
    collection: Collection,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    show_progress: bool = True,
) -> int:
    """Generate embeddings for chunks and upsert them into ChromaDB.

    Chunks are deduplicated by a deterministic ID built from
    ``book_title``, ``chapter_title``, and ``paragraph_range`` so
    that re-running the ingestion is idempotent.

    Args:
        chunks: List of chunk dicts as produced by the chunker.
        collection: Target ChromaDB collection.
        model_name: sentence-transformers model name/path.
        batch_size: Number of chunks to embed per forward pass.
        show_progress: Print progress to stdout.

    Returns:
        Number of chunks upserted.
    """
    if not chunks:
        return 0

    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    ids = [_make_chunk_id(c) for c in chunks]
    metadatas = [_make_metadata(c) for c in chunks]

    total = len(texts)
    upserted = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        batch_meta = metadatas[start:end]

        embeddings = model.encode(
            batch_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_meta,
        )

        upserted += len(batch_texts)
        if show_progress:
            print(f"  Embedded {upserted}/{total} chunks...", end="\r")

    if show_progress:
        print()  # newline after \r progress

    return upserted


def get_collection_stats(collection: Collection) -> dict[str, Any]:
    """Return basic statistics about a collection.

    Args:
        collection: A ChromaDB Collection object.

    Returns:
        Dict with ``name`` and ``count`` keys.
    """
    return {
        "name": collection.name,
        "count": collection.count(),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_chunk_id(chunk: dict) -> str:
    """Build a deterministic, stable ID for a chunk.

    Format: ``<book_slug>__<chapter_slug>__<start>-<end>``

    Args:
        chunk: Chunk dict with ``book_title``, ``chapter_title``,
               and ``paragraph_range`` keys.

    Returns:
        A URL-safe string ID.
    """
    book = _slugify(chunk["book_title"])
    chapter = _slugify(chunk["chapter_title"])
    start, end = chunk["paragraph_range"]
    return f"{book}__{chapter}__{start}-{end}"


def _make_metadata(chunk: dict) -> dict[str, Any]:
    """Extract ChromaDB-compatible metadata from a chunk dict.

    ChromaDB metadata values must be str, int, or float — no lists.
    ``paragraph_range`` is serialized as a JSON string.

    Args:
        chunk: Chunk dict.

    Returns:
        Flat metadata dict.
    """
    return {
        "book_title": chunk["book_title"],
        "chapter_title": chunk["chapter_title"],
        "paragraph_range": json.dumps(chunk["paragraph_range"]),
        "word_count": chunk["word_count"],
        "is_front_matter": int(is_front_matter(chunk["chapter_title"])),
    }


def _slugify(text: str) -> str:
    """Convert a string to a lowercase, underscore-separated slug.

    Args:
        text: Input string.

    Returns:
        Slugified string safe for use in IDs.
    """
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text[:80]  # cap length
