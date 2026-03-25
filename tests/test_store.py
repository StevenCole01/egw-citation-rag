"""Tests for WhiteRAG embedding store (Phase 3).

Uses a temporary in-memory ChromaDB client to avoid touching disk
and avoids loading a real sentence-transformers model by injecting
pre-computed dummy embeddings.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embeddings.store import (
    _make_chunk_id,
    _make_metadata,
    _slugify,
    embed_and_upsert,
    get_collection,
    get_collection_stats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_chunks(n: int = 3, chapter: str = "Chapter One") -> list[dict]:
    """Build n minimal chunk dicts for testing."""
    return [
        {
            "book_title": "Steps to Christ",
            "chapter_title": chapter,
            "paragraph_range": [i * 3 + 1, i * 3 + 3],
            "text": f"This is chunk number {i + 1} with enough words to be meaningful.",
            "word_count": 12,
        }
        for i in range(n)
    ]


@pytest.fixture()
def tmp_collection(tmp_path):
    """Return a fresh, isolated ChromaDB collection in a temp directory."""
    return get_collection(db_path=tmp_path / "chroma", collection_name="test_col")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestSlugify:
    def test_lowercases(self):
        assert _slugify("Steps To Christ") == "steps_to_christ"

    def test_removes_punctuation(self):
        assert _slugify("God's Love for Man!") == "gods_love_for_man"

    def test_collapses_spaces(self):
        assert _slugify("  too   many   spaces  ") == "too_many_spaces"

    def test_caps_at_80_chars(self):
        long = "a" * 100
        assert len(_slugify(long)) <= 80


class TestMakeChunkId:
    def test_format(self):
        chunk = {
            "book_title": "Steps to Christ",
            "chapter_title": "God's Love",
            "paragraph_range": [1, 3],
        }
        cid = _make_chunk_id(chunk)
        assert cid == "steps_to_christ__gods_love__1-3"

    def test_unique_per_chunk(self):
        chunks = _make_chunks(5)
        ids = [_make_chunk_id(c) for c in chunks]
        assert len(set(ids)) == len(ids)


class TestMakeMetadata:
    def test_all_fields_present(self):
        chunk = _make_chunks(1)[0]
        meta = _make_metadata(chunk)
        assert "book_title" in meta
        assert "chapter_title" in meta
        assert "paragraph_range" in meta
        assert "word_count" in meta

    def test_paragraph_range_is_json_string(self):
        chunk = _make_chunks(1)[0]
        meta = _make_metadata(chunk)
        # Must be a JSON string (ChromaDB only stores str/int/float)
        assert isinstance(meta["paragraph_range"], str)
        parsed = json.loads(meta["paragraph_range"])
        assert isinstance(parsed, list)

    def test_values_are_primitive_types(self):
        chunk = _make_chunks(1)[0]
        meta = _make_metadata(chunk)
        for v in meta.values():
            assert isinstance(v, (str, int, float))


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

class TestGetCollection:
    def test_creates_collection(self, tmp_path):
        col = get_collection(tmp_path / "chroma", "my_col")
        assert col.name == "my_col"

    def test_existing_collection_reloaded(self, tmp_path):
        db = tmp_path / "chroma"
        col1 = get_collection(db, "my_col")
        col2 = get_collection(db, "my_col")
        assert col1.name == col2.name

    def test_creates_directory(self, tmp_path):
        db = tmp_path / "deep" / "nested" / "chroma"
        get_collection(db, "test")
        assert db.exists()


# ---------------------------------------------------------------------------
# Embed and upsert (model is mocked — no GPU/download needed)
# ---------------------------------------------------------------------------

class TestEmbedAndUpsert:

    def _mock_model(self, chunks: list[dict]):
        """Return a mock SentenceTransformer that produces deterministic embeddings."""
        mock = MagicMock()
        # Return fake 384-dim embeddings, one per text
        mock.encode.return_value = np.random.rand(len(chunks), 384).astype(np.float32)
        return mock

    @patch("src.embeddings.store.SentenceTransformer")
    def test_returns_correct_count(self, MockST, tmp_collection):
        chunks = _make_chunks(3)
        MockST.return_value.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        n = embed_and_upsert(chunks, tmp_collection, show_progress=False)
        assert n == 3

    @patch("src.embeddings.store.SentenceTransformer")
    def test_collection_count_increases(self, MockST, tmp_collection):
        chunks = _make_chunks(4)
        MockST.return_value.encode.return_value = np.random.rand(4, 384).astype(np.float32)
        assert tmp_collection.count() == 0
        embed_and_upsert(chunks, tmp_collection, show_progress=False)
        assert tmp_collection.count() == 4

    @patch("src.embeddings.store.SentenceTransformer")
    def test_idempotent_upsert(self, MockST, tmp_collection):
        """Running twice with the same chunks should not duplicate records."""
        chunks = _make_chunks(3)
        MockST.return_value.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        embed_and_upsert(chunks, tmp_collection, show_progress=False)
        MockST.return_value.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        embed_and_upsert(chunks, tmp_collection, show_progress=False)
        assert tmp_collection.count() == 3  # not 6

    @patch("src.embeddings.store.SentenceTransformer")
    def test_empty_chunks_returns_zero(self, MockST, tmp_collection):
        n = embed_and_upsert([], tmp_collection, show_progress=False)
        assert n == 0
        assert tmp_collection.count() == 0

    @patch("src.embeddings.store.SentenceTransformer")
    def test_metadata_stored_correctly(self, MockST, tmp_collection):
        chunks = _make_chunks(1)
        MockST.return_value.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        embed_and_upsert(chunks, tmp_collection, show_progress=False)

        cid = _make_chunk_id(chunks[0])
        result = tmp_collection.get(ids=[cid], include=["metadatas", "documents"])
        assert result["metadatas"][0]["book_title"] == "Steps to Christ"
        assert result["metadatas"][0]["chapter_title"] == "Chapter One"

    @patch("src.embeddings.store.SentenceTransformer")
    def test_batching_handles_small_batch(self, MockST, tmp_collection):
        """Batch size larger than dataset should still work."""
        chunks = _make_chunks(2)
        MockST.return_value.encode.return_value = np.random.rand(2, 384).astype(np.float32)
        n = embed_and_upsert(chunks, tmp_collection, batch_size=100, show_progress=False)
        assert n == 2


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestGetCollectionStats:
    @patch("src.embeddings.store.SentenceTransformer")
    def test_stats_reflect_count(self, MockST, tmp_collection):
        chunks = _make_chunks(5)
        MockST.return_value.encode.return_value = np.random.rand(5, 384).astype(np.float32)
        embed_and_upsert(chunks, tmp_collection, show_progress=False)
        stats = get_collection_stats(tmp_collection)
        assert stats["count"] == 5
        assert stats["name"] == "test_col"
