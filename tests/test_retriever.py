"""Tests for WhiteRAG retrieval pipeline (Phase 4).

ChromaDB and SentenceTransformer are both mocked so tests run
instantly without touching disk or downloading models.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.retrieval.retriever import Retriever, SearchResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_chroma_response(n: int = 3) -> dict:
    """Build a fake ChromaDB query() response for n results."""
    documents = [f"Text of chunk {i}." for i in range(n)]
    metadatas = [
        {
            "book_title": "Steps to Christ",
            "chapter_title": f"Chapter {i + 1}",
            "paragraph_range": json.dumps([i * 3 + 1, i * 3 + 3]),
            "word_count": 50 + i * 10,
        }
        for i in range(n)
    ]
    # Distances (cosine): 0 = identical, 1 = orthogonal
    distances = [0.05 * (i + 1) for i in range(n)]
    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }


def _make_retriever(collection_count: int = 10) -> tuple[Retriever, MagicMock, MagicMock]:
    """Return a Retriever with mocked model and collection."""
    retriever = Retriever(db_path="/tmp/fake_chroma", collection_name="test")

    mock_col = MagicMock()
    mock_col.count.return_value = collection_count
    mock_col.name = "test"

    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(384).astype(np.float32)

    retriever._collection = mock_col
    retriever._model = mock_model

    return retriever, mock_col, mock_model


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class TestSearchResult:
    def _make_result(self, rank: int = 1, score: float = 0.92) -> SearchResult:
        return SearchResult(
            text="Some text here.",
            book_title="Steps to Christ",
            chapter_title="God's Love for Man",
            paragraph_range=[1, 3],
            word_count=45,
            score=score,
            rank=rank,
        )

    def test_to_dict_keys(self):
        r = self._make_result()
        d = r.to_dict()
        assert set(d.keys()) == {
            "rank", "score", "book_title", "chapter_title",
            "paragraph_range", "word_count", "text"
        }

    def test_format_citation_range(self):
        r = self._make_result()
        r.paragraph_range = [1, 3]
        assert r.format_citation() == "Steps to Christ — God's Love for Man [¶1–3]"

    def test_format_citation_single(self):
        r = self._make_result()
        r.paragraph_range = [5, 5]
        assert r.format_citation() == "Steps to Christ — God's Love for Man [¶5]"

    def test_score_rounded(self):
        r = self._make_result(score=0.9876543)
        assert r.to_dict()["score"] == round(0.9876543, 4)


# ---------------------------------------------------------------------------
# Retriever.search()
# ---------------------------------------------------------------------------

class TestRetrieverSearch:

    def test_returns_list_of_search_results(self):
        retriever, mock_col, mock_model = _make_retriever()
        mock_col.query.return_value = _make_chroma_response(3)
        results = retriever.search("What is faith?", top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_correct_number_of_results(self):
        retriever, mock_col, mock_model = _make_retriever()
        mock_col.query.return_value = _make_chroma_response(5)
        results = retriever.search("grace", top_k=5)
        assert len(results) == 5

    def test_ranks_are_sequential(self):
        retriever, mock_col, mock_model = _make_retriever()
        mock_col.query.return_value = _make_chroma_response(3)
        results = retriever.search("love", top_k=3)
        assert [r.rank for r in results] == [1, 2, 3]

    def test_score_is_one_minus_distance(self):
        retriever, mock_col, mock_model = _make_retriever()
        mock_col.query.return_value = _make_chroma_response(1)
        results = retriever.search("prayer", top_k=1)
        # First distance is 0.05 → score should be 0.95
        assert abs(results[0].score - 0.95) < 1e-3

    def test_metadata_parsed_correctly(self):
        retriever, mock_col, mock_model = _make_retriever()
        mock_col.query.return_value = _make_chroma_response(1)
        results = retriever.search("salvation", top_k=1)
        r = results[0]
        assert r.book_title == "Steps to Christ"
        assert r.chapter_title == "Chapter 1"
        assert isinstance(r.paragraph_range, list)
        assert len(r.paragraph_range) == 2

    def test_paragraph_range_deserialized(self):
        retriever, mock_col, mock_model = _make_retriever()
        mock_col.query.return_value = _make_chroma_response(1)
        results = retriever.search("test", top_k=1)
        assert results[0].paragraph_range == [1, 3]

    def test_raises_on_empty_collection(self):
        retriever, mock_col, mock_model = _make_retriever(collection_count=0)
        with pytest.raises(ValueError, match="empty"):
            retriever.search("anything")

    def test_raises_on_invalid_top_k(self):
        retriever, _, _ = _make_retriever()
        with pytest.raises(ValueError, match="top_k"):
            retriever.search("test", top_k=0)

    def test_query_embedding_called_once(self):
        retriever, mock_col, mock_model = _make_retriever()
        mock_col.query.return_value = _make_chroma_response(3)
        retriever.search("faith and works", top_k=3)
        mock_model.encode.assert_called_once_with(
            "faith and works", convert_to_numpy=True
        )

    def test_top_k_capped_at_collection_size(self):
        retriever, mock_col, mock_model = _make_retriever(collection_count=2)
        mock_col.query.return_value = _make_chroma_response(2)
        retriever.search("grace", top_k=10)
        # n_results passed to chroma should be min(10, 2) = 2
        call_kwargs = mock_col.query.call_args[1]
        assert call_kwargs["n_results"] == 2


# ---------------------------------------------------------------------------
# Retriever.collection_info()
# ---------------------------------------------------------------------------

class TestCollectionInfo:
    def test_returns_name_and_count(self):
        retriever, mock_col, _ = _make_retriever(collection_count=42)
        info = retriever.collection_info()
        assert info["name"] == "test"
        assert info["count"] == 42
