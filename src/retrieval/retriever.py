"""Retriever for WhiteRAG — similarity search against ChromaDB.

Converts a user query to an embedding, queries the vector store,
and returns the top-k matching chunks with citation metadata.

No LLM is involved at this phase — this module is purely for
validating retrieval quality before generation is added.
"""

import json
from dataclasses import dataclass

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()  # loads HF_TOKEN and other vars from .env if present

from src.embeddings.store import (
    DEFAULT_COLLECTION,
    DEFAULT_DB_PATH,
    DEFAULT_MODEL,
    get_collection,
)


@dataclass
class SearchResult:
    """A single retrieved chunk with its citation metadata and score."""

    text: str
    book_title: str
    chapter_title: str
    paragraph_range: list[int]
    word_count: int
    score: float          # cosine similarity (0–1, higher is more similar)
    rank: int             # 1-indexed position in result list

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "book_title": self.book_title,
            "chapter_title": self.chapter_title,
            "paragraph_range": self.paragraph_range,
            "word_count": self.word_count,
            "text": self.text,
        }

    def format_citation(self) -> str:
        """Return a human-readable citation string.

        Example: ``Steps to Christ — God's Love for Man [¶1–3]``
        """
        start, end = self.paragraph_range
        para = f"¶{start}" if start == end else f"¶{start}–{end}"
        return f"{self.book_title} — {self.chapter_title} [{para}]"


class Retriever:
    """Query a ChromaDB collection and return ranked search results.

    The retriever lazily loads the embedding model on first use so
    that importing the module carries no startup cost.

    Args:
        db_path: ChromaDB persistence directory.
        collection_name: Name of the collection to query.
        model_name: sentence-transformers model for query embedding.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self._db_path = db_path
        self._collection_name = collection_name
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._collection = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search the vector store for chunks relevant to the query.

        Args:
            query: Free-text user query.
            top_k: Number of results to return (default 5).

        Returns:
            List of SearchResult objects, ranked by similarity score.

        Raises:
            ValueError: If the collection is empty.
            ValueError: If top_k < 1.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        collection = self._get_collection()

        if collection.count() == 0:
            raise ValueError(
                "The vector store is empty. Run Phase 2 + 3 first:\n"
                "  python -m src.preprocessing.cli\n"
                "  python -m src.embeddings.cli"
            )

        model = self._get_model()
        query_embedding = model.encode(query, convert_to_numpy=True).tolist()

        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        return self._parse_results(raw)

    def collection_info(self) -> dict:
        """Return basic info about the underlying collection.

        Returns:
            Dict with ``name`` and ``count`` keys.
        """
        col = self._get_collection()
        return {"name": col.name, "count": col.count()}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        """Lazily load and cache the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _get_collection(self):
        """Lazily load and cache the ChromaDB collection."""
        if self._collection is None:
            self._collection = get_collection(self._db_path, self._collection_name)
        return self._collection

    @staticmethod
    def _parse_results(raw: dict) -> list[SearchResult]:
        """Convert raw ChromaDB query output to SearchResult objects.

        ChromaDB returns distances (lower = more similar for cosine).
        We convert them to similarity scores: ``score = 1 - distance``.

        Args:
            raw: Raw output from ``collection.query()``.

        Returns:
            List of SearchResult objects.
        """
        results: list[SearchResult] = []

        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        for rank, (doc, meta, dist) in enumerate(
            zip(documents, metadatas, distances), start=1
        ):
            para_range = json.loads(meta["paragraph_range"])
            results.append(
                SearchResult(
                    text=doc,
                    book_title=meta["book_title"],
                    chapter_title=meta["chapter_title"],
                    paragraph_range=para_range,
                    word_count=meta["word_count"],
                    score=round(1.0 - dist, 4),
                    rank=rank,
                )
            )

        return results
