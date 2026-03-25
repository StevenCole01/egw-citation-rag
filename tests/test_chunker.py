"""Tests for WhiteRAG chunking pipeline."""

import pytest

from src.preprocessing.chunker import chunk_paragraphs, Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paragraphs(
    texts: list[str],
    chapter: str = "Chapter One",
    book: str = "Test Book",
    start_id: int = 1,
) -> list[dict]:
    """Build a list of paragraph dicts for testing."""
    return [
        {
            "book_title": book,
            "chapter_title": chapter,
            "paragraph_id": start_id + i,
            "text": text,
        }
        for i, text in enumerate(texts)
    ]


# 50-word sentence used to build paragraphs of known size
FIFTY_WORDS = (
    "The quick brown fox jumps over the lazy dog and then continues "
    "running through the forest until it reaches a small clearing near "
    "the edge of the old farm where the farmer once lived long ago."
)

# 100-word paragraph
HUNDRED_WORDS = (FIFTY_WORDS + " " + FIFTY_WORDS)


# ---------------------------------------------------------------------------
# Basic chunking
# ---------------------------------------------------------------------------

class TestChunkParagraphs:

    def test_empty_input(self):
        assert chunk_paragraphs([]) == []

    def test_returns_list_of_dicts(self):
        paras = _make_paragraphs([HUNDRED_WORDS] * 4)
        result = chunk_paragraphs(paras, min_words=300, max_words=700, overlap_paragraphs=0)
        assert isinstance(result, list)
        assert all(isinstance(c, dict) for c in result)

    def test_output_fields(self):
        paras = _make_paragraphs([HUNDRED_WORDS] * 4)
        chunks = chunk_paragraphs(paras, min_words=300, max_words=700, overlap_paragraphs=0)
        required = {"book_title", "chapter_title", "paragraph_range", "text", "word_count"}
        for c in chunks:
            assert set(c.keys()) == required

    def test_metadata_preserved(self):
        paras = _make_paragraphs([HUNDRED_WORDS] * 4, book="Steps to Christ", chapter="Intro")
        chunks = chunk_paragraphs(paras, min_words=300, max_words=700, overlap_paragraphs=0)
        for c in chunks:
            assert c["book_title"] == "Steps to Christ"
            assert c["chapter_title"] == "Intro"

    def test_paragraph_range_is_two_element_list(self):
        paras = _make_paragraphs([HUNDRED_WORDS] * 4)
        chunks = chunk_paragraphs(paras, min_words=300, max_words=700, overlap_paragraphs=0)
        for c in chunks:
            r = c["paragraph_range"]
            assert isinstance(r, list)
            assert len(r) == 2
            assert r[0] <= r[1]

    def test_word_count_matches_text(self):
        paras = _make_paragraphs([HUNDRED_WORDS] * 4)
        chunks = chunk_paragraphs(paras, min_words=300, max_words=700, overlap_paragraphs=0)
        for c in chunks:
            assert c["word_count"] == len(c["text"].split())

    def test_chunks_cover_all_paragraphs_without_overlap(self):
        """All paragraph IDs should appear in at least one chunk (no overlap)."""
        paras = _make_paragraphs([HUNDRED_WORDS] * 6)
        chunks = chunk_paragraphs(paras, min_words=300, max_words=700, overlap_paragraphs=0)
        # Collect all paragraph IDs covered
        covered: set[int] = set()
        for c in chunks:
            start, end = c["paragraph_range"]
            covered.update(range(start, end + 1))
        expected = {p["paragraph_id"] for p in paras}
        assert expected.issubset(covered)

    def test_single_paragraph_still_produces_chunk(self):
        paras = _make_paragraphs([HUNDRED_WORDS])
        chunks = chunk_paragraphs(paras, min_words=300, max_words=700)
        assert len(chunks) == 1

    def test_chunk_text_matches_paragraph_text(self):
        """A single-paragraph chunk's text should equal the paragraph text."""
        paras = _make_paragraphs([HUNDRED_WORDS])
        chunks = chunk_paragraphs(paras, min_words=300, max_words=700)
        assert chunks[0]["text"] == HUNDRED_WORDS


# ---------------------------------------------------------------------------
# Chapter boundary handling
# ---------------------------------------------------------------------------

class TestChapterBoundaries:

    def test_chunk_never_spans_two_chapters(self):
        ch1 = _make_paragraphs([HUNDRED_WORDS] * 3, chapter="Chapter One", start_id=1)
        ch2 = _make_paragraphs([HUNDRED_WORDS] * 3, chapter="Chapter Two", start_id=4)
        all_paras = ch1 + ch2
        chunks = chunk_paragraphs(all_paras, min_words=50, max_words=700, overlap_paragraphs=0)
        for c in chunks:
            # All paragraphs in a chunk must share the same chapter
            assert c["chapter_title"] in ("Chapter One", "Chapter Two")

        chapter_titles = [c["chapter_title"] for c in chunks]
        # Chapter One chunks must appear before Chapter Two chunks
        ch1_idxs = [i for i, t in enumerate(chapter_titles) if t == "Chapter One"]
        ch2_idxs = [i for i, t in enumerate(chapter_titles) if t == "Chapter Two"]
        if ch1_idxs and ch2_idxs:
            assert max(ch1_idxs) < min(ch2_idxs)

    def test_chapter_change_resets_accumulation(self):
        # 2 × 200-word paragraphs per chapter (400 words total each chapter)
        two_hundred = HUNDRED_WORDS + " " + HUNDRED_WORDS
        ch1 = _make_paragraphs([two_hundred] * 2, chapter="Chapter One", start_id=1)
        ch2 = _make_paragraphs([two_hundred] * 2, chapter="Chapter Two", start_id=3)
        chunks = chunk_paragraphs(ch1 + ch2, min_words=300, max_words=700, overlap_paragraphs=0)
        titles = {c["chapter_title"] for c in chunks}
        assert "Chapter One" in titles
        assert "Chapter Two" in titles


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------

class TestOverlap:

    def test_overlap_paragraphs_appear_in_consecutive_chunks(self):
        """With overlap=1, the last paragraph of chunk N appears in chunk N+1."""
        paras = _make_paragraphs([HUNDRED_WORDS] * 9, start_id=1)
        chunks = chunk_paragraphs(paras, min_words=300, max_words=700, overlap_paragraphs=1)
        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")
        for i in range(len(chunks) - 1):
            end_of_current = chunks[i]["paragraph_range"][1]
            start_of_next = chunks[i + 1]["paragraph_range"][0]
            # The start of the next chunk should be <= the end of the current
            assert start_of_next <= end_of_current

    def test_no_overlap(self):
        """With overlap=0, consecutive chunks should not share paragraph IDs."""
        paras = _make_paragraphs([HUNDRED_WORDS] * 9, start_id=1)
        chunks = chunk_paragraphs(paras, min_words=300, max_words=700, overlap_paragraphs=0)
        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test no-overlap")
        for i in range(len(chunks) - 1):
            end_of_current = chunks[i]["paragraph_range"][1]
            start_of_next = chunks[i + 1]["paragraph_range"][0]
            assert start_of_next > end_of_current
