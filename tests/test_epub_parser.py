"""Tests for WhiteRAG EPUB ingestion pipeline."""

import json
import tempfile
from pathlib import Path

import pytest
from ebooklib import epub

from src.ingestion.epub_parser import parse_epub
from src.utils.text_cleaning import clean_text


# ---------------------------------------------------------------------------
# Tests for clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    """Unit tests for the text cleaning utility."""

    def test_strips_html_tags(self):
        assert clean_text("<b>bold</b> text") == "bold text"

    def test_strips_nested_tags(self):
        assert clean_text("<div><p>nested</p></div>") == "nested"

    def test_decodes_html_entities(self):
        assert clean_text("bread &amp; butter") == "bread & butter"
        # &lt;tag&gt; decodes to <tag>, which is then stripped as an HTML tag
        # This is expected behavior — the cleaner removes decoded tags too
        assert clean_text("&lt;tag&gt;") == ""
        assert clean_text("Tom &amp; Jerry &amp; friends") == "Tom & Jerry & friends"

    def test_collapses_whitespace(self):
        assert clean_text("hello    world") == "hello world"
        assert clean_text("line\n\n\nbreak") == "line break"
        assert clean_text("  tabs\t\there  ") == "tabs here"

    def test_removes_non_breaking_spaces(self):
        assert clean_text("hello\u00a0world") == "hello world"

    def test_removes_zero_width_spaces(self):
        assert clean_text("foo\u200bbar") == "foobar"

    def test_empty_input(self):
        assert clean_text("") == ""

    def test_whitespace_only(self):
        assert clean_text("   \n\t  ") == ""

    def test_plain_text_unchanged(self):
        assert clean_text("Hello, world.") == "Hello, world."


# ---------------------------------------------------------------------------
# EPUB fixture builder
# ---------------------------------------------------------------------------

def _create_test_epub(filepath: Path, title: str = "Test Book") -> None:
    """Create a minimal EPUB file with two chapters for testing.

    Args:
        filepath: Where to write the .epub file.
        title: Book title metadata.
    """
    book = epub.EpubBook()
    book.set_identifier("test-book-001")
    book.set_title(title)
    book.set_language("en")

    # Chapter 1
    ch1 = epub.EpubHtml(title="Chapter One", file_name="ch1.xhtml", lang="en")
    ch1.content = (
        "<html><body>"
        "<h1>Chapter One</h1>"
        "<p>First paragraph of chapter one.</p>"
        "<p>Second paragraph of chapter one.</p>"
        "</body></html>"
    )

    # Chapter 2
    ch2 = epub.EpubHtml(title="Chapter Two", file_name="ch2.xhtml", lang="en")
    ch2.content = (
        "<html><body>"
        "<h2>Chapter Two</h2>"
        "<p>First paragraph of chapter two.</p>"
        "<p>Second paragraph with <b>bold</b> and &amp; entity.</p>"
        "<p>Third paragraph of chapter two.</p>"
        "</body></html>"
    )

    book.add_item(ch1)
    book.add_item(ch2)

    # Define reading order
    book.spine = [ch1, ch2]

    # Add navigation (required by ebooklib)
    book.toc = [ch1, ch2]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    epub.write_epub(str(filepath), book)


# ---------------------------------------------------------------------------
# Tests for parse_epub
# ---------------------------------------------------------------------------

class TestParseEpub:
    """Integration tests for EPUB parsing."""

    @pytest.fixture()
    def sample_epub(self, tmp_path: Path) -> Path:
        """Create a sample EPUB file in a temporary directory."""
        filepath = tmp_path / "test_book.epub"
        _create_test_epub(filepath, title="Steps to Christ")
        return filepath

    def test_returns_list_of_dicts(self, sample_epub: Path):
        records = parse_epub(sample_epub)
        assert isinstance(records, list)
        assert all(isinstance(r, dict) for r in records)

    def test_correct_field_keys(self, sample_epub: Path):
        records = parse_epub(sample_epub)
        required_keys = {"book_title", "chapter_title", "paragraph_id", "text"}
        for record in records:
            assert set(record.keys()) == required_keys

    def test_book_title(self, sample_epub: Path):
        records = parse_epub(sample_epub)
        for record in records:
            assert record["book_title"] == "Steps to Christ"

    def test_chapter_detection(self, sample_epub: Path):
        records = parse_epub(sample_epub)
        ch1_records = [r for r in records if r["chapter_title"] == "Chapter One"]
        ch2_records = [r for r in records if r["chapter_title"] == "Chapter Two"]
        assert len(ch1_records) == 2
        assert len(ch2_records) == 3

    def test_paragraph_ids_are_sequential(self, sample_epub: Path):
        records = parse_epub(sample_epub)
        ids = [r["paragraph_id"] for r in records]
        assert ids == list(range(1, len(records) + 1))

    def test_text_is_clean(self, sample_epub: Path):
        records = parse_epub(sample_epub)
        for record in records:
            text = record["text"]
            assert "<" not in text, f"HTML tag found in text: {text}"
            assert "&amp;" not in text, f"HTML entity found in text: {text}"
            assert len(text) > 0

    def test_html_entities_decoded(self, sample_epub: Path):
        records = parse_epub(sample_epub)
        # Chapter 2, paragraph 2 has "&amp;"
        entity_para = [r for r in records if "&" in r["text"]]
        assert len(entity_para) > 0
        for r in entity_para:
            assert "&amp;" not in r["text"]

    def test_preserves_reading_order(self, sample_epub: Path):
        records = parse_epub(sample_epub)
        # Chapter 1 paragraphs should come before Chapter 2
        ch1_ids = [r["paragraph_id"] for r in records if r["chapter_title"] == "Chapter One"]
        ch2_ids = [r["paragraph_id"] for r in records if r["chapter_title"] == "Chapter Two"]
        assert max(ch1_ids) < min(ch2_ids)

    def test_total_paragraph_count(self, sample_epub: Path):
        records = parse_epub(sample_epub)
        assert len(records) == 5  # 2 + 3

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_epub(tmp_path / "nonexistent.epub")

    def test_json_serializable(self, sample_epub: Path):
        """Ensure output can be serialized to JSON (validates the contract)."""
        records = parse_epub(sample_epub)
        output = json.dumps(records, ensure_ascii=False, indent=2)
        reloaded = json.loads(output)
        assert reloaded == records
