"""EPUB parser for WhiteRAG ingestion pipeline.

Extracts structured text from EPUB files, preserving reading order
and producing citation-ready metadata (book, chapter, paragraph).
"""

from pathlib import Path
import warnings

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from src.utils.text_cleaning import clean_text


def parse_epub(filepath: str | Path) -> list[dict]:
    """Parse an EPUB file into structured paragraph records.

    Reads the EPUB spine in order, extracts chapter titles from
    heading tags (h1–h3) and collects each <p> tag as a paragraph.

    Args:
        filepath: Path to the .epub file.

    Returns:
        A list of dicts, each with keys:
            - book_title (str)
            - chapter_title (str)
            - paragraph_id (int) — 1-indexed, sequential across the book
            - text (str)

    Raises:
        FileNotFoundError: If the EPUB file does not exist.
        ebooklib.epub.EpubException: If the file is not a valid EPUB.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"EPUB file not found: {filepath}")

    book = epub.read_epub(str(filepath), options={"ignore_ncx": True})

    # Extract the book title from EPUB metadata
    book_title = _extract_book_title(book, filepath)

    records: list[dict] = []
    paragraph_id = 0
    current_chapter = "Untitled Chapter"

    # Iterate over spine-ordered document items
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content().decode("utf-8", errors="replace")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(content, "lxml")

        # Try to detect chapter title from heading tags
        chapter_heading = _extract_chapter_heading(soup)
        if chapter_heading:
            current_chapter = chapter_heading

        # Extract all paragraph elements
        for p_tag in soup.find_all("p"):
            text = clean_text(p_tag.get_text(separator=" "))

            # Skip empty paragraphs
            if not text:
                continue

            paragraph_id += 1
            records.append({
                "book_title": book_title,
                "chapter_title": current_chapter,
                "paragraph_id": paragraph_id,
                "text": text,
            })

    return records


def _extract_book_title(book: epub.EpubBook, filepath: Path) -> str:
    """Extract the book title from EPUB metadata, falling back to filename.

    Args:
        book: Parsed EpubBook object.
        filepath: Path to the EPUB file (used as fallback).

    Returns:
        The book title as a string.
    """
    title_meta = book.get_metadata("DC", "title")
    if title_meta:
        # Metadata returns list of (value, attrs) tuples
        return title_meta[0][0]
    return filepath.stem


def _extract_chapter_heading(soup: BeautifulSoup) -> str | None:
    """Extract the first heading (h1–h3) from an HTML document.

    Args:
        soup: Parsed BeautifulSoup object.

    Returns:
        The heading text, or None if no heading is found.
    """
    for tag in ["h1", "h2", "h3"]:
        heading = soup.find(tag)
        if heading:
            text = clean_text(heading.get_text(separator=" "))
            if text:
                return text
    return None
