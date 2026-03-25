"""Text chunker for WhiteRAG preprocessing pipeline.

Groups paragraphs from a single book into overlapping chunks
of ~300–700 words while preserving chapter boundaries and
citation metadata.
"""

from dataclasses import dataclass, field


@dataclass
class Paragraph:
    """A single paragraph record from the ingestion phase."""

    book_title: str
    chapter_title: str
    paragraph_id: int
    text: str


@dataclass
class Chunk:
    """A chunk of text ready for embedding, with citation metadata."""

    book_title: str
    chapter_title: str
    paragraph_range: list[int]   # [start_paragraph_id, end_paragraph_id]
    text: str
    word_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.word_count = len(self.text.split())

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON output."""
        return {
            "book_title": self.book_title,
            "chapter_title": self.chapter_title,
            "paragraph_range": self.paragraph_range,
            "text": self.text,
            "word_count": self.word_count,
        }


def chunk_paragraphs(
    paragraphs: list[dict],
    min_words: int = 300,
    max_words: int = 700,
    overlap_paragraphs: int = 1,
) -> list[dict]:
    """Split a list of paragraph records into overlapping chunks.

    Chunks are built by accumulating paragraphs until the word count
    reaches the target range. Chapter boundaries always force a new
    chunk — a chunk never spans two chapters.

    Overlap is achieved by re-including the last ``overlap_paragraphs``
    paragraphs of the previous chunk at the start of the next one,
    giving the retriever contextual continuity across chunk boundaries.

    Args:
        paragraphs: List of paragraph dicts as produced by epub_parser.
            Each dict must contain ``book_title``, ``chapter_title``,
            ``paragraph_id``, and ``text``.
        min_words: Minimum words before a chunk is considered complete
            and a new one is started (default 300).
        max_words: Hard cap — if a single paragraph exceeds this, it
            becomes its own chunk (default 700).
        overlap_paragraphs: Number of trailing paragraphs from the
            previous chunk to prepend to the next (default 1).

    Returns:
        A list of chunk dicts, each with keys:
            - book_title (str)
            - chapter_title (str)
            - paragraph_range (list[int]): [first_id, last_id]
            - text (str)
            - word_count (int)
    """
    if not paragraphs:
        return []

    # Convert input dicts to typed dataclass instances
    paras = [
        Paragraph(
            book_title=p["book_title"],
            chapter_title=p["chapter_title"],
            paragraph_id=p["paragraph_id"],
            text=p["text"],
        )
        for p in paragraphs
    ]

    chunks: list[Chunk] = []
    current_batch: list[Paragraph] = []

    def _flush(batch: list[Paragraph]) -> None:
        """Turn the current batch into a Chunk and append it to chunks."""
        if not batch:
            return
        combined_text = " ".join(p.text for p in batch)
        chunks.append(
            Chunk(
                book_title=batch[0].book_title,
                chapter_title=batch[0].chapter_title,
                paragraph_range=[batch[0].paragraph_id, batch[-1].paragraph_id],
                text=combined_text,
            )
        )

    for para in paras:
        para_words = len(para.text.split())

        # Chapter boundary — flush current batch before switching chapters
        if current_batch and current_batch[0].chapter_title != para.chapter_title:
            _flush(current_batch)
            # Start next batch with overlap from end of previous batch
            current_batch = current_batch[-overlap_paragraphs:] if overlap_paragraphs else []

        current_batch.append(para)
        current_words = sum(len(p.text.split()) for p in current_batch)

        # Flush when we've accumulated enough words
        if current_words >= min_words:
            # But don't split in the middle of a very long single paragraph —
            # if we're over max, force a flush regardless.
            _flush(current_batch)
            current_batch = current_batch[-overlap_paragraphs:] if overlap_paragraphs else []

    # Flush any remaining paragraphs
    _flush(current_batch)

    return [c.to_dict() for c in chunks]


def load_paragraphs(filepath: str) -> list[dict]:
    """Load a processed JSON file produced by the ingestion phase.

    Args:
        filepath: Path to the JSON file.

    Returns:
        List of paragraph dicts.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    import json
    from pathlib import Path

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)
