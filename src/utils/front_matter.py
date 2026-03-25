"""Utilities for detecting front-matter and structural sections in EGW books.

Front-matter sections (Preface, Foreword, Table of Contents, etc.) are structural
parts of a book that don't contain the author's primary theological content.
Filtering them out of retrieval results improves answer quality.

Sections that look like generic titles but ARE real content and should NOT be
filtered include: "Introduction", "Historical Prologue", "My First Vision",
"The Resurrection of Christ", chapter-title named sections, etc.
"""

import re

# ---------------------------------------------------------------------------
# Patterns — exact or prefix matches against chapter_title
# ---------------------------------------------------------------------------

# Exact titles that are always structural (case-insensitive)
FRONT_MATTER_EXACT: frozenset[str] = frozenset({
    "table of contents",
    "information about this book",
    "to the reader",
    "an explanation",
    "from the publisher",
    "publisher's note",
    "about the author",
    "about this book",
    "copyright",
    "dedication",
    "acknowledgments",
    "acknowledgements",
})

# Prefix patterns — chapter_title that START with these are structural
FRONT_MATTER_PREFIXES: tuple[str, ...] = (
    "preface",
    "foreword",
    "from the writings of",
    "from the bible",
    "compiled by",
    "editor's",
    "translator's",
)


def is_front_matter(chapter_title: str) -> bool:
    """Return True if the chapter title looks like a structural / front-matter section.

    Conservatively identifies only clearly non-content sections so that
    genuine chapters with descriptive titles (e.g. "My First Vision",
    "The Resurrection of Christ", "Introduction") are not filtered out.

    Args:
        chapter_title: The chapter_title field from a processed chunk.

    Returns:
        True if the section should typically be excluded from retrieval.

    Examples:
        >>> is_front_matter("Preface to the Second Edition")
        True
        >>> is_front_matter("Introduction")
        False
        >>> is_front_matter("The Resurrection of Christ")
        False
        >>> is_front_matter("Table of Contents")
        True
    """
    if not chapter_title:
        return False

    normalized = chapter_title.strip().lower()

    if normalized in FRONT_MATTER_EXACT:
        return True

    for prefix in FRONT_MATTER_PREFIXES:
        if normalized.startswith(prefix):
            return True

    return False
