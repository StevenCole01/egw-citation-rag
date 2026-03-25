"""Text cleaning utilities for WhiteRAG ingestion pipeline."""

import re
from html import unescape


def clean_text(raw: str) -> str:
    """Clean raw text extracted from HTML content.

    Strips residual HTML tags, decodes HTML entities,
    collapses whitespace, and trims leading/trailing space.

    Args:
        raw: Raw text that may contain HTML fragments.

    Returns:
        Cleaned, human-readable text.
    """
    # Decode HTML entities (&amp; → &, etc.)
    text = unescape(raw)

    # Remove any residual HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Replace non-breaking spaces and other unicode whitespace with regular space
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")  # zero-width space

    # Collapse multiple whitespace characters into a single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text
