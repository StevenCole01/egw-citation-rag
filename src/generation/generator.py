"""Answer generator for WhiteRAG (Phase 5).

Builds a context-grounded prompt from retrieved chunks, calls an LLM,
and returns a structured response with an answer and explicit citations.

Supported backends:
  - OpenAI (default): gpt-4o-mini, gpt-4o, etc.
"""

import os
import textwrap
from dataclasses import dataclass, field

from dotenv import load_dotenv

from src.retrieval.retriever import SearchResult

load_dotenv()


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    """A single source citation."""

    book_title: str
    chapter_title: str
    paragraph_range: list[int]

    def __str__(self) -> str:
        start, end = self.paragraph_range
        para = f"¶{start}" if start == end else f"¶{start}–{end}"
        return f"{self.book_title} — {self.chapter_title} [{para}]"


@dataclass
class GeneratedAnswer:
    """The full response from the generation step."""

    query: str
    answer: str
    citations: list[Citation]
    model: str
    context_chunks_used: int

    def format(self) -> str:
        """Format the answer and citations for display."""
        lines = [
            f"Q: {self.query}",
            "",
            f"A: {self.answer}",
            "",
            "─" * 60,
            "Sources:",
        ]
        for i, c in enumerate(self.citations, 1):
            lines.append(f"  [{i}] {c}")
        lines.append("─" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a research assistant that answers questions based ONLY on the 
    provided source excerpts from Ellen G. White's writings.

    Rules you MUST follow:
    1. Base your answer SOLELY on the provided context. Do not use any outside knowledge.
    2. If the context does not contain enough information to answer the question, 
       say "The provided sources do not contain sufficient information to answer this question."
    3. Do NOT invent, fabricate, or infer citations not present in the context.
    4. Be concise and direct. Aim for 2–4 sentences.
    5. Do not include phrases like "according to the context" or "the text says" — 
       just answer naturally.
""")


def build_prompt(query: str, results: list[SearchResult]) -> str:
    """Build a retrieval-augmented prompt from a query and search results.

    Each result is formatted as a numbered excerpt with its citation
    header so the model can reference the source material accurately.

    Args:
        query: The user's question.
        results: Retrieved chunks from the vector store.

    Returns:
        A formatted user-turn message string.
    """
    excerpts = []
    for r in results:
        start, end = r.paragraph_range
        para = f"¶{start}" if start == end else f"¶{start}–{end}"
        header = f"[Source {r.rank}] {r.book_title} — {r.chapter_title} [{para}]"
        excerpts.append(f"{header}\n{r.text}")

    context_block = "\n\n".join(excerpts)

    return (
        f"Use only the following source excerpts to answer the question.\n\n"
        f"--- BEGIN SOURCES ---\n{context_block}\n--- END SOURCES ---\n\n"
        f"Question: {query}"
    )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator:
    """LLM-backed answer generator grounded in retrieved context.

    Args:
        model: OpenAI model name (default: gpt-4o-mini).
        top_k: Number of retrieved chunks to include in the prompt.
        temperature: LLM temperature (0 = deterministic).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        top_k: int = 5,
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._top_k = top_k
        self._temperature = temperature
        self._client = None

    def generate(self, query: str, results: list[SearchResult]) -> GeneratedAnswer:
        """Generate a grounded answer from a query and retrieved chunks.

        Args:
            query: The user's question.
            results: Retrieved SearchResult objects from the Retriever.

        Returns:
            A GeneratedAnswer with the answer text and citations.

        Raises:
            ValueError: If results list is empty.
            RuntimeError: If OPENAI_API_KEY is not set.
        """
        if not results:
            raise ValueError("No retrieved results to generate from.")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file:\n"
                "  OPENAI_API_KEY=sk-..."
            )

        top_results = results[: self._top_k]
        user_prompt = build_prompt(query, top_results)
        client = self._get_client(api_key)

        response = client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer_text = response.choices[0].message.content.strip()

        citations = [
            Citation(
                book_title=r.book_title,
                chapter_title=r.chapter_title,
                paragraph_range=r.paragraph_range,
            )
            for r in top_results
        ]

        return GeneratedAnswer(
            query=query,
            answer=answer_text,
            citations=citations,
            model=self._model,
            context_chunks_used=len(top_results),
        )

    def _get_client(self, api_key: str):
        """Lazily create and cache the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise RuntimeError(
                    "openai package not installed. Run: pip install openai"
                )
            self._client = OpenAI(api_key=api_key)
        return self._client
