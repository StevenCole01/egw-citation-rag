"""Tests for WhiteRAG generation pipeline (Phase 5).

The OpenAI client is fully mocked so tests run offline with no API key.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.generation.generator import (
    Citation,
    GeneratedAnswer,
    Generator,
    build_prompt,
)
from src.retrieval.retriever import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(rank: int = 1, score: float = 0.85) -> SearchResult:
    return SearchResult(
        text=f"This is the text of result {rank}.",
        book_title="Steps to Christ",
        chapter_title=f"Chapter {rank}",
        paragraph_range=[rank * 3 - 2, rank * 3],
        word_count=10,
        score=score,
        rank=rank,
    )


def _make_results(n: int = 3) -> list[SearchResult]:
    return [_make_result(i + 1) for i in range(n)]


def _mock_openai_response(answer: str = "This is the generated answer."):
    """Build a mock OpenAI ChatCompletion response."""
    mock_msg = MagicMock()
    mock_msg.content = answer
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------

class TestCitation:
    def test_str_range(self):
        c = Citation("Steps to Christ", "God's Love", [1, 3])
        assert str(c) == "Steps to Christ — God's Love [¶1–3]"

    def test_str_single(self):
        c = Citation("Steps to Christ", "God's Love", [5, 5])
        assert str(c) == "Steps to Christ — God's Love [¶5]"


# ---------------------------------------------------------------------------
# GeneratedAnswer
# ---------------------------------------------------------------------------

class TestGeneratedAnswer:
    def _make_answer(self) -> GeneratedAnswer:
        return GeneratedAnswer(
            query="What is faith?",
            answer="Faith is trust in God.",
            citations=[Citation("Steps to Christ", "Chapter 1", [1, 3])],
            model="gpt-4o-mini",
            context_chunks_used=1,
        )

    def test_format_contains_query(self):
        a = self._make_answer()
        assert "What is faith?" in a.format()

    def test_format_contains_answer(self):
        a = self._make_answer()
        assert "Faith is trust in God." in a.format()

    def test_format_contains_citation(self):
        a = self._make_answer()
        assert "Steps to Christ" in a.format()

    def test_format_contains_source_header(self):
        a = self._make_answer()
        assert "Sources:" in a.format()


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_contains_query(self):
        results = _make_results(2)
        prompt = build_prompt("What is prayer?", results)
        assert "What is prayer?" in prompt

    def test_contains_source_headers(self):
        results = _make_results(3)
        prompt = build_prompt("test", results)
        assert "[Source 1]" in prompt
        assert "[Source 2]" in prompt
        assert "[Source 3]" in prompt

    def test_contains_chunk_text(self):
        results = _make_results(2)
        prompt = build_prompt("test", results)
        for r in results:
            assert r.text in prompt

    def test_contains_book_and_chapter(self):
        results = _make_results(1)
        prompt = build_prompt("test", results)
        assert "Steps to Christ" in prompt
        assert "Chapter 1" in prompt

    def test_empty_results_produces_empty_context(self):
        prompt = build_prompt("test", [])
        assert "BEGIN SOURCES" in prompt
        assert "[Source" not in prompt


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class TestGenerator:

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("src.generation.generator.Generator._get_client")
    def test_returns_generated_answer(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response()
        mock_get_client.return_value = mock_client

        gen = Generator(model="gpt-4o-mini", top_k=3)
        result = gen.generate("What is faith?", _make_results(3))

        assert isinstance(result, GeneratedAnswer)
        assert result.answer == "This is the generated answer."
        assert result.model == "gpt-4o-mini"
        assert result.context_chunks_used == 3

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("src.generation.generator.Generator._get_client")
    def test_citations_match_results(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response()
        mock_get_client.return_value = mock_client

        results = _make_results(3)
        gen = Generator(top_k=3)
        answer = gen.generate("test", results)

        assert len(answer.citations) == 3
        for i, (citation, result) in enumerate(zip(answer.citations, results)):
            assert citation.book_title == result.book_title
            assert citation.chapter_title == result.chapter_title
            assert citation.paragraph_range == result.paragraph_range

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("src.generation.generator.Generator._get_client")
    def test_top_k_limits_context(self, mock_get_client):
        """Generator should only pass top_k results to the prompt."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response()
        mock_get_client.return_value = mock_client

        results = _make_results(5)
        gen = Generator(top_k=2)
        answer = gen.generate("test", results)
        assert answer.context_chunks_used == 2
        assert len(answer.citations) == 2

    def test_raises_on_empty_results(self):
        gen = Generator()
        with pytest.raises(ValueError, match="No retrieved results"):
            gen.generate("test", [])

    @patch.dict("os.environ", {}, clear=True)
    def test_raises_without_api_key(self):
        gen = Generator()
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            gen.generate("test", _make_results(1))

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("src.generation.generator.Generator._get_client")
    def test_llm_called_with_system_prompt(self, mock_get_client):
        """System prompt must be passed in the messages array."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response()
        mock_get_client.return_value = mock_client

        gen = Generator()
        gen.generate("prayer", _make_results(1))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles
