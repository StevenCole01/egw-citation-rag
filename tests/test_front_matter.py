"""Tests for the front_matter utility module."""

import pytest
from src.utils.front_matter import is_front_matter


class TestIsFrontMatter:
    # --- Sections that SHOULD be filtered ---

    def test_preface(self):
        assert is_front_matter("Preface") is True

    def test_preface_variant(self):
        assert is_front_matter("Preface to the Second Edition") is True

    def test_preface_first_edition(self):
        assert is_front_matter("Preface to First Edition") is True

    def test_foreword(self):
        assert is_front_matter("Foreword") is True

    def test_table_of_contents(self):
        assert is_front_matter("Table of Contents") is True

    def test_information_about_this_book(self):
        assert is_front_matter("Information about this Book") is True

    def test_to_the_reader(self):
        assert is_front_matter("To the Reader") is True

    def test_an_explanation(self):
        assert is_front_matter("An Explanation") is True

    def test_from_the_writings_of(self):
        assert is_front_matter("From the Writings of Ellen G. White") is True

    def test_publisher_note(self):
        assert is_front_matter("Publisher's Note") is True

    def test_case_insensitive(self):
        assert is_front_matter("PREFACE") is True
        assert is_front_matter("table of contents") is True
        assert is_front_matter("FOREWORD") is True

    def test_leading_whitespace(self):
        assert is_front_matter("  Preface  ") is True

    # --- Real content sections that should NOT be filtered ---

    def test_introduction_not_filtered(self):
        """Introduction is real EGW content in most books."""
        assert is_front_matter("Introduction") is False

    def test_chapter_not_filtered(self):
        assert is_front_matter("Chapter 1—God's Love for Man") is False

    def test_descriptive_chapter_not_filtered(self):
        assert is_front_matter("The Resurrection of Christ") is False

    def test_my_first_vision_not_filtered(self):
        assert is_front_matter("My First Vision") is False

    def test_historical_prologue_not_filtered(self):
        assert is_front_matter("Historical Prologue") is False

    def test_appendix_not_filtered(self):
        """Appendix contains supporting EGW content, keep it."""
        assert is_front_matter("Appendix") is False

    def test_epilogue_not_filtered(self):
        assert is_front_matter("Epilogue") is False

    def test_empty_string(self):
        assert is_front_matter("") is False

    def test_none_equivalent(self):
        """Empty chapter title edge case."""
        assert is_front_matter("   ") is False
