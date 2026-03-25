"""CLI entry point for WhiteRAG chunking pipeline.

Usage:
    python -m src.preprocessing.cli [--input-dir DATA/PROCESSED] [--output-dir DATA/PROCESSED]
"""

import argparse
import json
import sys
from pathlib import Path

from src.preprocessing.chunker import chunk_paragraphs, load_paragraphs


def main(argv: list[str] | None = None) -> None:
    """Chunk all processed JSON files in the input directory.

    Reads each ``*_paragraphs.json`` (or ``*.json``) file produced by the
    ingestion CLI, runs the chunker, and writes ``<stem>_chunks.json``.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).
    """
    parser = argparse.ArgumentParser(
        description="Chunk processed paragraph JSON files for WhiteRAG."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing paragraph JSON files (default: data/processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for output chunk JSON files (default: data/processed)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=300,
        help="Minimum words per chunk (default: 300)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=700,
        help="Maximum words per chunk (default: 700)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=1,
        help="Number of paragraphs to overlap between chunks (default: 1)",
    )
    args = parser.parse_args(argv)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Discover paragraph JSON files (skip files already ending in _chunks.json)
    json_files = sorted(
        f for f in input_dir.glob("*.json")
        if not f.stem.endswith("_chunks") and f.name != ".gitkeep"
    )

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(json_files)} file(s) to chunk\n")

    for json_path in json_files:
        print(f"Processing: {json_path.name}")
        print("-" * 50)

        paragraphs = load_paragraphs(json_path)

        if not paragraphs:
            print("  ⚠ No paragraphs found. Skipping.\n")
            continue

        chunks = chunk_paragraphs(
            paragraphs,
            min_words=args.min_words,
            max_words=args.max_words,
            overlap_paragraphs=args.overlap,
        )

        word_counts = [c["word_count"] for c in chunks]
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0

        output_path = output_dir / (json_path.stem + "_chunks.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"  Paragraphs: {len(paragraphs)}")
        print(f"  Chunks:     {len(chunks)}")
        print(f"  Avg words:  {avg_words:.0f}")
        print(f"  Word range: {min(word_counts)}–{max(word_counts)}")
        print(f"  Output:     {output_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
