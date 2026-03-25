"""CLI entry point for WhiteRAG EPUB ingestion.

Usage:
    python -m src.ingestion.cli [--input-dir DATA/RAW] [--output-dir DATA/PROCESSED]
"""

import argparse
import json
import sys
from pathlib import Path

from src.ingestion.epub_parser import parse_epub


def main(argv: list[str] | None = None) -> None:
    """Run EPUB ingestion on all .epub files in the input directory.

    Discovers EPUB files, parses each one, and writes structured
    JSON output to the output directory (one file per book).

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).
    """
    parser = argparse.ArgumentParser(
        description="Ingest EPUB files into structured JSON for WhiteRAG."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing .epub files (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for output JSON files (default: data/processed)",
    )
    args = parser.parse_args(argv)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Discover EPUB files
    epub_files = sorted(input_dir.glob("*.epub"))
    if not epub_files:
        print(f"No .epub files found in {input_dir}")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(epub_files)} EPUB file(s) in {input_dir}\n")

    for epub_path in epub_files:
        print(f"Processing: {epub_path.name}")
        print("-" * 50)

        records = parse_epub(epub_path)

        if not records:
            print("  ⚠ No paragraphs extracted. Skipping.\n")
            continue

        # Collect stats
        book_title = records[0]["book_title"]
        chapters = sorted(set(r["chapter_title"] for r in records))
        total_paragraphs = len(records)

        # Write JSON output
        output_filename = epub_path.stem + ".json"
        output_path = output_dir / output_filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        # Print summary
        print(f"  Book:       {book_title}")
        print(f"  Chapters:   {len(chapters)}")
        print(f"  Paragraphs: {total_paragraphs}")
        print(f"  Output:     {output_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
