"""CLI entry point for WhiteRAG embedding pipeline.

Usage:
    python -m src.embeddings.cli [--chunks-dir DATA/PROCESSED] [--db-path DATA/CHROMA]
"""

import argparse
import json
import sys
from pathlib import Path

from src.embeddings.store import (
    DEFAULT_COLLECTION,
    DEFAULT_DB_PATH,
    DEFAULT_MODEL,
    embed_and_upsert,
    get_collection,
    get_collection_stats,
)


def main(argv: list[str] | None = None) -> None:
    """Embed all chunk JSON files and upsert them into ChromaDB.

    Discovers ``*_chunks.json`` files, embeds each one via
    sentence-transformers, and stores results in a persistent
    ChromaDB collection. Running again is safe — chunks are
    upserted by deterministic ID, so duplicates are never stored.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).
    """
    parser = argparse.ArgumentParser(
        description="Embed WhiteRAG chunks and store them in ChromaDB."
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing *_chunks.json files (default: data/processed)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path(DEFAULT_DB_PATH),
        help=f"ChromaDB persistence directory (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"sentence-transformers model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64)",
    )
    args = parser.parse_args(argv)

    chunks_dir: Path = args.chunks_dir

    if not chunks_dir.exists():
        print(f"Error: Chunks directory does not exist: {chunks_dir}")
        sys.exit(1)

    chunk_files = sorted(chunks_dir.glob("*_chunks.json"))
    if not chunk_files:
        print(f"No *_chunks.json files found in {chunks_dir}")
        print("Run Phase 2 (python -m src.preprocessing.cli) first.")
        sys.exit(0)

    print(f"Connecting to ChromaDB at: {args.db_path}")
    collection = get_collection(db_path=args.db_path, collection_name=args.collection)
    print(f"Collection '{args.collection}' ready.")
    print(f"Model: {args.model}\n")

    total_upserted = 0

    for chunk_file in chunk_files:
        print(f"Processing: {chunk_file.name}")
        print("-" * 50)

        with open(chunk_file, encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            print("  ⚠ No chunks found. Skipping.\n")
            continue

        n = embed_and_upsert(
            chunks,
            collection,
            model_name=args.model,
            batch_size=args.batch_size,
            show_progress=True,
        )
        total_upserted += n
        print(f"  Upserted: {n} chunks\n")

    stats = get_collection_stats(collection)
    print("=" * 50)
    print(f"Done. Collection '{stats['name']}' now has {stats['count']} vectors.")


if __name__ == "__main__":
    main()
