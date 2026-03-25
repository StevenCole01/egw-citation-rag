"""Interactive retrieval CLI for WhiteRAG (Phase 4).

Validates retrieval quality without any LLM involvement.
Runs as a REPL or accepts a one-shot --query argument.

Usage:
    # Interactive mode
    python -m src.retrieval.cli

    # One-shot mode
    python -m src.retrieval.cli --query "What is the nature of prayer?"
"""

import argparse
import sys

from src.retrieval.retriever import Retriever
from src.embeddings.store import DEFAULT_COLLECTION, DEFAULT_DB_PATH, DEFAULT_MODEL


def _print_results(results, query: str) -> None:
    """Pretty-print search results to stdout."""
    print(f"\n🔍 Query: {query}")
    print("=" * 60)

    if not results:
        print("No results found.")
        return

    for r in results:
        print(f"\n[{r.rank}] Score: {r.score:.4f}")
        print(f"    📖 {r.format_citation()}")
        print(f"    Words: {r.word_count}")
        print()
        # Print a preview (first 300 chars)
        preview = r.text[:300].rsplit(" ", 1)[0] + "…" if len(r.text) > 300 else r.text
        print(f"    {preview}")
        print("-" * 60)


def main(argv: list[str] | None = None) -> None:
    """Run the retrieval CLI in interactive or one-shot mode.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).
    """
    parser = argparse.ArgumentParser(
        description="Query the WhiteRAG vector store (no LLM — retrieval validation only)."
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Query string (omit for interactive REPL mode)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"ChromaDB directory (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Embedding model (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args(argv)

    print("Initializing retriever...")
    retriever = Retriever(
        db_path=args.db_path,
        collection_name=args.collection,
        model_name=args.model,
    )

    try:
        info = retriever.collection_info()
        print(f"Connected to '{info['name']}' — {info['count']} vectors indexed.\n")
    except Exception as e:
        print(f"Error connecting to vector store: {e}")
        sys.exit(1)

    if info["count"] == 0:
        print("The vector store is empty. Run Phases 2–3 first.")
        sys.exit(0)

    # One-shot mode
    if args.query:
        results = retriever.search(args.query, top_k=args.top_k)
        _print_results(results, args.query)
        return

    # Interactive REPL mode
    print("WhiteRAG Retrieval (Phase 4) — type a query, or 'exit' to quit.\n")
    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        results = retriever.search(query, top_k=args.top_k)
        _print_results(results, query)


if __name__ == "__main__":
    main()
