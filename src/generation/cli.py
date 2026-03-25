"""End-to-end CLI for WhiteRAG Phase 5 — Generation with Citations.

Chains retrieval → generation in a single command.

Usage:
    # One-shot
    python -m src.generation.cli --query "What does Ellen White say about prayer?"

    # Interactive REPL
    python -m src.generation.cli
"""

import argparse
import sys

from src.retrieval.retriever import Retriever
from src.generation.generator import Generator
from src.embeddings.store import DEFAULT_COLLECTION, DEFAULT_DB_PATH, DEFAULT_MODEL


def main(argv: list[str] | None = None) -> None:
    """Run the full RAG pipeline: retrieve then generate.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).
    """
    parser = argparse.ArgumentParser(
        description="WhiteRAG: retrieve + generate grounded answers from Ellen G. White writings."
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Query string (omit for interactive REPL)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
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
        "--embed-model",
        default=DEFAULT_MODEL,
        help=f"Embedding model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        default=True,
        help="Show retrieved source excerpts alongside the answer (default: True)",
    )
    args = parser.parse_args(argv)

    print("Initializing WhiteRAG...")
    retriever = Retriever(
        db_path=args.db_path,
        collection_name=args.collection,
        model_name=args.embed_model,
    )
    generator = Generator(model=args.model, top_k=args.top_k)

    try:
        info = retriever.collection_info()
        print(f"Vector store: '{info['name']}' — {info['count']} vectors\n")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if info["count"] == 0:
        print("Vector store is empty. Run Phases 1–3 first.")
        sys.exit(0)

    def _run(query: str) -> None:
        print(f"\nRetrieving top {args.top_k} chunks...")
        results = retriever.search(query, top_k=args.top_k)

        print(f"Generating answer with {args.model}...\n")
        response = generator.generate(query, results)

        print("=" * 60)
        print(response.format())

    # One-shot mode
    if args.query:
        _run(args.query)
        return

    # Interactive REPL
    print("WhiteRAG — type a question, or 'exit' to quit.\n")
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
        _run(query)


if __name__ == "__main__":
    main()
