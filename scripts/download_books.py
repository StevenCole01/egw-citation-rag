"""Download EGW EPUB files from egwwritings.org.

URL format: https://media2.egwwritings.org/epub/en_{CODE}.epub

Usage:
    # Download all books in the catalog
    python scripts/download_books.py

    # Download specific books by code
    python scripts/download_books.py --books SC DA GC

    # List available books without downloading
    python scripts/download_books.py --list

    # Download to a custom directory
    python scripts/download_books.py --output data/raw
"""

import argparse
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Book catalog — code: (title, category)
# ---------------------------------------------------------------------------

BOOK_CATALOG: dict[str, tuple[str, str]] = {
    # ── Conflict of the Ages Series ────────────────────────────────────────
    "PP":   ("Patriarchs and Prophets",          "Conflict of the Ages"),
    "PK":   ("Prophets and Kings",               "Conflict of the Ages"),
    "DA":   ("The Desire of Ages",               "Conflict of the Ages"),
    "AA":   ("The Acts of the Apostles",         "Conflict of the Ages"),
    "GC":   ("The Great Controversy",            "Conflict of the Ages"),

    # ── Steps to Christ & Devotional Classics ──────────────────────────────
    "SC":   ("Steps to Christ",                  "Devotional"),
    "MB":   ("Thoughts from the Mount of Blessing", "Devotional"),
    "COL":  ("Christ's Object Lessons",          "Devotional"),
    "MH":   ("The Ministry of Healing",          "Devotional"),
    "Ed":   ("Education",                        "Devotional"),
    "FW":   ("Faith and Works",                  "Devotional"),

    # ── Daily Devotionals ──────────────────────────────────────────────────
    "CC":   ("Conflict and Courage",             "Daily Devotional"),
    "HP":   ("In Heavenly Places",               "Daily Devotional"),
    "ML":   ("My Life Today",                    "Daily Devotional"),
    "OHC":  ("Our High Calling",                 "Daily Devotional"),
    "TDG":  ("This Day With God",                "Daily Devotional"),
    "UL":   ("The Upward Look",                  "Daily Devotional"),
    "RC":   ("Reflecting Christ",                "Daily Devotional"),
    "Mar":  ("Maranatha",                        "Daily Devotional"),
    "AG":   ("God's Amazing Grace",              "Daily Devotional"),
    "TMK":  ("That I May Know Him",              "Daily Devotional"),
    "FLB":  ("The Faith I Live By",              "Daily Devotional"),
    "RY":   ("Lift Him Up",                      "Daily Devotional"),

    # ── Testimonies for the Church ─────────────────────────────────────────
    "1T":   ("Testimonies for the Church Vol. 1", "Testimonies"),
    "2T":   ("Testimonies for the Church Vol. 2", "Testimonies"),
    "3T":   ("Testimonies for the Church Vol. 3", "Testimonies"),
    "4T":   ("Testimonies for the Church Vol. 4", "Testimonies"),
    "5T":   ("Testimonies for the Church Vol. 5", "Testimonies"),
    "6T":   ("Testimonies for the Church Vol. 6", "Testimonies"),
    "7T":   ("Testimonies for the Church Vol. 7", "Testimonies"),
    "8T":   ("Testimonies for the Church Vol. 8", "Testimonies"),
    "9T":   ("Testimonies for the Church Vol. 9", "Testimonies"),

    # ── Family & Christian Living ──────────────────────────────────────────
    "AH":   ("The Adventist Home",               "Christian Living"),
    "CG":   ("Child Guidance",                   "Christian Living"),
    "MYP":  ("Messages to Young People",         "Christian Living"),
    "CT":   ("Counsels to Parents, Teachers, and Students", "Christian Living"),
    "WM":   ("Welfare Ministry",                 "Christian Living"),
    "CH":   ("Counsels on Health",               "Christian Living"),
    "CD":   ("Counsels on Diet and Foods",       "Christian Living"),
    "Te":   ("Temperance",                       "Christian Living"),
    "HR":   ("Healthful Living",                 "Christian Living"),

    # ── Ministry & Evangelism ─────────────────────────────────────────────
    "Ev":   ("Evangelism",                       "Ministry"),
    "PM":   ("Pastoral Ministry",                "Ministry"),
    "MM":   ("Medical Ministry",                 "Ministry"),
    "TM":   ("Testimonies to Ministers",         "Ministry"),
    "GW":   ("Gospel Workers",                   "Ministry"),
    "Ev2":  ("Christian Service",                "Ministry"),
    "MinC": ("Ministry to the Cities",           "Ministry"),
    "SpM":  ("Spalding and Magan Collection",    "Ministry"),

    # ── Prophecy & Last Day Events ────────────────────────────────────────
    "EW":   ("Early Writings",                   "Prophecy"),
    "LDE":  ("Last Day Events",                  "Prophecy"),
    "PrT":  ("Prayer",                           "Prayer"),

    # ── Selected Messages ─────────────────────────────────────────────────
    "1SM":  ("Selected Messages Book 1",         "Selected Messages"),
    "2SM":  ("Selected Messages Book 2",         "Selected Messages"),
    "3SM":  ("Selected Messages Book 3",         "Selected Messages"),

    # ── Letters & Manuscripts ─────────────────────────────────────────────
    "VSS":  ("The Voice in Speech and Song",     "Speech & Music"),
}

BASE_URL = "https://media2.egwwritings.org/epub/en_{code}.epub"
DEFAULT_OUTPUT = Path("data/raw")
DELAY_BETWEEN_REQUESTS = 1.0   # seconds — be polite to the server


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def list_catalog() -> None:
    """Print the full book catalog grouped by category."""
    categories: dict[str, list[tuple[str, str]]] = {}
    for code, (title, category) in BOOK_CATALOG.items():
        categories.setdefault(category, []).append((code, title))

    print(f"\n{'CODE':<8} {'TITLE'}")
    print("─" * 70)
    for category, books in sorted(categories.items()):
        print(f"\n  [{category}]")
        for code, title in sorted(books):
            print(f"  {code:<8} {title}")
    print(f"\nTotal: {len(BOOK_CATALOG)} books\n")


def download_book(code: str, output_dir: Path, force: bool = False) -> bool:
    """Download a single EPUB by book code.

    Args:
        code: Book code from the catalog (e.g. 'SC', 'DA').
        output_dir: Directory to save the file.
        force: If True, overwrite existing files.

    Returns:
        True if download succeeded, False otherwise.
    """
    if code not in BOOK_CATALOG:
        print(f"  ✗ Unknown code '{code}'. Run --list to see available books.")
        return False

    title, _ = BOOK_CATALOG[code]
    filename = f"{code}.epub"
    dest = output_dir / filename

    if dest.exists() and not force:
        print(f"  ⏭  {code:<8} {title!r} — already exists (skip with --force to re-download)")
        return True

    url = BASE_URL.format(code=code)
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        size = 0
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                size += len(chunk)

        print(f"  ✓ {code:<8} {title!r} — {size / 1024:.0f} KB → {dest}")
        return True

    except requests.HTTPError as e:
        print(f"  ✗ {code:<8} {title!r} — HTTP {e.response.status_code}: {url}")
        return False
    except requests.RequestException as e:
        print(f"  ✗ {code:<8} {title!r} — {e}")
        return False


def download_books(
    codes: list[str],
    output_dir: Path,
    force: bool = False,
    delay: float = DELAY_BETWEEN_REQUESTS,
) -> tuple[int, int]:
    """Download multiple books by code.

    Args:
        codes: List of book codes to download.
        output_dir: Directory to save EPUBs.
        force: Re-download even if file exists.
        delay: Seconds to wait between requests.

    Returns:
        Tuple of (success_count, fail_count).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0

    for i, code in enumerate(codes):
        result = download_book(code, output_dir, force=force)
        if result:
            ok += 1
        else:
            fail += 1
        if i < len(codes) - 1:
            time.sleep(delay)

    return ok, fail


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Entry point for the book downloader CLI."""
    parser = argparse.ArgumentParser(
        description="Download EGW EPUB files from egwwritings.org."
    )
    parser.add_argument(
        "--books", "-b",
        nargs="+",
        metavar="CODE",
        help="Book codes to download (e.g. SC DA GC). Omit to download all.",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available books and their codes, then exit.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-download files that already exist.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DELAY_BETWEEN_REQUESTS,
        help=f"Seconds between requests (default: {DELAY_BETWEEN_REQUESTS})",
    )
    args = parser.parse_args(argv)

    if args.list:
        list_catalog()
        return

    codes = args.books if args.books else list(BOOK_CATALOG.keys())
    total = len(codes)

    print(f"\nDownloading {total} book(s) → {args.output}\n")
    ok, fail = download_books(codes, args.output, force=args.force, delay=args.delay)

    print(f"\n{'─' * 50}")
    print(f"Done — {ok} succeeded, {fail} failed.")

    if ok > 0:
        print(
            f"\nNext steps:\n"
            f"  python -m src.ingestion.cli\n"
            f"  python -m src.preprocessing.cli\n"
            f"  python -m src.embeddings.cli\n"
        )

    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
