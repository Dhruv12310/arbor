"""
Download public domain PDFs from arXiv for Arbor benchmarking.

Usage:
    python scripts/collect_pdfs.py
    python scripts/collect_pdfs.py --count 10
    python scripts/collect_pdfs.py --count 10 --resume
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

import feedparser
import requests

ARXIV_API = "http://export.arxiv.org/api/query"
CATEGORIES = ["cs.AI", "cs.CL", "cs.LG", "cs.IR"]
DATA_DIR = Path(__file__).parent.parent / "data"
PDF_DIR = DATA_DIR / "pdfs"
META_FILE = DATA_DIR / "metadata.json"
MAX_PAGES = 30
RATE_LIMIT = 3  # seconds between requests


def get_page_count(pdf_path: Path) -> int:
    """Count PDF pages using PyPDF2."""
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            return len(PyPDF2.PdfReader(f).pages)
    except Exception:
        return 0


def fetch_ids(category: str, count: int) -> list[dict]:
    """Query arXiv API and return list of {arxiv_id, title, url}."""
    results = []
    batch = 50
    for start in range(0, count, batch):
        params = (
            f"search_query=cat:{category}&start={start}"
            f"&max_results={min(batch, count - start)}&sortBy=submittedDate&sortOrder=descending"
        )
        url = f"{ARXIV_API}?{params}"
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                arxiv_id = entry.id.split("/abs/")[-1].replace("/", "_")
                pdf_url = next(
                    (l.href for l in entry.links if l.get("type") == "application/pdf"),
                    f"https://arxiv.org/pdf/{arxiv_id}"
                )
                results.append({
                    "arxiv_id": arxiv_id,
                    "title": entry.title.replace("\n", " ").strip(),
                    "category": category,
                    "url": pdf_url,
                })
        except Exception as e:
            print(f"  [warn] API fetch failed for {category} start={start}: {e}")
        time.sleep(RATE_LIMIT)
    return results


def download_pdf(url: str, dest: Path) -> bool:
    """Download a PDF, return True on success."""
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "ArborResearch/1.0"})
        r.raise_for_status()
        if b"%PDF" not in r.content[:10]:
            return False
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Collect arXiv PDFs for Arbor benchmarks")
    parser.add_argument("--count", type=int, default=50, help="Papers per category (default: 50)")
    parser.add_argument("--resume", action="store_true", help="Skip already-downloaded PDFs")
    args = parser.parse_args()

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing metadata if resuming
    existing: dict[str, dict] = {}
    if args.resume and META_FILE.exists():
        for entry in json.loads(META_FILE.read_text()):
            existing[entry["arxiv_id"]] = entry

    total_target = len(CATEGORIES) * args.count
    downloaded = 0
    skipped = 0
    failed = 0
    metadata: list[dict] = list(existing.values())

    print(f"Collecting {args.count} papers x {len(CATEGORIES)} categories = {total_target} total")
    print(f"Output: {PDF_DIR}\n")

    for category in CATEGORIES:
        print(f"[{category}] Fetching paper list...")
        candidates = fetch_ids(category, args.count * 2)  # fetch extra to account for skips

        cat_count = 0
        for paper in candidates:
            if cat_count >= args.count:
                break

            arxiv_id = paper["arxiv_id"]
            dest = PDF_DIR / f"{arxiv_id}.pdf"

            # Resume: skip if already downloaded
            if args.resume and arxiv_id in existing:
                cat_count += 1
                skipped += 1
                continue

            time.sleep(RATE_LIMIT)
            ok = download_pdf(paper["url"], dest)
            if not ok:
                print(f"  [skip] {arxiv_id} — download failed")
                failed += 1
                continue

            pages = get_page_count(dest)
            if pages > MAX_PAGES:
                dest.unlink()
                print(f"  [skip] {arxiv_id} — {pages} pages (>{MAX_PAGES})")
                failed += 1
                continue

            paper["pages"] = pages
            metadata.append(paper)
            cat_count += 1
            downloaded += 1
            total_so_far = downloaded + skipped
            print(f"  Downloaded {total_so_far}/{total_target}: {arxiv_id}.pdf ({pages} pages) — {paper['title'][:60]}")

    META_FILE.write_text(json.dumps(metadata, indent=2))
    print(f"\nDone. Downloaded: {downloaded}, Skipped (resume): {skipped}, Failed: {failed}")
    print(f"Metadata saved to {META_FILE}")


if __name__ == "__main__":
    main()
