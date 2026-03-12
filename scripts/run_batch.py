"""
Batch process all PDFs in data/pdfs/ through generate_training_pair.py.

Usage:
    python scripts/run_batch.py
    python scripts/run_batch.py --limit 10
    python scripts/run_batch.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time

from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
TRAINING_DIR = ROOT / "data" / "training"
ERRORS_DIR = ROOT / "data" / "errors"
LOG_FILE = ROOT / "data" / "batch_log.txt"
PROGRESS_FILE = ROOT / "data" / "batch_progress.json"
GENERATOR = Path(__file__).parent / "generate_training_pair.py"
RATE_LIMIT = 3  # seconds between documents


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def save_progress(processed: int, errors: int, skipped: int, total: int) -> None:
    PROGRESS_FILE.write_text(json.dumps({
        "processed": processed,
        "errors": errors,
        "skipped": skipped,
        "total": total,
        "updated": datetime.now(timezone.utc).isoformat(),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Batch process PDFs through Arbor")
    parser.add_argument("--limit", type=int, default=None, help="Max files to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run, don't process")
    parser.add_argument("--delay", type=int, default=5, help="Seconds between papers (default: 5)")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY") and not args.dry_run:
        sys.exit("GROQ_API_KEY environment variable not set")

    all_pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not all_pdfs:
        sys.exit(f"No PDFs found in {PDF_DIR}")

    already_done = {f.stem for f in TRAINING_DIR.glob("*.json")}
    already_errored = {f.stem for f in ERRORS_DIR.glob("*.json")}
    already_done |= already_errored

    pending = [p for p in all_pdfs if p.stem not in already_done]
    skipped = len(all_pdfs) - len(pending)

    if args.limit:
        pending = pending[:args.limit]

    total = len(all_pdfs)
    print(f"PDFs found: {total} | Already done: {skipped} | To process: {len(pending)}")

    if args.dry_run:
        for p in pending:
            print(f"  would process: {p.name}")
        print(f"\nDry run complete. Would process {len(pending)} file(s).")
        return

    processed = errors = 0

    for i, pdf in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] {pdf.name}")
        result = subprocess.run(
            [sys.executable, str(GENERATOR), str(pdf)],
            env=os.environ,
            capture_output=False,
        )
        # Check for error file written by the subprocess (it exits 0 on handled errors)
        if result.returncode != 0 or (ERRORS_DIR / f"{pdf.stem}.json").exists():
            errors += 1
            if result.returncode != 0:
                log(f"ERROR {pdf.stem}: subprocess exited with code {result.returncode}")
        else:
            processed += 1

        save_progress(processed, errors, skipped, total)

        if i < len(pending):
            time.sleep(args.delay)

    print(f"\nProcessed: {processed}/{len(pending)}, Errors: {errors}, Skipped: {skipped}")
    log(f"Batch complete - Processed: {processed}, Errors: {errors}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
