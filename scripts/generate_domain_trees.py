"""
Batch-generate JSON trees from domain PDFs using Gemini (GeminiProvider).

Reads  : data/domain_pdfs/<sector>/*.pdf   (from collect_domain_pdfs.py)
Writes : data/training_domain/<sector>/<stem>.json

Usage:
    python scripts/generate_domain_trees.py              # all sectors
    python scripts/generate_domain_trees.py --sector legal
    python scripts/generate_domain_trees.py --dry-run    # count only
    python scripts/generate_domain_trees.py --resume     # skip done files
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import arbor
from arbor.extraction.pdf_extractor import get_page_contents
from arbor.utils.tree_utils import count_nodes

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent.parent / "data"
PDF_DIR     = DATA_DIR / "domain_pdfs"
OUT_DIR     = DATA_DIR / "training_domain"
ERRORS_DIR  = DATA_DIR / "training_domain_errors"
PROGRESS_F  = DATA_DIR / "domain_trees_progress.json"

# ── Config ────────────────────────────────────────────────────────────────────
TIMEOUT_SEC   = 600   # 10 min per PDF
RATE_LIMIT    = 0.5   # seconds between Gemini calls
MIN_NODES     = 8     # skip trees with fewer nodes (too flat to be useful)

# Match generate_training_pair.py exactly — add_summaries=True is what produced good v6 trees
ARBORCONFIG = arbor.ArborConfig(
    add_node_ids   = True,
    add_summaries  = True,
    add_node_text  = False,
    max_tokens_per_node      = 8000,
    max_concurrent_llm_calls = 20,  # Max parallelism — paid tier handles 4K RPM
)


def load_progress() -> dict:
    if PROGRESS_F.exists():
        return json.loads(PROGRESS_F.read_text())
    return {}


def save_progress(progress: dict) -> None:
    PROGRESS_F.write_text(json.dumps(progress, indent=2))


async def process_pdf(pdf_path: Path, sector: str, provider, out_dir: Path) -> dict:
    stem    = pdf_path.stem
    out_f   = out_dir / f"{stem}.json"
    err_dir = ERRORS_DIR / sector
    err_f   = err_dir / f"{stem}.json"

    if out_f.exists():
        return {"status": "skipped", "stem": stem}

    # Retry up to 3 times with backoff for 503/429 errors
    last_err = ""
    for attempt in range(3):
        try:
            tree = await asyncio.wait_for(
                arbor.generate_tree(str(pdf_path), provider, ARBORCONFIG),
                timeout=TIMEOUT_SEC,
            )
            break  # success
        except asyncio.TimeoutError:
            err_dir.mkdir(parents=True, exist_ok=True)
            err_f.write_text(json.dumps({"error": "timeout", "pdf": str(pdf_path)}))
            return {"status": "error", "stem": stem, "reason": "timeout"}
        except Exception as e:
            last_err = str(e)
            # Retry on 503/429, fail fast on 404
            if "503" in last_err or "429" in last_err:
                wait = 10 * (2 ** attempt)  # 10s, 20s, 40s
                print(f"    [retry {attempt+1}/3 in {wait}s] {last_err[:80]}", flush=True)
                await asyncio.sleep(wait)
            else:
                break
    else:
        err_dir.mkdir(parents=True, exist_ok=True)
        err_f.write_text(json.dumps({"error": last_err, "pdf": str(pdf_path)}))
        return {"status": "error", "stem": stem, "reason": last_err[:120]}

    if "last_err" in dir() and last_err and tree is None:
        err_dir.mkdir(parents=True, exist_ok=True)
        err_f.write_text(json.dumps({"error": last_err, "pdf": str(pdf_path)}))
        return {"status": "error", "stem": stem, "reason": last_err[:120]}

    if tree is None:
        return {"status": "error", "stem": stem, "reason": "tree is None"}

    # Count nodes from tree.structure (same as generate_training_pair.py)
    node_count = count_nodes(tree.structure)

    if node_count < MIN_NODES:
        return {"status": "skipped_flat", "stem": stem, "nodes": node_count}

    # Save in same format as generate_training_pair.py / training_v6/
    pages = get_page_contents(str(pdf_path))
    document_text = "\n\n".join(p.text for p in pages)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_f.write_text(json.dumps({
        "arxiv_id":      stem,
        "document_text": document_text,
        "page_count":    len(pages),
        "tree":          tree.to_dict(),
        "metadata": {
            "sector":   sector,
            "provider": "gemini",
        },
    }, indent=2, ensure_ascii=True), encoding="utf-8")
    return {"status": "ok", "stem": stem, "nodes": node_count}


def main():
    parser = argparse.ArgumentParser(description="Generate trees from domain PDFs")
    parser.add_argument("--sector",  default="all", help="Sector name or 'all'")
    parser.add_argument("--dry-run", action="store_true", help="Count PDFs only, no API calls")
    parser.add_argument("--resume",  action="store_true", help="Skip already-processed PDFs")
    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY or GOOGLE_API_KEY in .env", flush=True)
        sys.exit(1)

    provider = arbor.GeminiProvider(api_key=api_key)

    # Collect PDFs
    if args.sector == "all":
        pdf_paths = sorted(PDF_DIR.rglob("*.pdf"))
    else:
        pdf_paths = sorted((PDF_DIR / args.sector).rglob("*.pdf"))

    if not pdf_paths:
        print(f"No PDFs found in {PDF_DIR}/{args.sector}", flush=True)
        sys.exit(1)

    print(f"Found {len(pdf_paths)} PDFs across sectors", flush=True)

    if args.dry_run:
        by_sector = {}
        for p in pdf_paths:
            sector = p.parent.name
            by_sector.setdefault(sector, 0)
            by_sector[sector] += 1
        print()
        for s, n in sorted(by_sector.items()):
            print(f"  {s:<20} {n} PDFs")
        print(f"\n  TOTAL: {len(pdf_paths)} PDFs")
        print(f"\n  Estimated cost: ~${len(pdf_paths) * 0.003:.2f} (Gemini 2.5 Flash Lite)")
        print(f"  Estimated time: ~{len(pdf_paths) * 2 // 60} min")
        return

    progress  = load_progress() if args.resume else {}
    done = 0; skipped = 0; errors = 0; flat = 0
    lock = asyncio.Lock() if False else None  # placeholder

    # Filter to only unprocessed PDFs
    pending = []
    for pdf_path in pdf_paths:
        sector = pdf_path.parent.name
        key    = f"{sector}/{pdf_path.stem}"
        if args.resume and key in progress and progress[key] == "ok":
            skipped += 1
        else:
            pending.append(pdf_path)

    print(f"Skipped (already done): {skipped}  |  To process: {len(pending)}", flush=True)

    PARALLEL = 5  # Process 5 PDFs at once — safe for 4K RPM paid tier

    async def run_all():
        nonlocal done, errors, flat
        semaphore = asyncio.Semaphore(PARALLEL)

        async def process_one(pdf_path, idx):
            nonlocal done, errors, flat
            sector = pdf_path.parent.name
            key    = f"{sector}/{pdf_path.stem}"
            async with semaphore:
                result = await process_pdf(pdf_path, sector, provider, OUT_DIR / sector)
            status = result["status"]
            progress[key] = status
            save_progress(progress)
            if status == "ok":
                done += 1
                print(f"[{idx}/{len(pending)}] OK   {key} ({result['nodes']} nodes)", flush=True)
            elif status == "skipped_flat":
                flat += 1
                print(f"[{idx}/{len(pending)}] FLAT {key} ({result['nodes']} nodes)", flush=True)
            elif status == "skipped":
                print(f"[{idx}/{len(pending)}] SKIP {key}", flush=True)
            else:
                errors += 1
                print(f"[{idx}/{len(pending)}] ERR  {key} — {result.get('reason','')}", flush=True)

        tasks = [process_one(p, i+1) for i, p in enumerate(pending)]
        await asyncio.gather(*tasks)

    asyncio.run(run_all())

    print(f"\nDone. OK={done}  Flat={flat}  Skipped={skipped}  Errors={errors}")
    print(f"Trees saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
