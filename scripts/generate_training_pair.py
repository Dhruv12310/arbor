"""
Process a single PDF through Arbor and save training data.

Usage:
    python scripts/generate_training_pair.py data/pdfs/2301.12345.pdf
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import arbor
from arbor.extraction.pdf_extractor import get_page_contents
from arbor.utils.tree_utils import count_nodes

TRAINING_DIR = Path(__file__).parent.parent / "data" / "training"
ERRORS_DIR = Path(__file__).parent.parent / "data" / "errors"


async def process(pdf_path: Path, provider: arbor.GeminiProvider, output_dir: Path = TRAINING_DIR) -> None:
    arxiv_id = pdf_path.stem
    out_file = output_dir / f"{arxiv_id}.json"
    err_file = ERRORS_DIR / f"{arxiv_id}.json"

    if out_file.exists():
        print(f"Already processed, skipping: {arxiv_id}")
        return

    pages = get_page_contents(str(pdf_path))
    document_text = "\n\n".join(p.text for p in pages)
    total_tokens = sum(p.token_count for p in pages)

    config = arbor.ArborConfig(
        add_node_ids=True,
        add_summaries=True,       # summaries essential for TreeGen training
        add_node_text=False,
        max_tokens_per_node=8000,
        max_concurrent_llm_calls=5,
    )

    t0 = time.monotonic()
    try:
        tree = await asyncio.wait_for(
            arbor.generate_tree(str(pdf_path), provider, config),
            timeout=1800,
        )
    except Exception as e:
        elapsed = time.monotonic() - t0
        ERRORS_DIR.mkdir(parents=True, exist_ok=True)
        err_file.write_text(json.dumps({
            "arxiv_id": arxiv_id,
            "error": type(e).__name__ if not str(e) else str(e),
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
        print(f"  [error] {arxiv_id}: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed = round(time.monotonic() - t0, 1)
    nodes = count_nodes(tree.structure)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "arxiv_id": arxiv_id,
        "document_text": document_text,
        "page_count": len(pages),
        "total_tokens": total_tokens,
        "tree": tree.to_dict(),
        "metadata": {
            "model": provider.model,
            "provider": "gemini",
            "generation_time_seconds": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }, indent=2))

    print(f"Generated tree for {arxiv_id} in {elapsed}s ({len(pages)} pages, {nodes} nodes)")


def main():
    parser = argparse.ArgumentParser(description="Generate Arbor training data for a PDF")
    parser.add_argument("pdf", type=Path, help="Path to PDF file")
    parser.add_argument("--output-dir", type=Path, default=TRAINING_DIR,
                        help="Directory to write output JSON (default: data/training)")
    args = parser.parse_args()

    if not args.pdf.exists():
        sys.exit(f"File not found: {args.pdf}")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY environment variable not set")

    provider = arbor.GeminiProvider(api_key=api_key, model="gemini-2.5-flash-lite")
    asyncio.run(process(args.pdf, provider, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
