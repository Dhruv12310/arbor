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


async def process(pdf_path: Path, provider: arbor.GroqProvider) -> None:
    arxiv_id = pdf_path.stem
    out_file = TRAINING_DIR / f"{arxiv_id}.json"
    err_file = ERRORS_DIR / f"{arxiv_id}.json"

    if out_file.exists():
        print(f"Already processed, skipping: {arxiv_id}")
        return

    pages = get_page_contents(str(pdf_path))
    document_text = "\n\n".join(p.text for p in pages)
    total_tokens = sum(p.token_count for p in pages)

    config = arbor.ArborConfig(
        add_node_ids=True,
        add_summaries=True,
        add_node_text=False,
        max_tokens_per_node=8000,   # Groq free tier: 12k TPM limit
    )

    t0 = time.monotonic()
    try:
        tree = await asyncio.wait_for(
            arbor.generate_tree(str(pdf_path), provider, config),
            timeout=300,
        )
    except Exception as e:
        elapsed = time.monotonic() - t0
        ERRORS_DIR.mkdir(parents=True, exist_ok=True)
        err_file.write_text(json.dumps({
            "arxiv_id": arxiv_id,
            "error": str(e),
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
        print(f"  [error] {arxiv_id}: {e}", file=sys.stderr)
        return

    elapsed = round(time.monotonic() - t0, 1)
    nodes = count_nodes(tree.structure)

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "arxiv_id": arxiv_id,
        "document_text": document_text,
        "page_count": len(pages),
        "total_tokens": total_tokens,
        "tree": tree.to_dict(),
        "metadata": {
            "model": provider.model,
            "provider": "groq",
            "generation_time_seconds": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }, indent=2))

    print(f"Generated tree for {arxiv_id} in {elapsed}s ({len(pages)} pages, {nodes} nodes)")


def main():
    parser = argparse.ArgumentParser(description="Generate Arbor training data for a PDF")
    parser.add_argument("pdf", type=Path, help="Path to PDF file")
    args = parser.parse_args()

    if not args.pdf.exists():
        sys.exit(f"File not found: {args.pdf}")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY environment variable not set")

    provider = arbor.GroqProvider(api_key=api_key)
    asyncio.run(process(args.pdf, provider))


if __name__ == "__main__":
    main()
