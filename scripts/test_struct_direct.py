"""
Quick test: StructDirect extraction + v9 model navigation on a local PDF.

Usage:
    python scripts/test_struct_direct.py --pdf data/financebench/pdfs/3M_2018_10K.pdf
    python scripts/test_struct_direct.py --pdf path/to/any.pdf --question "What is the revenue?"

Shows:
  - Which extraction strategy was used
  - How many sections were found
  - The section tree structure
  - v9 model navigation result (if --provider is set)
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from arbor.extraction.structure_extractor import extract_structure
from arbor.core.types import ArborConfig


def print_tree(nodes, indent=0):
    for node in nodes:
        marker = "  " * indent + "├─ "
        print(f"{marker}[{node.node_id}] {node.title} (pp. {node.start_index}–{node.end_index})")
        if node.nodes:
            print_tree(node.nodes, indent + 1)


async def run(pdf_path: str, question: str | None, provider_name: str | None):
    # ── Step 1: Extract structure (zero LLM calls) ────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Extracting structure from: {Path(pdf_path).name}")
    print(f"{'='*60}")

    import time
    t0 = time.monotonic()
    tree = extract_structure(pdf_path)
    elapsed = time.monotonic() - t0

    total_nodes = sum(1 for _ in _iter_nodes(tree.structure))
    print(f"  Extraction time : {elapsed*1000:.0f} ms")
    print(f"  Total sections  : {total_nodes}")
    print(f"  Page count      : {tree.structure[-1].end_index if tree.structure else '?'}")
    print()
    print("  Structure:")
    print_tree(tree.structure)

    # Optionally save the extracted tree
    out_path = ROOT / "data" / "struct_direct_test.json"
    out_path.write_text(json.dumps(tree.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Saved to {out_path}")

    # ── Step 2: Run v9 model navigation (optional) ────────────────────────────
    if not question or not provider_name:
        print("\n  (Skip navigation — pass --question and --provider to test search)")
        return

    print(f"\n{'='*60}")
    print(f"  Running v9 navigation")
    print(f"  Question: {question}")
    print(f"{'='*60}\n")

    import arbor
    from arbor.core.tree_searcher import search_tree

    if provider_name == "gemini":
        import os
        provider = arbor.GeminiProvider(api_key=os.environ["GEMINI_API_KEY"])
    elif provider_name == "groq":
        provider = arbor.GroqProvider()
    elif provider_name == "openai":
        provider = arbor.OpenAIProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Use: gemini, groq, openai")

    config = ArborConfig(
        max_hops=6,
        max_nodes_searched=0,
        timeout_sec=120.0,
        max_retries_on_bad_json=2,
    )

    t1 = time.monotonic()
    result = await search_tree(tree, question, provider, multihop=True, config=config)
    search_time = time.monotonic() - t1

    print(f"  Found nodes     : {result.node_ids}")
    print(f"  Search time     : {search_time:.1f}s")
    print()
    for node in result.nodes:
        print(f"  [{node.node_id}] {node.title} — pages {node.start_index}–{node.end_index}")


def _iter_nodes(nodes):
    for node in nodes:
        yield node
        yield from _iter_nodes(node.nodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf",      required=True, help="Path to PDF file")
    parser.add_argument("--question", default=None,  help="Question to search for")
    parser.add_argument("--provider", default=None,  help="LLM provider: gemini, groq, openai")
    args = parser.parse_args()

    asyncio.run(run(args.pdf, args.question, args.provider))
