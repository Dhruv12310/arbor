"""
StructDirect accuracy eval — all 13 FinanceBench questions across 7 docs.

Replaces LLM tree generation with native PDF structure extraction (<1 sec, 0 LLM calls).
Uses Gemini for tree search navigation (swap for v9 local model when available).

Usage:
    python scripts/eval_struct_direct.py --provider gemini
    python scripts/eval_struct_direct.py --provider groq

Results saved to: data/financebench/struct_direct_results.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import arbor
from arbor.core.tree_searcher import search_tree
from arbor.core.types import ArborConfig
from arbor.extraction.structure_extractor import extract_structure

QA_FILE  = ROOT / "data/financebench/financebench_open_source.jsonl"
PDF_DIR  = ROOT / "data/financebench/pdfs"
OUT_FILE = ROOT / "data/financebench/struct_direct_results.json"

# The 7 docs we have PDFs for
TARGET_DOCS = {
    "3M_2018_10K", "3M_2022_10K", "3M_2023Q2_10Q",
    "ACTIVISIONBLIZZARD_2019_10K",
    "ADOBE_2015_10K", "ADOBE_2016_10K", "ADOBE_2017_10K",
}


def make_provider(name: str):
    if name == "gemini":
        return arbor.GeminiProvider(api_key=os.environ["GEMINI_API_KEY"])
    if name == "groq":
        return arbor.GroqProvider()
    if name == "openai":
        return arbor.OpenAIProvider()
    raise ValueError(f"Unknown provider: {name}")


def score_retrieval(nodes, evidence_pages: list[int]) -> tuple[float, list[int]]:
    if not evidence_pages:
        return 1.0, []
    returned = set()
    for node in nodes:
        returned.update(range(node.start_index, node.end_index + 1))
    found = [p for p in evidence_pages if p in returned]
    return len(found) / len(evidence_pages), found


async def run(provider_name: str):
    # Load QA pairs for our 7 docs
    all_pairs = [
        json.loads(l)
        for l in QA_FILE.read_text(encoding="utf-8").strip().splitlines()
    ]
    qa_pairs = [q for q in all_pairs if q["doc_name"] in TARGET_DOCS]
    print(f"\nEvaluating {len(qa_pairs)} questions across {len(TARGET_DOCS)} docs")
    print(f"Provider: {provider_name}")
    print("=" * 60)

    provider = make_provider(provider_name)
    config = ArborConfig(
        max_hops=6,
        max_nodes_searched=0,
        timeout_sec=120.0,
        max_retries_on_bad_json=2,
    )

    tree_cache = {}
    extraction_times = []
    results = []

    for i, qa in enumerate(qa_pairs):
        doc  = qa["doc_name"]
        question = qa["question"]
        evidence_pages = [e["evidence_page_num"] for e in qa.get("evidence", [])]
        pdf_path = PDF_DIR / f"{doc}.pdf"

        print(f"\n[{i+1}/{len(qa_pairs)}] {doc}")
        print(f"  Q: {question[:80]}...")
        print(f"  Evidence pages: {evidence_pages}")

        result = {
            "id": qa["financebench_id"],
            "doc": doc,
            "question": question,
            "evidence_pages": evidence_pages,
            "recall": 0.0,
            "found_pages": [],
            "found_nodes": [],
            "strategy": None,
            "extraction_ms": 0,
            "search_sec": 0,
            "error": None,
        }

        try:
            # ── Step 1: Extract structure (zero LLM calls) ──────────────────
            if doc not in tree_cache:
                t0 = time.monotonic()
                tree = extract_structure(str(pdf_path))
                ms = (time.monotonic() - t0) * 1000
                extraction_times.append(ms)
                tree_cache[doc] = (tree, ms)
                print(f"  Extracted {sum(1 for _ in _iter(tree.structure))} sections in {ms:.0f}ms")
            else:
                tree, ms = tree_cache[doc]

            result["extraction_ms"] = ms
            # Detect which strategy was used (logged by extractor)
            result["strategy"] = "see_log"

            # ── Step 2: Navigate with model ──────────────────────────────────
            t1 = time.monotonic()
            sr = await search_tree(tree, question, provider, multihop=True, config=config)
            search_sec = time.monotonic() - t1

            result["found_nodes"] = sr.node_ids
            result["search_sec"] = round(search_sec, 1)

            # ── Step 3: Score retrieval recall ───────────────────────────────
            recall, found = score_retrieval(sr.nodes, evidence_pages)
            result["recall"] = recall
            result["found_pages"] = found

            # Also check ±1 page tolerance (offset from PDF vs printed pages)
            found_pages_set = set()
            for node in sr.nodes:
                found_pages_set.update(range(node.start_index, node.end_index + 1))
            found_tolerant = [p for p in evidence_pages
                              if any(abs(p - fp) <= 1 for fp in found_pages_set)]
            recall_tolerant = len(found_tolerant) / len(evidence_pages) if evidence_pages else 1.0

            status = "EXACT" if recall == 1.0 else ("+/-1" if recall_tolerant == 1.0 else "MISS")
            print(f"  Nodes: {sr.node_ids}")
            print(f"  Recall: {recall:.0%} (exact) | {recall_tolerant:.0%} (±1 page) | {status}")
            print(f"  Search: {search_sec:.1f}s")

        except Exception as e:
            result["error"] = str(e)
            print(f"  ERROR: {e}")

        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    evaluated = [r for r in results if r["error"] is None]
    n = len(evaluated)

    exact_recall   = sum(1 for r in evaluated if r["recall"] == 1.0)
    partial_recall = sum(1 for r in evaluated if 0 < r["recall"] < 1.0)
    zero_recall    = sum(1 for r in evaluated if r["recall"] == 0.0)

    # Tolerant recall (±1 page for PDF vs printed page offset)
    def tolerant(r):
        found_set = set()
        # Reconstruct from found_pages — approximate
        return r["recall"] > 0 or any(
            abs(ep - fp) <= 1
            for ep in r["evidence_pages"]
            for fp in r.get("found_pages", [])
        )

    avg_recall = sum(r["recall"] for r in evaluated) / n if n else 0
    avg_extract_ms = sum(extraction_times) / len(extraction_times) if extraction_times else 0
    avg_search_s   = sum(r["search_sec"] for r in evaluated) / n if n else 0

    print(f"\n{'='*60}")
    print(f"  STRUCT DIRECT RESULTS — {n} questions")
    print(f"{'='*60}")
    print(f"  Retrieval recall (exact)    : {avg_recall:.1%}")
    print(f"  Questions recall = 100%     : {exact_recall}/{n}")
    print(f"  Questions recall > 0%       : {exact_recall + partial_recall}/{n}")
    print(f"  Questions recall = 0%       : {zero_recall}/{n}")
    print(f"")
    print(f"  Avg extraction time         : {avg_extract_ms:.0f} ms  (0 LLM calls)")
    print(f"  Avg search time             : {avg_search_s:.1f} s")
    print(f"  Total pipeline per question : ~{avg_extract_ms/1000 + avg_search_s:.1f} s")
    print(f"")
    print(f"  FinanceBench baselines:")
    print(f"    GPT-4 zero-shot           : 68%")
    print(f"    GPT-4 naive RAG           : 72%")
    print(f"    GPT-4 long-context        : 78%")
    print(f"{'='*60}")

    OUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Results saved to {OUT_FILE}")

    # Per-question breakdown
    print(f"\n  Per-question breakdown:")
    for r in results:
        status = "OK " if r["recall"] == 1.0 else ("~~ " if r.get("recall", 0) > 0 else "XX ")
        err = f" [ERROR: {r['error'][:40]}]" if r.get("error") else ""
        print(f"    {status} p{r['evidence_pages']} | {r['doc'][:25]} | {r['question'][:50]}...{err}")


def _iter(nodes):
    for n in nodes:
        yield n
        yield from _iter(n.nodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="gemini", help="gemini | groq | openai")
    args = parser.parse_args()
    asyncio.run(run(args.provider))
