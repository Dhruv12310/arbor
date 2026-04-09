"""
Tree Quality Diagnostic — separates StructDirect problems from TreeSearch problems.

Uses Claude Haiku as an oracle navigator through the exact same search_tree(multihop=True)
pipeline. If Claude scores high (>70%), trees are structurally sound and model training
is the bottleneck. If Claude scores low, StructDirect is broken.

Each failure is classified as:
  UNREACHABLE — evidence pages not covered by any tree node (StructDirect bug)
  MISS        — pages exist in tree but Claude navigated wrong (model/hierarchy issue)
  HIT         — correctly found

Usage:
    # Full 150-question FinanceBench eval (~$0.85, ~5-8 min)
    python scripts/diagnose_tree_quality.py financebench

    # Quick sample (first N questions)
    python scripts/diagnose_tree_quality.py financebench --limit 20

    # Resume after interruption
    python scripts/diagnose_tree_quality.py financebench --resume

    # Use Claude Sonnet for stronger oracle
    python scripts/diagnose_tree_quality.py financebench --model sonnet

    # Interactive mode — any PDF
    python scripts/diagnose_tree_quality.py interactive data/financebench/pdfs/3M_2018_10K.pdf
    python scripts/diagnose_tree_quality.py interactive report.pdf --question "What is revenue?"

Results saved to: data/financebench/haiku_diagnostic_results.json
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

from arbor.core.tree_searcher import search_tree
from arbor.core.types import ArborConfig, DocumentTree, TreeNode
from arbor.extraction.structure_extractor import extract_structure
from arbor.providers.anthropic_provider import AnthropicProvider

QA_FILE   = ROOT / "data/financebench/financebench_open_source.jsonl"
PDF_DIR   = ROOT / "data/financebench/pdfs"
CACHE_DIR = ROOT / "data/financebench/trees_cache"
OUT_FILE  = ROOT / "data/financebench/haiku_diagnostic_results.json"


# ── Scoring helpers ────────────────────────────────────────────────────────────

def score_retrieval(nodes: list, evidence_pages: list[int]) -> tuple[float, list[int]]:
    """Exact recall: what fraction of evidence pages appear in returned nodes."""
    if not evidence_pages:
        return 1.0, []
    returned = set()
    for node in nodes:
        returned.update(range(node.start_index, node.end_index + 1))
    found = [p for p in evidence_pages if p in returned]
    return len(found) / len(evidence_pages), found


def check_reachability(tree: DocumentTree, evidence_pages: list[int]) -> float:
    """
    Can evidence pages be found within ANY node in the tree?
    If < 1.0, StructDirect failed to index those pages — no navigator can fix this.
    """
    if not evidence_pages:
        return 1.0
    covered: set[int] = set()
    for node in _iter_all_nodes(tree.structure):
        covered.update(range(node.start_index, node.end_index + 1))
    reachable = [p for p in evidence_pages if p in covered]
    return len(reachable) / len(evidence_pages)


def _iter_all_nodes(nodes: list):
    for node in nodes:
        yield node
        yield from _iter_all_nodes(node.nodes)


def _count_nodes(nodes: list) -> int:
    return sum(1 + _count_nodes(n.nodes) for n in nodes)


def _max_depth(nodes: list, depth: int = 1) -> int:
    if not nodes:
        return depth - 1
    return max(_max_depth(n.nodes, depth + 1) for n in nodes)


# ── Tree cache ─────────────────────────────────────────────────────────────────

def load_or_extract(doc_name: str) -> DocumentTree:
    """Load tree from cache or extract from PDF and save to cache."""
    cache_path = CACHE_DIR / f"{doc_name}.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return DocumentTree.from_dict(data)

    pdf_path = PDF_DIR / f"{doc_name}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            f"Run: python scripts/collect_financebench.py"
        )

    tree = extract_structure(str(pdf_path))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(tree.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return tree


# ── FinanceBench mode ──────────────────────────────────────────────────────────

async def run_financebench(args):
    model_label = args.model if args.model != "haiku" else "haiku (claude-haiku-4-5-20251001)"
    print(f"\nHAIKU ORACLE DIAGNOSTIC")
    print(f"Model   : {model_label}")
    print(f"Output  : {args.output}")
    print("=" * 65)

    # Load QA pairs
    qa_pairs = [
        json.loads(line)
        for line in QA_FILE.read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]
    if args.limit:
        qa_pairs = qa_pairs[: args.limit]
    print(f"Questions: {len(qa_pairs)}")

    # Load existing results for resume
    done_ids: set = set()
    existing_results: list = []
    if args.resume and args.output.exists():
        existing_results = json.loads(args.output.read_text(encoding="utf-8"))
        done_ids = {r["id"] for r in existing_results}
        print(f"Resuming: {len(done_ids)} already done, {len(qa_pairs) - len(done_ids)} remaining")

    provider = AnthropicProvider(model=args.model)
    config = ArborConfig(
        max_hops=args.max_hops,
        max_nodes_searched=0,
        timeout_sec=120.0,
        max_retries_on_bad_json=2,
    )

    results: list = list(existing_results)
    tree_cache: dict[str, DocumentTree] = {}

    for i, qa in enumerate(qa_pairs):
        qa_id = qa["financebench_id"]
        if qa_id in done_ids:
            continue

        doc = qa["doc_name"]
        question = qa["question"]
        evidence_pages = [e["evidence_page_num"] for e in qa.get("evidence", [])]

        print(f"\n[{i+1}/{len(qa_pairs)}] {doc}")
        print(f"  Q : {question[:80]}{'...' if len(question) > 80 else ''}")
        print(f"  Ev: {evidence_pages}")

        result = {
            "id": qa_id,
            "doc": doc,
            "question": question,
            "evidence_pages": evidence_pages,
            "reachability": 0.0,
            "recall": 0.0,
            "found_pages": [],
            "found_nodes": [],
            "strategy": None,
            "node_count": 0,
            "thinking": "",
            "search_sec": 0.0,
            "error": None,
        }

        try:
            # Step 1: Extract/load tree
            if doc not in tree_cache:
                t0 = time.monotonic()
                tree = load_or_extract(doc)
                ms = (time.monotonic() - t0) * 1000
                tree_cache[doc] = tree
                n = _count_nodes(tree.structure)
                print(f"  Tree: {n} nodes via {tree.extraction_strategy} ({ms:.0f}ms)")
            else:
                tree = tree_cache[doc]

            result["strategy"] = tree.extraction_strategy
            result["node_count"] = _count_nodes(tree.structure)

            # Step 2: Reachability check (is evidence even in the tree?)
            reachability = check_reachability(tree, evidence_pages)
            result["reachability"] = reachability

            # Step 3: Navigate with Claude
            t1 = time.monotonic()
            sr = await search_tree(tree, question, provider, multihop=True, config=config)
            search_sec = time.monotonic() - t1

            result["found_nodes"] = sr.node_ids
            result["thinking"] = sr.thinking[:400] if sr.thinking else ""
            result["search_sec"] = round(search_sec, 1)

            # Step 4: Score
            recall, found_pages = score_retrieval(sr.nodes, evidence_pages)
            result["recall"] = recall
            result["found_pages"] = found_pages

            # Classify
            if recall == 1.0:
                status = "HIT       "
            elif reachability < 1.0:
                status = "UNREACHABLE"
            else:
                status = "MISS      "

            print(f"  {status} | recall={recall:.0%} | reachable={reachability:.0%} | nodes={sr.node_ids} | {search_sec:.1f}s")

        except Exception as e:
            result["error"] = str(e)
            print(f"  ERROR: {e}")

        results.append(result)

        # Save after every question (resumable)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        # Small delay to avoid rate limits
        await asyncio.sleep(0.3)

    _print_summary(results, args.model, args.output)


def _print_summary(results: list, model: str, output_path: Path = OUT_FILE):
    evaluated = [r for r in results if r["error"] is None]
    n = len(evaluated)
    if n == 0:
        print("\nNo results to summarize.")
        return

    hits         = [r for r in evaluated if r["recall"] == 1.0]
    misses       = [r for r in evaluated if r["recall"] == 0.0 and r["reachability"] == 1.0]
    unreachable  = [r for r in evaluated if r["reachability"] < 1.0]
    partial      = [r for r in evaluated if 0.0 < r["recall"] < 1.0]
    errors       = [r for r in results if r.get("error")]

    oracle_pct   = len(hits) / n * 100
    model_pct    = 56.7  # v14b current best

    # By strategy
    from collections import defaultdict
    by_strat: dict = defaultdict(lambda: {"hits": 0, "total": 0})
    for r in evaluated:
        s = r.get("strategy") or "unknown"
        by_strat[s]["total"] += 1
        if r["recall"] == 1.0:
            by_strat[s]["hits"] += 1

    print(f"\n{'='*65}")
    print(f"  HAIKU ORACLE DIAGNOSTIC — {n} questions  (model: {model})")
    print(f"{'='*65}")
    print(f"  Claude oracle (exact recall = 100%) : {len(hits)}/{n} = {oracle_pct:.1f}%")
    print(f"  v14b fine-tuned model (reference)   : 82/150 = {model_pct:.1f}%")
    if n == 150:
        gap = oracle_pct - model_pct
        print(f"  Training headroom                   : +{gap:.1f}pp available")
    print()
    print(f"  Failure breakdown ({n - len(hits)} failures):")
    print(f"    UNREACHABLE (StructDirect bug)    : {len(unreachable)}")
    print(f"    MISS (Claude navigated wrong)     : {len(misses)}")
    print(f"    PARTIAL (some evidence found)     : {len(partial)}")
    print(f"    ERRORS                            : {len(errors)}")
    print()
    print(f"  By extraction strategy:")
    for strat, d in sorted(by_strat.items(), key=lambda x: -x[1]["total"]):
        pct = d["hits"] / d["total"] * 100 if d["total"] else 0
        print(f"    {strat:<20}: {d['hits']:>3}/{d['total']:<3} = {pct:.0f}%")
    print()

    # Verdict
    print(f"  VERDICT:")
    if len(unreachable) > 15:
        print(f"    StructDirect is missing pages in {len(unreachable)} questions.")
        print(f"    Fix extraction strategies before more training.")
    elif oracle_pct >= 70:
        print(f"    Trees are structurally sound ({oracle_pct:.0f}% oracle).")
        print(f"    The gap ({oracle_pct - model_pct:.0f}pp) is a model training problem.")
        print(f"    Keep training — target is matching oracle ceiling.")
    elif oracle_pct >= 60:
        print(f"    Trees are adequate but have room for improvement.")
        print(f"    Both StructDirect fixes AND model training will help.")
    else:
        print(f"    Trees are significantly limiting performance ({oracle_pct:.0f}% oracle).")
        print(f"    Fix StructDirect extraction before training v17.")

    print(f"{'='*65}")
    print(f"\n  Full results: {output_path}")

    # Show unreachable questions (most actionable)
    if unreachable:
        print(f"\n  UNREACHABLE questions (StructDirect missing these pages):")
        for r in unreachable[:15]:
            print(f"    [{r['doc'][:25]}] evid={r['evidence_pages']} | {r['question'][:55]}...")
        if len(unreachable) > 15:
            print(f"    ... and {len(unreachable) - 15} more")

    # Show miss questions (navigation failures)
    if misses:
        print(f"\n  MISS questions (pages exist, Claude still failed):")
        for r in misses[:15]:
            print(f"    [{r['doc'][:25]}] evid={r['evidence_pages']} | {r['strategy']} | {r['question'][:50]}...")
        if len(misses) > 15:
            print(f"    ... and {len(misses) - 15} more")


# ── Interactive mode ───────────────────────────────────────────────────────────

async def run_interactive(args):
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    model = args.model
    print(f"\nExtracting tree from: {pdf_path.name}")
    t0 = time.monotonic()
    tree = extract_structure(str(pdf_path))
    ms = (time.monotonic() - t0) * 1000

    n_nodes = _count_nodes(tree.structure)
    depth   = _max_depth(tree.structure)

    print(f"\n{'='*65}")
    print(f"  Document  : {pdf_path.name}")
    print(f"  Strategy  : {tree.extraction_strategy}")
    print(f"  Nodes     : {n_nodes}  |  Max depth: {depth}  |  Extracted in {ms:.0f}ms")
    print(f"  Top-level sections ({len(tree.structure)}):")
    for node in tree.structure[:20]:
        sub = f"  [{len(node.nodes)} sub-sections]" if node.nodes else ""
        print(f"    [{node.node_id}] {node.title} (pages {node.start_index}-{node.end_index}){sub}")
    if len(tree.structure) > 20:
        print(f"    ... and {len(tree.structure) - 20} more")
    print(f"{'='*65}")

    provider = AnthropicProvider(model=model)
    config = ArborConfig(
        max_hops=args.max_hops,
        max_nodes_searched=0,
        timeout_sec=120.0,
        max_retries_on_bad_json=2,
    )

    if args.question:
        questions = [args.question]
    else:
        questions = None  # interactive loop

    async def ask(question: str):
        print(f"\nQ: {question}")
        print("Navigating", end="", flush=True)
        t1 = time.monotonic()
        sr = await search_tree(tree, question, provider, multihop=True, config=config)
        elapsed = time.monotonic() - t1
        print(f" ({elapsed:.1f}s)")

        if sr.nodes:
            print(f"\nFound {len(sr.nodes)} section(s):")
            for node in sr.nodes:
                print(f"  [{node.node_id}] {node.title} (pages {node.start_index}-{node.end_index})")
        else:
            print("\nNo sections found.")

        if sr.thinking:
            lines = sr.thinking.replace(" | ", "\n          ").strip()
            print(f"\nThinking: {lines[:600]}")

    if questions:
        for q in questions:
            await ask(q)
    else:
        print("\nType a question (or 'quit' to exit):")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue
            await ask(q)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose whether failures are tree quality or model quality problems."
    )
    parser.add_argument("--model", default="haiku",
                        help="Claude model alias: haiku | sonnet | opus (default: haiku)")
    parser.add_argument("--max-hops", type=int, default=8,
                        help="Max navigation hops per question (default: 8)")

    sub = parser.add_subparsers(dest="mode", required=True)

    # financebench mode
    fb = sub.add_parser("financebench", help="Run on all 150 FinanceBench questions")
    fb.add_argument("--limit", type=int, default=0,
                    help="Run only first N questions (default: all)")
    fb.add_argument("--resume", action="store_true",
                    help="Skip questions already in output file")
    fb.add_argument("--output", type=Path, default=OUT_FILE,
                    help=f"Output JSON path (default: {OUT_FILE})")

    # interactive mode
    iv = sub.add_parser("interactive", help="Run interactively on any PDF")
    iv.add_argument("pdf", help="Path to PDF file")
    iv.add_argument("--question", "-q", help="Single question (skip interactive loop)")

    args = parser.parse_args()

    if args.mode == "financebench":
        asyncio.run(run_financebench(args))
    else:
        asyncio.run(run_interactive(args))


if __name__ == "__main__":
    main()
