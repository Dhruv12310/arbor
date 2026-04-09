"""
Domain PDF Quality Diagnostic — mirrors diagnose_tree_quality.py but for non-finance PDFs.

Strategy:
  1. Extract StructDirect tree from each PDF.
  2. Pick 3 nodes (spread across depth) as synthetic answer locations.
  3. Use Haiku to generate a factual question FROM that node's page content.
  4. Navigate with Haiku through search_tree(multihop=True).
  5. Classify: HIT / MISS / UNREACHABLE — same logic as FinanceBench diagnostic.

This separates StructDirect extraction bugs from navigation/model problems
across 7 non-financial domains (35 PDFs, 105 questions).

Usage:
    python scripts/diagnose_domain_quality.py
    python scripts/diagnose_domain_quality.py --limit 2        # 2 PDFs per domain
    python scripts/diagnose_domain_quality.py --resume
    python scripts/diagnose_domain_quality.py --model sonnet   # stronger oracle
    python scripts/diagnose_domain_quality.py --domain legal   # single domain

Results saved to: data/domain_pdfs/haiku_domain_diagnostic_results.json
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF

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

PDF_ROOT = ROOT / "data/domain_pdfs"
OUT_FILE = ROOT / "data/domain_pdfs/haiku_domain_diagnostic_results.json"

# Same 5 PDFs per domain as multidomaintest_v14.py
DOMAIN_PDFS: dict[str, list[str]] = {
    "automotive": [
        "2603.29165v1", "2603.29192v1", "2603.29213v1", "2603.29226v1", "2603.29227v1",
    ],
    "energy": [
        "2603.24489v1", "2603.24785v1", "2603.25192v1", "2603.25211v2", "2603.25259v1",
    ],
    "government": [
        "2603.27717v1", "2603.27801v1", "2603.27806v1", "2603.27994v1", "2603.28123v1",
    ],
    "healthcare": [
        "2603.24828v1", "2603.24846v1", "2603.25186v1", "2603.25780v1", "2603.25821v1",
    ],
    "insurance": [
        "2603.11897v1", "2603.12767v1", "2603.14546v2", "2603.15369v1", "2603.15839v1",
    ],
    "legal": [
        "2602.24130v1", "2603.04259v1", "2603.04982v2", "2603.09492v1", "2603.10653v1",
    ],
    "real_estate": [
        "2502.16810v5", "2502.21141v2", "2503.18332v2", "2504.13459v1", "2504.17113v1",
    ],
}

QUESTIONS_PER_DOC = 3


# ── Tree helpers ───────────────────────────────────────────────────────────────

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


def check_reachability(tree: DocumentTree, evidence_pages: list[int]) -> float:
    if not evidence_pages:
        return 1.0
    covered: set[int] = set()
    for node in _iter_all_nodes(tree.structure):
        covered.update(range(node.start_index, node.end_index + 1))
    reachable = [p for p in evidence_pages if p in covered]
    return len(reachable) / len(evidence_pages)


def score_retrieval(nodes: list, evidence_pages: list[int]) -> tuple[float, list[int]]:
    if not evidence_pages:
        return 1.0, []
    returned: set[int] = set()
    for node in nodes:
        returned.update(range(node.start_index, node.end_index + 1))
    found = [p for p in evidence_pages if p in returned]
    return len(found) / len(evidence_pages), found


def pick_sample_nodes(tree: DocumentTree, n: int = 3) -> list[TreeNode]:
    """
    Pick n nodes spread across the tree:
      - Prefer nodes with at least 1 page of content (end > start).
      - Spread across different top-level branches to get diverse questions.
    """
    all_nodes = [
        node for node in _iter_all_nodes(tree.structure)
        if node.end_index > node.start_index  # skip single-page stubs
        and node.end_index - node.start_index <= 10  # not huge chapters
    ]
    if not all_nodes:
        # Fall back: any node
        all_nodes = list(_iter_all_nodes(tree.structure))
    if len(all_nodes) <= n:
        return all_nodes

    # Try to pick from different top-level sections
    random.seed(42)
    random.shuffle(all_nodes)
    # Deduplicate by top-level section prefix (first 4 chars of node_id)
    seen_prefixes: set[str] = set()
    diverse: list[TreeNode] = []
    remaining: list[TreeNode] = []
    for node in all_nodes:
        prefix = node.node_id[:4]
        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            diverse.append(node)
        else:
            remaining.append(node)
    selected = diverse[:n]
    if len(selected) < n:
        selected += remaining[: n - len(selected)]
    return selected[:n]


def get_page_text(pdf_path: str, start_page: int, end_page: int, max_chars: int = 1500) -> str:
    """Extract text from a page range in the PDF (0-indexed internally)."""
    doc = fitz.open(pdf_path)
    total = doc.page_count
    texts = []
    for pg in range(start_page, min(end_page + 1, total)):
        texts.append(doc[pg].get_text())
    doc.close()
    combined = "\n".join(texts)
    return combined[:max_chars]


# ── Question generation ────────────────────────────────────────────────────────

async def generate_question(
    provider: AnthropicProvider,
    node: TreeNode,
    page_text: str,
    doc_title: str,
) -> str | None:
    """Ask Haiku to write a single factual question answered by this node's content."""
    if not page_text.strip():
        return None

    prompt = (
        f"Document title: {doc_title}\n"
        f"Section: {node.title} (pages {node.start_index}–{node.end_index})\n\n"
        f"Content excerpt:\n{page_text}\n\n"
        "Write ONE short factual question whose answer is clearly found in the excerpt above. "
        "The question should be specific enough that only this section would answer it. "
        "Reply with ONLY the question, no explanation."
    )
    try:
        question = await provider.complete(prompt, temperature=0.3, max_tokens=100)
        question = question.strip().strip('"').strip("'")
        return question if question else None
    except Exception:
        return None


# ── Main diagnostic ────────────────────────────────────────────────────────────

async def run_domain_diagnostic(args):
    model_label = AnthropicProvider.MODEL_ALIASES.get(args.model, args.model)
    print(f"\nDOMAIN PDF HAIKU DIAGNOSTIC")
    print(f"Model   : {model_label}")
    print(f"Output  : {args.output}")
    print("=" * 65)

    domains = DOMAIN_PDFS
    if args.domain:
        if args.domain not in domains:
            print(f"Unknown domain: {args.domain}. Options: {list(domains.keys())}")
            sys.exit(1)
        domains = {args.domain: DOMAIN_PDFS[args.domain]}

    limit = args.limit  # PDFs per domain (0 = all 5)

    # Load existing results for resume
    done_ids: set[str] = set()
    existing: list[dict] = []
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        done_ids = {r["id"] for r in existing}
        print(f"Resuming: {len(done_ids)} already done")

    provider = AnthropicProvider(model=args.model)
    config = ArborConfig(
        max_hops=args.max_hops,
        max_nodes_searched=0,
        timeout_sec=120.0,
        max_retries_on_bad_json=2,
    )

    results: list[dict] = list(existing)

    for domain, pdf_ids in domains.items():
        if limit:
            pdf_ids = pdf_ids[:limit]
        print(f"\n{'─'*65}")
        print(f"  DOMAIN: {domain.upper()}  ({len(pdf_ids)} PDFs × {QUESTIONS_PER_DOC} questions)")
        print(f"{'─'*65}")

        for pdf_id in pdf_ids:
            pdf_path = PDF_ROOT / domain / f"{pdf_id}.pdf"
            if not pdf_path.exists():
                print(f"\n  SKIP {pdf_id} — PDF not found at {pdf_path}")
                continue

            print(f"\n  [{pdf_id}]")

            # Extract tree once per PDF
            try:
                t0 = time.monotonic()
                tree = extract_structure(str(pdf_path))
                extract_ms = (time.monotonic() - t0) * 1000
                n_nodes = _count_nodes(tree.structure)
                depth = _max_depth(tree.structure)
                print(f"    Tree: {n_nodes} nodes | depth={depth} | strategy={tree.extraction_strategy} ({extract_ms:.0f}ms)")
            except Exception as e:
                print(f"    ERROR extracting tree: {e}")
                continue

            # Get document title from PDF metadata
            try:
                fitz_doc = fitz.open(str(pdf_path))
                doc_title = fitz_doc.metadata.get("title") or pdf_id
                fitz_doc.close()
            except Exception:
                doc_title = pdf_id

            # Pick sample nodes to generate questions from
            sample_nodes = pick_sample_nodes(tree, n=QUESTIONS_PER_DOC)
            if not sample_nodes:
                print(f"    SKIP — no usable nodes in tree")
                continue

            for node in sample_nodes:
                q_id = f"{domain}/{pdf_id}/{node.node_id}"
                if q_id in done_ids:
                    continue

                # Get page text for this node
                page_text = get_page_text(
                    str(pdf_path),
                    node.start_index,
                    node.end_index,
                )
                if not page_text.strip():
                    print(f"    SKIP node {node.node_id} — empty page text")
                    continue

                # Generate question
                question = await generate_question(provider, node, page_text, doc_title)
                if not question:
                    print(f"    SKIP node {node.node_id} — question generation failed")
                    continue

                evidence_pages = list(range(node.start_index, node.end_index + 1))

                print(f"    Node [{node.node_id}] p{node.start_index}-{node.end_index}: {question[:70]}{'...' if len(question)>70 else ''}")

                result: dict = {
                    "id": q_id,
                    "domain": domain,
                    "doc": pdf_id,
                    "section": node.title,
                    "question": question,
                    "evidence_pages": evidence_pages,
                    "reachability": 0.0,
                    "recall": 0.0,
                    "found_pages": [],
                    "found_nodes": [],
                    "strategy": tree.extraction_strategy,
                    "node_count": n_nodes,
                    "thinking": "",
                    "search_sec": 0.0,
                    "error": None,
                }

                try:
                    # Reachability: is this node's content even indexable?
                    reachability = check_reachability(tree, evidence_pages)
                    result["reachability"] = reachability

                    # Navigate with Haiku
                    t1 = time.monotonic()
                    sr = await search_tree(tree, question, provider, multihop=True, config=config)
                    search_sec = time.monotonic() - t1
                    result["search_sec"] = round(search_sec, 1)
                    result["found_nodes"] = sr.node_ids
                    result["thinking"] = sr.thinking[:400] if sr.thinking else ""

                    recall, found_pages = score_retrieval(sr.nodes, evidence_pages)
                    result["recall"] = recall
                    result["found_pages"] = found_pages

                    if recall == 1.0:
                        status = "HIT       "
                    elif reachability < 1.0:
                        status = "UNREACHABLE"
                    else:
                        status = "MISS      "

                    print(f"      {status} recall={recall:.0%} reach={reachability:.0%} nodes={sr.node_ids} {search_sec:.1f}s")

                except Exception as e:
                    result["error"] = str(e)
                    print(f"      ERROR: {e}")

                results.append(result)
                args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                await asyncio.sleep(0.3)

    _print_summary(results)


# ── Summary ────────────────────────────────────────────────────────────────────

def _print_summary(results: list[dict]):
    evaluated = [r for r in results if r.get("error") is None]
    n = len(evaluated)
    if n == 0:
        print("\nNo results to summarize.")
        return

    hits        = [r for r in evaluated if r["recall"] == 1.0]
    misses      = [r for r in evaluated if r["recall"] == 0.0 and r["reachability"] == 1.0]
    unreachable = [r for r in evaluated if r["reachability"] < 1.0]
    partial     = [r for r in evaluated if 0.0 < r["recall"] < 1.0]
    errors      = [r for r in results if r.get("error")]

    print(f"\n{'='*65}")
    print(f"  DOMAIN HAIKU DIAGNOSTIC — {n} questions across {len(set(r['doc'] for r in evaluated))} PDFs")
    print(f"{'='*65}")
    print(f"  HITs (oracle found answer)        : {len(hits)}/{n} = {len(hits)/n*100:.1f}%")
    print(f"  MISSes (pages exist, Claude wrong): {len(misses)}/{n} = {len(misses)/n*100:.1f}%")
    print(f"  UNREACHABLE (StructDirect gap)    : {len(unreachable)}/{n} = {len(unreachable)/n*100:.1f}%")
    print(f"  PARTIAL                           : {len(partial)}/{n} = {len(partial)/n*100:.1f}%")
    print(f"  ERRORS                            : {len(errors)}")
    print()

    # Per-domain breakdown
    from collections import defaultdict
    by_domain: dict = defaultdict(lambda: {"hits": 0, "total": 0, "unreachable": 0, "miss": 0})
    for r in evaluated:
        d = r["domain"]
        by_domain[d]["total"] += 1
        if r["recall"] == 1.0:
            by_domain[d]["hits"] += 1
        if r["reachability"] < 1.0:
            by_domain[d]["unreachable"] += 1
        if r["recall"] == 0.0 and r["reachability"] == 1.0:
            by_domain[d]["miss"] += 1

    print(f"  Per-domain breakdown:")
    for domain, d in sorted(by_domain.items()):
        pct = d["hits"] / d["total"] * 100 if d["total"] else 0
        print(f"    {domain:<14}: {d['hits']:>2}/{d['total']:<2} = {pct:.0f}%  (unreachable={d['unreachable']}, miss={d['miss']})")
    print()

    # Per-strategy breakdown
    by_strat: dict = defaultdict(lambda: {"hits": 0, "total": 0, "unreachable": 0})
    for r in evaluated:
        s = r.get("strategy") or "unknown"
        by_strat[s]["total"] += 1
        if r["recall"] == 1.0:
            by_strat[s]["hits"] += 1
        if r["reachability"] < 1.0:
            by_strat[s]["unreachable"] += 1

    print(f"  By extraction strategy:")
    for strat, d in sorted(by_strat.items(), key=lambda x: -x[1]["total"]):
        pct = d["hits"] / d["total"] * 100 if d["total"] else 0
        print(f"    {strat:<20}: {d['hits']:>3}/{d['total']:<3} = {pct:.0f}%  (unreachable={d['unreachable']})")
    print()

    # Verdict
    oracle_pct = len(hits) / n * 100
    print(f"  VERDICT:")
    if len(unreachable) > n * 0.15:
        print(f"    StructDirect is missing content in {len(unreachable)} questions ({len(unreachable)/n*100:.0f}%).")
        print(f"    Fix extraction strategies before training on domain PDFs.")
    elif oracle_pct >= 70:
        print(f"    Trees are structurally sound ({oracle_pct:.0f}% oracle).")
        print(f"    Domain PDFs are ready for training data generation.")
    elif oracle_pct >= 50:
        print(f"    Trees are adequate but navigation struggles ({oracle_pct:.0f}%).")
        print(f"    Check MISS cases — likely hierarchy or flat-tree issues.")
    else:
        print(f"    Poor oracle performance ({oracle_pct:.0f}%). Both StructDirect and")
        print(f"    navigation have problems on domain PDFs.")

    print(f"{'='*65}")
    print(f"\n  Full results: {OUT_FILE}")

    if unreachable:
        print(f"\n  UNREACHABLE (StructDirect missing these pages):")
        for r in unreachable[:10]:
            print(f"    [{r['domain']}/{r['doc'][:20]}] evid={r['evidence_pages'][:3]} section={r['section'][:40]}")
        if len(unreachable) > 10:
            print(f"    ... and {len(unreachable) - 10} more")

    if misses:
        print(f"\n  MISS (pages exist, Haiku still failed):")
        for r in misses[:10]:
            print(f"    [{r['domain']}/{r['doc'][:20]}] strat={r['strategy']} | {r['question'][:55]}")
        if len(misses) > 10:
            print(f"    ... and {len(misses) - 10} more")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose StructDirect + navigation quality on domain PDFs."
    )
    parser.add_argument("--model", default="haiku",
                        help="Claude model alias: haiku | sonnet | opus (default: haiku)")
    parser.add_argument("--max-hops", type=int, default=8,
                        help="Max navigation hops per question (default: 8)")
    parser.add_argument("--limit", type=int, default=0,
                        help="PDFs per domain (default: all 5)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip questions already in output file")
    parser.add_argument("--domain", default="",
                        help="Run only one domain (e.g. legal, healthcare)")
    parser.add_argument("--output", type=Path, default=OUT_FILE,
                        help=f"Output JSON path (default: {OUT_FILE})")
    args = parser.parse_args()

    asyncio.run(run_domain_diagnostic(args))


if __name__ == "__main__":
    main()
