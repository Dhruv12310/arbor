"""
Collect FinanceBench data and generate TreeSearch training pairs.

Pipeline:
  Phase 1 -- Download JSONL and PDFs from GitHub
  Phase 2 -- Generate Arbor trees for each PDF
  Phase 3 -- Map evidence page numbers to tree nodes
  Phase 4 -- Split and save training / benchmark JSONL files

Usage:
    python scripts/collect_financebench.py                          # full pipeline
    python scripts/collect_financebench.py --skip-download          # skip Phase 1
    python scripts/collect_financebench.py --skip-trees             # skip Phase 2
    python scripts/collect_financebench.py --skip-download --skip-trees  # Phase 3+4 only
    python scripts/collect_financebench.py --dry-run                # preview without writing
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
FB_DIR = ROOT / "data" / "financebench"
PDF_DIR = FB_DIR / "pdfs"
TREES_DIR = FB_DIR / "trees"
JSONL_FILE = FB_DIR / "financebench_open_source.jsonl"
TRAIN_FILE = FB_DIR / "treesearch_train.jsonl"
BENCHMARK_FILE = FB_DIR / "treesearch_benchmark.jsonl"
REPORT_FILE = FB_DIR / "mapping_report.json"
PROGRESS_FILE = FB_DIR / "progress.json"

GENERATOR = Path(__file__).parent / "generate_training_pair.py"

JSONL_URL = (
    "https://raw.githubusercontent.com/patronus-ai/financebench/main"
    "/data/financebench_open_source.jsonl"
)
PDF_BASE_URL = (
    "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"
)

PDF_RATE_LIMIT = 1  # seconds between PDF downloads
TREE_RATE_LIMIT = 0  # seconds between generate_training_pair.py calls

TREESEARCH_SYSTEM = (
    "You are a document retrieval expert. Given a document tree structure and a question, "
    "identify which nodes contain the answer. Reply with JSON: "
    '{"thinking": "...", "node_list": [...]}'
)


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def doc_name_to_stem(doc_name: str) -> str:
    """Return the filename stem (no extension) for a doc_name."""
    return Path(doc_name).stem if doc_name.lower().endswith(".pdf") else doc_name


def doc_name_to_filename(doc_name: str) -> str:
    """Ensure doc_name has a .pdf extension."""
    return doc_name if doc_name.lower().endswith(".pdf") else doc_name + ".pdf"


def strip_text_from_nodes(nodes: list[dict]) -> list[dict]:
    """Recursively remove the 'text' field from all nodes (keeps node_id, title,
    start_index, end_index, summary, nodes)."""
    out = []
    for n in nodes:
        cleaned = {k: v for k, v in n.items() if k != "text"}
        if "nodes" in cleaned:
            cleaned["nodes"] = strip_text_from_nodes(cleaned["nodes"])
        out.append(cleaned)
    return out


def find_nodes_for_page(nodes: list[dict], page_number: int) -> list[str]:
    """
    Return node_ids of every node whose page range contains page_number.

    Includes both parent and leaf nodes -- any node where
    start_index <= page_number <= end_index qualifies.
    """
    result = []
    for node in nodes:
        start = node.get("start_index", 0)
        end = node.get("end_index", 0)
        if start <= page_number <= end:
            node_id = node.get("node_id")
            if node_id:
                result.append(node_id)
            # Recurse into children regardless (they may also match)
            result.extend(find_nodes_for_page(node.get("nodes", []), page_number))
    return result


def get_node_by_id(nodes: list[dict], node_id: str) -> dict | None:
    """Find a node by node_id anywhere in the tree."""
    for node in nodes:
        if node.get("node_id") == node_id:
            return node
        found = get_node_by_id(node.get("nodes", []), node_id)
        if found:
            return found
    return None


# ── Phase 1: Download ─────────────────────────────────────────────────────────

def phase1_download(dry_run: bool) -> list[dict]:
    """Download JSONL + all unique PDFs. Returns loaded JSONL rows."""
    FB_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    # --- Download JSONL ---
    if JSONL_FILE.exists():
        print(f"JSONL already exists: {JSONL_FILE}")
    elif dry_run:
        print(f"[dry-run] Would download JSONL -> {JSONL_FILE}")
    else:
        print("Downloading FinanceBench JSONL...", end=" ", flush=True)
        r = requests.get(JSONL_URL, timeout=30, headers={"User-Agent": "ArborResearch/1.0"})
        r.raise_for_status()
        JSONL_FILE.write_bytes(r.content)
        print(f"OK ({len(r.content) // 1024} KB)")

    if dry_run and not JSONL_FILE.exists():
        print("[dry-run] JSONL not yet on disk -- cannot preview PDF list.")
        return []

    rows = load_jsonl(JSONL_FILE)

    # Unique PDF filenames from the dataset
    doc_names = sorted({row["doc_name"] for row in rows})
    print(f"\nFound {len(rows)} questions across {len(doc_names)} unique documents")

    downloaded = skipped = failed = 0
    for doc_name in doc_names:
        filename = doc_name_to_filename(doc_name)
        dest = PDF_DIR / filename

        if dest.exists():
            skipped += 1
            continue

        url = f"{PDF_BASE_URL}/{filename}"

        if dry_run:
            print(f"  [dry-run] Would download: {filename}")
            continue

        print(f"  Downloading {filename}...", end=" ", flush=True)
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "ArborResearch/1.0"})
            r.raise_for_status()
            if b"%PDF" not in r.content[:10]:
                print("SKIP (not a valid PDF)")
                failed += 1
            else:
                dest.write_bytes(r.content)
                print(f"OK ({len(r.content) // 1024} KB)")
                downloaded += 1
        except Exception as e:
            print(f"FAILED ({e})")
            failed += 1

        time.sleep(PDF_RATE_LIMIT)

    if not dry_run:
        print(
            f"\nPhase 1 complete -- Downloaded: {downloaded}, "
            f"Skipped (exists): {skipped}, Failed: {failed}"
        )

    return rows


# ── Phase 2: Generate trees ───────────────────────────────────────────────────

def phase2_generate_trees(dry_run: bool, tree_delay: int = TREE_RATE_LIMIT) -> None:
    """Run generate_training_pair.py on every PDF in data/financebench/pdfs/."""
    TREES_DIR.mkdir(parents=True, exist_ok=True)

    if not dry_run and not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GROQ_API_KEY environment variable not set")

    all_pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not all_pdfs:
        print(f"No PDFs found in {PDF_DIR} -- skipping tree generation")
        return

    already_done = {f.stem for f in TREES_DIR.glob("*.json")}
    pending = [p for p in all_pdfs if p.stem not in already_done]

    print(
        f"\nPhase 2: {len(all_pdfs)} PDFs total | "
        f"Already done: {len(already_done)} | To process: {len(pending)}"
    )

    if dry_run:
        for p in pending:
            print(f"  [dry-run] Would generate tree for: {p.name}")
        return

    processed = errors = 0
    times: list[float] = []
    phase_start = time.monotonic()

    def write_progress(current_pdf: str = "", status: str = "running") -> None:
        avg = sum(times) / len(times) if times else None
        remaining = len(pending) - len(times)
        eta_secs = int(avg * remaining) if avg else None
        PROGRESS_FILE.write_text(json.dumps({
            "status": status,
            "total": len(pending),
            "done": processed,
            "errors": errors,
            "left": remaining,
            "current": current_pdf,
            "avg_seconds_per_doc": round(avg, 1) if avg else None,
            "eta": f"{eta_secs // 60}m {eta_secs % 60}s" if eta_secs else None,
            "elapsed": f"{int(time.monotonic() - phase_start) // 60}m {int(time.monotonic() - phase_start) % 60}s",
        }, indent=2), encoding="utf-8")

    for i, pdf in enumerate(pending, 1):
        avg_str = f"{sum(times)/len(times):.0f}s/doc" if times else "--"
        eta_str = f"{int(sum(times)/len(times)*(len(pending)-i+1))//60}m {int(sum(times)/len(times)*(len(pending)-i+1))%60}s" if times else "--"
        print(
            f"\n[{i}/{len(pending)}] {pdf.stem}  |  "
            f"done={i-1}  left={len(pending)-i+1}  avg={avg_str}  ETA={eta_str}",
            flush=True,
        )
        write_progress(current_pdf=pdf.stem)

        doc_start = time.monotonic()
        result = subprocess.run(
            [
                sys.executable, str(GENERATOR),
                str(pdf),
                "--output-dir", str(TREES_DIR),
            ],
            env=os.environ,
            capture_output=False,
        )
        doc_elapsed = time.monotonic() - doc_start
        times.append(doc_elapsed)

        if result.returncode != 0:
            errors += 1
            print(f"  [error] {pdf.stem}: exited {result.returncode} ({doc_elapsed:.0f}s)", flush=True)
        else:
            processed += 1
            print(f"  [ok] {pdf.stem} done in {doc_elapsed:.0f}s", flush=True)

        if i < len(pending) and tree_delay > 0:
            time.sleep(tree_delay)

    total_elapsed = time.monotonic() - phase_start
    write_progress(status="complete")
    print(
        f"\nPhase 2 complete -- Processed: {processed}, Errors: {errors}, "
        f"Total time: {int(total_elapsed)//60}m {int(total_elapsed)%60}s",
        flush=True,
    )


# ── Phase 3: Map evidence to nodes ────────────────────────────────────────────

def phase3_map_evidence(rows: list[dict]) -> tuple[list[dict], dict]:
    """
    For each question, find which tree nodes contain the evidence pages.

    Returns (training_pairs, report_dict).
    """
    pairs: list[dict] = []
    skip_reasons: dict[str, int] = {}
    total = len(rows)

    for row in rows:
        doc_name = row["doc_name"]
        question = row.get("question", "")
        evidence_list = row.get("evidence", [])

        stem = doc_name_to_stem(doc_name)
        tree_file = TREES_DIR / f"{stem}.json"

        # --- Guard: tree must exist ---
        if not tree_file.exists():
            skip_reasons["tree_not_found"] = skip_reasons.get("tree_not_found", 0) + 1
            continue

        # --- Guard: tree must parse ---
        try:
            tree_data = json.loads(tree_file.read_text(encoding="utf-8"))
        except Exception:
            skip_reasons["tree_parse_error"] = skip_reasons.get("tree_parse_error", 0) + 1
            continue

        structure = tree_data.get("tree", {}).get("structure", [])
        if not structure:
            skip_reasons["empty_tree"] = skip_reasons.get("empty_tree", 0) + 1
            continue

        # --- Collect evidence page numbers ---
        evidence_pages: list[int] = []
        for ev in evidence_list:
            pg = ev.get("evidence_page_num") or ev.get("page_number")
            if pg is not None:
                try:
                    evidence_pages.append(int(pg))
                except (ValueError, TypeError):
                    pass

        if not evidence_pages:
            skip_reasons["no_evidence_pages"] = skip_reasons.get("no_evidence_pages", 0) + 1
            continue

        # --- Find matching node_ids (deduplicated, order-preserving) ---
        seen: set[str] = set()
        unique_node_ids: list[str] = []
        for page in evidence_pages:
            for nid in find_nodes_for_page(structure, page):
                if nid not in seen:
                    seen.add(nid)
                    unique_node_ids.append(nid)

        if not unique_node_ids:
            skip_reasons["no_matching_nodes"] = skip_reasons.get("no_matching_nodes", 0) + 1
            continue

        # --- Build thinking string from matched node metadata ---
        matched_nodes = [
            n for nid in unique_node_ids
            if (n := get_node_by_id(structure, nid)) is not None
        ]
        node_titles = ", ".join(f'"{n["title"]}"' for n in matched_nodes)
        pages_str = ", ".join(str(p) for p in sorted(set(evidence_pages)))
        thinking = (
            f"The question asks about {question.rstrip('?')}. "
            f"Based on the tree structure, the relevant sections are: {node_titles}. "
            f"These nodes span pages {pages_str} which contain the evidence."
        )

        # --- Build user message: tree JSON without text field ---
        tree_for_search = {"structure": strip_text_from_nodes(structure)}
        tree_json = json.dumps(tree_for_search, separators=(",", ":"))

        # --- Build assistant message ---
        assistant_content = json.dumps(
            {"thinking": thinking, "node_list": unique_node_ids},
            separators=(",", ":"),
        )

        pairs.append({"messages": [
            {"role": "system",    "content": TREESEARCH_SYSTEM},
            {"role": "user",      "content": f"Question: {question}\n\nDocument tree structure:\n{tree_json}"},
            {"role": "assistant", "content": assistant_content},
        ]})

    mapped = len(pairs)
    report = {
        "total_questions": total,
        "mapped": mapped,
        "skipped": total - mapped,
        "skip_reasons": skip_reasons,
    }
    return pairs, report


# ── Phase 4: Split and save ───────────────────────────────────────────────────

def phase4_save(pairs: list[dict], report: dict, dry_run: bool) -> tuple[int, int]:
    """
    Shuffle with seed 42, split 80/20 train/benchmark, write JSONL files.
    The benchmark set is a sacred holdout -- never used for training.
    """
    random.seed(42)
    shuffled = list(pairs)
    random.shuffle(shuffled)

    benchmark_n = max(1, int(len(shuffled) * 0.20))
    benchmark_set = shuffled[:benchmark_n]
    train_set = shuffled[benchmark_n:]

    if dry_run:
        print(
            f"\n[dry-run] Would save {len(train_set)} train pairs -> {TRAIN_FILE}"
        )
        print(
            f"[dry-run] Would save {len(benchmark_set)} benchmark pairs -> {BENCHMARK_FILE}"
        )
        return len(train_set), len(benchmark_set)

    FB_DIR.mkdir(parents=True, exist_ok=True)

    TRAIN_FILE.write_text(
        "\n".join(json.dumps(e) for e in train_set),
        encoding="utf-8",
    )
    BENCHMARK_FILE.write_text(
        "\n".join(json.dumps(e) for e in benchmark_set),
        encoding="utf-8",
    )

    report["train_count"] = len(train_set)
    report["benchmark_count"] = len(benchmark_set)
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return len(train_set), len(benchmark_set)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect FinanceBench data and build TreeSearch training pairs"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip Phase 1 (assume JSONL and PDFs already downloaded)",
    )
    parser.add_argument(
        "--skip-trees", action="store_true",
        help="Skip Phase 2 (assume trees already generated)",
    )
    parser.add_argument(
        "--tree-delay", type=int, default=TREE_RATE_LIMIT,
        help="Seconds between tree generation calls (default: 0, use 3 for Groq free tier)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be done without writing any files",
    )
    args = parser.parse_args()

    print("=== FinanceBench Collection Pipeline ===\n")

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    rows: list[dict] = []
    if args.skip_download:
        print("Phase 1: Skipped (--skip-download)")
        if JSONL_FILE.exists():
            rows = load_jsonl(JSONL_FILE)
            print(f"  Loaded {len(rows)} questions from {JSONL_FILE}")
        else:
            print(f"  WARNING: {JSONL_FILE} not found -- Phase 3 will have no data")
    else:
        rows = phase1_download(args.dry_run)

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    if args.skip_trees:
        tree_count = len(list(TREES_DIR.glob("*.json"))) if TREES_DIR.exists() else 0
        print(f"\nPhase 2: Skipped (--skip-trees) -- {tree_count} trees on disk")
    else:
        phase2_generate_trees(args.dry_run, tree_delay=args.tree_delay)

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    if not rows:
        print("\nNo JSONL data available -- cannot run Phase 3+4.")
        return

    print(f"\nPhase 3: Mapping {len(rows)} questions to tree nodes...")
    pairs, report = phase3_map_evidence(rows)
    print(
        f"  Mapped: {report['mapped']} / {report['total_questions']} questions"
    )
    if report["skip_reasons"]:
        for reason, count in sorted(report["skip_reasons"].items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    if not pairs:
        print("\nNo pairs to save -- exiting.")
        return

    print(f"\nPhase 4: Splitting {len(pairs)} pairs (80% train / 20% benchmark)...")
    train_n, bench_n = phase4_save(pairs, report, args.dry_run)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = report["total_questions"]
    mapped = report["mapped"]
    print(f"\n{'='*55}")
    print(
        f"  {total} questions  ->  {mapped} mapped  ->  "
        f"{train_n} train / {bench_n} benchmark"
    )
    print(f"{'='*55}")

    if not args.dry_run:
        print(f"\n  Train:     {TRAIN_FILE}")
        print(f"  Benchmark: {BENCHMARK_FILE}")
        print(f"  Report:    {REPORT_FILE}")


if __name__ == "__main__":
    main()
