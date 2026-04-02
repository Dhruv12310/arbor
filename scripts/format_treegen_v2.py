"""
Convert Arbor tree JSON files into proper TreeGen training data (v2).

Fixes vs v1 (format_training_data.py):
  - Adds [Page N] markers to input text
  - Uses rich system prompt with JSON schema + example
  - Proper size management (truncate at page boundary)
  - Quality filter (flat trees, too few nodes, duplicate summaries)
  - Token length distribution report

Usage:
    python scripts/format_treegen_v2.py
    python scripts/format_treegen_v2.py --input-dir data/training --pdf-dir data/pdfs
    python scripts/format_treegen_v2.py --max-seq-len 16384 --eval-ratio 0.1
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent

TREEGEN_SYSTEM = '''You are a document structure analyzer. Given document text with [Page N] markers, extract the hierarchical structure as a JSON tree.

Output format:
{
  "doc_name": "<filename>",
  "structure": [
    {
      "title": "<section title>",
      "node_id": "<4-digit zero-padded ID, e.g. 0001>",
      "start_index": <first page number>,
      "end_index": <last page number>,
      "summary": "<1-2 sentence summary of section content>",
      "nodes": [<child sections, same format>]
    }
  ]
}

Rules:
- node_id: Sequential in depth-first order starting from "0001"
- start_index / end_index: Physical page numbers (1-based) from [Page N] markers
- summary: Specific to THIS section, not the overall document
- nodes: Nested children. Leaf nodes have "nodes": []
- Create proper hierarchy: chapters > sections > subsections
- Page ranges must match where content actually appears

Example input:
[Page 1]
Deep Learning for NLP
Abstract: This survey covers recent advances...
1 Introduction
NLP has been transformed by deep learning...

[Page 2]
2 Background
2.1 Word Embeddings
Word2Vec revolutionized representation learning...
2.2 RNNs
RNNs process sequences by maintaining hidden state...

[Page 3]
3 Transformers
3.1 Self-Attention
The key innovation is self-attention...

Example output:
{"doc_name":"nlp_survey","structure":[{"title":"Abstract","node_id":"0001","start_index":1,"end_index":1,"summary":"Overview of recent deep learning advances for NLP tasks.","nodes":[]},{"title":"Introduction","node_id":"0002","start_index":1,"end_index":1,"summary":"How deep learning has transformed NLP research.","nodes":[]},{"title":"Background","node_id":"0003","start_index":2,"end_index":2,"summary":"Foundational concepts in neural NLP.","nodes":[{"title":"Word Embeddings","node_id":"0004","start_index":2,"end_index":2,"summary":"Word2Vec and distributed word representations.","nodes":[]},{"title":"RNNs","node_id":"0005","start_index":2,"end_index":2,"summary":"Recurrent networks for sequence processing.","nodes":[]}]},{"title":"Transformers","node_id":"0006","start_index":3,"end_index":3,"summary":"Transformer architecture and its innovations.","nodes":[{"title":"Self-Attention","node_id":"0007","start_index":3,"end_index":3,"summary":"Self-attention as the core transformer mechanism.","nodes":[]}]}]}'''


# ── Text helpers ───────────────────────────────────────────────────────────────

def get_paged_text_from_pdf(pdf_path: Path) -> str | None:
    """Re-extract per-page text from PDF with actual page boundaries."""
    try:
        from arbor.extraction.pdf_extractor import get_page_contents
        pages = get_page_contents(str(pdf_path))
        parts = []
        for i, page in enumerate(pages, 1):
            parts.append(f"[Page {i}]\n{page.text.strip()}")
        return "\n\n".join(parts)
    except Exception:
        return None


def get_paged_text_heuristic(doc_text: str, page_count: int) -> str:
    """Fallback: estimate page boundaries from character count."""
    if not doc_text:
        return ""
    if page_count <= 1:
        return f"[Page 1]\n{doc_text}"
    chars_per_page = max(1, len(doc_text) // page_count)
    parts = []
    for i in range(page_count):
        start = i * chars_per_page
        end = (i + 1) * chars_per_page if i < page_count - 1 else len(doc_text)
        # Try to break at paragraph boundary
        if i < page_count - 1:
            region_start = max(0, end - 200)
            region_end = min(len(doc_text), end + 200)
            search_region = doc_text[region_start:region_end]
            para_break = search_region.rfind("\n\n")
            if para_break > 0:
                end = region_start + para_break
        parts.append(f"[Page {i+1}]\n{doc_text[start:end].strip()}")
    return "\n\n".join(parts)


# ── Tree helpers ───────────────────────────────────────────────────────────────

def clean_node(node: dict) -> dict:
    return {
        "title": node.get("title", ""),
        "node_id": node.get("node_id", ""),
        "start_index": node.get("start_index", 0),
        "end_index": node.get("end_index", 0),
        "summary": (node.get("summary", "") or "")[:150],
        "nodes": [clean_node(c) for c in node.get("nodes", [])],
    }


def trim_depth(nodes: list, max_d: int, cur: int = 0) -> list:
    if cur >= max_d:
        return []
    result = []
    for n in nodes:
        trimmed = {k: v for k, v in n.items() if k != "nodes"}
        trimmed["nodes"] = trim_depth(n.get("nodes", []), max_d, cur + 1)
        result.append(trimmed)
    return result


def make_compact_tree(tree: dict, max_output_tokens: int = 8000) -> str | None:
    """Compact tree JSON, trimming depth if too large."""
    structure = tree.get("structure", [])
    max_chars = max_output_tokens * 4

    cleaned = {
        "doc_name": tree.get("doc_name", ""),
        "structure": [clean_node(n) for n in structure],
    }
    result = json.dumps(cleaned, separators=(",", ":"), ensure_ascii=False)
    if len(result) <= max_chars:
        return result

    for depth in [3, 2, 1]:
        trimmed_structure = trim_depth(structure, depth)
        trimmed_cleaned = {
            "doc_name": tree.get("doc_name", ""),
            "structure": [clean_node(n) for n in trimmed_structure],
        }
        result = json.dumps(trimmed_cleaned, separators=(",", ":"), ensure_ascii=False)
        if len(result) <= max_chars:
            return result

    return None


# ── Quality filter ─────────────────────────────────────────────────────────────

def count_nodes(nodes: list) -> int:
    total = 0
    for n in nodes:
        total += 1
        total += count_nodes(n.get("nodes", []))
    return total


def max_depth_fn(nodes: list, d: int = 0) -> int:
    mx = d
    for n in nodes:
        if n.get("nodes"):
            mx = max(mx, max_depth_fn(n["nodes"], d + 1))
    return mx


def collect_summaries(nodes: list, out: list | None = None) -> list:
    if out is None:
        out = []
    for n in nodes:
        s = n.get("summary", "")
        if s:
            out.append(s)
        collect_summaries(n.get("nodes", []), out)
    return out


def passes_quality(doc: dict) -> tuple[bool, str]:
    tree = doc.get("tree", {})
    structure = tree.get("structure", [])
    if not structure:
        return False, "empty_tree"

    total = count_nodes(structure)
    if total < 3:
        return False, "too_few_nodes"

    depth = max_depth_fn(structure)
    page_count = doc.get("page_count", 0)
    if depth == 0 and page_count > 5:
        return False, "flat_tree"

    summaries = collect_summaries(structure)
    if summaries and len(set(summaries)) < len(summaries) * 0.5:
        return False, "too_many_duplicate_summaries"

    return True, "ok"


# ── Example builder ────────────────────────────────────────────────────────────

def make_treegen_example(
    doc: dict,
    pdf_dir: Path | None = None,
    max_input_tokens: int = 6000,
    max_output_tokens: int = 8000,
) -> dict | None:
    """Build one training example. Returns None if it doesn't fit."""
    arxiv_id = doc.get("arxiv_id", "unknown")

    # 1. Get paged text
    paged_text = None
    if pdf_dir:
        pdf_path = pdf_dir / f"{arxiv_id}.pdf"
        if pdf_path.exists():
            paged_text = get_paged_text_from_pdf(pdf_path)

    if not paged_text:
        paged_text = get_paged_text_heuristic(
            doc.get("document_text", ""),
            doc.get("page_count", 1),
        )

    if not paged_text:
        return None

    # 2. Truncate input at page boundary if needed
    max_chars = max_input_tokens * 4
    if len(paged_text) > max_chars:
        truncated = paged_text[:max_chars]
        last_marker = truncated.rfind("[Page ")
        if last_marker > max_chars * 0.7:
            truncated = truncated[:last_marker]
        paged_text = truncated.rstrip() + "\n[Document truncated]"

    # 3. Build compact tree output
    tree = doc.get("tree", {})
    tree_json = make_compact_tree(tree, max_output_tokens)
    if tree_json is None:
        return None

    est_tokens = (len(TREEGEN_SYSTEM) + len(paged_text) + len(tree_json)) // 4

    return {
        "messages": [
            {"role": "system", "content": TREEGEN_SYSTEM},
            {"role": "user", "content": paged_text},
            {"role": "assistant", "content": tree_json},
        ],
        "est_tokens": est_tokens,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Format TreeGen v2 training data")
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "data" / "training",
        help="Directory with tree JSON files",
    )
    parser.add_argument(
        "--pdf-dir", type=Path, default=ROOT / "data" / "pdfs",
        help="Directory with source PDFs (for page extraction)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data" / "finetune",
        help="Directory to write JSONL files",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=16384,
        help="Max total tokens per example (default: 16384)",
    )
    parser.add_argument(
        "--eval-ratio", type=float, default=0.1,
        help="Fraction of data for eval set (default: 0.1)",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.json"))
    if not files:
        print(f"No JSON files found in {args.input_dir}")
        return

    print(f"Processing {len(files)} files from {args.input_dir} ...")

    examples = []
    skip_counts: dict[str, int] = {}

    for f in files:
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            skip_counts["parse_error"] = skip_counts.get("parse_error", 0) + 1
            continue

        ok, reason = passes_quality(doc)
        if not ok:
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
            continue

        ex = make_treegen_example(
            doc,
            pdf_dir=args.pdf_dir if args.pdf_dir.exists() else None,
            max_input_tokens=6000,
            max_output_tokens=8000,
        )
        if ex is None:
            skip_counts["too_large"] = skip_counts.get("too_large", 0) + 1
            continue

        if ex["est_tokens"] > args.max_seq_len:
            skip_counts["exceeds_max_seq_len"] = skip_counts.get("exceeds_max_seq_len", 0) + 1
            continue

        examples.append(ex)

    print(f"\nCollected {len(examples)} valid examples")
    if skip_counts:
        print("Skipped:")
        for reason, count in sorted(skip_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    if not examples:
        print("No examples to save.")
        return

    # Token distribution
    tokens = sorted(ex["est_tokens"] for ex in examples)
    n = len(tokens)
    print(f"\nToken length distribution:")
    print(f"  Min:    {tokens[0]}")
    print(f"  Median: {tokens[n//2]}")
    print(f"  p90:    {tokens[int(n*0.9)]}")
    print(f"  Max:    {tokens[-1]}")

    # Strip est_tokens before saving
    clean_examples = [{"messages": ex["messages"]} for ex in examples]

    # Shuffle + split
    random.seed(42)
    random.shuffle(clean_examples)
    eval_n = max(1, int(len(clean_examples) * args.eval_ratio))
    eval_set = clean_examples[:eval_n]
    train_set = clean_examples[eval_n:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "treegen_train.jsonl"
    eval_path = args.output_dir / "treegen_eval.jsonl"

    train_path.write_text("\n".join(json.dumps(e) for e in train_set), encoding="utf-8")
    eval_path.write_text("\n".join(json.dumps(e) for e in eval_set), encoding="utf-8")

    print(f"\nSaved:")
    print(f"  Train: {len(train_set)} examples -> {train_path}")
    print(f"  Eval:  {len(eval_set)} examples  -> {eval_path}")


if __name__ == "__main__":
    main()
