"""
Build v6 TreeSearch training dataset.

Regenerates trees for all 141 arXiv PDFs using Gemini 2.0 Flash (better than
the original Groq/Llama trees), then generates 5 varied questions per PDF
using Gemini (not templates).

Output:
  data/training_v6/ARXIV_ID.json  — one tree JSON per PDF (skips existing)
  data/finetune/treesearch_v6_train.jsonl
  data/finetune/treesearch_v6_eval.jsonl

Run:
  python scripts/build_v6_dataset.py
  python scripts/build_v6_dataset.py --questions-only  (skip tree gen, just redo questions)
"""

import argparse
import asyncio
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import arbor
from arbor.extraction.pdf_extractor import get_page_contents
from arbor.utils.tree_utils import count_nodes

ROOT         = Path(__file__).parent.parent
PDF_DIR      = ROOT / "data" / "pdfs"
TRAIN_DIR    = ROOT / "data" / "training_v6"
FINETUNE_DIR = ROOT / "data" / "finetune"

TREESEARCH_SYSTEM = (
    "You are a document retrieval expert. Given a document tree structure and a question, "
    "identify which nodes contain the answer. Reply with JSON: "
    '{"thinking": "...", "node_list": [...]}'
)

QUESTION_PROMPT = """\
You are given the hierarchical tree structure of an academic paper, including each node's summary of its actual content.
Generate 15 realistic questions a researcher might ask about this paper.

Rules:
- Questions must be grounded in the ACTUAL CONTENT described in the node summaries — not just the titles
- Each question should require 1-3 specific nodes to answer
- The question must be answerable ONLY by reading those specific nodes — not the whole paper
- Mix types: factual (specific numbers/results), conceptual (how/why), methodological, comparative
- Questions should sound natural — like a researcher asking about the paper
- NEVER include node IDs like [0034] in the question text — pure natural language only
- node_ids must appear exactly as listed below (only in your JSON response)

Tree with summaries:
{tree_summary}

Reply ONLY with a valid JSON array — no markdown fences, no extra text:
[
  {{"question": "...", "node_ids": ["0001"], "thinking": "one sentence explaining why these nodes"}},
  ...15 items...
]"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def summarize_tree(nodes: list, depth: int = 0) -> list[str]:
    lines = []
    for n in nodes:
        indent = "  " * depth
        summary = n.get("summary", "")
        # Truncate summary to first 120 chars to keep prompt manageable
        summary_short = (summary[:120] + "...") if len(summary) > 120 else summary
        lines.append(
            f"{indent}[{n['node_id']}] {n['title']} "
            f"(pages {n.get('start_index', '?')}-{n.get('end_index', '?')})"
            + (f"\n{indent}  → {summary_short}" if summary_short else "")
        )
        lines.extend(summarize_tree(n.get("nodes", []), depth + 1))
    return lines


def flatten_nodes(nodes: list) -> list:
    result = []
    for n in nodes:
        if n.get("node_id"):
            result.append(n)
        result.extend(flatten_nodes(n.get("nodes", [])))
    return result


def strip_text(nodes: list) -> list:
    out = []
    for n in nodes:
        c = {k: v for k, v in n.items() if k != "text"}
        if "nodes" in c:
            c["nodes"] = strip_text(c["nodes"])
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Tree generation
# ---------------------------------------------------------------------------

async def generate_tree_for_pdf(
    pdf_path: Path,
    provider: arbor.GeminiProvider,
    train_dir: Path,
    force: bool = False,
) -> dict | None:
    arxiv_id = pdf_path.stem
    out_file  = train_dir / f"{arxiv_id}.json"

    if out_file.exists() and not force:
        return json.loads(out_file.read_text())

    pages         = get_page_contents(str(pdf_path))
    document_text = "\n\n".join(p.text for p in pages)

    config = arbor.ArborConfig(
        add_node_ids=True,
        add_summaries=True,
        add_node_text=False,
        max_tokens_per_node=8000,
        max_concurrent_llm_calls=3,
    )

    t0 = time.monotonic()
    try:
        tree = await asyncio.wait_for(
            arbor.generate_tree(str(pdf_path), provider, config),
            timeout=1800,
        )
    except Exception as e:
        print(f"  [error] {arxiv_id}: {e}")
        return None

    elapsed = round(time.monotonic() - t0, 1)
    nodes   = count_nodes(tree.structure)

    doc = {
        "arxiv_id": arxiv_id,
        "document_text": document_text,
        "page_count": len(pages),
        "tree": tree.to_dict(),
        "metadata": {
            "model": provider.model,
            "generation_time_seconds": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    out_file.write_text(json.dumps(doc, indent=2))
    print(f"  [tree] {arxiv_id} — {elapsed}s, {len(pages)} pages, {nodes} nodes")
    return doc


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------

async def generate_questions(
    provider: arbor.GeminiProvider,
    structure: list,
    arxiv_id: str,
) -> list[dict]:
    node_map     = {n["node_id"]: n for n in flatten_nodes(structure)}
    tree_summary = "\n".join(summarize_tree(structure))[:8000]
    prompt       = QUESTION_PROMPT.format(tree_summary=tree_summary)

    try:
        raw = await provider.complete_with_retry(
            prompt, temperature=0.7, max_tokens=1024
        )
        # Strip markdown fences if model wraps output
        if "```" in raw:
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]
        pairs = json.loads(raw)

        results = []
        for pair in pairs:
            if not pair.get("question") or not pair.get("node_ids"):
                continue
            valid_ids = [nid for nid in pair["node_ids"] if nid in node_map]
            if not valid_ids:
                continue
            results.append({
                "question": pair["question"],
                "node_ids": valid_ids,
                "thinking": pair.get("thinking", ""),
            })
        return results

    except Exception as e:
        print(f"  [warn] question gen failed for {arxiv_id}: {e}")
        return []


# ---------------------------------------------------------------------------
# Training pair formatting
# ---------------------------------------------------------------------------

def make_treesearch_examples(doc: dict, questions: list[dict]) -> list[dict]:
    structure       = doc["tree"].get("structure", [])
    tree_for_search = {"structure": strip_text(structure)}
    tree_json       = json.dumps(tree_for_search, separators=(",", ":"))

    examples = []
    for q in questions:
        answer = json.dumps(
            {"thinking": q["thinking"], "node_list": q["node_ids"]},
            separators=(",", ":"),
        )
        examples.append({"messages": [
            {"role": "system",    "content": TREESEARCH_SYSTEM},
            {"role": "user",      "content": f"Question: {q['question']}\n\nDocument tree structure:\n{tree_json}"},
            {"role": "assistant", "content": answer},
        ]})
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(questions_only: bool = False) -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")

    provider = arbor.GeminiProvider(
        api_key=api_key,
        model="gemini-2.5-flash-lite",
        max_tokens=8192,
    )

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {PDF_DIR}")
    if questions_only:
        print("--questions-only: skipping tree generation, reusing existing JSONs")

    all_examples: list[dict] = []
    skipped = 0

    t_start = time.monotonic()

    for i, pdf_path in enumerate(pdfs):
        arxiv_id = pdf_path.stem
        elapsed_total = time.monotonic() - t_start
        if i > 0:
            avg = elapsed_total / i
            remaining = avg * (len(pdfs) - i)
            eta = f"  ETA {remaining/60:.0f}m"
        else:
            eta = ""
        print(f"\n[{i+1}/{len(pdfs)}] {arxiv_id}{eta}")

        # --- Tree ---
        if questions_only:
            tree_file = TRAIN_DIR / f"{arxiv_id}.json"
            if not tree_file.exists():
                print(f"  [skip] no tree JSON found, skipping")
                skipped += 1
                continue
            doc = json.loads(tree_file.read_text())
        else:
            doc = await generate_tree_for_pdf(pdf_path, provider, TRAIN_DIR)
            if doc is None:
                skipped += 1
                continue

        # --- Questions ---
        structure = doc["tree"].get("structure", [])
        if not structure:
            print(f"  [warn] empty tree, skipping")
            skipped += 1
            continue

        questions = await generate_questions(provider, structure, arxiv_id)
        print(f"  [questions] {len(questions)} generated")

        examples = make_treesearch_examples(doc, questions)
        all_examples.extend(examples)

        # Pace requests — Gemini free tier is 15 RPM
        if i < len(pdfs) - 1:
            await asyncio.sleep(2)

    # --- Split & write ---
    random.seed(42)
    random.shuffle(all_examples)
    split    = max(1, len(all_examples) // 10)
    eval_set  = all_examples[:split]
    train_set = all_examples[split:]

    train_path = FINETUNE_DIR / "treesearch_v6_train.jsonl"
    eval_path  = FINETUNE_DIR / "treesearch_v6_eval.jsonl"

    train_path.write_text("\n".join(json.dumps(e) for e in train_set))
    eval_path.write_text("\n".join(json.dumps(e) for e in eval_set))

    print(f"\n{'='*55}")
    print(f"  Done — {len(all_examples)} total examples ({skipped} PDFs skipped)")
    print(f"  Train: {len(train_set)} → {train_path}")
    print(f"  Eval:  {len(eval_set)} → {eval_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions-only", action="store_true",
        help="Skip tree generation; reuse existing data/training_v6/ JSONs and only redo questions"
    )
    args = parser.parse_args()
    asyncio.run(main(questions_only=args.questions_only))
