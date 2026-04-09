#!/usr/bin/env python3
# gen_diverse_questions.py
# ========================
# Generates ~19,000+ diverse training examples for TreeSearch by:
#   1. Selecting 43 PDFs (8 FinanceBench + 35 domain, all unique types/domains)
#   2. Extracting section text from each leaf node
#   3. Using Claude Haiku to generate 5 factual questions per section
#   4. For each question, generating 4 rephrasings (+ original = 5 styles)
#   5. Computing oracle navigation paths deterministically (no LLM)
#   6. Writing training-ready JSONL in the exact same format as structdirect_train.jsonl
#
# Usage:
#   cd C:\Users\dhruv\arbor
#   python scripts/gen_diverse_questions.py           # full run
#   python scripts/gen_diverse_questions.py --test    # single PDF dry run, inspect quality
#
# Output:
#   data/financebench/diverse_train.jsonl             (training-ready JSONL)
#   data/financebench/diverse_questions_raw.json      (raw questions, inspect before running)
#   data/financebench/diverse_gen_progress.json       (checkpoint — resume on interruption)
#
# Cost estimate: ~$5-7 total (Claude Haiku API)
# Requirements: pip install pymupdf anthropic tqdm

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FB_PDF_DIR     = REPO_ROOT / "data" / "financebench" / "pdfs"
DOMAIN_PDF_DIR = REPO_ROOT / "data" / "domain_pdfs"
OUT_JSONL      = REPO_ROOT / "data" / "financebench" / "diverse_train.jsonl"
OUT_RAW        = REPO_ROOT / "data" / "financebench" / "diverse_questions_raw.json"
PROGRESS_FILE  = REPO_ROOT / "data" / "financebench" / "diverse_gen_progress.json"

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass  # dotenv not installed — rely on env vars being set manually

from arbor.extraction.structure_extractor import extract_structure
from arbor.core.types import DocumentTree, TreeNode

try:
    import fitz as pymupdf
except ImportError:
    import pymupdf

try:
    import anthropic
except ImportError:
    sys.exit("anthropic package not found. Run: pip install anthropic")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("tip: pip install tqdm for a progress bar")

# ── PDF Selection ──────────────────────────────────────────────────────────────

# 8 FinanceBench PDFs — maximally diverse: 2 large 10-Ks, 2 10-Qs, 2 8-Ks, 2 earnings
FINANCEBENCH_PDFS = [
    ("MICROSOFT_2023_10K",                         FB_PDF_DIR / "MICROSOFT_2023_10K.pdf"),
    ("3M_2018_10K",                                FB_PDF_DIR / "3M_2018_10K.pdf"),
    ("JPMORGAN_2021Q1_10Q",                        FB_PDF_DIR / "JPMORGAN_2021Q1_10Q.pdf"),
    ("BESTBUY_2024Q2_10Q",                         FB_PDF_DIR / "BESTBUY_2024Q2_10Q.pdf"),
    ("AMCOR_2022_8K_dated-2022-07-01",             FB_PDF_DIR / "AMCOR_2022_8K_dated-2022-07-01.pdf"),
    ("JOHNSON_JOHNSON_2023_8K_dated-2023-08-30",   FB_PDF_DIR / "JOHNSON_JOHNSON_2023_8K_dated-2023-08-30.pdf"),
    ("MGMRESORTS_2022Q4_EARNINGS",                 FB_PDF_DIR / "MGMRESORTS_2022Q4_EARNINGS.pdf"),
    ("ULTABEAUTY_2023Q4_EARNINGS",                 FB_PDF_DIR / "ULTABEAUTY_2023Q4_EARNINGS.pdf"),
]

DOMAINS = ["automotive", "energy", "government", "healthcare", "insurance", "legal", "real_estate"]
DOMAIN_PDFS_PER_DOMAIN = 5  # 7 domains × 5 = 35 domain PDFs

# ── Training example format (identical to structdirect_train.jsonl) ────────────

SYSTEM_PROMPT = (
    "You are a document tree navigator. Given a question and a list of "
    "document sections at the current level, output JSON specifying which "
    "sections to explore next.\n"
    "Reply format: {\"thinking\": \"...\", \"navigate_to\": [\"XXXX\", ...]}"
)

MAX_NODES_PER_HOP = 20


def nodes_containing_pages(nodes: list[TreeNode], target_pages: list[int]) -> list[str]:
    return [n.node_id for n in nodes
            if any(n.start_index <= p <= n.end_index for p in target_pages)]


def get_children(tree_structure: list[TreeNode], parent_id: str) -> list[TreeNode]:
    def find(nodes):
        for n in nodes:
            if n.node_id == parent_id:
                return n.nodes
            r = find(n.nodes)
            if r is not None:
                return r
        return None
    return find(tree_structure) or []


def format_node_list(nodes: list[TreeNode]) -> str:
    lines = []
    for n in nodes:
        sub = " [has sub-sections]" if n.nodes else ""
        lines.append(f"[{n.node_id}] {n.title} (pages {n.start_index}-{n.end_index}){sub}")
    return "\n".join(lines)


def make_thinking(question: str, target_ids: list[str], nodes: list[TreeNode]) -> str:
    id_to_title = {n.node_id: n.title for n in nodes}
    targets = [id_to_title.get(t, t) for t in target_ids]
    return (
        f"The question asks: '{question[:80]}'. "
        f"'{', '.join(targets)}' is most likely to contain the answer."
    )


def make_example(doc: str, question: str, node_list: list[TreeNode], target_ids: list[str]) -> dict:
    """Build one training example, windowing to MAX_NODES_PER_HOP if needed."""
    if len(node_list) > MAX_NODES_PER_HOP:
        id_to_idx = {n.node_id: i for i, n in enumerate(node_list)}
        target_positions = [id_to_idx[t] for t in target_ids if t in id_to_idx]
        if target_positions:
            center = sum(target_positions) // len(target_positions)
            half   = MAX_NODES_PER_HOP // 2
            start  = max(0, center - half)
            end    = min(len(node_list), start + MAX_NODES_PER_HOP)
            start  = max(0, end - MAX_NODES_PER_HOP)
            node_list = node_list[start:end]
        else:
            node_list = node_list[:MAX_NODES_PER_HOP]

    return {"doc": doc, "messages": [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": f"Question: {question}\n\nSections at this level:\n{format_node_list(node_list)}"},
        {"role": "assistant", "content": json.dumps({
            "thinking":    make_thinking(question, target_ids, node_list),
            "navigate_to": target_ids,
        })},
    ]}


def add_hops(node_list: list[TreeNode], doc: str, question: str,
             evidence_pages: list[int], tree: DocumentTree,
             examples: list[dict], depth: int = 0) -> None:
    """Generate multi-hop training examples, recurse into children."""
    targets = nodes_containing_pages(node_list, evidence_pages)
    if not targets or depth > 3 or len(targets) > 4:
        return
    examples.append(make_example(doc, question, node_list, targets))
    shuffled = node_list[:]
    random.shuffle(shuffled)
    examples.append(make_example(doc, question, shuffled, targets))
    for t in targets:
        children = get_children(tree.structure, t)
        if children:
            add_hops(children, doc, question, evidence_pages, tree, examples, depth + 1)


def add_recovery_examples(tree: DocumentTree, question: str, evidence_pages: list[int],
                           examples: list[dict], doc: str, max_distractors: int = 2) -> None:
    """Teach the model to recover after exploring a wrong branch."""
    top_level    = tree.structure
    evidence_set = set(nodes_containing_pages(top_level, evidence_pages))
    if not evidence_set:
        return

    wrong_nodes = [n for n in top_level if n.node_id not in evidence_set]
    random.shuffle(wrong_nodes)

    for wrong_node in wrong_nodes[:max_distractors]:
        evidence_titles = [n.title for n in top_level if n.node_id in evidence_set]
        thinking = (
            f"I previously explored '{wrong_node.title}' which did not contain the answer. "
            f"The question asks: '{question[:60]}'. "
            f"The answer is more likely in '{', '.join(evidence_titles)}'."
        )
        node_list = top_level[:]
        if len(node_list) > MAX_NODES_PER_HOP:
            id_to_idx  = {n.node_id: i for i, n in enumerate(node_list)}
            positions  = [id_to_idx[t] for t in evidence_set if t in id_to_idx]
            if positions:
                center = sum(positions) // len(positions)
                half   = MAX_NODES_PER_HOP // 2
                start  = max(0, center - half)
                end    = min(len(node_list), start + MAX_NODES_PER_HOP)
                start  = max(0, end - MAX_NODES_PER_HOP)
                node_list = node_list[start:end]

        target_ids = [t for t in list(evidence_set) if any(n.node_id == t for n in node_list)]
        if not target_ids:
            continue

        user_content = (
            f"Question: {question}\n\n"
            f"Previously explored: [{wrong_node.node_id}] {wrong_node.title} "
            f"(pages {wrong_node.start_index}-{wrong_node.end_index}) — no relevant content found.\n\n"
            f"Sections at this level:\n"
            f"{format_node_list(node_list)}"
        )
        examples.append({"doc": doc, "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": json.dumps({
                "thinking":    thinking,
                "navigate_to": target_ids,
            })},
        ]})


def add_negative_examples(tree: DocumentTree, question: str, evidence_pages: list[int],
                           examples: list[dict], doc: str) -> None:
    """Hard negative: train the model to reject the closest wrong neighbor."""
    top_level    = tree.structure
    evidence_set = set(nodes_containing_pages(top_level, evidence_pages))
    if not evidence_set:
        return

    id_to_idx = {n.node_id: i for i, n in enumerate(top_level)}

    for correct_id in list(evidence_set):
        correct_idx = id_to_idx.get(correct_id)
        if correct_idx is None:
            continue
        best_neighbor, best_dist = None, float("inf")
        for n in top_level:
            if n.node_id in evidence_set:
                continue
            dist = abs(id_to_idx.get(n.node_id, 999) - correct_idx)
            if dist < best_dist:
                best_dist, best_neighbor = dist, n

        if best_neighbor is None:
            continue

        correct_node = next((n for n in top_level if n.node_id == correct_id), None)
        if correct_node is None:
            continue

        thinking = (
            f"Although '[{best_neighbor.node_id}] {best_neighbor.title}' "
            f"(pages {best_neighbor.start_index}-{best_neighbor.end_index}) is nearby, "
            f"it does not contain the answer. "
            f"The answer requires '[{correct_id}] {correct_node.title}' "
            f"(pages {correct_node.start_index}-{correct_node.end_index})."
        )
        node_list = top_level[:]
        if len(node_list) > MAX_NODES_PER_HOP:
            center = (id_to_idx.get(correct_id, 0) + id_to_idx.get(best_neighbor.node_id, 0)) // 2
            half   = MAX_NODES_PER_HOP // 2
            start  = max(0, center - half)
            end    = min(len(node_list), start + MAX_NODES_PER_HOP)
            start  = max(0, end - MAX_NODES_PER_HOP)
            node_list = node_list[start:end]

        target_ids = [t for t in list(evidence_set) if any(n.node_id == t for n in node_list)]
        if not target_ids:
            continue

        examples.append({"doc": doc, "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Question: {question}\n\nSections at this level:\n{format_node_list(node_list)}"},
            {"role": "assistant", "content": json.dumps({
                "thinking":    thinking,
                "navigate_to": target_ids,
            })},
        ]})
        break  # one hard negative per question


# ── Section text extraction ────────────────────────────────────────────────────

def extract_section_text(pdf_path: Path, start_page: int, end_page: int,
                          max_chars: int = 2000) -> str:
    """Extract and clean text from a section's page range. Caps at max_chars."""
    doc = pymupdf.open(str(pdf_path))
    total = len(doc)
    texts = []
    for pg in range(start_page, min(end_page + 1, total)):
        texts.append(doc[pg].get_text())
    doc.close()
    combined = "\n".join(texts)
    # Basic cleanup: collapse excessive whitespace, remove page numbers
    combined = re.sub(r'\n{3,}', '\n\n', combined)
    combined = re.sub(r'[ \t]+', ' ', combined)
    return combined[:max_chars].strip()


def get_leaf_sections(tree: DocumentTree) -> list[TreeNode]:
    """Return all leaf nodes that have meaningful content (>1 page span)."""
    def collect(nodes: list[TreeNode]) -> list[TreeNode]:
        result = []
        for n in nodes:
            if not n.nodes:
                result.append(n)
            else:
                result.extend(collect(n.nodes))
        return result
    return collect(tree.structure)


# ── Domain PDF selection ───────────────────────────────────────────────────────

def select_domain_pdfs() -> list[tuple[str, Path]]:
    """
    For each domain, pick the 5 PDFs with the most sections.
    Returns list of (doc_name, pdf_path) tuples.
    """
    selected = []
    for domain in DOMAINS:
        domain_dir = DOMAIN_PDF_DIR / domain
        if not domain_dir.exists():
            print(f"  WARNING: domain dir not found: {domain_dir}")
            continue

        pdfs = sorted(domain_dir.glob("*.pdf"))
        if not pdfs:
            print(f"  WARNING: no PDFs in {domain_dir}")
            continue

        # Score each PDF by section count (more sections = more training diversity)
        scored = []
        for pdf in pdfs:
            try:
                tree = extract_structure(str(pdf))
                leaves = get_leaf_sections(tree)
                scored.append((len(leaves), pdf))
            except Exception:
                pass

        scored.sort(reverse=True)
        top5 = scored[:DOMAIN_PDFS_PER_DOMAIN]

        for _, pdf in top5:
            doc_name = f"{domain}__{pdf.stem}"
            selected.append((doc_name, pdf))

    return selected


# ── Haiku question generation ──────────────────────────────────────────────────

QUESTION_GEN_PROMPT = """\
You are helping create training data for a document navigation model. \
Given the text from a specific section of a document, generate 5 distinct \
factual questions whose answers are clearly found in this text.

Requirements:
- Each question must ask about a DIFFERENT fact, topic, or data point in the text
- Questions should be specific enough that only THIS section would answer them
- Cover different aspects: numbers/metrics, concepts, comparisons, definitions, causes/reasons
- Do NOT mention the section title or page numbers in the questions
- Do NOT generate questions that require information from outside this text

Document: {doc_name}
Section: {section_title} (pages {start_page}-{end_page})

Text excerpt:
{page_text}

Respond with exactly 5 questions, one per line, numbered 1-5. No explanations."""


REPHRASE_PROMPT = """\
Rewrite this question in 4 different styles. Keep the exact same meaning but change how it is asked.

Original question: {question}

Write exactly 4 rewrites, one per line, numbered 1-4:
1. As a casual person with no technical background would ask (simple, conversational)
2. As a financial analyst or domain expert would ask (precise, technical terminology)
3. As an academic researcher would ask (formal, rigorous phrasing)
4. As a student new to the topic would ask (curious, exploratory)

Rules:
- Each rewrite MUST ask for the same information as the original
- Do NOT add facts or details not in the original question
- Keep each rewrite under 30 words
- No explanations, just the 4 numbered rewrites"""


def parse_numbered_lines(text: str, expected: int) -> list[str]:
    """Parse numbered list output from Haiku. Returns up to `expected` items."""
    lines = []
    for line in text.strip().splitlines():
        line = line.strip()
        # Strip leading number+period/dot: "1. ", "1) ", "1 - ", etc.
        cleaned = re.sub(r'^[0-9]+[.):\-]\s*', '', line).strip()
        if cleaned:
            lines.append(cleaned)
    return lines[:expected]


def validate_question(q: str) -> bool:
    """Basic sanity checks on a generated question."""
    q = q.strip()
    if len(q) < 10:
        return False
    if not q.endswith("?"):
        return False
    # Reject if it contains page number references
    if re.search(r'\bpage[s]?\s+\d+', q, re.IGNORECASE):
        return False
    # Reject meta-phrasing that references the document/text itself —
    # these are navigation-irrelevant and confuse the model
    meta_patterns = [
        r'^according to the (text|document|excerpt|passage)',
        r'^based on (the )?(text|document|excerpt|passage)',
        r'^per the (text|document|excerpt)',
        r'^from the (text|document|excerpt)',
        r'^in the (text|document|excerpt)',
        r'^as (stated|mentioned|described|noted) in',
        r'^what does the (text|document|excerpt) (say|state|indicate)',
    ]
    q_lower = q.lower()
    if any(re.match(p, q_lower) for p in meta_patterns):
        return False
    return True


async def generate_questions_for_section(
    client: anthropic.AsyncAnthropic,
    doc_name: str,
    node: TreeNode,
    page_text: str,
    semaphore: asyncio.Semaphore,
    test_mode: bool = False,
) -> list[str]:
    """Call Haiku once to get 5 questions for this section."""
    prompt = QUESTION_GEN_PROMPT.format(
        doc_name=doc_name,
        section_title=node.title,
        start_page=node.start_index,
        end_page=node.end_index,
        page_text=page_text,
    )
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=350,
                    temperature=0.4,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text
                questions = parse_numbered_lines(raw, 5)
                questions = [q for q in questions if validate_question(q)]
                return questions
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"  ERROR generating questions for {doc_name}/{node.title}: {e}")
                    return []
    return []


async def generate_rephrasings(
    client: anthropic.AsyncAnthropic,
    question: str,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    """Call Haiku to get 4 rephrasings. Returns original + 4 = 5 total."""
    prompt = REPHRASE_PROMPT.format(question=question)
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=300,
                    temperature=0.5,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text
                rephrasings = parse_numbered_lines(raw, 4)
                # Filter: must look like questions
                rephrasings = [r for r in rephrasings if len(r) > 8]
                # Return original + rephrasings (up to 5 total)
                return [question] + rephrasings[:4]
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return [question]  # fallback: just the original
    return [question]


# ── Main generation pipeline ───────────────────────────────────────────────────

async def process_pdf(
    doc_name: str,
    pdf_path: Path,
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    test_mode: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Process one PDF end-to-end.
    Returns (training_examples, raw_question_records).
    """
    # Extract tree
    try:
        tree = extract_structure(str(pdf_path))
    except Exception as e:
        print(f"  SKIP {doc_name}: extract_structure failed: {e}")
        return [], []

    leaves = get_leaf_sections(tree)

    # In test mode, only process first 3 sections
    if test_mode:
        leaves = leaves[:3]

    raw_records = []
    all_examples = []

    for node in leaves:
        # Skip tiny sections and preamble nodes
        if node.title == "Cover / Preamble":
            continue
        if node.end_index - node.start_index < 1:
            continue  # single-page sections often lack enough content

        # Extract text
        page_text = extract_section_text(pdf_path, node.start_index, node.end_index)
        if len(page_text) < 100:
            continue  # not enough text to generate meaningful questions

        # Generate 5 base questions
        questions = await generate_questions_for_section(
            client, doc_name, node, page_text, semaphore, test_mode
        )
        if not questions:
            continue

        # For each question, generate 4 rephrasings
        rephrase_tasks = [
            generate_rephrasings(client, q, semaphore)
            for q in questions
        ]
        all_variants = await asyncio.gather(*rephrase_tasks)

        # Flatten: all_variants is list of lists (one per question, each has original+4)
        for q_variants in all_variants:
            raw_records.append({
                "doc": doc_name,
                "section": node.title,
                "pages": [node.start_index, node.end_index],
                "variants": q_variants,
            })

        # Build oracle training examples for every question variant
        evidence_pages = list(range(node.start_index, node.end_index + 1))
        for q_variants in all_variants:
            for question in q_variants:
                if not question:
                    continue
                add_hops(tree.structure, doc_name, question, evidence_pages, tree, all_examples)
                add_recovery_examples(tree, question, evidence_pages, all_examples, doc_name)
                add_negative_examples(tree, question, evidence_pages, all_examples, doc_name)

    return all_examples, raw_records


async def run(test_mode: bool = False, limit: Optional[int] = None) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set. Export it before running.")

    client    = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(8)  # max 8 concurrent Haiku requests

    # ── Phase 1: Select PDFs ──────────────────────────────────────────────────
    print("=" * 60)
    print("Phase 1: Selecting PDFs")
    print("=" * 60)

    # FinanceBench: check which files exist
    fb_pdfs = [(name, path) for name, path in FINANCEBENCH_PDFS if path.exists()]
    missing = [name for name, path in FINANCEBENCH_PDFS if not path.exists()]
    if missing:
        print(f"  WARNING: {len(missing)} FinanceBench PDFs not found: {missing}")
    print(f"  FinanceBench PDFs: {len(fb_pdfs)}")

    # Domain PDFs: pick top 5 per domain by section count
    print("  Selecting domain PDFs (top 5 per domain by section count)...")
    domain_pdfs = select_domain_pdfs()
    print(f"  Domain PDFs selected: {len(domain_pdfs)}")

    if test_mode:
        # Test mode: only process first FinanceBench PDF
        all_pdfs = fb_pdfs[:1]
        print(f"\n  TEST MODE: processing only {all_pdfs[0][0]}")
    else:
        all_pdfs = fb_pdfs + domain_pdfs

    print(f"\n  Total PDFs to process: {len(all_pdfs)}")

    # ── Phase 2: Load checkpoint ──────────────────────────────────────────────
    progress: dict = {}
    if PROGRESS_FILE.exists() and not test_mode:
        progress = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        print(f"\nResuming from checkpoint: {len(progress)} PDFs already done")

    # ── Phase 3: Generate questions (async) ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 3: Generating questions via Haiku")
    print("=" * 60)

    all_raw_records: list[dict] = []
    all_training_examples: list[dict] = []

    # Load previously completed examples
    for doc_name, data in progress.items():
        all_raw_records.extend(data.get("raw_records", []))
        all_training_examples.extend(data.get("examples", []))

    pdfs_to_process = [(n, p) for n, p in all_pdfs if n not in progress]

    if limit is not None:
        print(f"\n  --limit {limit}: will process at most {limit} new PDFs this run "
              f"({len(pdfs_to_process)} remaining in queue)")
        pdfs_to_process = pdfs_to_process[:limit]

    iterator = tqdm(pdfs_to_process, desc="PDFs") if HAS_TQDM else pdfs_to_process

    for doc_name, pdf_path in iterator:
        if not HAS_TQDM:
            print(f"  Processing: {doc_name}")

        examples, raw_records = await process_pdf(
            doc_name, pdf_path, client, semaphore, test_mode
        )

        all_training_examples.extend(examples)
        all_raw_records.extend(raw_records)

        # Checkpoint: save after every PDF
        if not test_mode:
            progress[doc_name] = {
                "examples":    examples,
                "raw_records": raw_records,
            }
            PROGRESS_FILE.write_text(
                json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        q_count   = sum(len(r["variants"]) for r in raw_records)
        ex_count  = len(examples)
        sec_count = len(raw_records)
        if not HAS_TQDM:
            print(f"    {sec_count} sections, {q_count} question variants, {ex_count} training examples")

    # ── Phase 4: Save raw questions for manual inspection ─────────────────────
    print("\n" + "=" * 60)
    print("Phase 4: Saving raw questions")
    print("=" * 60)

    OUT_RAW.write_text(
        json.dumps(all_raw_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total_variants = sum(len(r["variants"]) for r in all_raw_records)
    print(f"  Saved {len(all_raw_records)} sections, {total_variants} question variants")
    print(f"  File: {OUT_RAW}")
    print(f"\n  IMPORTANT: Review {OUT_RAW.name} before proceeding.")
    print(f"  Check ~20 random entries to verify question quality.")

    if test_mode:
        print("\n  TEST MODE: printing sample questions:")
        for record in all_raw_records[:3]:
            print(f"\n  Section: {record['section']} (pages {record['pages'][0]}-{record['pages'][1]})")
            for i, v in enumerate(record["variants"][:5]):
                label = ["Original", "Casual", "Analyst", "Researcher", "Student"][i] if i < 5 else f"v{i+1}"
                print(f"    [{label}] {v}")
        return  # Stop here in test mode — don't write training data yet

    # ── Phase 5: Deduplicate, shuffle, write training JSONL ──────────────────
    print("\n" + "=" * 60)
    print("Phase 5: Writing training JSONL")
    print("=" * 60)

    seen   = set()
    unique = []
    for ex in all_training_examples:
        key = ex["messages"][1]["content"]
        h   = hashlib.md5(key.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(ex)

    random.seed(42)
    random.shuffle(unique)

    OUT_JSONL.write_text(
        "\n".join(json.dumps(ex, ensure_ascii=False) for ex in unique) + "\n",
        encoding="utf-8",
    )

    multi_node = [
        ex for ex in unique
        if len(json.loads(ex["messages"][2]["content"])["navigate_to"]) > 1
    ]

    print(f"  Total raw examples   : {len(all_training_examples)}")
    print(f"  After dedup          : {len(unique)}")
    print(f"  Multi-node examples  : {len(multi_node)} ({len(multi_node)/len(unique)*100:.1f}%)")
    print(f"  Saved to             : {OUT_JSONL}")
    print(f"\nDone!")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate diverse TreeSearch training data using Claude Haiku."
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: process only the first FinanceBench PDF (3 sections). "
             "Prints question samples without writing training JSONL. Costs ~$0.05."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after processing this many PDFs. Already-completed PDFs (from "
             "a previous run) are skipped and do NOT count toward this limit. "
             "Run again with the same --limit to process the next batch."
    )
    args = parser.parse_args()

    asyncio.run(run(test_mode=args.test, limit=args.limit))
