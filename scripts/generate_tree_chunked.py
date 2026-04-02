"""
Chunked TreeGen pipeline for large documents.

Implements the 2-phase architecture from CHUNKED_TREEGEN_ARCHITECTURE.md:
  Phase 1: Extract TOC skeleton (regex, <1s) or heading scan fallback
  Phase 2: Process each section IN PARALLEL with TreeGen
  Phase 3: Merge sub-trees + reassign node IDs sequentially

Does NOT import arbor — uses PDF extractor directly and calls the fine-tuned
model via standard transformers (not Unsloth, which has a RoPE inference bug).

Usage:
    python scripts/generate_tree_chunked.py data/financebench/pdfs/3M_2022_10K.pdf
    python scripts/generate_tree_chunked.py doc.pdf --output-dir data/financebench/trees
    python scripts/generate_tree_chunked.py doc.pdf --provider gemini
    python scripts/generate_tree_chunked.py doc.pdf --provider finetuned --model TStark12310/arbor-treegen-7b-v2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).parent.parent

# ── System prompts ─────────────────────────────────────────────────────────────

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
- Page ranges must match where content actually appears'''

SECTION_SUFFIX = """

IMPORTANT: You are generating the sub-tree for ONE SECTION of a larger document.
Context: {context}
Generate the internal structure (subsections, summary) of THIS section only.
The root of your output should be this section with its page range."""


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class PageContent:
    page_num: int
    text: str
    token_count: int = 0


@dataclass
class SkeletonSection:
    title: str
    start_page: int
    end_page: int
    children: list = field(default_factory=list)


# ── PDF extraction ─────────────────────────────────────────────────────────────

def extract_pages(pdf_path: Path) -> list[PageContent]:
    from arbor.extraction.pdf_extractor import get_page_contents
    raw = get_page_contents(str(pdf_path))
    return [PageContent(page_num=i + 1, text=p.text, token_count=p.token_count)
            for i, p in enumerate(raw)]


def pages_to_text(pages: list[PageContent]) -> str:
    return "\n\n".join(f"[Page {p.page_num}]\n{p.text.strip()}" for p in pages)


# ── TOC patterns ───────────────────────────────────────────────────────────────

TOC_PATTERNS = [
    # "Item 1. Business .............. 4"  (2+ dots)
    re.compile(r'^(.+?)\s*\.{2,}\s*(\d+)\s*$'),
    # "Item 1. Business    4"  (3+ spaces, no dots)
    re.compile(r'^(.{5,}?)\s{3,}(\d+)\s*$'),
    # "1.1 Introduction ... 12"
    re.compile(r'^(\d+(?:\.\d+)*\s+.+?)\s*\.{0,}\s*(\d+)\s*$'),
    # "ITEM 1. BUSINESS 4"  or  "Item 1A. Risk Factors 10"  (10-K style)
    re.compile(r'^((?:ITEM|PART)\s+\d+[A-Z]?\.?\s+.+?)\s+(\d+)\s*$', re.IGNORECASE),
    # "Item 1. Business\t4"  (tab-separated)
    re.compile(r'^(.+?)\t(\d+)\s*$'),
    # "Business  .  .  .  4"  (spaced dots)
    re.compile(r'^(.+?)\s+(?:\.\s+){2,}(\d+)\s*$'),
]

HEADING_PATTERNS = [
    re.compile(r'^[A-Z][A-Z\s\d\-\(\)\/]{4,60}$'),
    re.compile(r'^\d+(?:\.\d+)*\.?\s+\S'),
    re.compile(r'^(?:Abstract|Introduction|Background|Methodology|Methods|Results|'
               r'Discussion|Conclusion|References|Appendix|Acknowledgements?)\b',
               re.IGNORECASE),
    re.compile(r'^(?:PART|ITEM)\s+(?:\d+[A-Z]?|[IVX]+)\.?\b', re.IGNORECASE),
]


# ── Phase 1: Skeleton extraction ───────────────────────────────────────────────

def extract_toc_skeleton(pages: list[PageContent], max_toc_pages: int = 5) -> list[SkeletonSection] | None:
    entries: list[tuple[str, int]] = []
    candidates: list[str] = []

    for page in pages[:max_toc_pages]:
        for line in page.text.split('\n'):
            line = line.strip()
            if not line or len(line) < 5:
                continue
            for pat in TOC_PATTERNS:
                m = pat.match(line)
                if m and m.lastindex >= 2:
                    title = m.group(1).strip().rstrip('.')
                    try:
                        pg = int(m.group(2))
                        if 1 <= pg <= len(pages) + 5:
                            entries.append((title, pg))
                            candidates.append(f"  [{pg:>4}] {title}")
                    except ValueError:
                        pass
                    break

    print(f"[TOC debug] {len(candidates)} candidate lines matched in first {max_toc_pages} pages:")
    for c in candidates[:20]:
        print(c)
    if len(candidates) > 20:
        print(f"  ... and {len(candidates) - 20} more")

    if len(entries) < 3:
        return None

    seen: set[str] = set()
    unique: list[tuple[str, int]] = []
    for title, pg in entries:
        if title not in seen:
            seen.add(title)
            unique.append((title, pg))
    unique.sort(key=lambda x: x[1])

    skeleton = []
    for i, (title, start_pg) in enumerate(unique):
        end_pg = unique[i + 1][1] - 1 if i + 1 < len(unique) else len(pages)
        end_pg = max(start_pg, min(end_pg, len(pages)))
        skeleton.append(SkeletonSection(title=title, start_page=start_pg, end_page=end_pg))

    return skeleton or None


def build_skeleton_from_headings(pages: list[PageContent], max_pages_per_chunk: int = 10) -> list[SkeletonSection]:
    headings: list[tuple[str, int]] = []

    for page in pages:
        for line in page.text.split('\n'):
            line = line.strip()
            if not line or len(line) < 4 or len(line) > 100:
                continue
            for pat in HEADING_PATTERNS:
                if pat.match(line):
                    headings.append((line, page.page_num))
                    break

    if len(headings) < 3:
        return build_equal_splits(pages, max_pages_per_chunk)

    seen: set[str] = set()
    unique: list[tuple[str, int]] = []
    for title, pg in headings:
        if title not in seen:
            seen.add(title)
            unique.append((title, pg))

    if len(unique) > 30:
        print(f"  [heading scan] {len(unique)} headings found — too many, falling back to equal splits")
        return build_equal_splits(pages, max_pages_per_chunk)

    skeleton = []
    for i, (title, start_pg) in enumerate(unique):
        end_pg = unique[i + 1][1] - 1 if i + 1 < len(unique) else len(pages)
        end_pg = max(start_pg, min(end_pg, len(pages)))
        skeleton.append(SkeletonSection(title=title, start_page=start_pg, end_page=end_pg))

    return skeleton


def build_equal_splits(pages: list[PageContent], chunk_size: int = 10) -> list[SkeletonSection]:
    skeleton = []
    for i in range(0, len(pages), chunk_size):
        chunk = pages[i:i + chunk_size]
        skeleton.append(SkeletonSection(
            title=f"Pages {chunk[0].page_num}-{chunk[-1].page_num}",
            start_page=chunk[0].page_num,
            end_page=chunk[-1].page_num,
        ))
    return skeleton


def split_large_sections(skeleton: list[SkeletonSection], max_pages: int) -> list[SkeletonSection]:
    result = []
    for section in skeleton:
        size = section.end_page - section.start_page + 1
        if size <= max_pages:
            result.append(section)
        else:
            pg = section.start_page
            part = 1
            while pg <= section.end_page:
                end = min(pg + max_pages - 1, section.end_page)
                result.append(SkeletonSection(
                    title=f"{section.title} (part {part})",
                    start_page=pg,
                    end_page=end,
                ))
                pg = end  # 1-page overlap
                part += 1
    return result


def batch_short_sections(skeleton: list[SkeletonSection], min_pages: int = 2) -> list[SkeletonSection]:
    result = []
    pending: list[SkeletonSection] = []

    def flush():
        if not pending:
            return
        if len(pending) == 1:
            result.append(pending[0])
        else:
            result.append(SkeletonSection(
                title=" / ".join(s.title for s in pending),
                start_page=pending[0].start_page,
                end_page=pending[-1].end_page,
                children=list(pending),
            ))
        pending.clear()

    for section in skeleton:
        size = section.end_page - section.start_page + 1
        if size < min_pages:
            pending.append(section)
            if sum(s.end_page - s.start_page + 1 for s in pending) >= min_pages * 3:
                flush()
        else:
            flush()
            result.append(section)

    flush()
    return result


# ── LLM providers ──────────────────────────────────────────────────────────────

class GeminiTreeGenProvider:
    """Calls Gemini API for tree generation."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite"):
        from google import genai
        from google.genai import types as genai_types
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self._types = genai_types

    async def complete(self, messages: list[dict], max_tokens: int = 8192) -> str:
        contents = []
        system_text = None
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            elif m["role"] == "user":
                contents.append(m["content"])

        config = self._types.GenerateContentConfig(
            system_instruction=system_text,
            max_output_tokens=max_tokens,
            temperature=0.0,
        )
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents="\n\n".join(contents),
            config=config,
        )
        return response.text.strip()


class FineTunedTreeGenProvider:
    """Loads fine-tuned model directly via transformers. No HTTP server needed."""

    def __init__(self, model_path: str = "TStark12310/arbor-treegen-7b-v2",
                 max_new_tokens: int = 4096, max_seq_length: int = 8192):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"[FineTunedTreeGenProvider] Loading {model_path} ...")
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_length

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        print("[FineTunedTreeGenProvider] Ready.")

    def _generate_sync(self, messages: list[dict]) -> str:
        import torch
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=self.max_seq_length,
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
            )
        return self._tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

    async def complete(self, messages: list[dict], max_tokens: int = 4096) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_sync, messages)


# ── JSON parsing ───────────────────────────────────────────────────────────────

def parse_tree_json(text: str) -> dict | None:
    text = text.strip()
    if "```" in text:
        start = text.find("```")
        end = text.rfind("```")
        if start != end:
            inner = text[start + 3:end].strip()
            if inner.startswith("json"):
                inner = inner[4:].strip()
            text = inner

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


# ── Phase 2: Section-aware generation ─────────────────────────────────────────

async def treegen_for_section(
    section: SkeletonSection,
    pages: list[PageContent],
    provider,
    max_input_tokens: int = 6000,
) -> dict:
    section_pages = [p for p in pages if section.start_page <= p.page_num <= section.end_page]
    if not section_pages:
        return {"structure": [], "doc_name": section.title}

    paged_text = pages_to_text(section_pages)

    max_chars = max_input_tokens * 4
    if len(paged_text) > max_chars:
        truncated = paged_text[:max_chars]
        last_marker = truncated.rfind("[Page ")
        if last_marker > max_chars * 0.7:
            truncated = truncated[:last_marker]
        paged_text = truncated.rstrip() + "\n[Section truncated]"

    context = f"Section: '{section.title}' (pages {section.start_page}-{section.end_page})"
    system = TREEGEN_SYSTEM + SECTION_SUFFIX.format(context=context)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": paged_text},
    ]

    response = await provider.complete(messages)
    parsed = parse_tree_json(response)
    if parsed:
        return parsed

    # Fallback minimal node
    return {
        "doc_name": section.title,
        "structure": [{
            "title": section.title,
            "node_id": "0001",
            "start_index": section.start_page,
            "end_index": section.end_page,
            "summary": f"Content from pages {section.start_page}-{section.end_page}.",
            "nodes": [],
        }]
    }


# ── Phase 3: Merge + ID reassignment ──────────────────────────────────────────

def merge_skeleton_with_subtrees(skeleton: list[SkeletonSection], sub_trees: list[dict]) -> dict:
    structure = []
    for section, sub_tree in zip(skeleton, sub_trees):
        sub_structure = sub_tree.get("structure", [])

        if (len(sub_structure) == 1
                and sub_structure[0].get("title", "").strip().lower() == section.title.strip().lower()):
            root = sub_structure[0]
            node = {
                "title": section.title,
                "node_id": "0000",
                "start_index": section.start_page,
                "end_index": section.end_page,
                "summary": root.get("summary", ""),
                "nodes": root.get("nodes", []),
            }
        else:
            summary = sub_structure[0].get("summary", "") if sub_structure else ""
            node = {
                "title": section.title,
                "node_id": "0000",
                "start_index": section.start_page,
                "end_index": section.end_page,
                "summary": summary,
                "nodes": sub_structure,
            }
        structure.append(node)

    return {"structure": structure}


def reassign_node_ids(tree: dict) -> dict:
    counter = [1]

    def assign(nodes: list):
        for node in nodes:
            node["node_id"] = str(counter[0]).zfill(4)
            counter[0] += 1
            if node.get("nodes"):
                assign(node["nodes"])

    assign(tree.get("structure", []))
    return tree


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def generate_tree_chunked(
    pdf_path: Path,
    provider,
    max_pages_per_chunk: int = 10,
    verbose: bool = True,
) -> dict:
    def log(msg):
        if verbose:
            print(msg, flush=True)

    t0 = time.monotonic()

    log(f"Extracting pages from {pdf_path.name} ...")
    pages = extract_pages(pdf_path)
    log(f"  {len(pages)} pages extracted")

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    log("\nPhase 1: Extracting structure skeleton ...")

    skeleton = extract_toc_skeleton(pages)
    if skeleton:
        log(f"  TOC found: {len(skeleton)} top-level sections")
    else:
        log("  No TOC found — scanning headings ...")
        skeleton = build_skeleton_from_headings(pages, max_pages_per_chunk)
        log(f"  Heading scan: {len(skeleton)} sections")

    skeleton = split_large_sections(skeleton, max_pages_per_chunk)
    skeleton = batch_short_sections(skeleton, min_pages=2)
    log(f"  Final skeleton: {len(skeleton)} chunks")
    for s in skeleton:
        log(f"    [{s.start_page}-{s.end_page}] {s.title}")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    log(f"\nPhase 2: Processing {len(skeleton)} sections (max 5 concurrent) ...")
    t2 = time.monotonic()

    semaphore = asyncio.Semaphore(5)

    async def bounded(section):
        async with semaphore:
            return await treegen_for_section(section, pages, provider)

    sub_trees = await asyncio.gather(*[bounded(s) for s in skeleton])
    log(f"  Done in {round(time.monotonic() - t2, 1)}s")

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    log("\nPhase 3: Merging sub-trees ...")
    merged = merge_skeleton_with_subtrees(skeleton, sub_trees)
    merged = reassign_node_ids(merged)
    merged["doc_name"] = pdf_path.stem

    def count_nodes(nodes):
        return sum(1 + count_nodes(n.get("nodes", [])) for n in nodes)

    node_count = count_nodes(merged.get("structure", []))
    elapsed = round(time.monotonic() - t0, 1)
    log(f"\nDone: {node_count} nodes, {len(merged.get('structure', []))} top-level sections, {elapsed}s total")

    return merged


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chunked TreeGen for large documents")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-pages-per-chunk", type=int, default=10)
    parser.add_argument("--provider", choices=["gemini", "finetuned"], default="gemini")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if not args.pdf.exists():
        sys.exit(f"File not found: {args.pdf}")

    if args.provider == "finetuned":
        provider = FineTunedTreeGenProvider(
            model_path=args.model or "TStark12310/arbor-treegen-7b-v2"
        )
    else:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            sys.exit("GEMINI_API_KEY not set")
        provider = GeminiTreeGenProvider(api_key=api_key, model=args.model or "gemini-2.5-flash-lite")

    tree = asyncio.run(generate_tree_chunked(
        args.pdf,
        provider,
        max_pages_per_chunk=args.max_pages_per_chunk,
    ))

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / f"{args.pdf.stem}.json"
        out_path.write_text(json.dumps({"tree": tree}, indent=2), encoding="utf-8")
        print(f"Saved to {out_path}")
    else:
        print(json.dumps(tree, indent=2))


if __name__ == "__main__":
    main()
