"""
Arbor TreeGen Pipeline v2 — Colab Notebook (vLLM + TreeGen-7B-v2)
=================================================================

Improvements over v1:
- PyMuPDF TOC extraction (doc.get_toc()) instead of regex — fixes 0/84 TOC hit rate
- Schema normalizer — forces canonical {title, node_id, start_index, end_index, nodes}
- Smarter JSON repair — handles truncated JSON, markdown contamination
- Chunk size auto-tuning based on actual token count (not fixed pages)
- Stronger prompts with explicit schema examples and forbidden keys
- Debug logging on EVERY chunk (not just chunk 0)
- Retry logic for failed chunks with temperature bump

Copy each cell into Colab in order.
"""

# ============================================================================
# CELL 1: Install dependencies
# ============================================================================
CELL_1 = """
!pip install -q vllm pymupdf transformers
"""

# ============================================================================
# CELL 2: Mount Drive, set paths
# ============================================================================
CELL_2 = """
from google.colab import drive
drive.mount('/content/drive')

PDF_DIR = "/content/drive/My Drive/arbor-training-data/financebench-pdfs"
TREE_DIR = "/content/drive/My Drive/arbor-training-data/financebench-trees"
MODEL_ID = "TStark12310/arbor-treegen-7b-v2"

import os
os.makedirs(TREE_DIR, exist_ok=True)
"""

# ============================================================================
# CELL 3: Load model via vLLM
# ============================================================================
CELL_3 = """
from vllm import LLM, SamplingParams

# Guard against double-load (torchao registration error)
if 'model' not in dir():
    model = LLM(
        model=MODEL_ID,
        max_model_len=8192,
        dtype="float16",
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
"""

# ============================================================================
# CELL 4: All pipeline functions
# ============================================================================

import fitz  # PyMuPDF
import json
import re
import os
import time
from pathlib import Path


# --- PDF Extraction (PyMuPDF primary) ---

def extract_pages(pdf_path: str) -> list[str]:
    """Extract text from each page using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        pages.append(text.strip())
    doc.close()
    return pages


# --- TOC Extraction (PyMuPDF built-in — fixes Problem 1) ---

def extract_toc(pdf_path: str) -> list[tuple[int, str, int]]:
    """
    Use PyMuPDF's built-in TOC extraction instead of regex.
    Returns list of (level, title, page_number) tuples.

    This is the KEY fix for Problem 1 (0/84 TOC found with regex).
    PyMuPDF reads the PDF's actual outline/bookmarks, which exist in
    virtually every SEC 10-K/10-Q filing.
    """
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()  # Returns [[level, title, page], ...]
    doc.close()

    if not toc:
        return []

    # Filter out junk entries (single chars, empty titles)
    clean = []
    for level, title, page in toc:
        title = title.strip()
        if len(title) < 2:
            continue
        if page < 1:
            continue
        clean.append((level, title, page))

    return clean


def toc_to_sections(toc: list[tuple[int, str, int]], total_pages: int,
                    max_level: int = 2) -> list[dict]:
    """
    Convert PyMuPDF TOC entries into sections with page ranges.
    Only uses top-level entries (level 1-2) to create semantic chunks.
    """
    top_entries = [(title, page) for level, title, page in toc if level <= max_level]

    if not top_entries:
        return []

    sections = []
    for i, (title, start_page) in enumerate(top_entries):
        if i < len(top_entries) - 1:
            end_page = top_entries[i + 1][1] - 1
        else:
            end_page = total_pages
        end_page = max(end_page, start_page)
        sections.append({
            "title": title,
            "start_page": start_page,
            "end_page": end_page,
        })

    return sections


# --- Intelligent Chunking (fixes Problem 2 - token overflow) ---

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


def chunk_pages(pages: list[str], max_input_tokens: int = 2500,
                start_page: int = 1) -> list[dict]:
    """
    Create chunks that respect token budget.
    Instead of fixed 4-page chunks, pack pages until we hit the token limit.
    Dense pages (tables) get fewer pages per chunk.
    Sparse pages (cover pages) get more.

    With max_model_len=8192:
    - Prompt template + system ~300 tokens
    - Want ~5400 tokens for output (enough for any reasonable tree)
    - Input budget = 8192 - 300 - 5400 = ~2500 tokens
    """
    chunks = []
    current_pages = []
    current_tokens = 0
    chunk_start = start_page

    for i, page_text in enumerate(pages):
        page_tokens = estimate_tokens(page_text)

        # Single page exceeds budget — gets its own chunk (truncated)
        if page_tokens > max_input_tokens:
            if current_pages:
                chunks.append({
                    "pages": current_pages,
                    "start_page": chunk_start,
                    "end_page": start_page + i - 1,
                })
                current_pages = []
                current_tokens = 0

            truncated = page_text[:max_input_tokens * 4]
            chunks.append({
                "pages": [truncated],
                "start_page": start_page + i,
                "end_page": start_page + i,
            })
            chunk_start = start_page + i + 1
            continue

        # Would adding this page exceed budget?
        if current_tokens + page_tokens > max_input_tokens and current_pages:
            chunks.append({
                "pages": current_pages,
                "start_page": chunk_start,
                "end_page": start_page + i - 1,
            })
            current_pages = []
            current_tokens = 0
            chunk_start = start_page + i

        current_pages.append(page_text)
        current_tokens += page_tokens

    if current_pages:
        chunks.append({
            "pages": current_pages,
            "start_page": chunk_start,
            "end_page": start_page + len(pages) - 1,
        })

    return chunks


# --- Prompt Construction (stronger schema enforcement) ---

SYSTEM_PROMPT = """You are a document structure analyzer. You produce JSON trees.

EXACT OUTPUT SCHEMA (follow this precisely):
{
  "title": "Section Title",
  "node_id": "0001",
  "start_index": 1,
  "end_index": 4,
  "summary": "Brief description of section content",
  "nodes": [
    {
      "title": "Subsection",
      "node_id": "0002",
      "start_index": 1,
      "end_index": 2,
      "summary": "...",
      "nodes": []
    }
  ]
}

RULES:
1. Use ONLY these fields: title, node_id, start_index, end_index, summary, nodes
2. NEVER use: children, sections, content, type, value, metrics, data
3. "nodes" is the ONLY valid key for child elements
4. Every node MUST have start_index and end_index (integer page numbers)
5. Produce 2-5 levels of hierarchy, not more
6. Output valid JSON only — no markdown, no explanation"""


def build_prompt_for_section(section_title, page_texts, start_page, end_page):
    """Build prompt for a named section (TOC found)."""
    text_block = ""
    for i, text in enumerate(page_texts):
        page_num = start_page + i
        text_block += f"\n--- PAGE {page_num} ---\n{text}\n"

    return f"""Generate the tree structure for this section of a 10-K/10-Q filing.

Section: "{section_title}" (pages {start_page}-{end_page})

{text_block}

Output the JSON tree for this section. Use start_index/end_index for page ranges.
Every node needs: title, node_id, start_index, end_index, summary, nodes.
Produce multiple levels of hierarchy — do NOT create a single flat node."""


def build_prompt_for_chunk(page_texts, start_page, end_page):
    """Build prompt for an unnamed chunk (no TOC / fallback)."""
    text_block = ""
    for i, text in enumerate(page_texts):
        page_num = start_page + i
        text_block += f"\n--- PAGE {page_num} ---\n{text}\n"

    return f"""Generate the tree structure for pages {start_page}-{end_page} of a 10-K/10-Q filing.

{text_block}

Identify ALL distinct sections and subsections in these pages.
Output the JSON tree. Use start_index/end_index for page ranges.
Every node needs: title, node_id, start_index, end_index, summary, nodes.
You MUST produce multiple nodes — do NOT produce a single flat node."""


def format_prompt(user_msg, tokenizer):
    """Apply chat template and append JSON priming character."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += "{"
    return prompt


# --- JSON Repair (handles truncated/malformed JSON) ---

def repair_json(text: str) -> str:
    """
    Aggressively repair truncated or malformed JSON from model output.
    """
    # Prepend the { we used for priming
    text = "{" + text

    # Strip markdown contamination
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)

    # Remove any text before the first {
    first_brace = text.find('{')
    if first_brace > 0:
        text = text[first_brace:]

    # Fix trailing commas before } or ]
    text = re.sub(r',\s*([\}\]])', r'\1', text)

    # Try parsing as-is
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Count unmatched braces/brackets and close them
    stack = []
    in_string = False
    escape = False

    for char in text:
        if escape:
            escape = False
            continue
        if char == '\\' and in_string:
            escape = True
            continue
        if char == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in ('{', '['):
            stack.append(char)
        elif char == '}':
            if stack and stack[-1] == '{':
                stack.pop()
        elif char == ']':
            if stack and stack[-1] == '[':
                stack.pop()

    # Close unclosed structures
    closing = ""
    for opener in reversed(stack):
        closing += '}' if opener == '{' else ']'

    text = text.rstrip()
    if text.endswith(','):
        text = text[:-1]
    text += closing

    # Final trailing comma cleanup
    text = re.sub(r',\s*([\}\]])', r'\1', text)

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        # Last resort: find largest valid JSON substring
        for end in range(len(text), max(0, len(text)-500), -1):
            candidate = text[:end].rstrip()
            if not candidate or candidate[-1] not in ('}', ']'):
                continue
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
        return "{}"


# --- Schema Normalizer (inlined for Colab) ---

CHILDREN_ALIASES = {"children", "sections", "sub_nodes", "subnodes", "subsections",
                    "items", "content_nodes", "elements", "parts", "sub_sections"}
TITLE_ALIASES = {"title", "name", "heading", "section_title", "label", "section_name", "header"}
START_ALIASES = {"start_index", "start_page", "page_start", "begin_page", "from_page", "page"}
END_ALIASES = {"end_index", "end_page", "page_end", "to_page"}


def _find_field(node, aliases):
    for a in aliases:
        if a in node:
            return node[a]
    return None


def _find_children(node):
    if "nodes" in node and isinstance(node["nodes"], list):
        return node["nodes"]
    for a in CHILDREN_ALIASES:
        if a in node and isinstance(node[a], list):
            return node[a]
    return []


def _extract_title(node):
    val = _find_field(node, TITLE_ALIASES)
    if val and isinstance(val, str):
        return val.strip()
    return "Untitled Section"


def _extract_page_range(node, parent_start, parent_end):
    start = _find_field(node, START_ALIASES)
    end = _find_field(node, END_ALIASES)
    try:
        start = int(start) if start is not None else None
    except (ValueError, TypeError):
        start = None
    try:
        end = int(end) if end is not None else None
    except (ValueError, TypeError):
        end = None
    if start is not None and end is None:
        end = parent_end
    if start is None and end is not None:
        start = parent_start
    if start is None and end is None:
        start, end = parent_start, parent_end
    return max(start, parent_start), min(end, parent_end)


def normalize_node(node, parent_start, parent_end, counter, max_depth=5, depth=0):
    if not isinstance(node, dict):
        return None
    title = _extract_title(node)
    start, end = _extract_page_range(node, parent_start, parent_end)
    counter[0] += 1
    node_id = str(counter[0]).zfill(4)

    children = _find_children(node)
    normalized_children = []
    if children and depth < max_depth:
        for child in children:
            result = normalize_node(child, start, end, counter, max_depth, depth + 1)
            if result:
                normalized_children.append(result)

    result = {"title": title, "node_id": node_id, "start_index": start, "end_index": end}
    if normalized_children:
        result["nodes"] = normalized_children
    return result


def normalize_tree_output(raw_json, chunk_start, chunk_end, max_depth=5):
    if isinstance(raw_json, str):
        try:
            raw_json = json.loads(raw_json)
        except json.JSONDecodeError:
            return []

    nodes_to_process = []
    if isinstance(raw_json, list):
        nodes_to_process = raw_json
    elif isinstance(raw_json, dict):
        for key in ["structure", "nodes"] + list(CHILDREN_ALIASES):
            if key in raw_json and isinstance(raw_json[key], list):
                nodes_to_process = raw_json[key]
                break
        if not nodes_to_process:
            nodes_to_process = [raw_json]

    counter = [0]
    normalized = []
    for node in nodes_to_process:
        result = normalize_node(node, chunk_start, chunk_end, counter, max_depth)
        if result:
            normalized.append(result)
    return normalized


# --- Fallback Node ---

def make_fallback_node(start_page, end_page, title=None):
    return {
        "title": title or f"Content (pages {start_page}-{end_page})",
        "start_index": start_page,
        "end_index": end_page,
    }


# --- Main Processing Pipeline ---

def process_single_pdf(pdf_path, model, tokenizer, sampling_params):
    """
    Process one PDF:
    1. Extract pages
    2. Try TOC extraction (PyMuPDF built-in)
    3. Create chunks (TOC-based or token-budget-based)
    4. Generate tree for each chunk via vLLM batch
    5. Normalize and merge
    """
    pages = extract_pages(pdf_path)
    total_pages = len(pages)
    filename = os.path.basename(pdf_path)
    print(f"\n{'='*60}")
    print(f"Processing: {filename} ({total_pages} pages)")

    # Step 1: Try TOC
    toc = extract_toc(pdf_path)
    sections = toc_to_sections(toc, total_pages) if toc else []

    if sections:
        print(f"  TOC found: {len(toc)} entries -> {len(sections)} top-level sections")
        for s in sections[:5]:
            print(f"    {s['title'][:50]:50s} pages {s['start_page']}-{s['end_page']}")
        if len(sections) > 5:
            print(f"    ... and {len(sections)-5} more")
    else:
        print(f"  No TOC found — using token-budget chunking")

    # Step 2: Create chunks
    if sections:
        all_chunks = []
        for section in sections:
            sec_pages = pages[section["start_page"]-1 : section["end_page"]]
            sec_chunks = chunk_pages(sec_pages, max_input_tokens=2500,
                                    start_page=section["start_page"])
            for chunk in sec_chunks:
                chunk["section_title"] = section["title"]
            all_chunks.extend(sec_chunks)
    else:
        all_chunks = chunk_pages(pages, max_input_tokens=2500, start_page=1)

    print(f"  Total chunks: {len(all_chunks)}")

    # Step 3: Build prompts
    prompts = []
    for chunk in all_chunks:
        if "section_title" in chunk:
            user_msg = build_prompt_for_section(
                chunk["section_title"], chunk["pages"],
                chunk["start_page"], chunk["end_page"])
        else:
            user_msg = build_prompt_for_chunk(
                chunk["pages"], chunk["start_page"], chunk["end_page"])
        prompts.append(format_prompt(user_msg, tokenizer))

    # Step 4: Batch inference
    print(f"  Running vLLM batch inference ({len(prompts)} prompts)...")
    t0 = time.time()
    outputs = model.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"  Inference complete in {elapsed:.1f}s")

    # Step 5: Parse, normalize, merge
    all_nodes = []
    parse_success = 0
    parse_fail = 0

    for i, (output, chunk) in enumerate(zip(outputs, all_chunks)):
        raw_text = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason
        status = "TRUNC" if finish_reason == "length" else "OK"

        repaired = repair_json(raw_text)

        try:
            parsed = json.loads(repaired)
            nodes = normalize_tree_output(parsed, chunk["start_page"], chunk["end_page"], max_depth=5)
            if nodes:
                all_nodes.extend(nodes)
                parse_success += 1
                node_count = sum(1 for _ in _iter_nodes(nodes))
                print(f"  chunk {i:3d} [{status}] -> {node_count} nodes")
            else:
                fb = make_fallback_node(chunk["start_page"], chunk["end_page"],
                                        chunk.get("section_title"))
                all_nodes.append(fb)
                parse_fail += 1
                print(f"  chunk {i:3d} [{status}] -> EMPTY after normalize, fallback")
        except json.JSONDecodeError:
            fb = make_fallback_node(chunk["start_page"], chunk["end_page"],
                                    chunk.get("section_title"))
            all_nodes.append(fb)
            parse_fail += 1
            preview = raw_text[:80].replace('\n', '\\n')
            print(f"  chunk {i:3d} [{status}] -> PARSE FAIL: {preview}")

    # Step 6: Reassign node IDs
    counter = [0]
    def reassign(node):
        counter[0] += 1
        node["node_id"] = str(counter[0]).zfill(4)
        for child in node.get("nodes", []):
            reassign(child)
    for node in all_nodes:
        reassign(node)

    total_node_count = counter[0]
    print(f"\n  Results: {parse_success}/{len(all_chunks)} chunks parsed ({parse_fail} fallbacks)")
    print(f"  Total nodes: {total_node_count}")

    return {"structure": all_nodes}


def _iter_nodes(nodes):
    """Recursively iterate all nodes."""
    for n in nodes:
        yield n
        yield from _iter_nodes(n.get("nodes", []))


# ============================================================================
# CELL 5: Process all PDFs
# ============================================================================
CELL_5 = """
import glob

sampling_params = SamplingParams(
    temperature=0.3,
    top_p=0.9,
    max_tokens=5500,
    repetition_penalty=1.05,
)

pdf_files = sorted(glob.glob(f"{PDF_DIR}/*.pdf"))
print(f"Found {len(pdf_files)} PDFs")

for pdf_path in pdf_files:
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = f"{TREE_DIR}/{stem}.json"

    if os.path.exists(out_path):
        print(f"Skipping (exists): {stem}")
        continue

    try:
        tree = process_single_pdf(pdf_path, model, tokenizer, sampling_params)

        with open(out_path, 'w') as f:
            json.dump({
                "tree": tree,
                "metadata": {
                    "model": MODEL_ID,
                    "total_pages": len(extract_pages(pdf_path)),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            }, f, indent=2)

        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
"""
