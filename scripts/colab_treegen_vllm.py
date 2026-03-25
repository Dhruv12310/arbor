# Arbor TreeGen Pipeline v2 — vLLM Batched Inference
# =====================================================
# Run on Colab A100. Copy each CELL block into a separate Colab cell.
#
# Key improvements over v1:
#   - doc.get_toc() replaces broken regex TOC extraction (fixes 0/84 TOC rate)
#   - repair_json() closes truncated JSON (fixes ~50% parse failure)
#   - Schema normalizer: children→nodes, start_page→start_index, etc.
#   - Token-budget chunking: dense pages get fewer pages per chunk
#   - Per-chunk debug logging (not just chunk 0)


# ══════════════════════════════════════════════════════════════════
# CELL 1 — Install dependencies
# ══════════════════════════════════════════════════════════════════

!pip install vllm pymupdf transformers -q


# ══════════════════════════════════════════════════════════════════
# CELL 2 — Mount Drive + verify PDFs
# ══════════════════════════════════════════════════════════════════

from google.colab import drive
try:
    drive.mount('/content/drive')
except ValueError:
    import subprocess, os, shutil
    subprocess.run(["fusermount", "-uz", "/content/drive"], capture_output=True)
    if os.path.exists("/content/drive"):
        shutil.rmtree("/content/drive")
    os.makedirs("/content/drive")
    drive.mount('/content/drive')

from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/arbor-training-data")
PDF_DIR    = DRIVE_ROOT / "financebench-pdfs"
OUTPUT_DIR = DRIVE_ROOT / "financebench-trees"
MODEL_ID   = "TStark12310/arbor-treegen-7b-v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pdfs = sorted(PDF_DIR.glob("*.pdf"))
existing = {p.stem for p in OUTPUT_DIR.glob("*.json")}
todo = [p for p in pdfs if p.stem not in existing]
print(f"Found {len(pdfs)} PDFs  |  Done: {len(existing)}  |  Remaining: {len(todo)}")


# ══════════════════════════════════════════════════════════════════
# CELL 3 — Load model (run ONCE only)
# ══════════════════════════════════════════════════════════════════

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

if 'model' not in dir():
    print(f"Loading {MODEL_ID} ...")
    try:
        model = LLM(
            model=MODEL_ID,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
        print("Loaded with bitsandbytes.")
    except Exception as e:
        print(f"bitsandbytes failed ({e}), using float16 ...")
        model = LLM(
            model=MODEL_ID,
            dtype="float16",
            max_model_len=8192,
            gpu_memory_utilization=0.90,
            enforce_eager=True,
            trust_remote_code=True,
        )
        print("Loaded with float16.")
else:
    print("Model already loaded.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

sampling_params = SamplingParams(
    temperature=0.0,       # Greedy — deterministic, best for structured JSON
    max_tokens=5500,       # 8192 - ~2500 input budget = ~5700 available; leave small margin
    repetition_penalty=1.05,
)
print("Ready.")


# ══════════════════════════════════════════════════════════════════
# CELL 4 — Pipeline functions
# ══════════════════════════════════════════════════════════════════

import json
import re
import time
from pathlib import Path


# ── PDF extraction ──────────────────────────────────────────────────────────

def extract_pages(pdf_path):
    """Extract per-page text using PyMuPDF."""
    import pymupdf
    doc = pymupdf.open(str(pdf_path))
    pages = [{"page_num": i + 1, "text": (p.get_text() or "").strip()}
             for i, p in enumerate(doc)]
    doc.close()
    return pages


# ── TOC extraction (KEY FIX: PyMuPDF built-in, not regex) ──────────────────

def extract_toc(pdf_path, total_pages):
    """
    Read PDF bookmarks/outline via doc.get_toc().
    Returns list of Section dicts with title, start_page, end_page.

    This fixes the 0/84 TOC hit rate — PyMuPDF reads the PDF's actual
    outline metadata, not visual dot-leader text that PyMuPDF strips.
    Every SEC 10-K/10-Q filing has these bookmarks embedded.
    """
    import pymupdf
    doc = pymupdf.open(str(pdf_path))
    toc = doc.get_toc()   # [[level, title, page_number], ...]
    doc.close()

    if not toc:
        return []

    # Keep only top 2 levels (PART/ITEM level), filter junk
    top = [(title.strip(), page)
           for level, title, page in toc
           if level <= 2 and len(title.strip()) >= 3 and page >= 1]

    if len(top) < 3:
        return []

    # Reject TOCs where all entries cluster on pages 1-5 —
    # this happens when PDF bookmarks point to the TOC page itself, not section pages.
    unique_pages = {page for _, page in top}
    if max(unique_pages) <= 5:
        print(f"  TOC rejected — all {len(top)} bookmarks on pages 1-5 (point to TOC, not content)")
        return []

    # Deduplicate (same title at same page)
    seen, unique = set(), []
    for title, page in top:
        key = (title.lower(), page)
        if key not in seen:
            seen.add(key)
            unique.append((title, page))

    # Build page ranges
    sections = []
    for i, (title, start) in enumerate(unique):
        end = unique[i + 1][1] - 1 if i + 1 < len(unique) else total_pages
        end = max(start, min(end, total_pages))
        sections.append({"title": title, "start_page": start, "end_page": end})

    print(f"  TOC: {len(sections)} sections from {len(toc)} bookmarks")
    for s in sections[:8]:
        print(f"    [{s['start_page']:>4}-{s['end_page']:<4}] {s['title'][:60]}")
    if len(sections) > 8:
        print(f"    ... and {len(sections) - 8} more")
    return sections


# ── Token-budget chunking ───────────────────────────────────────────────────

def estimate_tokens(text):
    """~4 chars per token for English text."""
    return len(text) // 4


def chunk_section(pages, start_page, max_input_tokens=2500):
    """
    Split pages into chunks that respect the token budget.
    Dense pages (tables) → fewer pages per chunk.
    Sparse pages → more pages per chunk.
    Budget: 8192 total - ~300 prompt overhead - 5500 max_tokens = ~2392 input.
    """
    chunks = []
    current, current_tokens, chunk_start = [], 0, start_page

    for i, page in enumerate(pages):
        page_num = start_page + i
        tokens = estimate_tokens(page["text"])

        if tokens > max_input_tokens:
            # Single oversized page — flush pending and give it its own chunk
            if current:
                chunks.append({"pages": current, "start_page": chunk_start,
                                "end_page": page_num - 1})
                current, current_tokens = [], 0
            chunks.append({"pages": [{"page_num": page_num,
                                       "text": page["text"][:max_input_tokens * 4]}],
                            "start_page": page_num, "end_page": page_num})
            chunk_start = page_num + 1
            continue

        if current_tokens + tokens > max_input_tokens and current:
            chunks.append({"pages": current, "start_page": chunk_start,
                            "end_page": page_num - 1})
            current, current_tokens, chunk_start = [], 0, page_num

        current.append(page)
        current_tokens += tokens

    if current:
        chunks.append({"pages": current, "start_page": chunk_start,
                        "end_page": start_page + len(pages) - 1})
    return chunks


# ── Prompt builder ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a document structure analyzer. Output JSON trees only.

EXACT SCHEMA — use ONLY these fields:
{
  "title": "Section Title",
  "node_id": "0001",
  "start_index": 1,
  "end_index": 4,
  "summary": "Brief description of this section's content.",
  "nodes": [
    {"title": "Subsection", "node_id": "0002", "start_index": 1,
     "end_index": 2, "summary": "...", "nodes": []}
  ]
}

RULES:
1. Use ONLY: title, node_id, start_index, end_index, summary, nodes
2. NEVER use: children, sections, content, type, value, metrics, data
3. "nodes" is the ONLY valid key for children — never "children"
4. Every node MUST have start_index and end_index as integers
5. Produce 2-4 levels of hierarchy
6. Output JSON only — no markdown, no explanations, no prose"""


def build_prompt(pages, start_page, end_page, section_title=None):
    text_block = "\n\n".join(
        f"[Page {p['page_num']}]\n{p['text']}" for p in pages if p["text"]
    )
    if section_title:
        user = (f'Generate the JSON tree for section: "{section_title}" '
                f'(pages {start_page}-{end_page}).\n\n{text_block}\n\n'
                f'Output the hierarchical structure with multiple levels. '
                f'Do NOT produce a single flat node.')
    else:
        user = (f'Generate the JSON tree for pages {start_page}-{end_page} '
                f'of a 10-K/10-Q filing.\n\n{text_block}\n\n'
                f'Identify ALL distinct sections and subsections. '
                f'You MUST produce multiple nodes with hierarchy.')

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
    # Append "{" to force JSON output (prevents Markdown)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True) + "{"


# ── JSON repair ─────────────────────────────────────────────────────────────

def repair_json(raw_text):
    """
    Repair truncated/malformed JSON from model output.
    The model output is the rest of the JSON after our "{" primer.
    """
    text = ("{" + raw_text.strip())

    # Strip markdown fences
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)

    # Find outermost JSON object
    start = text.find("{")
    if start > 0:
        text = text[start:]

    # Fix trailing commas
    text = re.sub(r',\s*([\}\]])', r'\1', text)

    # Try as-is first
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Close unclosed brackets/braces
    stack, in_string, escape = [], False, False
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
        elif char == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif char == ']' and stack and stack[-1] == '[':
            stack.pop()

    text = text.rstrip().rstrip(',')
    for opener in reversed(stack):
        text += '}' if opener == '{' else ']'

    text = re.sub(r',\s*([\}\]])', r'\1', text)

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Last resort: find largest valid JSON suffix
    for end in range(len(text), max(0, len(text) - 500), -1):
        candidate = text[:end].rstrip()
        if candidate and candidate[-1] in ('}', ']'):
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
    return "{}"


# ── Schema normalizer ───────────────────────────────────────────────────────

_CHILDREN_ALIASES = {"children", "sections", "sub_nodes", "subnodes",
                     "subsections", "items", "elements", "parts"}
_TITLE_ALIASES    = {"title", "name", "heading", "section_title",
                     "label", "section_name", "header"}
_START_ALIASES    = {"start_index", "start_page", "page_start",
                     "begin_page", "from_page", "page"}
_END_ALIASES      = {"end_index", "end_page", "page_end", "to_page"}
_SUMMARY_ALIASES  = {"summary", "description", "overview", "detail"}


def _find(node, aliases):
    for a in aliases:
        if a in node:
            return node[a]
    return None


def _get_children(node):
    if "nodes" in node and isinstance(node["nodes"], list):
        return node["nodes"]
    for a in _CHILDREN_ALIASES:
        if a in node and isinstance(node[a], list):
            return node[a]
    return []


def _get_page_range(node, p_start, p_end):
    start = _find(node, _START_ALIASES)
    end   = _find(node, _END_ALIASES)
    try: start = int(start)
    except (TypeError, ValueError): start = None
    try: end = int(end)
    except (TypeError, ValueError): end = None
    if start is None: start = p_start
    if end is None:   end = p_end
    return max(start, p_start), min(end, p_end)


def _is_placeholder(text):
    """Detect $X, $XX style placeholders only — not real financial figures."""
    if not text:
        return False
    return bool(re.search(r'\$X+\b|\[placeholder\]|TBD|Lorem ipsum', text, re.IGNORECASE))


def _normalize_node(node, p_start, p_end, counter, max_depth=5, depth=0):
    if not isinstance(node, dict):
        return None
    title = (_find(node, _TITLE_ALIASES) or "Untitled Section")
    if isinstance(title, str):
        title = title.strip()
    if _is_placeholder(title):
        return None

    start, end = _get_page_range(node, p_start, p_end)
    if start > end:
        end = start

    counter[0] += 1
    node_id = str(counter[0]).zfill(4)

    summary = _find(node, _SUMMARY_ALIASES) or ""
    if isinstance(summary, str):
        summary = summary.strip()
    else:
        summary = ""

    children = _get_children(node)
    norm_children = []
    if children and depth < max_depth:
        for child in children:
            nc = _normalize_node(child, start, end, counter, max_depth, depth + 1)
            if nc:
                norm_children.append(nc)
        # If all children inherited same range, distribute evenly
        if (norm_children and len(norm_children) > 1
                and all(c["start_index"] == norm_children[0]["start_index"]
                        and c["end_index"] == norm_children[0]["end_index"]
                        for c in norm_children)):
            span = end - start + 1
            ppn = max(1, span // len(norm_children))
            for i, c in enumerate(norm_children):
                c["start_index"] = start + i * ppn
                c["end_index"] = (start + (i + 1) * ppn - 1
                                  if i < len(norm_children) - 1 else end)
                # Ensure end >= start after distribution
                c["end_index"] = max(c["end_index"], c["start_index"])

    result = {"title": title, "node_id": node_id,
              "start_index": start, "end_index": end}
    if summary:
        result["summary"] = summary
    if norm_children:
        result["nodes"] = norm_children
    return result


def normalize_output(raw_json, chunk_start, chunk_end, counter, max_depth=5):
    """Normalize raw model JSON into canonical TreeSearch schema."""
    nodes_to_process = []

    if isinstance(raw_json, list):
        nodes_to_process = raw_json
    elif isinstance(raw_json, dict):
        for key in ["structure", "nodes"] + list(_CHILDREN_ALIASES):
            if key in raw_json and isinstance(raw_json[key], list):
                nodes_to_process = raw_json[key]
                break
        if not nodes_to_process:
            nodes_to_process = [raw_json]

    result = []
    for node in nodes_to_process:
        n = _normalize_node(node, chunk_start, chunk_end, counter, max_depth)
        if n:
            result.append(n)
    return result


def count_nodes(nodes):
    return sum(1 + count_nodes(n.get("nodes", [])) for n in nodes)


print("Pipeline functions loaded.")


# ══════════════════════════════════════════════════════════════════
# CELL 5 — Process all PDFs
# ══════════════════════════════════════════════════════════════════

from pathlib import Path
import json, time

DRIVE_ROOT = Path("/content/drive/MyDrive/arbor-training-data")
PDF_DIR    = DRIVE_ROOT / "financebench-pdfs"
OUTPUT_DIR = DRIVE_ROOT / "financebench-trees"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pdfs = sorted(PDF_DIR.glob("*.pdf"))
existing = {p.stem for p in OUTPUT_DIR.glob("*.json")}
todo = [p for p in pdfs if p.stem not in existing]
print(f"Total: {len(pdfs)}  |  Done: {len(existing)}  |  Remaining: {len(todo)}\n")

success, errors = 0, 0
total_t0 = time.time()

for i, pdf_path in enumerate(todo, 1):
    stem = pdf_path.stem
    print(f"[{i}/{len(todo)}] {stem}")
    out_file = OUTPUT_DIR / f"{stem}.json"
    t0 = time.time()

    try:
        pages = extract_pages(pdf_path)
        total_pages = len(pages)
        print(f"  {total_pages} pages")

        # Phase 1: TOC
        sections = extract_toc(pdf_path, total_pages)

        # Phase 2: Chunking
        if sections:
            all_chunks = []
            for sec in sections:
                sec_pages = [p for p in pages
                             if sec["start_page"] <= p["page_num"] <= sec["end_page"]]
                for chunk in chunk_section(sec_pages, sec["start_page"]):
                    chunk["section_title"] = sec["title"]
                    all_chunks.append(chunk)
        else:
            print("  No TOC — token-budget equal splits")
            all_chunks = chunk_section(pages, start_page=1)

        print(f"  {len(all_chunks)} chunks")

        # Phase 3: Build prompts (with JSON priming)
        prompts = [
            build_prompt(
                chunk["pages"], chunk["start_page"], chunk["end_page"],
                chunk.get("section_title")
            )
            for chunk in all_chunks
        ]

        # Phase 4: Batch vLLM inference
        outputs = model.generate(prompts, sampling_params)

        # Phase 5: Parse, normalize, merge
        all_nodes = []
        counter = [0]
        parse_ok, parse_fail = 0, 0

        for j, (output, chunk) in enumerate(zip(outputs, all_chunks)):
            raw = output.outputs[0].text
            finish = output.outputs[0].finish_reason
            truncated = (finish == "length")

            repaired = repair_json(raw)
            try:
                parsed = json.loads(repaired)
                nodes = normalize_output(
                    parsed, chunk["start_page"], chunk["end_page"], counter
                )
                if nodes:
                    all_nodes.extend(nodes)
                    parse_ok += 1
                    print(f"  chunk {j:3d} {'[TRUNC]' if truncated else '[OK]   '}"
                          f" -> {count_nodes(nodes)} nodes"
                          f"  {chunk.get('section_title', '')[:40]}")
                else:
                    raise ValueError("empty after normalize")
            except Exception:
                # Fallback node
                counter[0] += 1
                fb = {"title": chunk.get("section_title") or
                               f"Pages {chunk['start_page']}-{chunk['end_page']}",
                      "node_id": str(counter[0]).zfill(4),
                      "start_index": chunk["start_page"],
                      "end_index": chunk["end_page"]}
                all_nodes.append(fb)
                parse_fail += 1
                print(f"  chunk {j:3d} {'[TRUNC]' if truncated else '[FAIL] '}"
                      f" -> fallback | first 80: {repr(raw[:80])}")

        # Phase 6: Reassign IDs sequentially across whole doc
        final_counter = [0]
        def reassign(node):
            final_counter[0] += 1
            node["node_id"] = str(final_counter[0]).zfill(4)
            for child in node.get("nodes", []):
                reassign(child)
        for node in all_nodes:
            reassign(node)

        tree = {"doc_name": stem, "total_pages": total_pages, "structure": all_nodes}
        out_file.write_text(
            json.dumps({"tree": tree}, indent=2), encoding="utf-8"
        )

        elapsed = time.time() - t0
        total_nodes = count_nodes(all_nodes)
        print(f"\n  DONE: {total_nodes} nodes | {parse_ok}/{len(all_chunks)} parsed"
              f" | {elapsed:.0f}s")
        success += 1

    except Exception as e:
        import traceback
        print(f"  [ERROR] {e}")
        traceback.print_exc()
        errors += 1

    avg = (time.time() - total_t0) / i
    print(f"  Running: {success} ok, {errors} err | ETA {avg*(len(todo)-i)/60:.1f} min\n")

print(f"Done! {success} trees saved. Total: {(time.time()-total_t0)/60:.1f} min")
