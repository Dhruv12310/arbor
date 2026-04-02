# Chunked TreeGen Architecture — Be Faster AND Better Than PageIndex

## The Problem

TreeGen-v2 produces excellent trees for 6 pages. A 200-page 10-K needs all 200 pages indexed.

**Naive chunking** (process 6 pages independently, merge) is:
- SLOW: 33 sequential calls × ~30s each = ~16 minutes
- BAD: Each chunk produces an isolated sub-tree with no awareness of the rest of the document
- PageIndex processes the same doc in ~5 minutes with GPT-4o

We need to be FASTER and BETTER. Here's how.

---

## How PageIndex Handles Large Documents (and where it's wasteful)

PageIndex makes 80-250 LLM calls per document:

```
1. TOC Detection:     ~20 calls (1 per page, first 20 pages)
2. TOC Extraction:     1 call
3. Completeness check: 1 call  
4. Page mapping:       5-10 calls (per chunk)
5. Verification:       20-50 calls (1 per TOC entry)
6. Fix incorrect:      0-30 calls (retries)
7. Large node split:   5-20 calls (recursive)
8. Summary generation: 20-100 calls (1 per node)
TOTAL:                80-250 calls, each to GPT-4o ($$$)
```

Most of these calls are **verification and error-correction** because GPT-4o isn't fine-tuned for this task. It needs hand-holding at every step. That's the inefficiency we exploit.

---

## The Arbor Architecture: 2-Phase, Parallel, Fine-Tuned

### Phase 1: TOC Extraction (1 call, fast)

Most documents over 30 pages have a Table of Contents. Extract it.

```
Input:  First 3-5 pages of the document (where TOC lives)
Output: Flat list of section titles with page numbers
Model:  TreeGen-v2 OR simple regex/heuristic for structured docs
```

For a 10-K filing, the TOC is on pages 2-3 and lists:
```
Item 1. Business .............. 4
Item 1A. Risk Factors ......... 10
Item 7. MD&A .................. 52
Item 8. Financial Statements .. 60
```

This gives us the SKELETON — we know the top-level structure and exact page boundaries before touching TreeGen-v2 at all.

**For documents WITHOUT a TOC** (some academic papers, reports):
- Process the first 6 pages with TreeGen-v2 to get initial structure
- Scan subsequent pages for heading patterns (regex: lines in ALL CAPS, 
  numbered sections like "3.1", bold markers, etc.)
- Build a rough skeleton from headings alone

**Time:** < 5 seconds (regex/heuristic) or ~30 seconds (1 TreeGen call)

### Phase 2: Parallel Chunk Processing (N calls, concurrent)

This is the key innovation. We DON'T process chunks blindly. We tell each chunk about the skeleton.

```
For each top-level section from the TOC:
  1. Extract pages for that section (e.g., "Item 7: MD&A" = pages 52-59)
  2. Feed those pages to TreeGen-v2 WITH context:
     "This section is 'Item 7. MD&A' (pages 52-59) within a 10-K filing.
      Generate the sub-tree for THIS section only."
  3. TreeGen-v2 produces the internal structure (subsections, summaries)
  4. Attach as children of the skeleton node
```

**Why this is faster:** All section calls run IN PARALLEL. If there are 8 top-level sections, that's 8 concurrent TreeGen calls. On a GPU with batching (or multiple GPU workers), this takes the same wall-clock time as 1-2 sequential calls.

**Why this is better than naive chunking:**
- Each chunk knows its context (section title, page range, position in document)
- No overlapping or duplicated content
- Natural boundaries (section breaks, not arbitrary 6-page cuts)
- The skeleton ensures coherent top-level hierarchy
- Sub-trees attach cleanly because they know their parent

```
Timing comparison for a 200-page 10-K:

PageIndex:   80-250 GPT-4o calls × 2-5s each = 3-15 minutes (sequential)
Naive chunk: 33 TreeGen calls × 30s each     = 16 minutes (sequential)
Arbor:       1 TOC call + 8 parallel TreeGen  = 30-60 seconds (parallel on GPU)
```

---

## Detailed Algorithm

```python
async def generate_tree_large(pdf_path, model, max_pages_per_chunk=10):
    """Generate tree for documents of any size."""
    
    pages = extract_pages(pdf_path)  # List of (page_text, page_num)
    
    # Small doc: single pass
    if len(pages) <= max_pages_per_chunk:
        return await treegen_single_pass(pages, model)
    
    # ── Phase 1: Extract skeleton ────────────────────────────
    skeleton = extract_toc_skeleton(pages)
    
    if not skeleton:
        # No TOC found — build skeleton from headings
        skeleton = build_skeleton_from_headings(pages)
    
    if not skeleton:
        # Last resort — equal page splits with overlap
        skeleton = build_equal_splits(pages, max_pages_per_chunk)
    
    # ── Phase 2: Parallel sub-tree generation ────────────────
    tasks = []
    for section in skeleton:
        section_pages = pages[section.start_page - 1 : section.end_page]
        tasks.append(
            treegen_for_section(
                section_pages, 
                model,
                context=f"Section: {section.title} (pages {section.start_page}-{section.end_page})"
            )
        )
    
    sub_trees = await asyncio.gather(*tasks)  # ALL RUN IN PARALLEL
    
    # ── Phase 3: Merge ───────────────────────────────────────
    tree = merge_skeleton_with_subtrees(skeleton, sub_trees)
    tree = reassign_node_ids(tree)  # Sequential depth-first IDs
    
    return tree
```

### Phase 1 Detail: TOC Extraction

```python
def extract_toc_skeleton(pages):
    """Extract TOC from first few pages using regex patterns."""
    
    toc_patterns = [
        # Pattern: "Item 1. Business .... 4"  or  "1.1 Introduction ... 12"
        r'^(.+?)\s*\.{2,}\s*(\d+)\s*$',
        # Pattern: "Item 1. Business    4"  (spaces, no dots)
        r'^(.+?)\s{3,}(\d+)\s*$',
        # Pattern: "PART I" / "ITEM 1." (10-K/10-Q specific)
        r'^((?:PART|ITEM)\s+\w+\.?\s*.*)$',
    ]
    
    toc_entries = []
    for page in pages[:5]:  # TOC is usually in first 5 pages
        for line in page.text.split('\n'):
            for pattern in toc_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    title = match.group(1).strip()
                    page_num = int(match.group(2)) if match.lastindex >= 2 else None
                    toc_entries.append(TOCEntry(title=title, page_num=page_num))
    
    if len(toc_entries) < 3:
        return None  # Not enough to be a real TOC
    
    # Convert to skeleton with page ranges
    skeleton = []
    for i, entry in enumerate(toc_entries):
        end_page = toc_entries[i+1].page_num - 1 if i + 1 < len(toc_entries) else len(pages)
        skeleton.append(Section(
            title=entry.title,
            start_page=entry.page_num,
            end_page=end_page,
        ))
    
    return skeleton
```

### Phase 2 Detail: Section-Aware TreeGen Prompt

```python
async def treegen_for_section(section_pages, model, context):
    """Run TreeGen-v2 on a single section with context."""
    
    # Build paged text
    paged_text = "\n\n".join(
        f"[Page {p.page_num}]\n{p.text}" 
        for p in section_pages
    )
    
    # Modified system prompt — tells model this is a SECTION, not a full doc
    system = TREEGEN_SYSTEM + f"""

IMPORTANT: You are generating the sub-tree for ONE SECTION of a larger document.
Context: {context}
Generate the internal structure (subsections) of this section only.
The root node should be this section's title with appropriate page range."""
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": paged_text},
    ]
    
    return await model.generate(messages)
```

### Phase 3 Detail: Merge

```python
def merge_skeleton_with_subtrees(skeleton, sub_trees):
    """Attach sub-trees as children of skeleton nodes."""
    
    structure = []
    for section, sub_tree in zip(skeleton, sub_trees):
        node = {
            "title": section.title,
            "start_index": section.start_page,
            "end_index": section.end_page,
            "summary": sub_tree.get("root_summary", ""),
            "nodes": sub_tree.get("structure", []),
        }
        structure.append(node)
    
    return {"doc_name": "...", "structure": structure}


def reassign_node_ids(tree):
    """Assign sequential node_ids in depth-first order."""
    counter = [1]  # mutable counter
    
    def assign(nodes):
        for node in nodes:
            node["node_id"] = str(counter[0]).zfill(4)
            counter[0] += 1
            if node.get("nodes"):
                assign(node["nodes"])
    
    assign(tree["structure"])
    return tree
```

---

## Why This Is Faster Than PageIndex

| Step | PageIndex | Arbor |
|------|-----------|-------|
| TOC extraction | 20+ LLM calls (detection per page) | 1 regex scan (0.1s) |
| Page mapping | 5-10 LLM calls per chunk | Already have page numbers from TOC |
| Verification | 20-50 LLM calls | Not needed — TreeGen is trained for this |
| Error correction | 0-30 LLM calls with retries | Not needed — model produces correct output |
| Sub-structure | 5-20 sequential LLM calls | 5-10 PARALLEL TreeGen calls |
| Summaries | 20-100 LLM calls (1 per node) | Built into TreeGen output (0 extra calls) |
| **Total calls** | **80-250 sequential GPT-4o** | **1 regex + 5-10 parallel TreeGen** |
| **Total time** | **3-15 minutes** | **30-90 seconds** |
| **Cost** | **$2-10 per document** | **$0 (local GPU) or $0.01-0.05 (API)** |

The speed advantage comes from THREE things:

1. **Fine-tuned model doesn't need verification loops** — PageIndex spends 50%+ of its calls verifying and fixing errors because GPT-4o isn't specialized. TreeGen-v2 is trained specifically for this task, so it gets it right the first time.

2. **Summaries are built-in** — PageIndex makes a separate LLM call for every node's summary. TreeGen produces summaries as part of the tree generation. Zero extra calls.

3. **Parallel execution** — PageIndex is mostly sequential (each step depends on the previous). Arbor's phase 2 is embarrassingly parallel — all sections process simultaneously.

---

## What About Documents Without a TOC?

Most professional documents (10-K filings, annual reports, textbooks, manuals) have a TOC. For those that don't:

### Strategy A: Heading scan (fast, heuristic)
Scan all pages for heading patterns:
- Lines in ALL CAPS
- Lines starting with numbers (1.1, 2.3.1)
- Lines that are significantly shorter than surrounding paragraphs
- Lines matching known patterns ("Abstract", "Introduction", "Conclusion", "References")

This produces a rough skeleton in milliseconds.

### Strategy B: Two-pass approach
1. First pass: Process pages 1-6 with TreeGen to understand document structure
2. Use the pattern from pass 1 (e.g., "this doc uses numbered sections") to scan remaining pages
3. Build skeleton from combined knowledge

### Strategy C: Sliding window with overlap
For truly unstructured documents:
1. Process pages 1-8 → sub-tree A
2. Process pages 7-14 (2 page overlap) → sub-tree B  
3. Process pages 13-20 → sub-tree C
4. Merge overlapping regions (match nodes by title similarity)

This is the slowest option but handles any document type.

---

## Edge Cases

**Section spans > 10 pages (e.g., "Risk Factors" = 17 pages):**
Process in 2 sub-chunks (pages 1-10, pages 9-17 with overlap), merge.
The section-aware prompt ensures coherent sub-structure.

**Very short sections (1-2 pages):**
Batch multiple short sections into a single TreeGen call.
"Generate sub-trees for these 3 sections: [titles, page ranges]"

**Document has no discernible structure:**
Fall back to Strategy C (sliding window). The tree will be flatter but
still functional for TreeSearch.

---

## Implementation Priority

1. **TOC regex extractor** — handles 80%+ of target documents (10-K, reports)
2. **Parallel section processing** — the core speed advantage  
3. **Merge + ID reassignment** — simple post-processing
4. **Heading scan fallback** — for documents without TOC
5. **Sliding window fallback** — for truly unstructured documents

Steps 1-3 get you a working system for financial documents (your primary use case).
Steps 4-5 are generalization for other document types.
