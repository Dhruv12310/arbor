# multidomaintest.py — Multi-Domain PDF Retrieval Test
# ======================================================
# Tests v13 TreeSearch + StructDirect across 8 categories:
#   - 7 domain categories × 5 arXiv PDFs = 35 PDFs
#   - 5 genuinely failed FinanceBench PDFs (0% accuracy on prior eval)
# Total: 40 PDFs × 3 questions = 120 questions
#
# Mode B: Claude generates questions → v13 navigates → Claude judges answer
#
# For each FAIL the script reports:
#   - Extraction strategy used (embedded_toc / multiline_toc / font_headings / regex_toc / page_chunks)
#   - Failure type: NAVIGATION_ERROR vs EXTRACTION_GAP vs BAD_QUESTION
#   - Diagnosis: which node should have been picked (navigation) or what's missing (extraction)
#
# Run CELL 1 → 2 → 3 → 4 → 5 → 6 → 7 in order.


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 1 — Install dependencies                          ║
# ╚══════════════════════════════════════════════════════════╝
"""
!pip install -q "unsloth[colab-new]" pymupdf trl transformers accelerate bitsandbytes peft anthropic
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2 — Mount Drive + Clone Repo + Paths              ║
# ╚══════════════════════════════════════════════════════════╝
"""
from google.colab import drive
drive.mount('/content/drive')

import subprocess, sys, os

# Clone or update repo
if not os.path.exists("/content/arbor"):
    subprocess.run(["git", "clone", "https://github.com/Dhruv12310/arbor.git", "/content/arbor"], check=True)
else:
    subprocess.run(["git", "-C", "/content/arbor", "pull"], check=True)
sys.path.insert(0, "/content/arbor")

DRIVE_DIR       = "/content/drive/MyDrive/arbor-training-data"
MODEL_DIR       = f"{DRIVE_DIR}/models/treesearch-v13"
FINANCE_PDF_DIR = f"{DRIVE_DIR}/financebench-pdfs"
DOMAIN_PDF_DIR  = "/content/domain_pdfs"          # arXiv PDFs downloaded here
TREE_CACHE_DIR  = f"{DRIVE_DIR}/multidomaintest_trees"
QUESTIONS_CACHE = f"{DRIVE_DIR}/multidomaintest_questions.json"
RESULTS_FILE    = f"{DRIVE_DIR}/multidomaintest_results.json"

os.makedirs(DOMAIN_PDF_DIR, exist_ok=True)
os.makedirs(TREE_CACHE_DIR, exist_ok=True)

# ── Anthropic API key ─────────────────────────────────────────────────────────
# Set your key here OR use: import os; os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
ANTHROPIC_API_KEY = ""   # ← PASTE YOUR KEY HERE
if not ANTHROPIC_API_KEY:
    raise ValueError("Set ANTHROPIC_API_KEY above before running this cell")

import anthropic as _anthropic_module
claude = _anthropic_module.Anthropic(api_key=ANTHROPIC_API_KEY)
print("Drive mounted, repo cloned, Claude client ready")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 3 — Define all 40 PDFs                            ║
# ╚══════════════════════════════════════════════════════════╝
"""
# ── Domain PDFs: first 5 arXiv papers per category ───────────────────────────
# These are arXiv IDs — we download them directly from arxiv.org

DOMAIN_PDFS = {
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

# ── FinanceBench PDFs: 5 docs with genuine 0% accuracy ───────────────────────
# Questions sourced from financebench_open_source.jsonl
FINANCEBENCH_DOCS = [
    "PEPSICO_2022_10K",
    "CVSHEALTH_2022_10K",
    "VERIZON_2022_10K",
    "AES_2022_10K",
    "PFIZER_2021_10K",
]

# Flat list of all PDFs
ALL_PDFS = []   # [{id, domain, pdf_path, source}]

for domain, ids in DOMAIN_PDFS.items():
    for arxiv_id in ids:
        ALL_PDFS.append({
            "id":       arxiv_id,
            "domain":   domain,
            "pdf_path": f"{DOMAIN_PDF_DIR}/{arxiv_id}.pdf",
            "source":   "arxiv",
        })

for doc in FINANCEBENCH_DOCS:
    ALL_PDFS.append({
        "id":       doc,
        "domain":   "financebench",
        "pdf_path": f"{FINANCE_PDF_DIR}/{doc}.pdf",
        "source":   "financebench",
    })

print(f"Total PDFs: {len(ALL_PDFS)}")
print(f"  arXiv domain PDFs : {sum(1 for p in ALL_PDFS if p['source'] == 'arxiv')}")
print(f"  FinanceBench PDFs : {sum(1 for p in ALL_PDFS if p['source'] == 'financebench')}")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 4 — Download arXiv PDFs + Load v13 model          ║
# ╚══════════════════════════════════════════════════════════╝
"""
import requests, time, torch, warnings
warnings.filterwarnings("ignore", message="Both `max_new_tokens`.*")

# ── Download arXiv PDFs (skip if already cached) ──────────────────────────────
print("Downloading arXiv PDFs...")
download_errors = []
for entry in ALL_PDFS:
    if entry["source"] != "arxiv":
        continue
    path = entry["pdf_path"]
    if os.path.exists(path):
        continue
    url = f"https://arxiv.org/pdf/{entry['id']}"
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"  Downloaded: {entry['id']} ({len(r.content)//1024}KB)")
        time.sleep(0.5)   # polite rate limit for arXiv
    except Exception as e:
        print(f"  FAILED: {entry['id']}: {e}")
        download_errors.append(entry["id"])

print(f"\nDownload complete. Errors: {len(download_errors)}")

# ── Check FinanceBench PDFs exist ─────────────────────────────────────────────
missing_finance = [e["id"] for e in ALL_PDFS
                   if e["source"] == "financebench" and not os.path.exists(e["pdf_path"])]
if missing_finance:
    print(f"\nWARNING: Missing FinanceBench PDFs: {missing_finance}")
    print("These should be at Drive/arbor-training-data/financebench-pdfs/")

# ── Load v13 model ────────────────────────────────────────────────────────────
from unsloth import FastLanguageModel
from arbor.providers.base import LLMProvider

MAX_SEQ_LEN = 4096
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_DIR,
    max_seq_length = MAX_SEQ_LEN,
    dtype          = None,
    load_in_4bit   = True,
)
FastLanguageModel.for_inference(model)

class V13Provider(LLMProvider):
    @property
    def name(self): return "treesearch-v13"

    async def complete(self, prompt, temperature=0.0, max_tokens=None, chat_history=None):
        result, _ = await self.complete_with_finish_reason(prompt, temperature, max_tokens, chat_history)
        return result

    async def complete_with_finish_reason(self, prompt, temperature=0.0, max_tokens=None, chat_history=None):
        messages = list(chat_history or []) + [{"role": "user", "content": prompt}]
        enc = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = enc.input_ids if hasattr(enc, "input_ids") else enc
        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            output = model.generate(
                input_ids          = input_ids,
                max_new_tokens     = max_tokens or 256,
                do_sample          = False,
                repetition_penalty = 1.1,
            )
        new_tokens = output[0][input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        finish = "length" if len(new_tokens) >= (max_tokens or 256) else "stop"
        return text, finish

provider = V13Provider()
print("v13 model ready")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 5 — Extract structures (cached to Drive)          ║
# ╚══════════════════════════════════════════════════════════╝
"""
import json, time as _time
from arbor.extraction.structure_extractor import extract_structure
from arbor.core.types import DocumentTree

tree_cache = {}    # id -> DocumentTree
skipped    = []

for entry in ALL_PDFS:
    pdf_id    = entry["id"]
    pdf_path  = entry["pdf_path"]
    cache_path = f"{TREE_CACHE_DIR}/{pdf_id}.json"

    if not os.path.exists(pdf_path):
        print(f"SKIP (no PDF): {pdf_id}")
        skipped.append(pdf_id)
        continue

    # Load from cache
    if os.path.exists(cache_path):
        raw = json.loads(open(cache_path, encoding="utf-8").read())
        tree_cache[pdf_id] = DocumentTree.from_dict(raw)
        strat = tree_cache[pdf_id].extraction_strategy or "unknown"
        print(f"CACHED [{strat:25s}] {pdf_id}")
        continue

    # Extract fresh
    try:
        t0   = _time.monotonic()
        tree = extract_structure(pdf_path)
        ms   = (_time.monotonic() - t0) * 1000

        def _count(ns): return sum(1 + _count(n.nodes) for n in ns)
        n_sec = _count(tree.structure)

        tree_cache[pdf_id] = tree
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(tree.to_dict(), f, ensure_ascii=False, indent=2)

        strat = tree.extraction_strategy or "unknown"
        print(f"OK    [{strat:25s}] {pdf_id}: {n_sec} sections, {ms:.0f}ms")
    except Exception as e:
        print(f"ERROR {pdf_id}: {e}")
        skipped.append(pdf_id)

print(f"\nExtracted: {len(tree_cache)} trees | Skipped: {len(skipped)}")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 6 — Generate questions per PDF (cached)           ║
# ╚══════════════════════════════════════════════════════════╝
"""
import time, pymupdf

def extract_first_pages_text(pdf_path: str, n_pages: int = 3) -> str:
    doc = pymupdf.open(pdf_path)
    text = ""
    for i in range(min(n_pages, len(doc))):
        text += doc[i].get_text()
    doc.close()
    return text[:4000]   # cap at 4000 chars so prompt stays small

def generate_questions(pdf_id: str, domain: str, pdf_path: str) -> list[dict]:
    """
    Ask Claude to generate 3 factual questions answerable from this PDF.
    Returns [{question, expected_topic}, ...]
    """
    text = extract_first_pages_text(pdf_path, n_pages=4)
    prompt = f"""You are creating a retrieval test for a document.

Domain: {domain}
Document ID: {pdf_id}
First pages text:
---
{text}
---

Generate exactly 3 factual questions whose answers are DEFINITELY present somewhere in this document (not just the first pages). Each question should:
1. Require finding a specific number, name, date, or claim from the document
2. Relate to different topics/sections so they test different parts of the document
3. Be answerable with 1-3 sentences

Return ONLY valid JSON, no preamble:
[
  {{"question": "...", "expected_section": "brief description of where in the doc the answer likely is"}},
  {{"question": "...", "expected_section": "..."}},
  {{"question": "...", "expected_section": "..."}}
]"""

    resp = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = resp.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

# Load or generate question cache
if os.path.exists(QUESTIONS_CACHE):
    question_cache = json.loads(open(QUESTIONS_CACHE, encoding="utf-8").read())
    print(f"Loaded {len(question_cache)} cached question sets")
else:
    question_cache = {}

changed = False
for entry in ALL_PDFS:
    pdf_id = entry["id"]
    if pdf_id not in tree_cache:
        continue   # skip PDFs that failed extraction
    if pdf_id in question_cache:
        print(f"  CACHED {pdf_id}")
        continue

    # For FinanceBench: use existing questions from the dataset
    if entry["source"] == "financebench":
        fb_questions = []
        fb_qa_path = "/content/arbor/data/financebench/financebench_open_source.jsonl"
        all_qa = [json.loads(l) for l in open(fb_qa_path, encoding="utf-8").read().strip().splitlines()]
        doc_qa = [q for q in all_qa if q["doc_name"] == pdf_id]
        for qa in doc_qa[:3]:   # take up to 3 questions
            ev_pages = [e["evidence_page_num"] for e in qa.get("evidence", [])]
            fb_questions.append({
                "question":         qa["question"],
                "expected_section": f"evidence pages: {ev_pages}",
                "evidence_pages":   ev_pages,
                "financebench_id":  qa["financebench_id"],
            })
        question_cache[pdf_id] = fb_questions
        changed = True
        print(f"  FB    {pdf_id}: {len(fb_questions)} questions")
        continue

    # For arXiv PDFs: generate with Claude
    try:
        qs = generate_questions(pdf_id, entry["domain"], entry["pdf_path"])
        for q in qs:
            q["evidence_pages"] = []   # unknown for arXiv
        question_cache[pdf_id] = qs
        changed = True
        print(f"  GEN   {pdf_id}: {len(qs)} questions")
        time.sleep(0.3)   # gentle rate limit
    except Exception as e:
        print(f"  ERROR {pdf_id}: {e}")
        question_cache[pdf_id] = []

if changed:
    with open(QUESTIONS_CACHE, "w", encoding="utf-8") as f:
        json.dump(question_cache, f, ensure_ascii=False, indent=2)
    print(f"\nSaved question cache → {QUESTIONS_CACHE}")

total_q = sum(len(v) for v in question_cache.values())
print(f"\nTotal questions: {total_q}")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 7 — Run v13 navigation + Claude judgment          ║
# ╚══════════════════════════════════════════════════════════╝
"""
import asyncio, pymupdf
from arbor.core.tree_searcher import search_tree
from arbor.core.types import ArborConfig

config = ArborConfig(
    max_hops                = 8,
    max_nodes_searched      = 0,
    timeout_sec             = 180.0,
    max_retries_on_bad_json = 2,
)

def get_node_text(pdf_path: str, start_page: int, end_page: int, max_chars: int = 3000) -> str:
    """Extract raw text from pages start_page..end_page (1-indexed, inclusive)."""
    try:
        doc = pymupdf.open(pdf_path)
        text = ""
        for pg in range(start_page - 1, min(end_page, len(doc))):
            text += doc[pg].get_text()
            if len(text) >= max_chars:
                break
        doc.close()
        return text[:max_chars]
    except Exception as e:
        return f"[TEXT EXTRACTION ERROR: {e}]"

def get_all_node_titles(nodes, depth=0) -> list[str]:
    """Flat list of 'indent node_id: title (pages X-Y)' for all nodes."""
    result = []
    for n in nodes:
        prefix = "  " * depth
        result.append(f"{prefix}[{n.node_id}] {n.title} (pages {n.start_index}-{n.end_index})")
        result.extend(get_all_node_titles(n.nodes, depth + 1))
    return result

def check_extraction_coverage(tree, target_pages: list[int]) -> dict:
    """For each target page, find which node covers it (or none)."""
    def find_node_for_page(nodes, page):
        for n in nodes:
            if n.start_index <= page <= n.end_index:
                sub = find_node_for_page(n.nodes, page)
                return sub if sub else n
        return None

    coverage = {}
    for pg in target_pages:
        node = find_node_for_page(tree.structure, pg)
        coverage[pg] = node.node_id if node else None
    return coverage

def judge_answer(question: str, retrieved_text: str, pdf_id: str, domain: str) -> dict:
    """
    Two-pass Claude judge (more reliable than combined reasoning+verdict call).
    Pass 1: strict YES/NO verdict (max_tokens=5, no hedging possible).
    Pass 2: diagnosis only on NO — one sentence on what section would have the answer.
    Returns {answered: bool, confidence: str, reasoning: str, correct_section_hint: str}
    """
    # Pass 1 — strict verdict only
    verdict_prompt = (
        f"PDF: {pdf_id} (domain: {domain})\n"
        f"Question: {question}\n\n"
        f"Retrieved text:\n---\n{retrieved_text[:2500]}\n---\n\n"
        f"Does this text DIRECTLY answer the question? Reply with exactly: YES or NO"
    )
    resp1 = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5,
        messages=[{"role": "user", "content": verdict_prompt}]
    )
    verdict = resp1.content[0].text.strip().upper()
    answered = verdict.startswith("YES")

    if answered:
        return {
            "answered": True,
            "confidence": "high",
            "reasoning": "Text directly answers the question.",
            "correct_section_hint": "",
        }

    # Pass 2 — diagnosis only (runs on FAIL cases)
    diag_prompt = (
        f"PDF: {pdf_id} (domain: {domain})\n"
        f"Question: {question}\n\n"
        f"The retrieved text did NOT answer this question. "
        f"In one sentence, what topic or section of the document would contain the answer?"
    )
    resp2 = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        messages=[{"role": "user", "content": diag_prompt}]
    )
    hint = resp2.content[0].text.strip()

    return {
        "answered": False,
        "confidence": "high",
        "reasoning": hint,
        "correct_section_hint": hint,
    }

def diagnose_failure(pdf_path: str, tree, question: str,
                     retrieved_nodes, correct_section_hint: str,
                     evidence_pages: list[int]) -> str:
    """
    Determine failure type:
    - EXTRACTION_GAP: evidence pages not covered by any node in tree
    - NAVIGATION_ERROR: pages exist in tree, model navigated to wrong section
    - BAD_QUESTION: question too vague or requires external context
    """
    # Check evidence pages (FinanceBench only)
    if evidence_pages:
        coverage = check_extraction_coverage(tree, evidence_pages)
        uncovered = [pg for pg, nid in coverage.items() if nid is None]
        if uncovered:
            return f"EXTRACTION_GAP (pages {uncovered} not in any tree node)"
        # Pages exist in tree but model didn't find them
        covered_nodes = [coverage[pg] for pg in evidence_pages if coverage[pg]]
        retrieved_ids = {n.node_id for n in retrieved_nodes}
        if not any(nid in retrieved_ids for nid in covered_nodes):
            return f"NAVIGATION_ERROR (evidence in node(s) {covered_nodes}, model retrieved {[n.node_id for n in retrieved_nodes]})"
        return "PARTIAL_MISS (retrieved section contains evidence page but text wasn't surfaced)"

    # arXiv PDFs: use the correct_section_hint from Claude
    if "vague" in correct_section_hint.lower() or "external" in correct_section_hint.lower():
        return "BAD_QUESTION"
    if correct_section_hint:
        return f"NAVIGATION_ERROR (model retrieved wrong section; answer likely in: {correct_section_hint[:80]})"
    return "UNKNOWN_FAILURE"


async def run_full_eval():
    results = []
    q_num   = 0
    total_q = sum(len(v) for v in question_cache.values() if v)

    for entry in ALL_PDFS:
        pdf_id  = entry["id"]
        domain  = entry["domain"]
        source  = entry["source"]

        if pdf_id not in tree_cache:
            print(f"SKIP (no tree): {pdf_id}")
            continue
        if pdf_id not in question_cache or not question_cache[pdf_id]:
            print(f"SKIP (no questions): {pdf_id}")
            continue

        tree     = tree_cache[pdf_id]
        strategy = tree.extraction_strategy or "unknown"

        for qa in question_cache[pdf_id]:
            q_num += 1
            question       = qa["question"]
            evidence_pages = qa.get("evidence_pages", [])

            result = {
                "pdf_id":         pdf_id,
                "domain":         domain,
                "source":         source,
                "strategy":       strategy,
                "question":       question,
                "evidence_pages": evidence_pages,
                "status":         "ERROR",
                "failure_type":   None,
                "retrieved_nodes": [],
                "node_titles":    [],
                "claude_judgment": None,
                "diagnosis":      None,
                "error":          None,
            }

            print(f"\n[{q_num:3d}/{total_q}] {pdf_id[:40]:40s} [{strategy:18s}]")
            print(f"         Q: {question[:90]}")

            try:
                # ── Navigation ───────────────────────────────────────────────
                sr = await search_tree(tree, question, provider, multihop=True, config=config)

                retrieved_nodes = sr.nodes
                result["retrieved_nodes"] = sr.node_ids
                result["node_titles"] = [
                    f"[{n.node_id}] {n.title} (p{n.start_index}-{n.end_index})"
                    for n in retrieved_nodes
                ]

                # ── Extract text from retrieved pages ─────────────────────────
                all_text = ""
                for node in retrieved_nodes:
                    all_text += get_node_text(entry["pdf_path"], node.start_index, node.end_index)
                    all_text += "\n\n"

                # ── Claude judgment ───────────────────────────────────────────
                judgment = judge_answer(question, all_text, pdf_id, domain)
                result["claude_judgment"] = judgment

                if judgment["answered"]:
                    result["status"] = "PASS"
                    print(f"         ✓ PASS  (confidence: {judgment['confidence']})")
                else:
                    # ── Failure diagnosis ─────────────────────────────────────
                    failure_type = diagnose_failure(
                        entry["pdf_path"], tree, question,
                        retrieved_nodes, judgment.get("correct_section_hint", ""),
                        evidence_pages,
                    )
                    result["status"]       = "FAIL"
                    result["failure_type"] = failure_type
                    result["diagnosis"]    = judgment["reasoning"]
                    print(f"         ✗ FAIL  [{failure_type[:60]}]")
                    print(f"           Claude: {judgment['reasoning'][:100]}")
                    if evidence_pages:
                        coverage = check_extraction_coverage(tree, evidence_pages)
                        print(f"           Evidence pages coverage: {coverage}")

            except Exception as e:
                result["error"] = str(e)
                print(f"         ERROR: {e}")

            results.append(result)
            import time; time.sleep(0.2)   # breathing room between Claude calls

    # ── Save results ──────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n\nResults saved → {RESULTS_FILE}")

    return results


results = await run_full_eval()
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 8 — Analysis Report                               ║
# ╚══════════════════════════════════════════════════════════╝
"""
from collections import defaultdict

ok     = [r for r in results if r["error"] is None]
passed = [r for r in ok if r["status"] == "PASS"]
failed = [r for r in ok if r["status"] == "FAIL"]
errors = [r for r in results if r["error"] is not None]

n = len(ok)
print(f"\n{'='*70}")
print(f"  MULTI-DOMAIN TEST — {n} questions  ({len(errors)} errors)")
print(f"{'='*70}")
print(f"  PASS : {len(passed):3d}/{n}  ({len(passed)/n:.1%})")
print(f"  FAIL : {len(failed):3d}/{n}  ({len(failed)/n:.1%})")
print(f"  ERROR: {len(errors):3d}/{len(results)}")

# ── By domain ─────────────────────────────────────────────────────────────────
print(f"\n  {'Domain':<20} {'Q':>4}  {'Pass':>5}  {'Fail':>5}  {'Rate':>7}")
print(f"  {'-'*50}")
by_domain = defaultdict(lambda: {"pass": 0, "fail": 0})
for r in ok:
    by_domain[r["domain"]]["pass" if r["status"] == "PASS" else "fail"] += 1
for domain in sorted(by_domain):
    p = by_domain[domain]["pass"]
    f = by_domain[domain]["fail"]
    t = p + f
    print(f"  {domain:<20} {t:4d}  {p:5d}  {f:5d}  {p/t:7.1%}")

# ── By extraction strategy ────────────────────────────────────────────────────
print(f"\n  {'Strategy':<28} {'Q':>4}  {'Pass':>5}  {'Fail':>5}  {'Miss%':>7}")
print(f"  {'-'*55}")
by_strat = defaultdict(lambda: {"pass": 0, "fail": 0})
for r in ok:
    by_strat[r["strategy"]]["pass" if r["status"] == "PASS" else "fail"] += 1
for strat in sorted(by_strat, key=lambda s: -by_strat[s]["fail"]):
    p = by_strat[strat]["pass"]
    f = by_strat[strat]["fail"]
    t = p + f
    print(f"  {strat:<28} {t:4d}  {p:5d}  {f:5d}  {f/t:7.1%}")

# ── By failure type ───────────────────────────────────────────────────────────
print(f"\n  Failure type breakdown ({len(failed)} failures):")
fail_types = defaultdict(int)
for r in failed:
    ft = r["failure_type"] or "UNKNOWN"
    # Normalize to top-level category
    if "NAVIGATION_ERROR" in ft:
        fail_types["NAVIGATION_ERROR"] += 1
    elif "EXTRACTION_GAP" in ft:
        fail_types["EXTRACTION_GAP"] += 1
    elif "BAD_QUESTION" in ft:
        fail_types["BAD_QUESTION"] += 1
    elif "PARTIAL" in ft:
        fail_types["PARTIAL_MISS"] += 1
    else:
        fail_types["UNKNOWN"] += 1

for ft, count in sorted(fail_types.items(), key=lambda x: -x[1]):
    print(f"    {ft:<25}: {count:3d}  ({count/len(failed):.1%} of failures)")

# ── Full failure list ─────────────────────────────────────────────────────────
print(f"\n  {'='*70}")
print(f"  DETAILED FAILURE LIST")
print(f"  {'='*70}")
for r in failed:
    print(f"\n  [{r['domain'].upper():<12}] {r['pdf_id'][:45]}")
    print(f"  Strategy : {r['strategy']}")
    print(f"  Question : {r['question'][:100]}")
    print(f"  Retrieved: {r['node_titles']}")
    if r['evidence_pages']:
        print(f"  Ev.Pages : {r['evidence_pages']}")
    print(f"  Failure  : {r['failure_type']}")
    print(f"  Diagnosis: {r['diagnosis']}")

# ── Summary insight ───────────────────────────────────────────────────────────
nav_count  = fail_types.get("NAVIGATION_ERROR", 0)
extr_count = fail_types.get("EXTRACTION_GAP", 0)
print(f"\n{'='*70}")
print(f"  KEY INSIGHT:")
if nav_count > extr_count:
    pct = nav_count / len(failed) * 100
    print(f"  {pct:.0f}% of failures are NAVIGATION errors → fix requires v14 training data")
    print(f"  {extr_count} EXTRACTION gaps → StructDirect improvements needed for those docs")
else:
    pct = extr_count / len(failed) * 100
    print(f"  {pct:.0f}% of failures are EXTRACTION gaps → StructDirect is the bottleneck")
    print(f"  {nav_count} navigation errors → model training also needed")

strat_miss_rate = {s: d["fail"]/(d["pass"]+d["fail"]) for s, d in by_strat.items()}
worst_strat = max(strat_miss_rate, key=strat_miss_rate.get)
best_strat  = min(strat_miss_rate, key=strat_miss_rate.get)
print(f"\n  Worst strategy: {worst_strat} ({strat_miss_rate[worst_strat]:.1%} miss rate)")
print(f"  Best strategy : {best_strat}  ({strat_miss_rate[best_strat]:.1%} miss rate)")
print(f"{'='*70}")
"""
