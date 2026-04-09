# eval_interactive_v17.py — Interactive PDF Q&A with the v17 TreeSearch model
# =========================================================================
# Colab notebook. Copy each CELL block into a separate Colab cell.
# Run cells in order: 1 → 2 → 3 → then repeat CELL 4 for any question.
#
# USAGE:
#   - CELL 3: set PDF_PATH to your PDF on Drive, set a friendly PDF_NAME
#   - CELL 4: run it, type a question when prompted, press Enter
#   - Re-run CELL 4 as many times as you want with different questions
#   - To switch PDFs: change PDF_PATH/PDF_NAME and re-run CELL 3 only


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Install + Mount + Clone                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
import subprocess, os, sys

subprocess.run([
    "pip", "install", "-q",
    "transformers", "peft", "accelerate", "bitsandbytes",
    "safetensors", "pymupdf",
    "unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git"
], check=True)

from google.colab import drive
drive.mount("/content/drive")

if not os.path.exists("/content/arbor"):
    subprocess.run(
        ["git", "clone", "https://github.com/Dhruv12310/arbor.git", "/content/arbor"],
        check=True
    )
else:
    subprocess.run(["git", "-C", "/content/arbor", "pull"], check=True)

sys.path.insert(0, "/content/arbor")
print("Setup complete.")
"""


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Load v17 model (run once, takes ~60s)                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
import torch
from unsloth import FastLanguageModel

DRIVE_DIR  = "/content/drive/MyDrive/arbor-training-data"
MODEL_PATH = f"{DRIVE_DIR}/models/treesearch-v17"

print("Loading v17 model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_PATH,
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print("v17 model loaded and ready.")
"""


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Load PDF + build tree (re-run to switch documents)        ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
import json, sys
import fitz  # PyMuPDF
from arbor.extraction.structure_extractor import extract_structure
from arbor.providers.base import LLMProvider
from arbor.core.tree_searcher import search_tree
from arbor.core.types import ArborConfig

# ── SET THESE for each PDF you want to test ───────────────────────────────────
#
#   Option A — PDF is already on Google Drive:
#     PDF_PATH = "/content/drive/MyDrive/YOUR_FILE.pdf"
#
#   Option B — Upload via Colab file picker:
#     from google.colab import files
#     uploaded = files.upload()
#     PDF_PATH = list(uploaded.keys())[0]
#
PDF_PATH = "/content/drive/MyDrive/arbor-training-data/pdfs/APPLE_2022_10K.pdf"
PDF_NAME = "Apple_2022_10K"   # friendly label shown in output

# ── Extract tree ──────────────────────────────────────────────────────────────
print(f"Opening: {PDF_PATH}")
_pdf_doc = fitz.open(PDF_PATH)
_tree    = extract_structure(_pdf_doc, PDF_NAME)

total_pages    = max((n.end_index for n in _tree.structure), default=0)
total_sections = len(_tree.structure)
total_nodes    = sum(1 + len(n.nodes or []) for n in _tree.structure)

print(f"\nTree extracted:")
print(f"  Document         : {PDF_NAME}")
print(f"  Pages            : {total_pages}")
print(f"  Top-level items  : {total_sections}")
print(f"  Total nodes      : {total_nodes}")
print(f"  Strategy         : {_tree.extraction_strategy}")
print()
print("Top-level sections:")
for node in _tree.structure[:15]:
    sub = f" [+{len(node.nodes)} sub-sections]" if node.nodes else ""
    print(f"  [{node.node_id}] {node.title}  (pages {node.start_index}-{node.end_index}){sub}")
if len(_tree.structure) > 15:
    print(f"  ... and {len(_tree.structure) - 15} more sections")
print()
print("Ready. Run CELL 4 to ask a question.")


# ── Provider (reused across all CELL 4 runs) ──────────────────────────────────
class _LocalProvider(LLMProvider):
    def __init__(self, model, tokenizer):
        self._model     = model
        self._tokenizer = tokenizer

    @property
    def name(self):
        return "treesearch-v17-local"

    async def complete(self, prompt, temperature=0.0, max_tokens=None, chat_history=None):
        result, _ = await self.complete_with_finish_reason(prompt, temperature, max_tokens, chat_history)
        return result

    async def complete_with_finish_reason(self, prompt, temperature=0.0, max_tokens=None, chat_history=None):
        messages = list(chat_history or []) + [{"role": "user", "content": prompt}]
        enc = self._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = enc.input_ids if hasattr(enc, "input_ids") else enc
        input_ids = input_ids.to(self._model.device)
        with torch.no_grad():
            output = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens or 256,
                do_sample=False,
                repetition_penalty=1.1,
            )
        new_tokens = output[0][input_ids.shape[1]:]
        text       = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        finish     = "length" if len(new_tokens) >= (max_tokens or 256) else "stop"
        return text, finish

_provider = _LocalProvider(model, tokenizer)
_config   = ArborConfig(max_hops=10, max_nodes_searched=0, timeout_sec=300.0, max_retries_on_bad_json=2)
"""


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 4 — Automated benchmark: 10 preset questions with ground truth ║
# ║  Set PDF_PATH/PDF_NAME in CELL 3 to Apple_2024_10K before running   ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
import asyncio

# ── Apple 2024 10K benchmark ──────────────────────────────────────────────────
# Tree structure (after StructDirect financial-statement header fix):
#   [0001] Item 1.  Business                           p1-4
#   [0002] Item 1A. Risk Factors                       p5-16
#   [0009] Item 7.  MD&A                               p21-26
#   [0010] Item 7A. Quantitative Disclosures           p27-27
#   [0011] Item 8.  Financial Statements               p28-50
#     [0012]   Pages 28-30  (index/auditor)            p28-30
#     [0013]   Consolidated Statements Of Operations   p32-32   ← income statement
#     [0014]   Consolidated Statements Of Comprehensive Income  p33-33
#     [0015]   Consolidated Balance Sheets             p34-35   ← balance sheet
#     [0016]   Consolidated Statements Of Cash Flows  p36-36
#     [0017]   Notes to Consolidated Financial Statements p37-50
#   [0029] Item 16. Form 10-K Summary                  p56-121
#
# Ground truth: "correct" = model's returned pages overlap the answer page(s)

BENCHMARK = [
    {
        "question": "What was Apple's total net sales revenue for fiscal year 2024?",
        "answer_pages": {32},          # Income statement: Total net sales $391,035M on p32
        "answer_hint": "p32 — Income Statement: Total net sales $391,035M",
    },
    {
        "question": "What was Apple's net income for fiscal year 2024?",
        "answer_pages": {32},          # Income statement: Net income $93,736M on p32
        "answer_hint": "p32 — Income Statement: Net income $93,736M",
    },
    {
        "question": "What are Apple's two revenue categories reported on the income statement?",
        "answer_pages": {32},          # Products vs Services split on p32
        "answer_hint": "p32 — Income Statement: Products $294,866M, Services $96,169M",
    },
    {
        "question": "What were Apple's total assets as of September 28, 2024?",
        "answer_pages": {34},          # Balance sheet: Total assets $364,980M on p34
        "answer_hint": "p34 — Balance Sheet: Total assets $364,980M",
    },
    {
        "question": "How much cash and cash equivalents did Apple hold at the end of fiscal 2024?",
        "answer_pages": {34},          # Balance sheet: Cash $29,943M on p34
        "answer_hint": "p34 — Balance Sheet: Cash and equivalents $29,943M",
    },
    {
        "question": "What new Mac products did Apple announce in the first quarter of fiscal 2024?",
        "answer_pages": {24},          # MD&A product announcements: MacBook Pro 14-in, 16-in, iMac on p24
        "answer_hint": "p24 — MD&A: MacBook Pro 14-in, MacBook Pro 16-in, iMac",
    },
    {
        "question": "What was Apple's net sales in the Americas segment for fiscal year 2024?",
        "answer_pages": {25},          # MD&A segment table: Americas $167,045M on p25
        "answer_hint": "p25 — MD&A Segment table: Americas $167,045M",
    },
    {
        "question": "What was Apple's effective tax rate for fiscal year 2024?",
        "answer_pages": {28},          # Tax provision table: effective rate 24.1% on p28
        "answer_hint": "p28 — Tax note: Effective tax rate 24.1%",
    },
    {
        "question": "What is Apple's stock ticker symbol and on which exchange is it traded?",
        "answer_pages": {22},          # Item 5 Market for Common Equity: AAPL on Nasdaq p22
        "answer_hint": "p22 — Item 5: AAPL on Nasdaq Stock Market LLC",
    },
    {
        "question": "What are the primary risk factors related to Apple's dependence on third-party manufacturers?",
        "answer_pages": set(range(5, 17)),   # Item 1A Risk Factors p5-16
        "answer_hint": "p5-16 — Item 1A Risk Factors (manufacturing/supply chain risks)",
    },
]

# ── Run benchmark ─────────────────────────────────────────────────────────────
print("=" * 70)
print(f"  APPLE 2024 10K BENCHMARK  —  {len(BENCHMARK)} questions")
print(f"  Model: treesearch-v17 | Doc: {PDF_NAME}")
print("=" * 70)

results = []

for i, bench in enumerate(BENCHMARK, 1):
    question    = bench["question"]
    answer_pages = bench["answer_pages"]
    hint        = bench["answer_hint"]

    sr = await search_tree(
        _tree, question, _provider,
        multihop=True, config=_config, doc_type=None
    )

    returned_pages = set()
    for node in sr.nodes:
        returned_pages.update(range(node.start_index, node.end_index + 1))

    hit = bool(returned_pages & answer_pages)
    results.append(hit)

    status = "PASS" if hit else "FAIL"
    sections = ", ".join(f"[{n.node_id}]{n.title[:30]}" for n in sr.nodes)

    print(f"\nQ{i:02d}: {question}")
    print(f"  Expected : {hint}")
    print(f"  Got      : pages {sorted(returned_pages)} — {sections}")
    print(f"  Result   : {'✓ ' + status if hit else '✗ ' + status}")

# ── Summary ───────────────────────────────────────────────────────────────────
correct = sum(results)
total   = len(results)
print()
print("=" * 70)
print(f"  SCORE: {correct}/{total}  ({correct/total*100:.0f}%)")
print()
for i, (bench, hit) in enumerate(zip(BENCHMARK, results), 1):
    mark = "PASS" if hit else "FAIL"
    print(f"  Q{i:02d} {mark}  {bench['question'][:60]}")
print("=" * 70)
"""
