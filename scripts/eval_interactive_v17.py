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
# ║  CELL 4 — Ask a question (re-run as many times as you want)         ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
import asyncio

# ── Type your question here OR use input() for interactive prompt ─────────────
QUESTION = input("Question: ").strip()
# Or hardcode:  QUESTION = "What was the total revenue in 2022?"

if not QUESTION:
    print("No question entered.")
else:
    print()
    print("=" * 70)
    print(f"  Doc      : {PDF_NAME}")
    print(f"  Question : {QUESTION}")
    print("=" * 70)

    sr = await search_tree(
        _tree, QUESTION, _provider,
        multihop=True, config=_config, doc_type=None
    )

    # Pages covered by returned nodes
    returned_pages = set()
    for node in sr.nodes:
        returned_pages.update(range(node.start_index, node.end_index + 1))

    # ── Sections found ────────────────────────────────────────────────────────
    print(f"\nSections found ({len(sr.nodes)}):")
    for node in sr.nodes:
        print(f"  [{node.node_id}] {node.title}")
        print(f"           Pages {node.start_index}–{node.end_index}")

    print(f"\nPages to check: {sorted(returned_pages)}")

    # ── Navigation trace ──────────────────────────────────────────────────────
    if sr.thinking:
        print(f"\nNavigation reasoning:")
        for hop in sr.thinking.split(" | "):
            if hop.strip():
                print(f"  → {hop.strip()}")

    # ── Text preview from found pages ─────────────────────────────────────────
    print(f"\nText preview (up to 400 chars per section):\n")
    for node in sr.nodes:
        page_idx = node.start_index - 1  # PyMuPDF is 0-indexed
        if 0 <= page_idx < len(_pdf_doc):
            text = _pdf_doc[page_idx].get_text()[:400].strip()
            print(f"--- [{node.node_id}] {node.title} (page {node.start_index}) ---")
            print(text)
            print()

    print("=" * 70)
    print(f"  Open the PDF and navigate to page(s): {sorted(returned_pages)}")
    print("=" * 70)
"""
