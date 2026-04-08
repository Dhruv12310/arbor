# eval_unknown_pdf.py — Test v15 on an unknown PDF from Google Drive
# =========================================================================
# Fresh session script. Copy each CELL block into a separate Colab cell.
# Run cells in order: 1 → 2 → 3 → 4
#
# BEFORE RUNNING:
#   - Upload your PDF to Google Drive ROOT (not a subfolder)
#   - Set PDF_FILENAME below to match your file name
#   - Set QUESTION to whatever you want to ask


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 1 — Install + Mount + Clone                       ║
# ╚══════════════════════════════════════════════════════════╝
"""
import subprocess, os, sys

subprocess.run([
    "pip", "install", "-q",
    "transformers", "peft", "accelerate", "bitsandbytes",
    "safetensors", "pymupdf", "unsloth[colab-new]"
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


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2 — Load v15 model                                ║
# ╚══════════════════════════════════════════════════════════╝
"""
import torch
from unsloth import FastLanguageModel

DRIVE_DIR  = "/content/drive/MyDrive/arbor-training-data"
MODEL_PATH = f"{DRIVE_DIR}/models/treesearch-v15"

print("Loading v15 model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_PATH,
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print("v15 loaded.")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 3 — Build provider + extract tree from PDF        ║
# ╚══════════════════════════════════════════════════════════╝
"""
import json, torch, asyncio, sys
from arbor.providers.base import LLMProvider
from arbor.core.tree_searcher import search_tree
from arbor.core.types import ArborConfig
from arbor.extraction.structure_extractor import extract_structure
import fitz  # PyMuPDF

# ── Config ────────────────────────────────────────────────────────────────────
# PDF must be uploaded to Drive ROOT (MyDrive/your_file.pdf)
PDF_FILENAME = "c87043b9-5d89-4717-9f49-c4f9663d0061.pdf"   # change if needed
PDF_PATH     = f"/content/drive/MyDrive/{PDF_FILENAME}"
PDF_NAME     = "Apple_2024_10K"   # friendly name for the tree

QUESTION = "What is the company stock performance in 2024?"

# ── Local provider wrapping the fine-tuned model ──────────────────────────────
class LocalProvider(LLMProvider):
    def __init__(self, model, tokenizer):
        self._model     = model
        self._tokenizer = tokenizer

    @property
    def name(self):
        return "treesearch-v15-local"

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

provider = LocalProvider(model, tokenizer)

# ── Extract tree from PDF ─────────────────────────────────────────────────────
print(f"Opening PDF: {PDF_PATH}")
doc  = fitz.open(PDF_PATH)
tree = extract_structure(doc, PDF_NAME)

total_pages   = max((n.end_index for n in tree.structure), default=0) if tree.structure else 0
total_sections = sum(1 for _ in tree.structure)

print(f"Tree extracted:")
print(f"  Document : {PDF_NAME}")
print(f"  Pages    : {total_pages}")
print(f"  Top-level sections : {total_sections}")
print(f"  Extraction strategy: {tree.extraction_strategy}")
print()

# Preview first 10 top-level sections
print("Top-level sections (first 10):")
for node in tree.structure[:10]:
    sub = " [+]" if node.nodes else ""
    print(f"  [{node.node_id}] {node.title} (pages {node.start_index}-{node.end_index}){sub}")
if len(tree.structure) > 10:
    print(f"  ... and {len(tree.structure) - 10} more")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 4 — Run search + show results with citations      ║
# ╚══════════════════════════════════════════════════════════╝
"""
config = ArborConfig(
    max_hops=10,
    max_nodes_searched=0,
    timeout_sec=300.0,
    max_retries_on_bad_json=2,
)

print("=" * 65)
print(f"  Document : {PDF_NAME}")
print(f"  Question : {QUESTION}")
print("=" * 65)

# Run search
sr = await search_tree(
    tree, QUESTION, provider,
    multihop=True, config=config, doc_type=None
)

# Collect all pages covered by returned nodes
returned_pages = set()
for node in sr.nodes:
    returned_pages.update(range(node.start_index, node.end_index + 1))

# ── Print results ─────────────────────────────────────────────────────────────
print(f"\nSections found ({len(sr.nodes)}):")
for node in sr.nodes:
    print(f"  [{node.node_id}] {node.title}")
    print(f"           Pages {node.start_index}–{node.end_index}")

print(f"\nAll pages covered: {sorted(returned_pages)}")

print(f"\nNavigation reasoning:")
for hop in sr.thinking.split(" | "):
    if hop.strip():
        print(f"  {hop.strip()}")

print()
print("=" * 65)
print("  WHAT TO DO NEXT:")
print("=" * 65)
print(f"  Open '{PDF_FILENAME}' and go to page(s): {sorted(returned_pages)}")
print(f"  The answer to your question should be there.")
print(f"  If the section title and page match → v15 is working on new PDFs.")
print(f"  If it navigated to the wrong section → we have a failure to fix.")
print("=" * 65)

# ── Also extract a text snippet from the found pages ─────────────────────────
print("\nText preview from found sections (first 500 chars each):\n")
for node in sr.nodes:
    try:
        # Extract text from first page of this node
        page_idx = node.start_index - 1  # PyMuPDF is 0-indexed
        if 0 <= page_idx < len(doc):
            text = doc[page_idx].get_text()[:500].strip()
            print(f"--- [{node.node_id}] {node.title} (page {node.start_index}) ---")
            print(text)
            print()
    except Exception as e:
        print(f"  (Could not extract text: {e})")
"""
