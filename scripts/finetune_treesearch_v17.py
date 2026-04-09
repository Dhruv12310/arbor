# Arbor TreeSearch v17 — Diverse Training + DAgger Corrections
# =========================================================================
# Copy each CELL block into a separate Colab cell and run in order.
# Run cells 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 in order.
#
# WHAT CHANGED FROM v16:
#
#   1. DIVERSE TRAINING DATA (new):
#      - 38,826 examples generated from 43 PDFs (8 FinanceBench + 35 domain)
#      - Haiku reads each section's actual text and writes 5 questions × 5 phrasings
#      - Covers question styles the model never saw: casual, researcher, student, expert
#      - Prevents navigation failures caused by unfamiliar phrasing
#
#   2. REFRESHED DAgger targeting ALL 56 v16 failures:
#      - dagger_v17_targeted.jsonl: 69 examples for 34 of 56 v16 failures
#      - 16 failures are structurally unreachable (evidence on page 0 or in coverage gaps)
#      - Combined with dagger_v15 (139), total corrective signal: 208 unique examples
#
#   3. REFRESHED success replay from v16 correct answers:
#      - v17_success_replay.jsonl: 118 examples from 59 of 94 v16 successes
#      - Prevents forgetting while learning new styles
#
#   4. Training mix (all mixing done offline by build_v17_dataset.py):
#      - v17_train.jsonl: 63,438 examples (pre-shuffled, pre-mixed)
#        diverse             38,826  ×1  = 38,826  (61.2%)
#        structdirect         1,329  ×8  = 10,632  (16.8%)
#        dagger_v15             139  ×40 =  5,560   (8.8%)
#        replay_v17             118  ×30 =  3,540   (5.6%)
#        dagger_v17              69  ×40 =  2,760   (4.4%)
#        replay_v14b            106  ×20 =  2,120   (3.3%)
#      DAgger combined share: 13.2% — every batch of 16 has ~2 correction rows
#
#   5. 2 epochs (not 3) — more data per epoch, overfitting risk higher with 40× DAgger
#   6. load_best_model_at_end=False — NEVER CHANGE THIS
#
# DATASETS NEEDED ON DRIVE (MyDrive/arbor-training-data/):
#   - v17_train.jsonl       ← UPLOAD (63,438 examples, built by build_v17_dataset.py)
#
# REQUIREMENTS:
#   - Google Colab Pro (A100 40GB recommended)
#
# EXPECTED RUNTIME on A100:
#   63,438 examples × 2 epochs / 16 batch = ~7,930 steps ≈ 90-120 minutes
#
# OUTPUT:
#   - Saved to Drive: MyDrive/arbor-training-data/models/treesearch-v17/


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 1 — Install dependencies                          ║
# ╚══════════════════════════════════════════════════════════╝
"""
!pip install -q "unsloth[colab-new]" trl transformers accelerate bitsandbytes peft
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2 — Mount Drive + verify files                    ║
# ╚══════════════════════════════════════════════════════════╝
"""
from google.colab import drive
drive.mount('/content/drive')

import os, subprocess

DRIVE_DIR  = "/content/drive/MyDrive/arbor-training-data"
TRAIN_FILE = f"{DRIVE_DIR}/v17_train.jsonl"
OUTPUT_DIR = f"{DRIVE_DIR}/models/treesearch-v17"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Verify training file exists and count lines
if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(
        f"v17_train.jsonl not found at {TRAIN_FILE}\n"
        "Build it locally with: python scripts/build_v17_dataset.py\n"
        "Then upload to Drive."
    )

result = subprocess.run(["wc", "-l", TRAIN_FILE], capture_output=True, text=True)
print(f"  v17_train.jsonl : {result.stdout.strip()}")
print(f"  Output dir      : {OUTPUT_DIR}")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 3 — Load base model + LoRA                        ║
# ╚══════════════════════════════════════════════════════════╝
"""
from unsloth import FastLanguageModel
import torch

MAX_SEQ_LEN  = 4096
DTYPE        = None   # Auto-detect (bfloat16 on A100)
LOAD_IN_4BIT = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Qwen2.5-3B-Instruct",
    max_seq_length = MAX_SEQ_LEN,
    dtype          = DTYPE,
    load_in_4bit   = LOAD_IN_4BIT,
)

# r=16 — same rank as v14b/v15/v16, proven to fit A100 with full-layer coverage
model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = 16,
    lora_dropout   = 0,
    bias           = "none",
    use_gradient_checkpointing = "unsloth",
    random_state   = 42,
    use_rslora     = False,
)
print(model.print_trainable_parameters())
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 4 — Load and format dataset                       ║
# ╚══════════════════════════════════════════════════════════╝
"""
import json, random
from datasets import Dataset

# v17_train.jsonl was pre-mixed and pre-shuffled by build_v17_dataset.py.
# All oversampling multipliers are already baked in — no re-mixing here.
# We only normalize the system prompt to match inference-time wording.

_INFERENCE_SYSTEM = (
    "You are a document tree navigator. "
    "Given a question and a list of document sections at the current level, "
    "select which sections to explore next to find the answer.\n\n"
    "Always reply with valid JSON:\n"
    '{"thinking": "brief reasoning", "navigate_to": ["node_id1", "node_id2"]}'
)


def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def normalize_system_prompt(examples):
    """Unify all training system prompts to match the inference-time wording."""
    for ex in examples:
        msgs = ex.get("messages", [])
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] = _INFERENCE_SYSTEM
        elif msgs and msgs[0]["role"] != "system":
            msgs.insert(0, {"role": "system", "content": _INFERENCE_SYSTEM})
    return examples


def format_example(example):
    """Apply Qwen2.5 chat template (no generation prompt — this is training, not inference)."""
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


print("Loading v17_train.jsonl...")
all_train = load_jsonl(TRAIN_FILE)
print(f"  Loaded: {len(all_train):,} examples")

normalize_system_prompt(all_train)

# Re-shuffle after system prompt normalization (paranoia: ensure no ordering artifacts)
random.seed(42)
random.shuffle(all_train)

print("Applying chat template...")
train_texts = [format_example(ex) for ex in all_train]

# Token length distribution — warn if p99 exceeds MAX_SEQ_LEN
sample = random.sample(train_texts, min(500, len(train_texts)))
token_lengths = sorted([len(tokenizer.encode(t)) for t in sample])
p50 = token_lengths[len(token_lengths) // 2]
p90 = token_lengths[int(len(token_lengths) * 0.9)]
p99 = token_lengths[int(len(token_lengths) * 0.99)]
max_len = token_lengths[-1]
print(f"  Token lengths — p50: {p50} | p90: {p90} | p99: {p99} | max: {max_len}")
if p99 >= MAX_SEQ_LEN:
    print(f"  WARNING: p99={p99} >= MAX_SEQ_LEN={MAX_SEQ_LEN} — some examples will be truncated")
else:
    print(f"  OK: p99 < {MAX_SEQ_LEN}")

train_dataset = Dataset.from_dict({"text": train_texts})
print(f"\nDataset ready: {len(train_dataset):,} training examples")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 5 — Train                                         ║
# ╚══════════════════════════════════════════════════════════╝
"""
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# 2 epochs: DAgger examples appear 2 × 40 = 80 times each; more epochs risk
# overfitting on the 69 unique DAgger rows while degrading the general signal.
NUM_EPOCHS   = 2
BATCH_SIZE   = 16  # 4 per device × 4 gradient accumulation steps
TOTAL_STEPS  = (len(train_dataset) * NUM_EPOCHS) // BATCH_SIZE
# Warmup over first 5% of training — longer warmup helps with large diverse dataset
WARMUP_STEPS = max(100, TOTAL_STEPS // 20)

print(f"Training plan:")
print(f"  Examples      : {len(train_dataset):,}")
print(f"  Epochs        : {NUM_EPOCHS}")
print(f"  Effective batch: {BATCH_SIZE}")
print(f"  Total steps   : {TOTAL_STEPS:,}")
print(f"  Warmup steps  : {WARMUP_STEPS}")
print(f"  Est. runtime  : ~{TOTAL_STEPS * 0.9 / 60:.0f} min (A100 @ ~0.9 sec/step)")

trainer = SFTTrainer(
    model              = model,
    train_dataset      = train_dataset,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
    dataset_num_proc   = 2,
    packing            = False,   # Keep False — packing mixes DAgger signal with unrelated examples
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,      # Effective batch = 16
        warmup_steps                = WARMUP_STEPS,
        num_train_epochs            = NUM_EPOCHS,
        learning_rate               = 2e-4,   # Proven LR from v14b/v15/v16
        fp16                        = not is_bfloat16_supported(),
        bf16                        = is_bfloat16_supported(),
        logging_steps               = 25,
        save_strategy               = "steps",
        save_steps                  = 500,
        save_total_limit            = 2,          # Keep only last 2 checkpoints — prevents Drive overflow
        output_dir                  = OUTPUT_DIR,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "cosine",
        max_grad_norm               = 0.3,
        seed                        = 42,
        load_best_model_at_end      = False,   # NEVER CHANGE — causes checkpoint/adapter mismatch
        report_to                   = "none",
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_vram = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
max_vram   = round(gpu_stats.total_memory / 1024 ** 3, 3)
print(f"\nGPU: {gpu_stats.name} | VRAM: {max_vram}GB | Reserved: {start_vram}GB")

trainer_stats = trainer.train()
print(f"\nTraining complete.")
print(f"  Runtime : {trainer_stats.metrics.get('train_runtime', 0)/60:.1f} min")
print(f"  Loss    : {trainer_stats.metrics.get('train_loss', 'N/A')}")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 6 — Save adapter to Drive                         ║
# ╚══════════════════════════════════════════════════════════╝
"""
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 7 — Smoke test (4 questions)                      ║
# ╚══════════════════════════════════════════════════════════╝
"""
import json, torch
from unsloth import FastLanguageModel

DRIVE_DIR  = "/content/drive/MyDrive/arbor-training-data"
MODEL_PATH = f"{DRIVE_DIR}/models/treesearch-v17"

model_test, tokenizer_test = FastLanguageModel.from_pretrained(
    MODEL_PATH, max_seq_length=4096, dtype=None, load_in_4bit=True
)
model_test.eval()

SYSTEM = (
    "You are a document tree navigator. "
    "Given a question and a list of document sections at the current level, "
    "select which sections to explore next to find the answer.\n\n"
    "Always reply with valid JSON:\n"
    '{"thinking": "brief reasoning", "navigate_to": ["node_id1", "node_id2"]}'
)

SMOKE_TESTS = [
    # (name, expected_node_id, user_content)
    (
        "Revenue — Financial Statements",
        "0004",
        (
            "Question: What was the company's total revenue for fiscal year 2022?\n\n"
            "Sections at this level:\n"
            "[0001] Letter to Shareholders (pages 1-3)\n"
            "[0002] Business Overview (pages 4-12) [has sub-sections]\n"
            "[0003] Risk Factors (pages 13-28) [has sub-sections]\n"
            "[0004] Financial Statements and Results (pages 29-54) [has sub-sections]\n"
            "[0005] Corporate Governance (pages 55-61)\n"
            "[0006] Executive Compensation (pages 62-74)\n\n"
            "Which sections should we explore next?"
        ),
    ),
    (
        "Stock performance — Item 5",
        "0008",
        (
            "Question: What is the company stock performance in 2024?\n\n"
            "Sections at this level:\n"
            "[0001] Item 1. Business (pages 1-9) [has sub-sections]\n"
            "[0002] Item 1A. Risk Factors (pages 10-19)\n"
            "[0003] Item 1B. Unresolved Staff Comments (pages 19-19)\n"
            "[0004] Item 1C. Cybersecurity (pages 19-19)\n"
            "[0005] Item 2. Properties (pages 19-19)\n"
            "[0006] Item 3. Legal Proceedings (pages 19-19)\n"
            "[0007] Item 4. Mine Safety Disclosures (pages 19-19)\n"
            "[0008] Item 5. Market for Registrant's Common Equity (pages 20-20)\n"
            "[0009] Item 6. [Reserved] (pages 20-20)\n"
            "[0010] Item 7. Management's Discussion and Analysis (pages 21-38) [has sub-sections]\n"
            "[0011] Item 7A. Quantitative and Qualitative Disclosures About Market Risk (pages 39-39)\n"
            "[0012] Item 8. Financial Statements and Supplementary Data (pages 40-73) [has sub-sections]\n"
            "[0013] Item 9. Changes in and Disagreements With Accountants (pages 73-73)\n\n"
            "Which sections should we explore next?"
        ),
    ),
    (
        "Casual question — Risk Factors",
        "0003",
        (
            "Question: What could go wrong with this company?\n\n"
            "Sections at this level:\n"
            "[0001] Item 1. Business (pages 1-15) [has sub-sections]\n"
            "[0002] Item 1A. Risk Factors (pages 16-34) [has sub-sections]\n"
            "[0003] Item 1B. Unresolved Staff Comments (pages 35-35)\n"
            "[0004] Item 2. Properties (pages 35-36)\n"
            "[0005] Item 3. Legal Proceedings (pages 36-37)\n"
            "[0006] Item 4. Mine Safety (pages 37-37)\n"
            "[0007] Item 5. Market for Common Equity (pages 38-39)\n"
            "[0008] Item 6. Reserved (pages 39-39)\n"
            "[0009] Item 7. MD&A (pages 40-68) [has sub-sections]\n"
            "[0010] Item 8. Financial Statements (pages 69-130) [has sub-sections]\n\n"
            "Which sections should we explore next?"
        ),
    ),
    (
        "Domain paper — Methodology",
        "0004",
        (
            "Question: How was the data collection methodology designed for this study?\n\n"
            "Sections at this level:\n"
            "[0001] Abstract (pages 0-0)\n"
            "[0002] Introduction (pages 1-2)\n"
            "[0003] Related Work (pages 3-4)\n"
            "[0004] Methodology (pages 5-7) [has sub-sections]\n"
            "[0005] Experiments (pages 8-10) [has sub-sections]\n"
            "[0006] Results (pages 11-12)\n"
            "[0007] Discussion (pages 13-14)\n"
            "[0008] Conclusion (pages 15-15)\n"
            "[0009] References (pages 16-18)\n\n"
            "Which sections should we explore next?"
        ),
    ),
]

print("Running smoke tests...\n")
all_pass = True
for name, expected, user_content in SMOKE_TESTS:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": user_content},
    ]
    enc = tokenizer_test.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = (enc.input_ids if hasattr(enc, "input_ids") else enc).to("cuda")
    with torch.no_grad():
        out = model_test.generate(
            input_ids        = input_ids,
            max_new_tokens   = 128,
            do_sample        = False,
            repetition_penalty = 1.1,
            pad_token_id     = tokenizer_test.eos_token_id,
            use_cache        = False,
        )
    response = tokenizer_test.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    try:
        parsed = json.loads(response)
        nav    = parsed.get("navigate_to", [])
        passed = expected in nav
        status = "PASS" if passed else f"FAIL (expected {expected})"
        if not passed:
            all_pass = False
    except Exception:
        status = "PARSE FAIL"
        nav = [response[:80]]
        all_pass = False
    print(f"  [{status}] {name}")
    print(f"    navigate_to: {nav}")
    print(f"    thinking   : {parsed.get('thinking', '')[:100]}\n")

print("All smoke tests PASSED" if all_pass else "WARNING: some smoke tests FAILED — check model output")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 8 — FinanceBench end-to-end eval (THE REAL METRIC)║
# ╚══════════════════════════════════════════════════════════╝
"""
import subprocess, os, json, torch, warnings
warnings.filterwarnings("ignore", message="Both `max_new_tokens`.*")

from google.colab import drive
drive.mount('/content/drive')

# Clone / pull latest arbor (gets fixed StructDirect code)
if not os.path.exists("/content/arbor"):
    subprocess.run(
        ["git", "clone", "https://github.com/Dhruv12310/arbor.git", "/content/arbor"],
        check=True
    )
else:
    subprocess.run(["git", "-C", "/content/arbor", "pull"], check=True)

subprocess.run(
    ["pip", "install", "-q", "unsloth[colab-new]", "trl", "transformers",
     "accelerate", "bitsandbytes", "peft", "pymupdf"],
    check=True
)

import sys
sys.path.insert(0, "/content/arbor")

from unsloth import FastLanguageModel

DRIVE_DIR    = "/content/drive/MyDrive/arbor-training-data"
ADAPTER_PATH = f"{DRIVE_DIR}/models/treesearch-v17"
PDF_DIR      = f"{DRIVE_DIR}/financebench-pdfs"     # PDFs on Drive

print("Loading v17 model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    ADAPTER_PATH, max_seq_length=4096, dtype=None, load_in_4bit=True
)
model.eval()
print("v17 loaded.\n")

from arbor.providers.base import LLMProvider
from arbor.extraction.structure_extractor import extract_structure
from arbor.core.tree_searcher import search_tree
from arbor.core.types import ArborConfig, DocumentTree


class V17Provider(LLMProvider):
    @property
    def name(self):
        return "treesearch-v17"

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
        text       = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        finish     = "length" if len(new_tokens) >= (max_tokens or 256) else "stop"
        return text, finish


# ── Tree loading: extract fresh from PDFs using fixed StructDirect code ──────
# We deliberately skip pre-cached JSON trees because older cache files may
# have been extracted with buggy code (before the 04-08 StructDirect fixes).
# Extracting fresh ensures the model is evaluated on correct, complete trees.

QA_FILE  = "/content/arbor/data/financebench/financebench_open_source.jsonl"
qa_pairs = [
    json.loads(line)
    for line in open(QA_FILE, encoding="utf-8").read().strip().splitlines()
    if line.strip()
]

all_docs   = sorted({q["doc_name"] for q in qa_pairs})
tree_cache = {}

print(f"Extracting trees for {len(all_docs)} docs (fresh from PDFs)...")
failed_docs = []
for doc in all_docs:
    pdf_path = f"{PDF_DIR}/{doc}.pdf"
    if not os.path.exists(pdf_path):
        failed_docs.append(doc)
        continue
    try:
        tree = extract_structure(pdf_path)
        tree_cache[doc] = tree
    except Exception as e:
        print(f"  ERROR {doc}: {e}")
        failed_docs.append(doc)

if failed_docs:
    print(f"\nWARNING: {len(failed_docs)} PDFs not found — those questions will be skipped.")
    print(f"  Missing: {failed_docs}")

qa_pairs_filtered = [q for q in qa_pairs if q["doc_name"] in tree_cache]
print(f"\nTrees extracted: {len(tree_cache)}/{len(all_docs)}")
print(f"QA pairs       : {len(qa_pairs_filtered)}/150\n")

config   = ArborConfig(
    max_hops=8, max_nodes_searched=0, timeout_sec=300.0, max_retries_on_bad_json=2,
)
provider = V17Provider()


async def run_financebench_v17():
    results = []

    for i, qa in enumerate(qa_pairs_filtered):
        doc            = qa["doc_name"]
        question       = qa["question"]
        evidence_pages = [e["evidence_page_num"] for e in qa.get("evidence", [])]
        tree           = tree_cache[doc]

        try:
            sr = await search_tree(
                tree, question, provider, multihop=True, config=config, doc_type=None
            )
            returned_pages = set()
            for node in sr.nodes:
                returned_pages.update(range(node.start_index, node.end_index + 1))
            found   = [p for p in evidence_pages if p in returned_pages]
            recall  = len(found) / len(evidence_pages) if evidence_pages else 1.0
            perfect = recall == 1.0
            results.append({
                "q":        i + 1,
                "doc":      doc,
                "recall":   recall,
                "found":    found,
                "evidence": evidence_pages,
                "nodes":    sr.node_ids,
                "perfect":  perfect,
            })
            status = "+" if perfect else "-"
            print(
                f"[{i+1:03d}/{len(qa_pairs_filtered)}] {status} {doc[:35]:<35} | "
                f"recall={recall:.0%} | evid={evidence_pages}"
            )
        except Exception as e:
            print(f"[{i+1:03d}] ERROR {doc}: {e}")
            results.append({
                "q": i + 1, "doc": doc, "recall": 0.0,
                "error": str(e), "perfect": False,
            })

    n          = len(results)
    avg_recall = sum(r["recall"] for r in results) / n if n else 0.0
    perfect    = sum(1 for r in results if r.get("perfect", False))
    partial    = sum(1 for r in results if 0 < r.get("recall", 0) < 1.0)
    zero       = sum(1 for r in results if r.get("recall", 0) == 0.0)

    print(f"\n{'='*58}")
    print(f"  v17 FinanceBench Retrieval Results")
    print(f"  Questions evaluated : {n}/150")
    print(f"  Avg recall          : {avg_recall:.1%}")
    print(f"  Perfect recall (1.0): {perfect}/{n}  ({perfect/n:.1%})")
    print(f"  Partial recall      : {partial}/{n}")
    print(f"  Zero recall         : {zero}/{n}")
    print(f"{'='*58}")
    print(f"  v16 baseline        : 62.7% (94/150)")
    print(f"  Target              : 70%+  (105/150)")
    delta = perfect - 94
    sign  = "+" if delta >= 0 else ""
    print(f"  Delta vs v16        : {sign}{delta} questions  ({sign}{delta/150*100:.1f}pp)")
    if perfect >= 105:
        print(f"  Status              : TARGET REACHED")
    elif perfect > 94:
        print(f"  Status              : IMPROVEMENT OVER v16")
    else:
        print(f"  Status              : NO IMPROVEMENT OVER v16 — investigate failures")
    print(f"{'='*58}")

    results_path = f"{DRIVE_DIR}/financebench_v17_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")
    print("Download and commit to arbor repo for analysis.")

    return results


results = await run_financebench_v17()
"""
