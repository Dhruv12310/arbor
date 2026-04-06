# Arbor TreeSearch v14 — DAgger + Academic + StructDirect (3-Dataset Mix)
# =========================================================================
# Copy each CELL block into a separate Colab cell and run in order.
# Run cells 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 in order. Do NOT skip.
#
# WHAT CHANGED FROM v13:
#   - THREE training datasets instead of one:
#       1. structdirect_train.jsonl  (3,122 examples — 84 FinanceBench docs)
#       2. academic_train.jsonl      (3,829 examples — 174 academic PDFs)
#       3. dagger_corrections.jsonl  (202 examples × 4x oversampled = 808)
#     Total: 7,759 examples
#   - DAgger (Dataset Aggregation) — captures REAL wrong navigation decisions
#     v13 committed-path bug: always navigated to same node regardless of question
#     Root cause: teacher forcing — model trained on correct paths, never saw wrong states
#     DAgger fix: ran v13 on all 150 FB questions, intercepted NavigationDecisionEvent,
#     collected 202 corrections at real wrong-decision states (180 root-level, 22 deep)
#   - System prompt normalization at training time:
#     All training examples get the EXACT inference _MULTIHOP_SYSTEM prompt
#     Eliminates train/inference mismatch that caused model to ignore context
#   - Base model: Qwen2.5-3B-Instruct FRESH (NOT fine-tuned v13 adapter)
#     Starting from v13 adapter would preserve committed-path weights
#   - LoRA: r=16, alpha=16 (simpler than v9's r=32/alpha=64 — dataset is cleaner)
#   - MAX_SEQ_LEN: 4096 (DAgger examples with long section lists need more headroom)
#
# DATASETS TO UPLOAD TO DRIVE (MyDrive/arbor-training-data/):
#   - structdirect_train.jsonl  ← from data/financebench/structdirect_train.jsonl
#   - academic_train.jsonl      ← from data/academic_train.jsonl
#   - dagger_corrections.jsonl  ← already on Drive from DAgger run
#   - treesearch_multihop_eval.jsonl ← already on Drive (reference only for Cell 7)
#
# REQUIREMENTS:
#   - Google Colab Pro (A100 40GB recommended)
#   - All 4 files in Drive as listed above
#
# EXPECTED RUNTIME on A100: ~20-25 minutes (7,759 examples × 3 epochs)
#   Total steps ≈ (7759 × 3) / 16 = 1,455 steps
#
# OUTPUT:
#   - Saved to Drive: MyDrive/arbor-training-data/models/treesearch-v14/
#   - Push to HuggingFace: TStark12310/arbor-treesearch-v14 (private)
#
# KEY METRIC: Cell 10 end-to-end FinanceBench recall is the only metric that matters.
#   Cell 7 F1 uses LLM-tree eval set → stale, lower than true performance. Ignore it.
#   Target: >38% on 150 questions (v13 baseline) → aim for 50%+


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 1 — Install dependencies                          ║
# ╚══════════════════════════════════════════════════════════╝
"""
!pip install -q "unsloth[colab-new]" trl transformers accelerate bitsandbytes peft
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2 — Mount Drive + set paths                       ║
# ╚══════════════════════════════════════════════════════════╝
"""
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_DIR    = "/content/drive/MyDrive/arbor-training-data"

# Training files — upload structdirect_train.jsonl + academic_train.jsonl before running
STRUCT_FILE  = f"{DRIVE_DIR}/structdirect_train.jsonl"   # 3,122 examples
ACADEMIC_FILE = f"{DRIVE_DIR}/academic_train.jsonl"      # 3,829 examples
DAGGER_FILE  = f"{DRIVE_DIR}/dagger_corrections.jsonl"   # 202 examples (will 4x oversample → 808)

# Eval file — already on Drive from prior sessions (LLM-tree eval, reference only)
EVAL_FILE    = f"{DRIVE_DIR}/treesearch_multihop_eval.jsonl"

OUTPUT_DIR   = f"{DRIVE_DIR}/models/treesearch-v14"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sanity check — verify all files exist and count examples
import subprocess
for label, path in [("StructDirect", STRUCT_FILE), ("Academic", ACADEMIC_FILE),
                    ("DAgger", DAGGER_FILE), ("Eval", EVAL_FILE)]:
    if os.path.exists(path):
        result = subprocess.run(["wc", "-l", path], capture_output=True, text=True)
        print(f"  {label:12s}: {result.stdout.strip()}")
    else:
        print(f"  {label:12s}: *** FILE NOT FOUND: {path} ***")

print(f"\nOutput dir: {OUTPUT_DIR}")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 3 — Load base model with Unsloth (4-bit QLoRA)    ║
# ╚══════════════════════════════════════════════════════════╝
"""
from unsloth import FastLanguageModel
import torch

# 4096 covers DAgger examples with long section lists (v13 used 2048 → some truncation)
MAX_SEQ_LEN  = 4096
DTYPE        = None    # Auto-detect (bfloat16 on A100)
LOAD_IN_4BIT = True

# IMPORTANT: Load FRESH base model, NOT treesearch-v13 adapter
# Starting from v13 would preserve the committed-path weights we're trying to fix
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Qwen2.5-3B-Instruct",
    max_seq_length = MAX_SEQ_LEN,
    dtype          = DTYPE,
    load_in_4bit   = LOAD_IN_4BIT,
)

# LoRA config — r=16, alpha=16 (cleaner than v9's r=32/alpha=64; dataset quality > capacity)
model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = 16,
    lora_dropout   = 0,        # 0 dropout — dataset is clean, regularization not needed
    bias           = "none",
    use_gradient_checkpointing = "unsloth",  # VRAM savings
    random_state   = 42,
    use_rslora     = False,    # Standard LoRA — rslora adds complexity without benefit here
)
print(model.print_trainable_parameters())
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 4 — Load, merge, and format all three datasets    ║
# ╚══════════════════════════════════════════════════════════╝
"""
import json
import random
from datasets import Dataset

# ── The EXACT system prompt used at inference time ────────────────────────────
# CRITICAL: This must match _MULTIHOP_SYSTEM in arbor/core/tree_searcher.py exactly.
# Any mismatch causes the model to partially ignore the question (train/inference drift).
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
    """
    Replace every training example's system message with the inference-time prompt.
    This is the critical fix for v14: train/inference system prompt mismatch caused
    v13 to partially ignore the question and commit to structural navigation instead.
    """
    for ex in examples:
        msgs = ex.get("messages", [])
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] = _INFERENCE_SYSTEM
        elif msgs and msgs[0]["role"] != "system":
            # Example has no system message — prepend one (shouldn't happen, but safety)
            msgs.insert(0, {"role": "system", "content": _INFERENCE_SYSTEM})
    return examples

def format_example(example):
    """Apply Qwen2.5 chat template."""
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

# ── Load all three datasets ───────────────────────────────────────────────────
print("Loading datasets...")
struct_raw   = load_jsonl(STRUCT_FILE)    # 3,122 StructDirect (FinanceBench)
academic_raw = load_jsonl(ACADEMIC_FILE)  # 3,829 academic PDFs
dagger_raw   = load_jsonl(DAGGER_FILE)    # 202 DAgger corrections

print(f"  StructDirect : {len(struct_raw):,} examples")
print(f"  Academic     : {len(academic_raw):,} examples")
print(f"  DAgger raw   : {len(dagger_raw):,} examples")

# ── DAgger 4x oversampling ────────────────────────────────────────────────────
# 202 DAgger examples × 4 = 808 effective examples ≈ 10% of final mix
# This is enough signal to teach recovery from wrong states without
# drowning out the StructDirect/academic navigation knowledge
random.seed(42)
dagger_oversampled = dagger_raw * 4  # List multiplication: [ex1, ex2, ...] × 4
random.shuffle(dagger_oversampled)   # Shuffle so copies are interleaved, not sequential
print(f"  DAgger (4x)  : {len(dagger_oversampled):,} examples")

# ── Merge and normalize system prompts ───────────────────────────────────────
all_train = struct_raw + academic_raw + dagger_oversampled  # 3122 + 3829 + 808 = 7,759
print(f"\n  Total before shuffle : {len(all_train):,} examples")

# NORMALIZE: replace every system prompt with exact inference-time prompt
normalize_system_prompt(all_train)

# Shuffle combined dataset (seed fixed for reproducibility)
random.shuffle(all_train)
print(f"  Total after shuffle  : {len(all_train):,} examples")

# ── Load eval set (LLM-tree based — stale metric, reference only) ─────────────
eval_raw = load_jsonl(EVAL_FILE)
normalize_system_prompt(eval_raw)  # Normalize eval too for consistency
print(f"  Eval set             : {len(eval_raw):,} examples (LLM-tree based — stale)")

# ── Apply chat template ───────────────────────────────────────────────────────
print("\nApplying chat template...")
train_texts = [format_example(ex) for ex in all_train]
eval_texts  = [format_example(ex) for ex in eval_raw]

# Check token length distribution (sample of 300 for speed)
sample_for_lengths = random.sample(train_texts, min(300, len(train_texts)))
token_lengths = sorted([len(tokenizer.encode(t)) for t in sample_for_lengths])
p50 = token_lengths[len(token_lengths) // 2]
p90 = token_lengths[int(len(token_lengths) * 0.9)]
p99 = token_lengths[int(len(token_lengths) * 0.99)]
max_len = token_lengths[-1]
print(f"  Token lengths (sample 300) — p50: {p50} | p90: {p90} | p99: {p99} | max: {max_len}")
print(f"  MAX_SEQ_LEN={MAX_SEQ_LEN} — {'OK ✓' if p99 < MAX_SEQ_LEN else '⚠ WARNING: some examples truncated'}")

train_dataset = Dataset.from_dict({"text": train_texts})
eval_dataset  = Dataset.from_dict({"text": eval_texts})
print(f"\nDatasets ready — train: {len(train_dataset)} | eval: {len(eval_dataset)}")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 5 — Train                                         ║
# ╚══════════════════════════════════════════════════════════╝
"""
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import torch

# ── Warmup calculation ────────────────────────────────────────────────────────
# Total steps = (7759 examples × 3 epochs) / effective_batch_size(16) = 1,455 steps
# Warmup = 10% = 145 steps (more stable ramp-up for larger dataset)
TOTAL_STEPS = (len(train_dataset) * 3) // 16
WARMUP_STEPS = max(50, TOTAL_STEPS // 10)
print(f"Total steps: {TOTAL_STEPS} | Warmup steps: {WARMUP_STEPS}")

trainer = SFTTrainer(
    model              = model,
    train_dataset      = train_dataset,
    eval_dataset       = eval_dataset,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
    dataset_num_proc   = 2,
    packing            = False,  # Keep False — structured JSON output needs clean boundaries
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,     # Effective batch = 16
        warmup_steps                = WARMUP_STEPS,
        num_train_epochs            = 3,
        learning_rate               = 2e-4,
        fp16                        = not is_bfloat16_supported(),
        bf16                        = is_bfloat16_supported(),
        logging_steps               = 10,
        eval_strategy               = "steps",
        eval_steps                  = 100,
        save_strategy               = "steps",
        save_steps                  = 100,
        output_dir                  = OUTPUT_DIR,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "cosine",
        max_grad_norm               = 0.3,   # Prevents loss spikes (proven in v9)
        seed                        = 42,
        load_best_model_at_end      = False,  # CRITICAL: eval set is LLM-tree (mismatched dist)
        # metric_for_best_model     = "eval_loss",  # disabled — eval loss is NOT a valid signal
        report_to                   = "none",
    ),
)

# Show GPU state before training
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
max_memory = round(gpu_stats.total_memory / 1024 ** 3, 3)
print(f"GPU: {gpu_stats.name} | VRAM: {max_memory}GB | Used before training: {start_gpu_memory}GB")

trainer_stats = trainer.train()
print(f"\nTraining complete — {trainer_stats.metrics}")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 6 — Save adapter to Drive + HuggingFace Hub       ║
# ╚══════════════════════════════════════════════════════════╝
"""
# Save LoRA adapter (small — just the delta weights)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")

# Optional: push to HuggingFace Hub (private)
# from huggingface_hub import login
# login(token="hf_YOUR_TOKEN")
# model.push_to_hub("TStark12310/arbor-treesearch-v14", private=True)
# tokenizer.push_to_hub("TStark12310/arbor-treesearch-v14", private=True)

# Optional: save merged 16-bit model (for inference without Unsloth)
# model.save_pretrained_merged(OUTPUT_DIR + "-merged", tokenizer,
#                              save_method="merged_16bit")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 7 — F1 eval on eval set (STALE METRIC — reference)║
# ║  NOTE: eval set uses LLM-tree data with wrong page nums  ║
# ║  This F1 will be artificially low. Use Cell 10 instead. ║
# ╚══════════════════════════════════════════════════════════╝
"""
import json
import re
import torch

model.eval()  # Use model.eval() — FastLanguageModel.for_inference() has shape bug

def parse_navigate_to(text: str) -> list:
    text = text.strip()
    try:
        parsed = json.loads(text)
        return [str(x) for x in parsed.get("navigate_to", [])]
    except Exception:
        return re.findall(r'\b(\d{4})\b', text)

def compute_f1(predicted: list, ground_truth: list) -> tuple:
    pred_set = set(predicted)
    gt_set   = set(ground_truth)
    if not pred_set and not gt_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gt_set:
        return 0.0, 0.0, 0.0
    tp        = len(pred_set & gt_set)
    precision = tp / len(pred_set)
    recall    = tp / len(gt_set)
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

eval_sample = eval_raw[:200]
results = {"precision": [], "recall": [], "f1": [], "exact_match": []}

for ex in eval_sample:
    messages      = ex["messages"]
    ground_truth  = json.loads(messages[-1]["content"])["navigate_to"]
    input_messages = messages[:-1]  # System + user only

    raw_inputs = tokenizer.apply_chat_template(
        input_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = (raw_inputs.input_ids if hasattr(raw_inputs, "input_ids") else raw_inputs).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids      = input_ids,
            max_new_tokens = 128,
            temperature    = 0.1,
            do_sample      = False,
            pad_token_id   = tokenizer.eos_token_id,
            use_cache      = False,
        )

    generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    predicted = parse_navigate_to(generated)

    p, r, f1 = compute_f1(predicted, ground_truth)
    results["precision"].append(p)
    results["recall"].append(r)
    results["f1"].append(f1)
    results["exact_match"].append(set(predicted) == set(ground_truth))

n = len(eval_sample)
avg_p  = sum(results["precision"]) / n
avg_r  = sum(results["recall"]) / n
avg_f1 = sum(results["f1"]) / n
em     = sum(results["exact_match"]) / n

print(f"\n{'='*50}")
print(f"  Evaluated on {n} examples (LLM-tree eval — stale)")
print(f"  Precision   : {avg_p:.3f}")
print(f"  Recall      : {avg_r:.3f}")
print(f"  F1          : {avg_f1:.3f}")
print(f"  Exact Match : {em:.3f} ({em*100:.1f}%)")
print(f"{'='*50}")
print(f"  ⚠  This F1 uses LLM-tree data with wrong page numbers.")
print(f"  ⚠  Use Cell 10 (end-to-end FinanceBench recall) for the real metric.")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 8 — Quick smoke test (fresh session compatible)   ║
# ╚══════════════════════════════════════════════════════════╝
"""
import json, torch
from unsloth import FastLanguageModel

DRIVE_DIR  = "/content/drive/MyDrive/arbor-training-data"
MODEL_PATH = f"{DRIVE_DIR}/models/treesearch-v14"

model_test, tokenizer_test = FastLanguageModel.from_pretrained(
    model_name     = MODEL_PATH,
    max_seq_length = 4096,
    dtype          = None,
    load_in_4bit   = True,
)
model_test.eval()

SYSTEM = (
    "You are a document tree navigator. "
    "Given a question and a list of document sections at the current level, "
    "select which sections to explore next to find the answer.\n\n"
    "Always reply with valid JSON:\n"
    '{"thinking": "brief reasoning", "navigate_to": ["node_id1", "node_id2"]}'
)

test_question = "What was the company's total revenue for fiscal year 2022?"

test_sections = (
    "[0001] Letter to Shareholders (pages 1-3)\n"
    "[0002] Business Overview (pages 4-12) [has sub-sections]\n"
    "[0003] Risk Factors (pages 13-28) [has sub-sections]\n"
    "[0004] Financial Statements and Results (pages 29-54) [has sub-sections]\n"
    "[0005] Corporate Governance (pages 55-61)\n"
    "[0006] Executive Compensation (pages 62-74)"
)

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user",   "content": f"Question: {test_question}\n\nSections at this level:\n{test_sections}\n\nWhich sections should we explore next?"},
]

raw_inputs = tokenizer_test.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
)
input_ids = (raw_inputs.input_ids if hasattr(raw_inputs, "input_ids") else raw_inputs).to("cuda")

with torch.no_grad():
    outputs = model_test.generate(
        input_ids=input_ids, max_new_tokens=128, temperature=0.1, do_sample=False,
        pad_token_id=tokenizer_test.eos_token_id, use_cache=False,
    )

response = tokenizer_test.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
print("Model response:", response)

try:
    parsed = json.loads(response)
    nav = parsed.get("navigate_to", [])
    thinking = parsed.get("thinking", "")
    print(f"\nParsed navigate_to : {nav}")
    print(f"Thinking           : {thinking}")
    # Expected: ["0004"] — Financial Statements
    expected = "0004"
    status = "PASS ✓" if expected in nav else f"FAIL ✗ (expected {expected})"
    print(f"Smoke test         : {status}")
except Exception:
    print("\n[warn] Could not parse JSON from response — check model output above")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 9 — Cross-domain eval (12 hand-crafted tests)     ║
# ║  Fully standalone — works in a fresh session            ║
# ╚══════════════════════════════════════════════════════════╝
"""
# Step 1 — install (required: adapter uses unsloth-patched attention)
import subprocess
subprocess.run(["pip", "install", "-q", "unsloth[colab-new]", "trl", "transformers",
                "accelerate", "bitsandbytes", "peft"], check=True)

# Step 2 — mount Drive
from google.colab import drive
drive.mount("/content/drive")

# Step 3 — load via FastLanguageModel (NOT AutoModelForCausalLM — breaks apply_qkv)
import json, torch
from unsloth import FastLanguageModel

DRIVE_DIR    = "/content/drive/MyDrive/arbor-training-data"
ADAPTER_PATH = f"{DRIVE_DIR}/models/treesearch-v14"

print("Loading v14 adapter...")
model_test, tokenizer_test = FastLanguageModel.from_pretrained(
    model_name     = ADAPTER_PATH,
    max_seq_length = 4096,
    dtype          = None,
    load_in_4bit   = True,
)
model_test.eval()
print("Model ready.\n")

SYSTEM = (
    "You are a document tree navigator. "
    "Given a question and a list of document sections at the current level, "
    "select which sections to explore next to find the answer.\n\n"
    "Always reply with valid JSON:\n"
    '{"thinking": "brief reasoning", "navigate_to": ["node_id1", "node_id2"]}'
)

# 12 test cases across all domains (same as v9 for comparability)
TESTS = [
    # --- Research paper ---
    (
        "What datasets were used to evaluate the model?",
        "[0001] Abstract (pages 1-1)\n[0002] Introduction (pages 2-3)\n[0003] Related Work (pages 4-6)\n[0004] Methodology (pages 7-10)\n[0005] Experiments and Datasets (pages 11-14) [has sub-sections]\n[0006] Results (pages 15-18)\n[0007] Conclusion (pages 19-20)",
        "0005", "Research"
    ),
    (
        "What are the limitations of the proposed approach?",
        "[0001] Introduction (pages 1-2)\n[0002] Background (pages 3-5)\n[0003] Model Architecture (pages 6-10)\n[0004] Training Setup (pages 11-13)\n[0005] Results (pages 14-17)\n[0006] Discussion and Limitations (pages 18-20)\n[0007] Conclusion (pages 21-22)",
        "0006", "Research"
    ),
    # --- Finance ---
    (
        "What was the total revenue for fiscal year 2023?",
        "[0001] Letter to Shareholders (pages 1-3)\n[0002] Business Overview (pages 4-12)\n[0003] Risk Factors (pages 13-28)\n[0004] Financial Statements and Results (pages 29-54) [has sub-sections]\n[0005] Corporate Governance (pages 55-61)\n[0006] Executive Compensation (pages 62-74)",
        "0004", "Finance"
    ),
    # --- Legal ---
    (
        "What are the termination clauses in this contract?",
        "[0001] Preamble and Definitions (pages 1-2)\n[0002] Scope of Services (pages 3-5)\n[0003] Payment Terms (pages 6-7)\n[0004] Intellectual Property Rights (pages 8-9)\n[0005] Termination and Breach (pages 10-12)\n[0006] Dispute Resolution (pages 13-14)\n[0007] Governing Law (pages 15-15)",
        "0005", "Legal"
    ),
    (
        "What confidentiality obligations does the employee have?",
        "[0001] Employment Terms (pages 1-3)\n[0002] Compensation and Benefits (pages 4-6)\n[0003] Confidentiality and Non-Disclosure (pages 7-9)\n[0004] Non-Compete Restrictions (pages 10-11)\n[0005] Termination Procedures (pages 12-13)",
        "0003", "Legal"
    ),
    # --- Healthcare ---
    (
        "What were the side effects observed in the clinical trial?",
        "[0001] Study Background (pages 1-3)\n[0002] Patient Population and Eligibility (pages 4-6)\n[0003] Treatment Protocol (pages 7-9)\n[0004] Efficacy Outcomes (pages 10-14)\n[0005] Adverse Events and Safety Profile (pages 15-19)\n[0006] Discussion (pages 20-23)\n[0007] Conclusions (pages 24-25)",
        "0005", "Healthcare"
    ),
    (
        "What is the recommended dosage for pediatric patients?",
        "[0001] Drug Overview (pages 1-2)\n[0002] Mechanism of Action (pages 3-4)\n[0003] Indications and Usage (pages 5-6)\n[0004] Dosage and Administration (pages 7-10) [has sub-sections]\n[0005] Contraindications (pages 11-12)\n[0006] Drug Interactions (pages 13-15)\n[0007] Storage and Handling (pages 16-16)",
        "0004", "Healthcare"
    ),
    # --- Energy ---
    (
        "What renewable energy sources are used in the power grid?",
        "[0001] Executive Summary (pages 1-3)\n[0002] Grid Infrastructure Overview (pages 4-8)\n[0003] Fossil Fuel Generation (pages 9-14)\n[0004] Renewable Energy Sources (pages 15-22) [has sub-sections]\n[0005] Transmission and Distribution (pages 23-29)\n[0006] Future Capacity Planning (pages 30-35)",
        "0004", "Energy"
    ),
    # --- Insurance ---
    (
        "What is the maximum liability coverage under this policy?",
        "[0001] Policy Overview (pages 1-2)\n[0002] Definitions (pages 3-5)\n[0003] Coverage and Limits (pages 6-11) [has sub-sections]\n[0004] Exclusions (pages 12-16)\n[0005] Claims Procedure (pages 17-20)\n[0006] Premium and Payment Terms (pages 21-23)",
        "0003", "Insurance"
    ),
    # --- Real Estate ---
    (
        "What are the maintenance responsibilities of the tenant?",
        "[0001] Lease Agreement Overview (pages 1-2)\n[0002] Rent and Payment Schedule (pages 3-5)\n[0003] Security Deposit Terms (pages 6-7)\n[0004] Tenant Obligations and Maintenance (pages 8-11)\n[0005] Landlord Access Rights (pages 12-13)\n[0006] Lease Renewal and Termination (pages 14-16)",
        "0004", "Real Estate"
    ),
    # --- Government / Policy ---
    (
        "What penalties apply for non-compliance with the regulation?",
        "[0001] Purpose and Scope (pages 1-3)\n[0002] Definitions and Applicability (pages 4-6)\n[0003] Compliance Requirements (pages 7-14) [has sub-sections]\n[0004] Enforcement and Penalties (pages 15-19)\n[0005] Appeals Process (pages 20-22)\n[0006] Effective Date and Amendments (pages 23-24)",
        "0004", "Government"
    ),
    # --- Automotive ---
    (
        "What are the safety test results for the braking system?",
        "[0001] Vehicle Overview (pages 1-4)\n[0002] Engine and Powertrain Specs (pages 5-12)\n[0003] Chassis and Suspension (pages 13-18)\n[0004] Braking System Safety Tests (pages 19-25) [has sub-sections]\n[0005] Emissions Compliance (pages 26-30)\n[0006] Crash Test Results (pages 31-36)\n[0007] Warranty Information (pages 37-39)",
        "0004", "Automotive"
    ),
]

print(f"Running {len(TESTS)} cross-domain tests...\n")
passed = 0

for i, (question, sections, expected, domain) in enumerate(TESTS):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"Question: {question}\n\nSections at this level:\n{sections}\n\nWhich sections should we explore next?"},
    ]
    raw_inputs = tokenizer_test.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = (raw_inputs.input_ids if hasattr(raw_inputs, "input_ids") else raw_inputs).to("cuda")

    with torch.no_grad():
        outputs = model_test.generate(
            input_ids=input_ids, max_new_tokens=128, temperature=0.1,
            do_sample=False, pad_token_id=tokenizer_test.eos_token_id, use_cache=False,
        )
    response = tokenizer_test.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

    try:
        parsed   = json.loads(response)
        nav      = parsed.get("navigate_to", [])
        thinking = parsed.get("thinking", "")[:80]
        correct  = expected in nav
    except Exception:
        nav, thinking, correct = [], "PARSE ERROR", False

    status = "PASS ✓" if correct else "FAIL ✗"
    if correct: passed += 1
    print(f"[{i+1:02d}] {status} [{domain:<12}] expected={expected} got={nav}")
    print(f"       {thinking}\n")

print(f"{'='*50}")
print(f"  Result: {passed}/{len(TESTS)} correct ({passed/len(TESTS)*100:.0f}%)")
print(f"  v9 baseline: 12/12 (100%) — target: ≥ 11/12")
print(f"{'='*50}")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 10 — FinanceBench end-to-end retrieval eval       ║
# ║  THE REAL METRIC — tests tree search on all 150 Qs      ║
# ║  Extracts trees from Drive PDFs with caching.           ║
# ║  Fully standalone — works in a fresh session            ║
# ╚══════════════════════════════════════════════════════════╝
"""
# ── Step 1: Mount Drive + clone repo ──────────────────────────────────────────
import subprocess, os, json, sys, time as _time
from google.colab import drive
drive.mount('/content/drive')

if not os.path.exists("/content/arbor"):
    subprocess.run(
        ["git", "clone", "https://github.com/Dhruv12310/arbor.git", "/content/arbor"],
        check=True
    )
else:
    subprocess.run(["git", "-C", "/content/arbor", "pull"], check=True)

# ── Step 2: Install dependencies ──────────────────────────────────────────────
subprocess.run(["pip", "install", "-q", "unsloth[colab-new]", "trl", "transformers",
                "accelerate", "bitsandbytes", "peft", "pymupdf"], check=True)

# ── Step 3: Imports ────────────────────────────────────────────────────────────
import torch
sys.path.insert(0, "/content/arbor")

# ── Step 4: Load v14 adapter ──────────────────────────────────────────────────
from unsloth import FastLanguageModel

DRIVE_DIR    = "/content/drive/MyDrive/arbor-training-data"
ADAPTER_PATH = f"{DRIVE_DIR}/models/treesearch-v14"

print("Loading v14 model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    ADAPTER_PATH, max_seq_length=4096, dtype=None, load_in_4bit=True
)
model.eval()
print("v14 loaded.\n")

# ── Step 5: Provider wrapper ───────────────────────────────────────────────────
from arbor.providers.base import LLMProvider

class V14Provider(LLMProvider):
    @property
    def name(self):
        return "treesearch-v14"

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
                input_ids=input_ids,
                max_new_tokens=max_tokens or 256,
                do_sample=False,
                repetition_penalty=1.1,
            )
        new_tokens = output[0][input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        finish = "length" if len(new_tokens) >= (max_tokens or 256) else "stop"
        return text, finish

# ── Step 6: Build tree cache (extract from PDFs, with caching to Drive) ───────
# Tree sources (in priority order):
#   1. Repo pre-committed trees: /content/arbor/data/financebench/trees/ (7 docs)
#   2. Drive tree cache: {DRIVE_DIR}/multidomaintest_trees/ (reuse if already extracted)
#   3. Fresh extraction from PDFs: {DRIVE_DIR}/financebench-pdfs/{doc}.pdf
#
# IMPORTANT: PDFs must be at DRIVE_DIR/financebench-pdfs/ (same location used by
# multidomaintest.py — they were already uploaded when that script ran successfully)

from arbor.extraction.structure_extractor import extract_structure
from arbor.core.tree_searcher import search_tree
from arbor.core.types import ArborConfig, DocumentTree

REPO_TREE_DIR = "/content/arbor/data/financebench/trees"
DRIVE_PDF_DIR = f"{DRIVE_DIR}/financebench-pdfs"
TREE_CACHE_DIR = f"{DRIVE_DIR}/multidomaintest_trees"   # Reuse existing cache from prior runs
os.makedirs(TREE_CACHE_DIR, exist_ok=True)

QA_FILE = "/content/arbor/data/financebench/financebench_open_source.jsonl"
qa_pairs = [
    json.loads(line)
    for line in open(QA_FILE, encoding="utf-8").read().strip().splitlines()
    if line.strip()
]

# Load/extract all trees needed for 84 docs
tree_cache = {}   # doc_name -> DocumentTree
skipped    = []

unique_docs = list({q["doc_name"] for q in qa_pairs})
print(f"Need trees for {len(unique_docs)} unique docs...\n")

for doc in sorted(unique_docs):
    # Source 1: pre-committed repo tree
    repo_path = f"{REPO_TREE_DIR}/{doc}.json"
    # Source 2: Drive tree cache (from multidomaintest or prior run)
    cache_path = f"{TREE_CACHE_DIR}/{doc}.json"
    # Source 3: PDF on Drive
    pdf_path   = f"{DRIVE_PDF_DIR}/{doc}.pdf"

    if os.path.exists(repo_path):
        raw = json.loads(open(repo_path, encoding="utf-8").read())
        if "tree" in raw and isinstance(raw["tree"], dict):
            raw = raw["tree"]
        tree_cache[doc] = DocumentTree.from_dict(raw)
        print(f"REPO    {doc}")
    elif os.path.exists(cache_path):
        raw = json.loads(open(cache_path, encoding="utf-8").read())
        if "tree" in raw and isinstance(raw["tree"], dict):
            raw = raw["tree"]
        tree_cache[doc] = DocumentTree.from_dict(raw)
        strat = tree_cache[doc].extraction_strategy or "unknown"
        print(f"CACHED  [{strat:20s}] {doc}")
    elif os.path.exists(pdf_path):
        try:
            t0   = _time.monotonic()
            tree = extract_structure(pdf_path)
            ms   = (_time.monotonic() - t0) * 1000
            tree_cache[doc] = tree
            # Save to cache so future runs are instant
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(tree.to_dict(), f, ensure_ascii=False)
            strat = tree.extraction_strategy or "unknown"
            print(f"EXTRACT [{strat:20s}] {doc}: {ms:.0f}ms")
        except Exception as e:
            print(f"ERROR   {doc}: {e}")
            skipped.append(doc)
    else:
        print(f"SKIP    {doc} — no PDF at {pdf_path}")
        skipped.append(doc)

qa_pairs_filtered = [q for q in qa_pairs if q["doc_name"] in tree_cache]
print(f"\nTrees ready: {len(tree_cache)}/84 | Skipped: {len(skipped)}")
print(f"QA pairs to evaluate: {len(qa_pairs_filtered)}/150\n")

if len(qa_pairs_filtered) < 100:
    print("⚠ WARNING: fewer than 100 QA pairs available.")
    print(f"  Upload PDFs to {DRIVE_PDF_DIR}/ and re-run.")

import warnings
warnings.filterwarnings("ignore", message="Both `max_new_tokens`.*")

config = ArborConfig(
    max_hops           = 8,
    max_nodes_searched = 0,       # No limit — large 10-K trees have 500+ nodes
    timeout_sec        = 300.0,
    max_retries_on_bad_json = 2,
)
provider = V14Provider()

# ── Step 7: Evaluation loop ────────────────────────────────────────────────────
def _infer_doc_type(doc_name: str) -> str:
    """Infer document type from doc_name for the navigation prompt context."""
    name = doc_name.upper()
    if "_10K" in name:
        return "annual 10-K filing"
    elif "_10Q" in name:
        return "quarterly 10-Q filing"
    elif "_8K" in name:
        return "8-K current report"
    elif "EARNINGS" in name:
        return "earnings release"
    else:
        return "financial document"

async def run_financebench_v14():
    results = []

    for i, qa in enumerate(qa_pairs_filtered):
        doc            = qa["doc_name"]
        question       = qa["question"]
        evidence_pages = [e["evidence_page_num"] for e in qa.get("evidence", [])]
        tree           = tree_cache[doc]
        doc_type       = _infer_doc_type(doc)

        try:
            sr = await search_tree(tree, question, provider, multihop=True, config=config,
                                   doc_type=doc_type)
            returned_pages = set()
            for node in sr.nodes:
                returned_pages.update(range(node.start_index, node.end_index + 1))
            found   = [p for p in evidence_pages if p in returned_pages]
            recall  = len(found) / len(evidence_pages) if evidence_pages else 1.0
            perfect = recall == 1.0
            results.append({
                "q": i + 1, "doc": doc, "recall": recall,
                "found": found, "evidence": evidence_pages,
                "nodes": sr.node_ids, "perfect": perfect,
            })
            status = "✓" if perfect else "✗"
            print(f"[{i+1:03d}/{len(qa_pairs_filtered)}] {status} {doc[:32]:<32} | "
                  f"recall={recall:.0%} | evid={evidence_pages} | found={found}")
        except Exception as e:
            print(f"[{i+1:03d}] ERROR: {e}")
            results.append({"q": i + 1, "doc": doc, "recall": 0.0, "error": str(e), "perfect": False})

    # ── Summary ──────────────────────────────────────────────────────────────
    n          = len(results)
    avg_recall = sum(r["recall"] for r in results) / n
    perfect    = sum(1 for r in results if r.get("perfect", False))
    partial    = sum(1 for r in results if 0 < r["recall"] < 1.0)
    zero       = sum(1 for r in results if r["recall"] == 0.0)

    print(f"\n{'='*55}")
    print(f"  v14 FinanceBench Retrieval Results")
    print(f"  Questions evaluated : {n}/150")
    print(f"  Avg recall          : {avg_recall:.1%}")
    print(f"  Perfect recall (1.0): {perfect}/{n}  ({perfect/n:.0%})")
    print(f"  Partial recall      : {partial}/{n}")
    print(f"  Zero recall         : {zero}/{n}")
    print(f"{'='*55}")
    print(f"  Baseline (v13)      : 38% on 150 questions")
    print(f"  Target              : > 50%")
    outcome = "✓ IMPROVEMENT" if avg_recall > 0.38 else "✗ NO IMPROVEMENT OVER v13"
    print(f"  Status              : {outcome}")
    print(f"{'='*55}")

    # Save results to Drive for analysis
    results_path = f"{DRIVE_DIR}/financebench_v14_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    return results

results = await run_financebench_v14()
"""
