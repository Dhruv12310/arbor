# Arbor TreeSearch v7 — Multi-hop Navigation Fine-tune
# ======================================================
# Copy each CELL block into a separate Colab cell and run in order.
#
# WHY v7 IS DIFFERENT FROM v1-v6:
#   - v1-v6: one-shot (full 3000-node tree → pick answer nodes) → 52% truncation, F1=0.23
#   - v7: multi-hop (question + 10-20 nodes at current level → which to explore next)
#         → zero truncation, simple task, expected F1=0.70-0.80
#
# REQUIREMENTS:
#   - Google Colab (free T4 works, A100 preferred)
#   - Google Drive mounted at /content/drive/MyDrive/arbor/
#   - data/finetune/treesearch_multihop_train.jsonl (from build_multihop_dataset.py)
#   - data/finetune/treesearch_multihop_eval.jsonl
#
# EXPECTED RUNTIME:
#   - T4 (free): ~90 minutes for ~2500 examples × 3 epochs
#   - A100 (Pro): ~25 minutes
#
# OUTPUT:
#   - Saved to Google Drive: arbor/models/treesearch-v7/
#   - Optionally pushed to HuggingFace Hub


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
DRIVE_DIR = "/content/drive/MyDrive/arbor-training-data"
TRAIN_FILE = f"{DRIVE_DIR}/treesearch_multihop_train.jsonl"
EVAL_FILE  = f"{DRIVE_DIR}/treesearch_multihop_eval.jsonl"
OUTPUT_DIR = f"{DRIVE_DIR}/models/treesearch-v8"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sanity check
import subprocess
result = subprocess.run(["wc", "-l", TRAIN_FILE, EVAL_FILE], capture_output=True, text=True)
print(result.stdout)
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 3 — Load model with Unsloth (4-bit QLoRA)         ║
# ╚══════════════════════════════════════════════════════════╝
"""
from unsloth import FastLanguageModel
import torch

MAX_SEQ_LEN = 2048  # p90=384 tokens, but max=4273 from wide FinanceBench windows. 2048 covers 99%+.
DTYPE       = None  # Auto-detect
LOAD_IN_4BIT = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name   = "unsloth/Qwen2.5-3B-Instruct",  # Fresh base — NOT the broken v1
    max_seq_length = MAX_SEQ_LEN,
    dtype          = DTYPE,
    load_in_4bit   = LOAD_IN_4BIT,
)

# LoRA config
model = FastLanguageModel.get_peft_model(
    model,
    r              = 32,       # Higher rank than v1 (was 16) — more capacity
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = 64,       # 2× rank — standard scaling
    lora_dropout   = 0.05,
    bias           = "none",
    use_gradient_checkpointing = "unsloth",  # Saves VRAM on T4
    random_state   = 42,
    use_rslora     = True,     # Rank-stabilized LoRA — better convergence
)
print(model.print_trainable_parameters())
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 4 — Load and format dataset                       ║
# ╚══════════════════════════════════════════════════════════╝
"""
import json
from datasets import Dataset

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def format_example(example):
    '''Apply Qwen2.5 chat template to each example.'''
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
# Load
train_raw = load_jsonl(TRAIN_FILE)
eval_raw  = load_jsonl(EVAL_FILE)
print(f"Train: {len(train_raw)} | Eval: {len(eval_raw)}")

# Format
train_texts = [format_example(ex) for ex in train_raw]
eval_texts  = [format_example(ex) for ex in eval_raw]

# Check token length distribution
token_lengths = [len(tokenizer.encode(t)) for t in train_texts[:200]]
token_lengths.sort()
p50 = token_lengths[len(token_lengths)//2]
p90 = token_lengths[int(len(token_lengths)*0.9)]
p99 = token_lengths[int(len(token_lengths)*0.99)]
print(f"Token lengths (sample of 200) — p50: {p50}, p90: {p90}, p99: {p99}")
print(f"MAX_SEQ_LEN={MAX_SEQ_LEN} — {'OK' if p99 < MAX_SEQ_LEN else 'WARNING: some examples will be truncated'}")

train_dataset = Dataset.from_dict({"text": train_texts})
eval_dataset  = Dataset.from_dict({"text": eval_texts})
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 5 — Train                                         ║
# ╚══════════════════════════════════════════════════════════╝
"""
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model              = model,
    train_dataset      = train_dataset,
    eval_dataset       = eval_dataset,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
    dataset_num_proc   = 2,
    packing            = False,  # Keep False — structured output needs clean boundaries
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,   # Effective batch = 16
        warmup_steps                = 36,  # ~5% of ~712 total steps (3800 examples, 3 epochs, bs=16)
        num_train_epochs            = 3,
        learning_rate               = 2e-4,
        fp16                        = not is_bfloat16_supported(),
        bf16                        = is_bfloat16_supported(),
        logging_steps               = 10,
        eval_strategy               = "steps",
        eval_steps                  = 50,
        save_strategy               = "steps",
        save_steps                  = 50,
        output_dir                  = OUTPUT_DIR,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "cosine",
        seed                        = 42,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        report_to                   = "none",
    ),
)

# Show starting GPU memory
import torch
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
max_memory = round(gpu_stats.total_memory / 1024**3, 3)
print(f"GPU: {gpu_stats.name} | VRAM: {max_memory}GB | Used: {start_gpu_memory}GB")

trainer_stats = trainer.train()
print(f"\\nTraining complete — {trainer_stats.metrics}")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 6 — Save and push to HuggingFace Hub              ║
# ╚══════════════════════════════════════════════════════════╝
"""
# Save adapter (LoRA weights only — small file)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")

# Optional: push to HuggingFace Hub
# from huggingface_hub import login
# login(token="hf_YOUR_TOKEN")
# model.push_to_hub("TStark12310/arbor-treesearch-v8", private=True)
# tokenizer.push_to_hub("TStark12310/arbor-treesearch-v8", private=True)

# Optional: save merged 16-bit model (for inference without Unsloth)
# model.save_pretrained_merged(OUTPUT_DIR + "-merged", tokenizer,
#                              save_method="merged_16bit")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 7 — Evaluate (F1 on eval set)                     ║
# ╚══════════════════════════════════════════════════════════╝
"""
import json
import re
import torch

model.eval()  # Use model.eval() — FastLanguageModel.for_inference() has shape bug in current Unsloth

def parse_navigate_to(text: str) -> list[str]:
    '''Extract navigate_to list from model output.'''
    text = text.strip()
    try:
        parsed = json.loads(text)
        return [str(x) for x in parsed.get("navigate_to", [])]
    except Exception:
        # Fallback: find node ID patterns like "0003"
        return re.findall(r'\\b(\\d{4})\\b', text)

def compute_f1(predicted: list, ground_truth: list) -> tuple[float, float, float]:
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

# Evaluate on up to 200 examples
eval_sample = eval_raw[:200]
results = {"precision": [], "recall": [], "f1": [], "exact_match": []}

for ex in eval_sample:
    messages      = ex["messages"]
    ground_truth  = json.loads(messages[-1]["content"])["navigate_to"]
    input_messages = messages[:-1]  # System + user only

    # Tokenize input — handle Transformers 5.x returning BatchEncoding
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

    generated = tokenizer.decode(
        outputs[0][input_ids.shape[1]:], skip_special_tokens=True
    )
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

print(f"\\n{'='*45}")
print(f"  Evaluated on {n} examples")
print(f"  Precision   : {avg_p:.3f}")
print(f"  Recall      : {avg_r:.3f}")
print(f"  F1          : {avg_f1:.3f}")
print(f"  Exact Match : {em:.3f} ({em*100:.1f}%)")
print(f"{'='*45}")
print(f"\\n  Target: F1 ≥ 0.88 | EM ≥ 0.82")
print(f"  Status: {'✓ PASS' if avg_f1 >= 0.88 else '✗ BELOW TARGET'}")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 8 — Standalone test (works in a fresh notebook)   ║
# ╚══════════════════════════════════════════════════════════╝
"""
import json, torch
from unsloth import FastLanguageModel

# Load the saved v8 adapter from Drive
DRIVE_DIR  = "/content/drive/MyDrive/arbor-training-data"
MODEL_PATH = f"{DRIVE_DIR}/models/treesearch-v8"

model_test, tokenizer_test = FastLanguageModel.from_pretrained(
    model_name     = MODEL_PATH,
    max_seq_length = 2048,
    dtype          = None,
    load_in_4bit   = True,
)
model_test.eval()

SYSTEM = (
    "You are a document tree navigator. "
    "Given a question and a list of document sections at the current level, "
    "select which sections to explore next to find the answer.\\n\\n"
    "Always reply with valid JSON:\\n"
    '{"thinking": "brief reasoning", "navigate_to": ["node_id1", "node_id2"]}'
)

test_question = "What is the main methodology proposed in this paper?"

test_sections = (
    "[0001] Introduction (pages 1-2) [has sub-sections]\\n"
    "[0002] Related Work (pages 3-5) [has sub-sections]\\n"
    "[0003] Methodology (pages 6-12) [has sub-sections]\\n"
    "[0004] Experimental Setup (pages 13-16)\\n"
    "[0005] Results and Analysis (pages 17-22) [has sub-sections]\\n"
    "[0006] Discussion (pages 23-24)\\n"
    "[0007] Conclusion (pages 25-26)"
)

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user",   "content": f"Question: {test_question}\\n\\nSections at this level:\\n{test_sections}\\n\\nWhich sections should we explore next?"},
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
    print("\\nParsed navigate_to:", parsed.get("navigate_to"))
    print("Thinking:          ", parsed.get("thinking"))
except Exception:
    print("\\n[warn] Could not parse JSON from response")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 9 — Cross-domain evaluation (12 hand-crafted)     ║
# ║  Fully standalone — works in a fresh session            ║
# ╚══════════════════════════════════════════════════════════╝
"""
# Step 1 — install (skip if already done)
import subprocess
subprocess.run(["pip", "install", "-q", "peft", "transformers", "accelerate", "bitsandbytes"], check=True)

# Step 2 — mount Drive
from google.colab import drive
drive.mount("/content/drive")

# Step 3 — load model
import json, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DRIVE_DIR    = "/content/drive/MyDrive/arbor-training-data"
ADAPTER_PATH = f"{DRIVE_DIR}/models/treesearch-v8"

print("Loading base model...")
tokenizer_test = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-3B-Instruct")
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-3B-Instruct", torch_dtype=torch.float16, device_map="cuda"
)
print("Applying LoRA adapter...")
model_test = PeftModel.from_pretrained(base_model, ADAPTER_PATH, local_files_only=True)
model_test.eval()
print("Model ready.\\n")

SYSTEM = (
    "You are a document tree navigator. "
    "Given a question and a list of document sections at the current level, "
    "select which sections to explore next to find the answer.\\n\\n"
    "Always reply with valid JSON:\\n"
    '{"thinking": "brief reasoning", "navigate_to": ["node_id1", "node_id2"]}'
)

# 12 test cases across all domains — (question, sections, expected_node_id, domain)
TESTS = [
    # --- Research paper ---
    (
        "What datasets were used to evaluate the model?",
        "[0001] Abstract (pages 1-1)\\n[0002] Introduction (pages 2-3)\\n[0003] Related Work (pages 4-6)\\n[0004] Methodology (pages 7-10)\\n[0005] Experiments and Datasets (pages 11-14) [has sub-sections]\\n[0006] Results (pages 15-18)\\n[0007] Conclusion (pages 19-20)",
        "0005", "Research"
    ),
    (
        "What are the limitations of the proposed approach?",
        "[0001] Introduction (pages 1-2)\\n[0002] Background (pages 3-5)\\n[0003] Model Architecture (pages 6-10)\\n[0004] Training Setup (pages 11-13)\\n[0005] Results (pages 14-17)\\n[0006] Discussion and Limitations (pages 18-20)\\n[0007] Conclusion (pages 21-22)",
        "0006", "Research"
    ),
    # --- Finance ---
    (
        "What was the total revenue for fiscal year 2023?",
        "[0001] Letter to Shareholders (pages 1-3)\\n[0002] Business Overview (pages 4-12)\\n[0003] Risk Factors (pages 13-28)\\n[0004] Financial Statements and Results (pages 29-54) [has sub-sections]\\n[0005] Corporate Governance (pages 55-61)\\n[0006] Executive Compensation (pages 62-74)",
        "0004", "Finance"
    ),
    # --- Legal ---
    (
        "What are the termination clauses in this contract?",
        "[0001] Preamble and Definitions (pages 1-2)\\n[0002] Scope of Services (pages 3-5)\\n[0003] Payment Terms (pages 6-7)\\n[0004] Intellectual Property Rights (pages 8-9)\\n[0005] Termination and Breach (pages 10-12)\\n[0006] Dispute Resolution (pages 13-14)\\n[0007] Governing Law (pages 15-15)",
        "0005", "Legal"
    ),
    (
        "What confidentiality obligations does the employee have?",
        "[0001] Employment Terms (pages 1-3)\\n[0002] Compensation and Benefits (pages 4-6)\\n[0003] Confidentiality and Non-Disclosure (pages 7-9)\\n[0004] Non-Compete Restrictions (pages 10-11)\\n[0005] Termination Procedures (pages 12-13)",
        "0003", "Legal"
    ),
    # --- Healthcare ---
    (
        "What were the side effects observed in the clinical trial?",
        "[0001] Study Background (pages 1-3)\\n[0002] Patient Population and Eligibility (pages 4-6)\\n[0003] Treatment Protocol (pages 7-9)\\n[0004] Efficacy Outcomes (pages 10-14)\\n[0005] Adverse Events and Safety Profile (pages 15-19)\\n[0006] Discussion (pages 20-23)\\n[0007] Conclusions (pages 24-25)",
        "0005", "Healthcare"
    ),
    (
        "What is the recommended dosage for pediatric patients?",
        "[0001] Drug Overview (pages 1-2)\\n[0002] Mechanism of Action (pages 3-4)\\n[0003] Indications and Usage (pages 5-6)\\n[0004] Dosage and Administration (pages 7-10) [has sub-sections]\\n[0005] Contraindications (pages 11-12)\\n[0006] Drug Interactions (pages 13-15)\\n[0007] Storage and Handling (pages 16-16)",
        "0004", "Healthcare"
    ),
    # --- Energy ---
    (
        "What renewable energy sources are used in the power grid?",
        "[0001] Executive Summary (pages 1-3)\\n[0002] Grid Infrastructure Overview (pages 4-8)\\n[0003] Fossil Fuel Generation (pages 9-14)\\n[0004] Renewable Energy Sources (pages 15-22) [has sub-sections]\\n[0005] Transmission and Distribution (pages 23-29)\\n[0006] Future Capacity Planning (pages 30-35)",
        "0004", "Energy"
    ),
    # --- Insurance ---
    (
        "What is the maximum liability coverage under this policy?",
        "[0001] Policy Overview (pages 1-2)\\n[0002] Definitions (pages 3-5)\\n[0003] Coverage and Limits (pages 6-11) [has sub-sections]\\n[0004] Exclusions (pages 12-16)\\n[0005] Claims Procedure (pages 17-20)\\n[0006] Premium and Payment Terms (pages 21-23)",
        "0003", "Insurance"
    ),
    # --- Real Estate ---
    (
        "What are the maintenance responsibilities of the tenant?",
        "[0001] Lease Agreement Overview (pages 1-2)\\n[0002] Rent and Payment Schedule (pages 3-5)\\n[0003] Security Deposit Terms (pages 6-7)\\n[0004] Tenant Obligations and Maintenance (pages 8-11)\\n[0005] Landlord Access Rights (pages 12-13)\\n[0006] Lease Renewal and Termination (pages 14-16)",
        "0004", "Real Estate"
    ),
    # --- Government / Policy ---
    (
        "What penalties apply for non-compliance with the regulation?",
        "[0001] Purpose and Scope (pages 1-3)\\n[0002] Definitions and Applicability (pages 4-6)\\n[0003] Compliance Requirements (pages 7-14) [has sub-sections]\\n[0004] Enforcement and Penalties (pages 15-19)\\n[0005] Appeals Process (pages 20-22)\\n[0006] Effective Date and Amendments (pages 23-24)",
        "0004", "Government"
    ),
    # --- Automotive ---
    (
        "What are the safety test results for the braking system?",
        "[0001] Vehicle Overview (pages 1-4)\\n[0002] Engine and Powertrain Specs (pages 5-12)\\n[0003] Chassis and Suspension (pages 13-18)\\n[0004] Braking System Safety Tests (pages 19-25) [has sub-sections]\\n[0005] Emissions Compliance (pages 26-30)\\n[0006] Crash Test Results (pages 31-36)\\n[0007] Warranty Information (pages 37-39)",
        "0004", "Automotive"
    ),
]

print(f"Running {len(TESTS)} cross-domain tests...\\n")
passed = 0

for i, (question, sections, expected, domain) in enumerate(TESTS):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"Question: {question}\\n\\nSections at this level:\\n{sections}\\n\\nWhich sections should we explore next?"},
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
        parsed    = json.loads(response)
        nav       = parsed.get("navigate_to", [])
        thinking  = parsed.get("thinking", "")[:80]
        correct   = expected in nav
    except Exception:
        nav, thinking, correct = [], "PARSE ERROR", False

    status = "PASS" if correct else "FAIL"
    if correct: passed += 1
    print(f"[{i+1:02d}] {status} [{domain:<12}] expected={expected} got={nav}")
    print(f"       {thinking}\\n")

print(f"{'='*50}")
print(f"  Result: {passed}/{len(TESTS)} correct ({passed/len(TESTS)*100:.0f}%)")
print(f"{'='*50}")
"""
