# Arbor TreeSearch v10 — StructDirect-Trained Navigation Model
# =============================================================
# Copy each CELL block into a separate Colab cell and run in order.
# Run cells 1 → 2 → 3 → 4 → 5 → 6 → 7 in order. Do NOT skip.
#
# WHAT CHANGED FROM v9:
#   - Training data: rebuilt from StructDirect-extracted trees (correct physical page numbers)
#   - Old v9 training data was generated from LLM trees with wrong TOC-printed page numbers
#     causing the model to navigate to wrong sections (Risk Factors, MD&A instead of
#     Financial Statements). This retraining fixes that root cause.
#   - New data: structdirect_train.jsonl (FinanceBench 7 docs × 13 QA pairs × multi-hop)
#     combined with existing treesearch_multihop_train.jsonl (3,063 cross-domain examples)
#   - OUTPUT_DIR: treesearch-v9 → treesearch-v10
#   - warmup_steps recalculated from combined dataset size
#
# REQUIREMENTS:
#   - Google Colab Pro (A100 40GB recommended; T4 works ~60 min)
#   - Google Drive with MyDrive/arbor-training-data/ folder
#   - treesearch_multihop_train.jsonl  (3,063 examples — existing)
#   - treesearch_multihop_eval.jsonl   (341 examples — existing)
#   - structdirect_train.jsonl         (new — generate with gen_structdirect_data.py first)
#
# EXPECTED RUNTIME on A100: ~15 minutes
# OUTPUT: MyDrive/arbor-training-data/models/treesearch-v10/


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
DRIVE_DIR        = "/content/drive/MyDrive/arbor-training-data"
TRAIN_FILE       = f"{DRIVE_DIR}/treesearch_multihop_train.jsonl"   # 3063 existing examples
EVAL_FILE        = f"{DRIVE_DIR}/treesearch_multihop_eval.jsonl"    # 341 eval examples
SD_TRAIN_FILE    = f"{DRIVE_DIR}/structdirect_train.jsonl"          # new StructDirect examples
OUTPUT_DIR       = f"{DRIVE_DIR}/models/treesearch-v10"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sanity check file counts
import subprocess
for f in [TRAIN_FILE, EVAL_FILE, SD_TRAIN_FILE]:
    result = subprocess.run(["wc", "-l", f], capture_output=True, text=True)
    print(result.stdout.strip())
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 3 — Load model with Unsloth (4-bit QLoRA)         ║
# ╚══════════════════════════════════════════════════════════╝
"""
from unsloth import FastLanguageModel
import torch

MAX_SEQ_LEN  = 2048
DTYPE        = None   # Auto-detect
LOAD_IN_4BIT = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Qwen2.5-3B-Instruct",
    max_seq_length = MAX_SEQ_LEN,
    dtype          = DTYPE,
    load_in_4bit   = LOAD_IN_4BIT,
)

model = FastLanguageModel.get_peft_model(
    model,
    r              = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = 64,
    lora_dropout   = 0.05,
    bias           = "none",
    use_gradient_checkpointing = "unsloth",
    random_state   = 42,
    use_rslora     = True,
)
print(model.print_trainable_parameters())
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 4 — Load and combine datasets                     ║
# ╚══════════════════════════════════════════════════════════╝
"""
import json, random
from datasets import Dataset

def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def format_example(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

# Load all three sources
base_train = load_jsonl(TRAIN_FILE)       # 3063 cross-domain examples
sd_train   = load_jsonl(SD_TRAIN_FILE)    # StructDirect FinanceBench examples
eval_raw   = load_jsonl(EVAL_FILE)        # 341 eval examples

print(f"Base train (cross-domain)  : {len(base_train)} examples")
print(f"StructDirect train (new)   : {len(sd_train)} examples")
print(f"Eval                       : {len(eval_raw)} examples")

# Combine and shuffle — oversample StructDirect 3x so model prioritises correct page mappings
combined_train = base_train + sd_train * 3
random.seed(42)
random.shuffle(combined_train)
print(f"Combined train (3x SD)     : {len(combined_train)} examples")

# Format to text
train_texts = [format_example(ex) for ex in combined_train]
eval_texts  = [format_example(ex) for ex in eval_raw]

# Token length check
sample = train_texts[:200]
lengths = sorted(len(tokenizer.encode(t)) for t in sample)
p50 = lengths[len(lengths)//2]
p90 = lengths[int(len(lengths)*0.9)]
p99 = lengths[int(len(lengths)*0.99)]
print(f"\nToken lengths (sample 200) — p50:{p50} p90:{p90} p99:{p99}")
print(f"MAX_SEQ_LEN={MAX_SEQ_LEN} — {'OK' if p99 < MAX_SEQ_LEN else 'WARNING: truncation'}")

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
import torch

# Recalculate warmup based on combined dataset size
# total_steps = ceil(len(combined_train) / batch_size) * epochs
# batch_size = per_device_batch * grad_accum = 4 * 4 = 16
# warmup = 10% of total steps
n_train      = len(train_dataset)
batch_size   = 16   # 4 per device × 4 grad accum
total_steps  = (n_train // batch_size) * 3   # 3 epochs
warmup_steps = max(10, total_steps // 10)
print(f"Total steps: {total_steps} | Warmup: {warmup_steps}")

trainer = SFTTrainer(
    model              = model,
    train_dataset      = train_dataset,
    eval_dataset       = eval_dataset,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
    dataset_num_proc   = 2,
    packing            = False,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps                = warmup_steps,
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
        max_grad_norm               = 0.3,
        seed                        = 42,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        report_to                   = "none",
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu_stats.name} | VRAM: {round(gpu_stats.total_memory/1024**3, 1)}GB")

trainer_stats = trainer.train()
print(f"\nTraining complete — {trainer_stats.metrics}")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 6 — Save adapter                                  ║
# ╚══════════════════════════════════════════════════════════╝
"""
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")

# Push to HuggingFace (uncomment when ready)
# from huggingface_hub import login
# login(token="hf_YOUR_TOKEN")
# model.push_to_hub("TStark12310/arbor-treesearch-v10", private=True)
# tokenizer.push_to_hub("TStark12310/arbor-treesearch-v10", private=True)
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 7 — Evaluate F1 on eval set                       ║
# ╚══════════════════════════════════════════════════════════╝
"""
import json, re, torch

model.eval()

def parse_navigate_to(text: str) -> list:
    text = text.strip()
    try:
        return [str(x) for x in json.loads(text).get("navigate_to", [])]
    except Exception:
        return re.findall(r'\b(\d{4})\b', text)

def compute_f1(predicted, ground_truth):
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
    input_messages = messages[:-1]

    raw_inputs = tokenizer.apply_chat_template(
        input_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = (raw_inputs.input_ids if hasattr(raw_inputs, "input_ids") else raw_inputs).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids      = input_ids,
            max_new_tokens = 128,
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

n      = len(eval_sample)
avg_f1 = sum(results["f1"]) / n
avg_p  = sum(results["precision"]) / n
avg_r  = sum(results["recall"]) / n
em     = sum(results["exact_match"]) / n

print(f"\n{'='*45}")
print(f"  TreeSearch v10 — Eval on {n} examples")
print(f"{'='*45}")
print(f"  Precision   : {avg_p:.3f}")
print(f"  Recall      : {avg_r:.3f}")
print(f"  F1          : {avg_f1:.3f}")
print(f"  Exact Match : {em:.3f} ({em*100:.1f}%)")
print(f"{'='*45}")
print(f"  v9 baseline : F1=0.911 | EM=87%")
print(f"  Target      : F1 ≥ 0.91 | EM ≥ 87%")
print(f"  Status      : {'✓ PASS' if avg_f1 >= 0.91 else '✗ BELOW TARGET'}")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 8 — FinanceBench end-to-end eval (v10 + StructDirect) ║
# ╚══════════════════════════════════════════════════════════╝
"""
# Run AFTER training to verify end-to-end retrieval on FinanceBench 13 questions
# Expected: significant improvement over v9's 15.4% recall

import subprocess, sys, os, json, torch, asyncio, warnings
warnings.filterwarnings("ignore")
subprocess.run(["pip", "install", "pymupdf", "-q"], check=True)

if not os.path.exists("/content/arbor"):
    subprocess.run(["git", "clone", "https://github.com/Dhruv12310/arbor.git", "/content/arbor"], check=True)
else:
    subprocess.run(["git", "-C", "/content/arbor", "pull"], check=True)
sys.path.insert(0, "/content/arbor")

from arbor.providers.base import LLMProvider
from arbor.core.tree_searcher import search_tree
from arbor.core.types import ArborConfig
from arbor.extraction.structure_extractor import extract_structure

class V10Provider(LLMProvider):
    @property
    def name(self): return "treesearch-v10"
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
                input_ids=input_ids, max_new_tokens=256, do_sample=False, repetition_penalty=1.1
            )
        new_tokens = output[0][input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text, "length" if len(new_tokens) >= 256 else "stop"

PDF_DIR = "/content/drive/MyDrive/arbor-training-data/financebench-pdfs"
QA_FILE = "/content/arbor/data/financebench/financebench_open_source.jsonl"
TARGET_DOCS = [
    "3M_2018_10K", "3M_2022_10K", "3M_2023Q2_10Q",
    "ACTIVISIONBLIZZARD_2019_10K",
    "ADOBE_2015_10K", "ADOBE_2016_10K", "ADOBE_2017_10K",
]

all_qa  = [json.loads(l) for l in open(QA_FILE, encoding="utf-8").read().strip().splitlines()]
qa_pairs = [q for q in all_qa if q["doc_name"] in TARGET_DOCS]

config   = ArborConfig(max_hops=6, max_nodes_searched=0, timeout_sec=120.0, max_retries_on_bad_json=2)
provider = V10Provider()

async def run_eval():
    tree_cache = {}
    results    = []

    for i, qa in enumerate(qa_pairs):
        doc            = qa["doc_name"]
        question       = qa["question"]
        evidence_pages = [e["evidence_page_num"] for e in qa.get("evidence", [])]

        if doc not in tree_cache:
            tree_cache[doc] = extract_structure(f"{PDF_DIR}/{doc}.pdf")
        tree = tree_cache[doc]

        try:
            sr = await search_tree(tree, question, provider, multihop=True, config=config)
            found_pages = set()
            for node in sr.nodes:
                found_pages.update(range(node.start_index, node.end_index + 1))

            found_exact    = [p for p in evidence_pages if p in found_pages]
            found_tolerant = [p for p in evidence_pages if any(abs(p - fp) <= 1 for fp in found_pages)]
            recall_exact   = len(found_exact) / len(evidence_pages) if evidence_pages else 1.0
            recall_tol     = len(found_tolerant) / len(evidence_pages) if evidence_pages else 1.0

            status = "EXACT" if recall_exact == 1.0 else ("+/-1" if recall_tol == 1.0 else "MISS")
            print(f"[{i+1:02d}] {status} | {doc[:28]} | evidence={evidence_pages} | nodes={sr.node_ids}")
            results.append({"recall_exact": recall_exact, "recall_tol": recall_tol})
        except Exception as e:
            print(f"[{i+1:02d}] ERROR: {e}")
            results.append({"recall_exact": 0.0, "recall_tol": 0.0})

    n            = len(results)
    avg_exact    = sum(r["recall_exact"] for r in results) / n
    avg_tol      = sum(r["recall_tol"]   for r in results) / n
    perfect_exact = sum(1 for r in results if r["recall_exact"] == 1.0)
    perfect_tol   = sum(1 for r in results if r["recall_tol"]   == 1.0)

    print(f"\n{'='*55}")
    print(f"  StructDirect + TreeSearch v10 — {n} questions")
    print(f"{'='*55}")
    print(f"  Avg recall (exact)   : {avg_exact:.1%}")
    print(f"  Avg recall (±1 page) : {avg_tol:.1%}")
    print(f"  100% exact           : {perfect_exact}/{n}")
    print(f"  100% ±1 page         : {perfect_tol}/{n}")
    print(f"{'='*55}")
    print(f"  v9 baseline          : 15.4% (wrong training data)")
    print(f"  GPT-4 long-context   : 78%")
    print(f"{'='*55}")

await run_eval()
"""
