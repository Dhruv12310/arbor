# Arbor TreeSearch v16 — Fixed StructDirect Foundation + Regenerated Training Data
# =========================================================================
# Copy each CELL block into a separate Colab cell and run in order.
# Run cells 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 in order.
#
# WHAT CHANGED FROM v15:
#   1. StructDirect foundation bug FIXED:
#      - _try_multiline_toc() previously deduped TOC entries by PAGE NUMBER alone
#      - Items 3, 4, 5 (short sections sharing a page) were silently dropped from
#        ALL 49 multiline_toc 10-K filings
#      - Fix: dedup on (title, page) pair → all sections preserved
#      - Impact: +716 sections across 59 FinanceBench docs (Apple 10-K: 12 → 22 sections)
#
#   2. ALL training data regenerated from fixed trees:
#      - structdirect_train.jsonl: 3,287 examples (was 3,122) — +165 new
#        Multi-node examples: 1,316/3,287 (40%) vs near 0% before
#      - dagger_v15_targeted.jsonl: 139 examples (same count, improved quality)
#        Multi-evidence merging: paths to Items 3,4,5 now valid
#      - v14b_success_replay.jsonl: 106 examples (same count, improved quality)
#
#   3. Training mix (v16):
#      - structdirect_train.jsonl:    3,287  (base navigation, ~60%)
#      - dagger_corrections.jsonl:    202 × 4 = 808  (v14b DAgger, ~15%)
#      - dagger_v15_targeted.jsonl:   139 × 4 = 556  (targeted fixes, ~10%)
#      - v14b_success_replay.jsonl:   106    (anti-forgetting, ~2%)
#      Total: ~4,757 examples
#
#   4. doc_type=None at inference (no change from v15)
#   5. load_best_model_at_end=False — NEVER CHANGE THIS
#
# WHY v16 SHOULD OUTPERFORM v15 (87/150 = 58%):
#   - Complete trees → oracle paths to previously-impossible sections
#   - 40% multi-node training (vs ~0%) → model learns multi-section retrieval
#   - Questions requiring Items 3,4,5 can now be answered
#   - Foundation is correct: the model navigates COMPLETE, ACCURATE trees
#
# DATASETS NEEDED ON DRIVE (MyDrive/arbor-training-data/):
#   - structdirect_train.jsonl         ← UPLOAD NEW VERSION (3,287 examples)
#   - dagger_corrections.jsonl         ← already on Drive (202 examples)
#   - dagger_v15_targeted.jsonl        ← UPLOAD NEW VERSION (139 examples)
#   - v14b_success_replay.jsonl        ← UPLOAD NEW VERSION (106 examples)
#
# REQUIREMENTS:
#   - Google Colab Pro (A100 40GB recommended)
#
# EXPECTED RUNTIME on A100: ~15-18 minutes (~4,757 examples × 3 epochs)
#   Total steps ≈ (4757 × 3) / 16 ≈ 892 steps
#
# OUTPUT:
#   - Saved to Drive: MyDrive/arbor-training-data/models/treesearch-v16/


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
DRIVE_DIR     = "/content/drive/MyDrive/arbor-training-data"

STRUCT_FILE   = f"{DRIVE_DIR}/structdirect_train.jsonl"        # 3,287 examples (fixed trees)
DAGGER_FILE   = f"{DRIVE_DIR}/dagger_corrections.jsonl"        # 202  (4x → 808)
DAGGER_V15    = f"{DRIVE_DIR}/dagger_v15_targeted.jsonl"       # 139  (4x → 556)
REPLAY_FILE   = f"{DRIVE_DIR}/v14b_success_replay.jsonl"       # 106  anti-forgetting

EVAL_FILE     = f"{DRIVE_DIR}/treesearch_multihop_eval.jsonl"  # stale, reference only
OUTPUT_DIR    = f"{DRIVE_DIR}/models/treesearch-v16"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import subprocess
for label, path in [
    ("StructDirect", STRUCT_FILE),
    ("DAgger v14b",  DAGGER_FILE),
    ("DAgger v15",   DAGGER_V15),
    ("Replay",       REPLAY_FILE),
    ("Eval",         EVAL_FILE),
]:
    if os.path.exists(path):
        result = subprocess.run(["wc", "-l", path], capture_output=True, text=True)
        print(f"  {label:<14}: {result.stdout.strip()}")
    else:
        print(f"  {label:<14}: *** FILE NOT FOUND: {path} ***")

print(f"\nOutput dir: {OUTPUT_DIR}")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 3 — Load base model                               ║
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

# r=16 same as v14b/v15 — proven to fit A100 and give good results
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
# ║  CELL 4 — Load and format datasets                      ║
# ╚══════════════════════════════════════════════════════════╝
"""
import json, random
from datasets import Dataset

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
    """Replace every training example's system message with the inference-time prompt."""
    for ex in examples:
        msgs = ex.get("messages", [])
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] = _INFERENCE_SYSTEM
        elif msgs and msgs[0]["role"] != "system":
            msgs.insert(0, {"role": "system", "content": _INFERENCE_SYSTEM})
    return examples


def format_example(example):
    """Apply Qwen2.5 chat template."""
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


print("Loading datasets...")

struct_raw  = load_jsonl(STRUCT_FILE)   # 3,287 StructDirect examples (fixed trees)
dagger_raw  = load_jsonl(DAGGER_FILE)   # 202 v14b DAgger corrections
eval_raw    = load_jsonl(EVAL_FILE)

print(f"  StructDirect   : {len(struct_raw):,}")
print(f"  DAgger v14b    : {len(dagger_raw):,}")

if os.path.exists(DAGGER_V15):
    dagger_v15_raw = load_jsonl(DAGGER_V15)
    print(f"  DAgger v15     : {len(dagger_v15_raw):,}")
else:
    dagger_v15_raw = []
    print(f"  DAgger v15     : NOT FOUND — training without it")

if os.path.exists(REPLAY_FILE):
    replay_raw = load_jsonl(REPLAY_FILE)
    print(f"  Replay buffer  : {len(replay_raw):,}")
else:
    replay_raw = []
    print(f"  Replay buffer  : NOT FOUND — training without it")

# Oversample DAgger 4x (same as v14b/v15)
random.seed(42)
dagger_oversampled    = dagger_raw * 4
dagger_v15_oversampled = dagger_v15_raw * 4

random.shuffle(dagger_oversampled)
random.shuffle(dagger_v15_oversampled)

print(f"\n  DAgger v14b 4x : {len(dagger_oversampled):,}")
print(f"  DAgger v15  4x : {len(dagger_v15_oversampled):,}")

# Build final training mix
all_train = struct_raw + dagger_oversampled + dagger_v15_oversampled + replay_raw
total = len(all_train)
print(f"\n  Total examples before shuffle: {total:,}")
print(f"  Mix breakdown:")
print(f"    StructDirect   : {len(struct_raw)/total*100:.1f}%")
print(f"    DAgger v14b 4x : {len(dagger_oversampled)/total*100:.1f}%")
print(f"    DAgger v15  4x : {len(dagger_v15_oversampled)/total*100:.1f}%")
print(f"    Replay         : {len(replay_raw)/total*100:.1f}%")

normalize_system_prompt(all_train)
normalize_system_prompt(eval_raw)

random.shuffle(all_train)
print(f"  Total after shuffle: {len(all_train):,}")

print("\nApplying chat template...")
train_texts = [format_example(ex) for ex in all_train]
eval_texts  = [format_example(ex) for ex in eval_raw]

sample_for_lengths = random.sample(train_texts, min(300, len(train_texts)))
token_lengths = sorted([len(tokenizer.encode(t)) for t in sample_for_lengths])
p50 = token_lengths[len(token_lengths) // 2]
p90 = token_lengths[int(len(token_lengths) * 0.9)]
p99 = token_lengths[int(len(token_lengths) * 0.99)]
max_len = token_lengths[-1]
print(f"  Token lengths — p50: {p50} | p90: {p90} | p99: {p99} | max: {max_len}")
print(f"  MAX_SEQ_LEN={MAX_SEQ_LEN} — {'OK' if p99 < MAX_SEQ_LEN else 'WARNING: some examples truncated'}")

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

TOTAL_STEPS  = (len(train_dataset) * 3) // 16
WARMUP_STEPS = max(50, TOTAL_STEPS // 10)
print(f"Total steps: {TOTAL_STEPS} | Warmup steps: {WARMUP_STEPS}")

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
        max_grad_norm               = 0.3,
        seed                        = 42,
        load_best_model_at_end      = False,
        report_to                   = "none",
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_vram = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
max_vram   = round(gpu_stats.total_memory / 1024 ** 3, 3)
print(f"GPU: {gpu_stats.name} | VRAM: {max_vram}GB | Used: {start_vram}GB")

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
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 7 — Smoke test                                    ║
# ╚══════════════════════════════════════════════════════════╝
"""
import json, torch
from unsloth import FastLanguageModel

DRIVE_DIR  = "/content/drive/MyDrive/arbor-training-data"
MODEL_PATH = f"{DRIVE_DIR}/models/treesearch-v16"

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

# Test: stock performance (the Apple 10-K failure that exposed the bug)
messages_stock = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": (
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
    )},
]

# Test: revenue (basic, should always work)
messages_revenue = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": (
        "Question: What was the company's total revenue for fiscal year 2022?\n\n"
        "Sections at this level:\n"
        "[0001] Letter to Shareholders (pages 1-3)\n"
        "[0002] Business Overview (pages 4-12) [has sub-sections]\n"
        "[0003] Risk Factors (pages 13-28) [has sub-sections]\n"
        "[0004] Financial Statements and Results (pages 29-54) [has sub-sections]\n"
        "[0005] Corporate Governance (pages 55-61)\n"
        "[0006] Executive Compensation (pages 62-74)\n\n"
        "Which sections should we explore next?"
    )},
]

for name, msgs, expected in [
    ("Stock performance (Item 5)", messages_stock, "0008"),
    ("Revenue (Financial Statements)", messages_revenue, "0004"),
]:
    enc = tokenizer_test.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = (enc.input_ids if hasattr(enc, "input_ids") else enc).to("cuda")
    with torch.no_grad():
        out = model_test.generate(input_ids=input_ids, max_new_tokens=128,
                                  temperature=0.1, do_sample=False,
                                  pad_token_id=tokenizer_test.eos_token_id, use_cache=False)
    response = tokenizer_test.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    try:
        parsed = json.loads(response)
        nav = parsed.get("navigate_to", [])
        status = "PASS" if expected in nav else f"FAIL (expected {expected}, got {nav})"
    except Exception:
        status = "PARSE FAIL"
        nav = response[:80]
    print(f"[{name}] {status}")
    print(f"  navigate_to: {nav}")
    print(f"  thinking: {parsed.get('thinking', '')[:100]}")
    print()
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 8 — FinanceBench end-to-end eval (THE REAL METRIC)║
# ╚══════════════════════════════════════════════════════════╝
"""
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

subprocess.run(["pip", "install", "-q", "unsloth[colab-new]", "trl", "transformers",
                "accelerate", "bitsandbytes", "peft", "pymupdf"], check=True)

import torch, sys, warnings
sys.path.insert(0, "/content/arbor")
warnings.filterwarnings("ignore", message="Both `max_new_tokens`.*")

from unsloth import FastLanguageModel

DRIVE_DIR    = "/content/drive/MyDrive/arbor-training-data"
ADAPTER_PATH = f"{DRIVE_DIR}/models/treesearch-v16"

print("Loading v16 model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    ADAPTER_PATH, max_seq_length=4096, dtype=None, load_in_4bit=True
)
model.eval()
print("v16 loaded.\n")

from arbor.providers.base import LLMProvider
from arbor.extraction.structure_extractor import extract_structure
from arbor.core.tree_searcher import search_tree
from arbor.core.types import ArborConfig, DocumentTree


class V16Provider(LLMProvider):
    @property
    def name(self):
        return "treesearch-v16"

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
                input_ids=input_ids, max_new_tokens=max_tokens or 256,
                do_sample=False, repetition_penalty=1.1,
            )
        new_tokens = output[0][input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        finish = "length" if len(new_tokens) >= (max_tokens or 256) else "stop"
        return text, finish


REPO_TREE_DIR  = "/content/arbor/data/financebench/trees"
DRIVE_PDF_DIR  = f"{DRIVE_DIR}/financebench-pdfs"
TREE_CACHE_DIR = f"{DRIVE_DIR}/multidomaintest_trees"
os.makedirs(TREE_CACHE_DIR, exist_ok=True)

QA_FILE = "/content/arbor/data/financebench/financebench_open_source.jsonl"
qa_pairs = [
    json.loads(line)
    for line in open(QA_FILE, encoding="utf-8").read().strip().splitlines()
    if line.strip()
]

tree_cache = {}
for doc in sorted({q["doc_name"] for q in qa_pairs}):
    for p in [f"{REPO_TREE_DIR}/{doc}.json", f"{TREE_CACHE_DIR}/{doc}.json"]:
        if os.path.exists(p):
            raw = json.loads(open(p, encoding="utf-8").read())
            if "tree" in raw and isinstance(raw["tree"], dict):
                raw = raw["tree"]
            tree_cache[doc] = DocumentTree.from_dict(raw)
            break
    else:
        pdf_path = f"{DRIVE_PDF_DIR}/{doc}.pdf"
        if os.path.exists(pdf_path):
            try:
                tree = extract_structure(pdf_path)
                tree_cache[doc] = tree
                with open(f"{TREE_CACHE_DIR}/{doc}.json", "w", encoding="utf-8") as f:
                    json.dump(tree.to_dict(), f, ensure_ascii=False)
            except Exception as e:
                print(f"ERROR {doc}: {e}")

qa_pairs_filtered = [q for q in qa_pairs if q["doc_name"] in tree_cache]
print(f"Trees: {len(tree_cache)}/84 | QA pairs: {len(qa_pairs_filtered)}/150\n")

config = ArborConfig(
    max_hops=8, max_nodes_searched=0, timeout_sec=300.0, max_retries_on_bad_json=2,
)
provider = V16Provider()


async def run_financebench_v16():
    results = []

    for i, qa in enumerate(qa_pairs_filtered):
        doc            = qa["doc_name"]
        question       = qa["question"]
        evidence_pages = [e["evidence_page_num"] for e in qa.get("evidence", [])]
        tree           = tree_cache[doc]
        try:
            sr = await search_tree(tree, question, provider, multihop=True, config=config,
                                   doc_type=None)
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
            status = "+" if perfect else "-"
            print(f"[{i+1:03d}/{len(qa_pairs_filtered)}] {status} {doc[:32]:<32} | "
                  f"recall={recall:.0%} | evid={evidence_pages}")
        except Exception as e:
            print(f"[{i+1:03d}] ERROR: {e}")
            results.append({"q": i+1, "doc": doc, "recall": 0.0, "error": str(e), "perfect": False})

    n          = len(results)
    avg_recall = sum(r["recall"] for r in results) / n
    perfect    = sum(1 for r in results if r.get("perfect", False))
    partial    = sum(1 for r in results if 0 < r["recall"] < 1.0)
    zero       = sum(1 for r in results if r["recall"] == 0.0)

    print(f"\n{'='*55}")
    print(f"  v16 FinanceBench Retrieval Results")
    print(f"  Questions evaluated : {n}/150")
    print(f"  Avg recall          : {avg_recall:.1%}")
    print(f"  Perfect recall (1.0): {perfect}/{n}  ({perfect/n:.0%})")
    print(f"  Partial recall      : {partial}/{n}")
    print(f"  Zero recall         : {zero}/{n}")
    print(f"{'='*55}")
    print(f"  v14b baseline       : 56.7% (82/150)")
    print(f"  v15 baseline        : 58.0% (87/150)")
    print(f"  Target              : 65%+")
    outcome = "IMPROVEMENT OVER v15" if perfect/n > 0.58 else "NO IMPROVEMENT OVER v15"
    print(f"  Status              : {outcome}")
    print(f"{'='*55}")

    results_path = f"{DRIVE_DIR}/financebench_v16_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    return results

results = await run_financebench_v16()
"""
