"""
Fine-tune TreeGen model using Unsloth (v2 — fixed hyperparameters).

Key fixes vs v1:
  - Fresh base model (Qwen2.5-7B-Instruct), not broken v1
  - MAX_SEQ_LEN=16384 (was 2048 — caused 97% truncation)
  - packing=False (long structured outputs must not be concatenated)
  - LR=2e-5 (was 1e-4 — too aggressive for structured output learning)
  - eval_strategy="steps" with load_best_model_at_end=True
  - 10% warmup ratio

Usage (Google Colab):
    !pip install unsloth
    # Upload treegen_train.jsonl and treegen_eval.jsonl
    # Then run this script or paste cells into Colab

Usage (local):
    python scripts/finetune_treegen_v2.py --hf-repo TStark12310/arbor-treegen-7b-v2 --hf-token hf_xxx
"""

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# ── Config (change these) ─────────────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen2.5-7B-Instruct"   # Fresh base — NOT the broken v1
LORA_RANK   = 16
LORA_ALPHA  = 32
MAX_SEQ_LEN = 16384

EPOCHS      = 3
BATCH_SIZE  = 1
GRAD_ACCUM  = 16     # Effective batch = 16
LR          = 2e-5   # Gentle LR for structured output learning
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
# ──────────────────────────────────────────────────────────────────────────────


def detect_paths(hf_repo: str) -> tuple[Path, Path, Path, Path]:
    """Auto-detect data and output paths for Colab/Kaggle/local."""
    # Colab
    if Path("/content/drive/MyDrive").exists():
        drive = Path("/content/drive/MyDrive/arbor-training-data")
        train = drive / "treegen_train.jsonl"
        eval_ = drive / "treegen_eval.jsonl"
        adapter = Path("/content/treegen-adapter")
        merged = drive / "treegen-merged"
    # Kaggle
    elif Path("/kaggle/input").exists():
        data = Path("/kaggle/input/arbor-training-data")
        train = data / "treegen_train.jsonl"
        eval_ = data / "treegen_eval.jsonl"
        adapter = Path("/kaggle/working/treegen-adapter")
        merged = Path("/kaggle/working/treegen-merged")
    # Local
    else:
        root = Path(__file__).parent.parent
        train = root / "data" / "finetune" / "treegen_train.jsonl"
        eval_ = root / "data" / "finetune" / "treegen_eval.jsonl"
        adapter = root / "models" / "treegen-adapter"
        merged = root / "models" / "treegen-merged"

    return train, eval_, adapter, merged


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TreeGen v2 with Unsloth")
    parser.add_argument("--hf-repo", default="TStark12310/arbor-treegen-7b-v2")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--base-model", default=BASE_MODEL)
    args = parser.parse_args()

    train_file, eval_file, adapter_dir, merged_dir = detect_paths(args.hf_repo)

    print(f"Train file: {train_file}")
    print(f"Eval file:  {eval_file}")

    train_raw = load_jsonl(train_file)
    eval_raw  = load_jsonl(eval_file)
    print(f"Train: {len(train_raw)}  |  Eval: {len(eval_raw)}")

    # Load model with Unsloth
    print(f"\nLoading {args.base_model} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model.print_trainable_parameters()

    # Format datasets
    def fmt(ex):
        return {"text": tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )}

    train_ds = Dataset.from_list(train_raw).map(fmt)
    eval_ds  = Dataset.from_list(eval_raw).map(fmt)

    # Train
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            output_dir=str(adapter_dir),
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            bf16=True,
            packing=False,              # CRITICAL: no packing for structured outputs
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="none",
        ),
    )

    torch.cuda.empty_cache()
    print(f"\nTraining for {args.epochs} epoch(s)...")
    trainer.train()
    trainer.save_model(str(adapter_dir))
    print(f"Adapter saved -> {adapter_dir}")

    # Merge and save
    print("\nMerging adapter into base model...")
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    print(f"Merged model saved -> {merged_dir}")

    # Push to HuggingFace
    if args.hf_token and args.hf_repo:
        print(f"\nPushing to {args.hf_repo} ...")
        model.push_to_hub_merged(
            args.hf_repo, tokenizer,
            save_method="merged_16bit",
            token=args.hf_token,
        )
        print(f"Done: https://huggingface.co/{args.hf_repo}")
    else:
        print("\nNo HF token — skipping push. Set --hf-token to push to HuggingFace.")

    # Cleanup
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
