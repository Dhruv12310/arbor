"""
QLoRA fine-tuning for the tree generation model.

Usage:
    pip install transformers peft trl datasets accelerate bitsandbytes
    python scripts/finetune_treegen.py
    python scripts/finetune_treegen.py --epochs 3 --batch-size 2
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRAIN_FILE = ROOT / "data" / "finetune" / "treegen_train.jsonl"
EVAL_FILE  = ROOT / "data" / "finetune" / "treegen_eval.jsonl"
ADAPTER_DIR = ROOT / "models" / "treegen-adapter"
MERGED_DIR  = ROOT / "models" / "treegen-merged"

BASE_MODEL   = "Qwen/Qwen2.5-7B-Instruct"
LORA_RANK    = 16
LORA_ALPHA   = 32
MAX_SEQ_LEN  = 4096


def load_jsonl(path: Path):
    from datasets import Dataset
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def format_messages(example, tokenizer):
    return {"text": tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )}


def main():
    global TRAIN_FILE, EVAL_FILE, ADAPTER_DIR, MERGED_DIR

    KAGGLE_INPUT = Path("/kaggle/input/arbor-training-data")
    if KAGGLE_INPUT.exists():
        TRAIN_FILE  = KAGGLE_INPUT / "treegen_train.jsonl"
        EVAL_FILE   = KAGGLE_INPUT / "treegen_eval.jsonl"
        ADAPTER_DIR = Path("/kaggle/working/treegen-adapter")
        MERGED_DIR  = Path("/kaggle/working/treegen-merged")

    parser = argparse.ArgumentParser(description="QLoRA fine-tune tree generation model")
    parser.add_argument("--epochs",     type=int,   default=4)
    parser.add_argument("--batch-size", type=int,   default=1)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int,   default=8)
    parser.add_argument("--hf-repo",    type=str,   default=None,
        help="HuggingFace repo to push merged model e.g. stark12310/arbor-treegen-7b")
    parser.add_argument("--hf-token",   type=str,   default=None,
        help="HuggingFace API token for pushing model")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    print(f"Loading base model: {BASE_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    fmt = lambda ex: format_messages(ex, tokenizer)
    train_ds = load_jsonl(TRAIN_FILE).map(fmt)
    eval_ds  = load_jsonl(EVAL_FILE).map(fmt)
    print(f"Train: {len(train_ds)} examples  |  Eval: {len(eval_ds)} examples")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            output_dir=str(ADAPTER_DIR),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            fp16=True,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(str(ADAPTER_DIR))
    print(f"Adapter saved to {ADAPTER_DIR}")

    print("Freeing training model from memory...")
    del trainer
    del model
    torch.cuda.empty_cache()

    print("Merging adapter into base model...")
    from peft import PeftModel
    merged = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto"),
        str(ADAPTER_DIR),
    ).merge_and_unload()
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))
    print(f"Merged model saved to {MERGED_DIR}")

    if args.hf_repo:
        print(f"Pushing merged model to HuggingFace Hub: {args.hf_repo}")
        if args.hf_token:
            from huggingface_hub import login
            login(token=args.hf_token)
        merged.push_to_hub(args.hf_repo)
        tokenizer.push_to_hub(args.hf_repo)
        print(f"Pushed to https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
