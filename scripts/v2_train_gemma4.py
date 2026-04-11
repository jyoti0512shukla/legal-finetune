#!/usr/bin/env python3
"""Fresh QLoRA fine-tune of Gemma 4 26B-A4B on the v2 dataset only.

This script is intended to run on a RunPod A100 80GB pod. It does NOT
continue from any previous adapter — it starts from base
google/gemma-4-26B-A4B-it and trains a new adapter purely on the v2 data.

Why fresh? We want a clean comparison: does our small (~275 example) but
information-dense v2 corpus produce a model that beats the existing
v1 adapter trained on ~1,827 short EDGAR clauses? The only way to know is
to train a separate adapter on v2 alone.

Hyperparameters are tuned for a small dataset:
  - per-device batch size 1, gradient accumulation 4 (effective batch 4)
  - 5 epochs to give the small dataset enough exposure
  - learning rate 1e-4 with 20-step warmup
  - LoRA rank 64, alpha 128 (same as v1 for capacity parity)
  - max_seq_length 8192 (longer than v1's 4096 because v2 has full-contract
    examples up to ~10K tokens)

Run:
    python scripts/v2_train_gemma4.py \\
        --hf-token $HF_TOKEN \\
        --output-dir /workspace/gemma4-legal-v2-only \\
        --hf-repo jyoti0512shuklaorg/gemma4-legal-v2-only
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token", required=True, help="HuggingFace token")
    parser.add_argument("--repo-dir", default="/workspace/legal-finetune",
                        help="Local clone of legal-finetune repo")
    parser.add_argument("--train-file", default="data/v2/training/v2_train.jsonl",
                        help="Train file path inside repo-dir")
    parser.add_argument("--val-file", default="data/v2/training/v2_val.jsonl",
                        help="Validation file path inside repo-dir")
    parser.add_argument("--output-dir", default="/workspace/gemma4-legal-v2-only",
                        help="Where to save adapter checkpoints")
    parser.add_argument("--hf-repo", default="jyoti0512shuklaorg/gemma4-legal-v2-only",
                        help="HuggingFace repo to push adapter to")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push-to-hub", action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 70)
    print("Gemma 4 26B-A4B v2 fine-tune (fresh from base, v2 data only)")
    print("=" * 70)
    print(f"  repo_dir:       {args.repo_dir}")
    print(f"  train_file:     {args.train_file}")
    print(f"  val_file:       {args.val_file}")
    print(f"  output_dir:     {args.output_dir}")
    print(f"  hf_repo:        {args.hf_repo}")
    print(f"  max_seq_length: {args.max_seq_length}")
    print(f"  batch_size:     {args.batch_size}")
    print(f"  grad_accum:     {args.grad_accum}  (effective batch {args.batch_size * args.grad_accum})")
    print(f"  epochs:         {args.epochs}")
    print(f"  lr:             {args.lr}")
    print(f"  lora_r/alpha:   {args.lora_r} / {args.lora_alpha}")
    print()

    # ── 1. Login to HuggingFace ─────────────────────────────────────────
    from huggingface_hub import login
    login(token=args.hf_token)
    print("✓ HuggingFace login")

    # ── 2. Load base model with Unsloth ─────────────────────────────────
    from unsloth import FastLanguageModel
    import torch

    print("Loading base Gemma 4 26B-A4B (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = "google/gemma-4-26B-A4B-it",
        max_seq_length = args.max_seq_length,
        dtype          = None,
        load_in_4bit   = True,
        token          = args.hf_token,
    )
    print(f"✓ Base model loaded")

    model = FastLanguageModel.get_peft_model(
        model,
        r              = args.lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha     = args.lora_alpha,
        lora_dropout   = 0.05,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = args.seed,
    )
    print(f"✓ LoRA adapter attached (r={args.lora_r}, alpha={args.lora_alpha})")

    # ── 3. Load and format training data ───────────────────────────────
    from datasets import load_dataset
    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(tokenizer, chat_template="gemma")

    train_path = os.path.join(args.repo_dir, args.train_file)
    val_path = os.path.join(args.repo_dir, args.val_file)
    print(f"Loading train: {train_path}")
    print(f"Loading val:   {val_path}")

    raw_train = load_dataset("json", data_files=train_path, split="train")
    raw_val = load_dataset("json", data_files=val_path, split="train")
    print(f"✓ Loaded {len(raw_train)} train + {len(raw_val)} val examples")

    def format_chat(examples):
        return {
            "text": [
                tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
                for c in examples["conversations"]
            ]
        }

    train_data = raw_train.map(format_chat, batched=True, remove_columns=raw_train.column_names)
    val_data = raw_val.map(format_chat, batched=True, remove_columns=raw_val.column_names)
    print(f"✓ Formatted with Gemma chat template")

    # Sanity check token length distribution after tokenization
    sample_lens = [len(tokenizer(t)["input_ids"]) for t in train_data["text"][:50]]
    print(f"  Sample token lengths (first 50): min={min(sample_lens)}, "
          f"median={sorted(sample_lens)[len(sample_lens)//2]}, max={max(sample_lens)}")

    # ── 4. Train ────────────────────────────────────────────────────────
    from trl import SFTTrainer
    from transformers import TrainingArguments

    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = train_data,
        eval_dataset       = val_data,
        dataset_text_field = "text",
        max_seq_length     = args.max_seq_length,
        args = TrainingArguments(
            per_device_train_batch_size  = args.batch_size,
            per_device_eval_batch_size   = args.batch_size,
            gradient_accumulation_steps  = args.grad_accum,
            num_train_epochs             = args.epochs,
            learning_rate                = args.lr,
            warmup_steps                 = args.warmup_steps,
            bf16                         = True,
            optim                        = "adamw_8bit",
            weight_decay                 = 0.01,
            lr_scheduler_type            = "cosine",
            logging_steps                = args.logging_steps,
            save_steps                   = args.save_steps,
            save_total_limit             = 3,
            eval_strategy                = "steps",
            eval_steps                   = args.save_steps,
            output_dir                   = args.output_dir,
            seed                         = args.seed,
            report_to                    = "none",
        ),
    )

    print()
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)
    trainer_stats = trainer.train()
    print()
    print(f"✓ Training complete: {trainer_stats}")

    # ── 5. Save adapter locally ─────────────────────────────────────────
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"✓ Adapter saved to {final_dir}")

    # ── 6. Push to HuggingFace ──────────────────────────────────────────
    if args.push_to_hub:
        print(f"Pushing adapter to {args.hf_repo}...")
        model.push_to_hub(args.hf_repo, private=True, token=args.hf_token)
        tokenizer.push_to_hub(args.hf_repo, private=True, token=args.hf_token)
        print(f"✓ Adapter pushed to https://huggingface.co/{args.hf_repo}")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
