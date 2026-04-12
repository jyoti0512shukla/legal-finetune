#!/usr/bin/env python3
"""QLoRA fine-tune of Gemma 4 26B-A4B on the v3 combined multi-task dataset.

This script is intended to run on a RunPod A100 80GB pod. It starts from base
google/gemma-4-26B-A4B-it and trains a new adapter on the combined dataset
(21K examples across 7 tasks and 8 contract types + 4 external augmentations).

Dataset: data/v2/training/v2_combined_train.jsonl (19,002 examples)
         data/v2/training/v2_combined_val.jsonl (2,110 examples)

Hyperparameters tuned for 19K examples (vs v2 script's 275):
  - per-device batch size 1, gradient accumulation 8 (effective batch 8)
  - 2 epochs (19K × 2 = ~4,750 steps/epoch × 2 = ~9,500 total)
  - learning rate 2e-4 with 100-step warmup and cosine decay
  - LoRA rank 64, alpha 128
  - max_seq_length 8192 (examples beyond this get truncated — affects ~5% of
    CUAD contracts, acceptable since most signal is in the first 8K tokens)

Run:
    python scripts/v3_train_gemma4.py \\
        --hf-token $HF_TOKEN \\
        --output-dir /workspace/gemma4-legal-v3 \\
        --hf-repo jyoti0512shuklaorg/gemma4-legal-v3
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


def print_dataset_stats(dataset, label="Dataset"):
    """Print a quick summary of the dataset composition."""
    tasks = Counter()
    types = Counter()
    for ex in dataset:
        convs = ex.get("conversations", [])
        tasks[ex.get("task", "?")] += 1
        ct = ex.get("contract_type") or "?"
        types[ct] += 1
    print(f"\n  {label}: {len(dataset)} examples")
    print(f"  Tasks: {dict(sorted(tasks.items(), key=lambda x:-x[1]))}")
    top_types = sorted(types.items(), key=lambda x:-x[1])[:10]
    print(f"  Top sources: {dict(top_types)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token", required=True, help="HuggingFace token")
    parser.add_argument("--repo-dir", default="/workspace/legal-finetune",
                        help="Local clone of legal-finetune repo")
    parser.add_argument("--train-file",
                        default="data/v2/training/v2_combined_train.jsonl",
                        help="Train JSONL (Gemma chat format)")
    parser.add_argument("--val-file",
                        default="data/v2/training/v2_combined_val.jsonl",
                        help="Val JSONL (Gemma chat format)")
    parser.add_argument("--output-dir", default="/workspace/gemma4-legal-v3",
                        help="Where to save adapter checkpoints")
    parser.add_argument("--hf-repo",
                        default="jyoti0512shuklaorg/gemma4-legal-v3",
                        help="HuggingFace repo to push adapter to")
    parser.add_argument("--max-seq-length", type=int, default=8192,
                        help="Max sequence length in tokens. Examples longer "
                             "than this are truncated. 8192 fits A100 80GB "
                             "comfortably; 12288 may work with gradient "
                             "checkpointing.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps. "
                             "Effective batch = batch-size × grad-accum")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs. 2 is good for 19K "
                             "examples; 3 if you see val loss still dropping")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Peak learning rate. 2e-4 is the unsloth "
                             "recommended default for QLoRA")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Linear warmup steps before cosine decay")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--save-steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=500,
                        help="Run eval every N steps")
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push-to-hub", action="store_true", default=True)
    parser.add_argument("--no-push", dest="push_to_hub", action="store_false")
    args = parser.parse_args()

    eff_batch = args.batch_size * args.grad_accum
    print("=" * 70)
    print("Gemma 4 26B-A4B — v3 multi-task legal fine-tune")
    print("=" * 70)
    print(f"  repo_dir:         {args.repo_dir}")
    print(f"  train_file:       {args.train_file}")
    print(f"  val_file:         {args.val_file}")
    print(f"  output_dir:       {args.output_dir}")
    print(f"  hf_repo:          {args.hf_repo}")
    print(f"  max_seq_length:   {args.max_seq_length}")
    print(f"  batch_size:       {args.batch_size}")
    print(f"  grad_accum:       {args.grad_accum}  (effective batch {eff_batch})")
    print(f"  epochs:           {args.epochs}")
    print(f"  lr:               {args.lr}")
    print(f"  warmup_steps:     {args.warmup_steps}")
    print(f"  lora_r/alpha:     {args.lora_r} / {args.lora_alpha}")
    print(f"  save/eval_steps:  {args.save_steps}")
    print(f"  push_to_hub:      {args.push_to_hub}")
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
    print(f"✓ Base model loaded (max_seq_length={args.max_seq_length})")

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
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✓ LoRA adapter attached (r={args.lora_r}, alpha={args.lora_alpha})")
    print(f"  Trainable params: {trainable:,} / {total:,} "
          f"({trainable/total*100:.2f}%)")

    # ── 3. Load and format training data ───────────────────────────────
    from datasets import load_dataset
    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(tokenizer, chat_template="gemma")

    train_path = os.path.join(args.repo_dir, args.train_file)
    val_path = os.path.join(args.repo_dir, args.val_file)
    print(f"\nLoading train: {train_path}")
    print(f"Loading val:   {val_path}")

    raw_train = load_dataset("json", data_files=train_path, split="train")
    raw_val = load_dataset("json", data_files=val_path, split="train")

    # Print dataset composition
    print_dataset_stats(raw_train, "Train")
    print_dataset_stats(raw_val, "Val")

    def format_chat(examples):
        texts = []
        for c in examples["conversations"]:
            try:
                texts.append(
                    tokenizer.apply_chat_template(
                        c, tokenize=False, add_generation_prompt=False
                    )
                )
            except Exception as e:
                # Fallback: manual Gemma format if template fails
                parts = []
                for turn in c:
                    role = turn.get("role", "user")
                    content = turn.get("content", "")
                    if role == "user":
                        parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
                    else:
                        parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
                texts.append("\n".join(parts))
        return {"text": texts}

    train_data = raw_train.map(
        format_chat, batched=True, remove_columns=raw_train.column_names,
        num_proc=4, desc="Formatting train"
    )
    val_data = raw_val.map(
        format_chat, batched=True, remove_columns=raw_val.column_names,
        num_proc=4, desc="Formatting val"
    )
    print(f"\n✓ Formatted with Gemma chat template")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Val:   {len(val_data)} examples")

    # Token length distribution (sample for speed)
    import random
    sample_idx = random.sample(range(len(train_data)), min(200, len(train_data)))
    sample_lens = [
        len(tokenizer(train_data[i]["text"], truncation=False)["input_ids"])
        for i in sample_idx
    ]
    sample_lens.sort()
    n = len(sample_lens)
    print(f"\n  Token length distribution (sample of {n}):")
    print(f"    min={sample_lens[0]}, p25={sample_lens[n//4]}, "
          f"median={sample_lens[n//2]}, p75={sample_lens[3*n//4]}, "
          f"p95={sample_lens[int(n*0.95)]}, max={sample_lens[-1]}")
    over = sum(1 for l in sample_lens if l > args.max_seq_length)
    print(f"    Truncated (>{args.max_seq_length} tokens): "
          f"~{over}/{n} ({over/n*100:.1f}%)")

    # Estimate training steps
    steps_per_epoch = len(train_data) // eff_batch
    total_steps = steps_per_epoch * args.epochs
    print(f"\n  Estimated steps: {steps_per_epoch}/epoch × {args.epochs} epochs "
          f"= {total_steps} total")
    est_hours = total_steps * 1.5 / 3600  # rough: ~1.5s/step on A100
    print(f"  Estimated time: ~{est_hours:.1f} hours on A100 80GB")

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
        packing            = False,  # Don't pack — examples vary hugely in length
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
            eval_steps                   = args.eval_steps,
            load_best_model_at_end       = True,
            metric_for_best_model        = "eval_loss",
            greater_is_better            = False,
            output_dir                   = args.output_dir,
            seed                         = args.seed,
            report_to                    = "none",
            dataloader_num_workers       = 4,
        ),
    )

    print()
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)
    trainer_stats = trainer.train()
    print()
    print(f"✓ Training complete")
    print(f"  Total steps: {trainer_stats.global_step}")
    print(f"  Train loss:  {trainer_stats.training_loss:.4f}")

    # ── 5. Save best adapter locally ────────────────────────────────────
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
