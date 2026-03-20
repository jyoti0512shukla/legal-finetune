"""QLoRA fine-tuning — trains a LoRA adapter on top of Saul-7B / Mistral.

Uses 4-bit quantization (QLoRA) to fit on a single A100 or even T4 GPU.
Training config is loaded from YAML. Produces a LoRA adapter that can be
merged with the base model for deployment.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration loaded from YAML."""
    # Model
    base_model: str = "Equall/Saul-Instruct-v1"
    max_seq_length: int = 2048
    # QLoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = None
    # Training
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    # Infrastructure
    bf16: bool = True
    output_dir: str = "outputs/saul-7b-legal-lora"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3

    def __post_init__(self):
        if self.target_modules is None:
            # Default Mistral attention + MLP modules for LoRA
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"]

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        # Merge base config if specified
        if "extends" in data:
            base_path = Path(path).parent / data.pop("extends")
            with open(base_path) as f:
                base_data = yaml.safe_load(f)
            base_data.update(data)
            data = base_data
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def train(config: TrainConfig, dataset: DatasetDict) -> str:
    """Run QLoRA fine-tuning. Returns path to the saved adapter."""

    logger.info("Loading base model: %s", config.base_model)
    logger.info("Training on %d examples, validating on %d",
                len(dataset["train"]), len(dataset["validation"]))

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info("Trainable parameters: %s / %s (%.2f%%)",
                f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        report_to="none",  # set to "wandb" if you want W&B logging
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )

    # SFT Trainer from TRL
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
        max_seq_length=config.max_seq_length,
    )

    logger.info("Starting QLoRA training...")
    trainer.train()

    # Save adapter
    adapter_path = f"{config.output_dir}/final_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("Adapter saved to %s", adapter_path)

    return adapter_path
