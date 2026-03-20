"""HuggingFace Dataset builder — converts TrainingExamples into a format
ready for QLoRA fine-tuning with the Mistral chat template.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict
import jsonlines

from src.data.schema import TrainingExample

logger = logging.getLogger(__name__)

# Mistral instruct format — matches the chat template used in production
MISTRAL_TEMPLATE = "<s>[INST] {instruction} [/INST] {response}</s>"


def format_example(example: TrainingExample) -> dict:
    """Format a training example into Mistral chat format."""
    text = MISTRAL_TEMPLATE.format(
        instruction=example.instruction.strip(),
        response=example.response.strip(),
    )
    return {
        "text": text,
        "clause_type": example.clause_type.value,
        "contract_type": example.contract_type.value,
        "jurisdiction": example.jurisdiction.value,
        "source": example.source,
        "quality_score": example.quality_score or 0.0,
    }


def build_dataset(
    examples: list[TrainingExample],
    val_split: float = 0.1,
    output_dir: Optional[Path] = None,
) -> DatasetDict:
    """Build a HuggingFace DatasetDict from training examples.

    Splits into train/validation. Saves to disk if output_dir provided.
    """
    formatted = [format_example(ex) for ex in examples]

    ds = Dataset.from_list(formatted)
    ds = ds.shuffle(seed=42)
    split = ds.train_test_split(test_size=val_split, seed=42)
    dd = DatasetDict({"train": split["train"], "validation": split["test"]})

    logger.info(
        "Dataset built: %d train, %d validation (%.0f%% split)",
        len(dd["train"]), len(dd["validation"]), val_split * 100,
    )

    # Log distribution
    for split_name in ["train", "validation"]:
        from collections import Counter
        types = Counter(dd[split_name]["clause_type"])
        sources = Counter(dd[split_name]["source"])
        logger.info("  %s: types=%s", split_name, dict(types))
        logger.info("  %s: sources=%s", split_name, dict(sources))

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        dd.save_to_disk(str(output_dir))

        # Also save as JSONL for inspection
        for split_name in ["train", "validation"]:
            jsonl_path = output_dir / f"{split_name}.jsonl"
            with jsonlines.open(str(jsonl_path), mode="w") as writer:
                for row in dd[split_name]:
                    writer.write(row)
            logger.info("Saved %s to %s", split_name, jsonl_path)

    return dd


def load_examples_from_jsonl(path: Path) -> list[TrainingExample]:
    """Load previously saved training examples from JSONL."""
    examples = []
    with jsonlines.open(str(path)) as reader:
        for row in reader:
            examples.append(TrainingExample(**row))
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples
