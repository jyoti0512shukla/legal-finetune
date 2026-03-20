#!/usr/bin/env python3
"""Step 5: Combine all data sources into a unified training dataset.

Merges EDGAR clauses, CUAD clauses, and Claude-distilled examples into
a single HuggingFace Dataset with train/validation splits.
"""

import argparse
import logging
from pathlib import Path

import jsonlines

from src.data.schema import (
    ClauseType, ContractType, Jurisdiction, TrainingExample,
)
from src.distill.quality_filter import filter_examples
from src.train.dataset import build_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/training")


def load_clause_jsonl(path: Path, source: str) -> list[TrainingExample]:
    """Convert clause JSONL into TrainingExamples by generating instructions."""
    examples = []
    if not path.exists():
        logger.warning("File not found: %s", path)
        return examples

    with jsonlines.open(str(path)) as reader:
        for row in reader:
            clause_type = row.get("clause_type", "UNKNOWN")
            contract_type = row.get("contract_type", "Other")
            text = row.get("text", "")

            if clause_type == "UNKNOWN" or len(text) < 200:
                continue

            # Generate a natural instruction
            ct_display = clause_type.replace("_", " ").title()
            instruction = f"Draft a {ct_display} clause for a {contract_type}."

            try:
                ct_enum = ClauseType(clause_type)
                ctype_enum = ContractType(contract_type)
            except ValueError:
                ct_enum = ClauseType.UNKNOWN
                ctype_enum = ContractType.OTHER

            examples.append(TrainingExample(
                instruction=instruction,
                response=text,
                clause_type=ct_enum,
                contract_type=ctype_enum,
                jurisdiction=Jurisdiction.CALIFORNIA,  # default for EDGAR
                source=source,
                quality_score=row.get("quality_score"),
            ))

    return examples


def load_distilled_jsonl(path: Path) -> list[TrainingExample]:
    """Load Claude-distilled examples from JSONL."""
    examples = []
    if not path.exists():
        logger.warning("File not found: %s", path)
        return examples

    with jsonlines.open(str(path)) as reader:
        for row in reader:
            try:
                examples.append(TrainingExample(
                    instruction=row["instruction"],
                    response=row["response"],
                    clause_type=ClauseType(row["clause_type"]),
                    contract_type=ContractType(row["contract_type"]),
                    jurisdiction=Jurisdiction(row["jurisdiction"]),
                    source=row.get("source", "distilled"),
                    quality_score=row.get("quality_score"),
                ))
            except (ValueError, KeyError) as e:
                logger.debug("Skipping malformed row: %s", e)

    return examples


def main():
    parser = argparse.ArgumentParser(description="Build unified training dataset")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--min-quality", type=float, default=3.0)
    args = parser.parse_args()

    all_examples = []

    # Layer 1: EDGAR clauses
    edgar_path = PROCESSED_DIR / "clauses" / "edgar_clauses.jsonl"
    edgar_examples = load_clause_jsonl(edgar_path, source="edgar")
    logger.info("EDGAR: %d examples", len(edgar_examples))
    all_examples.extend(edgar_examples)

    # Layer 2: CUAD clauses
    cuad_path = PROCESSED_DIR / "cuad" / "cuad_clauses.jsonl"
    cuad_examples = load_clause_jsonl(cuad_path, source="cuad")
    logger.info("CUAD: %d examples", len(cuad_examples))
    all_examples.extend(cuad_examples)

    # Layer 3: Claude-distilled gold examples
    distilled_path = PROCESSED_DIR / "distilled" / "claude_examples.jsonl"
    distilled_examples = load_distilled_jsonl(distilled_path)
    logger.info("Distilled: %d examples", len(distilled_examples))
    all_examples.extend(distilled_examples)

    # Layer 4: Augmented examples (if they exist)
    augmented_path = PROCESSED_DIR / "distilled" / "augmented_examples.jsonl"
    if augmented_path.exists():
        augmented_examples = load_distilled_jsonl(augmented_path)
        logger.info("Augmented: %d examples", len(augmented_examples))
        all_examples.extend(augmented_examples)

    logger.info("Total before filtering: %d examples", len(all_examples))

    # Quality filter
    all_examples = filter_examples(all_examples, min_score=args.min_quality)

    # Build HuggingFace dataset
    dataset = build_dataset(all_examples, val_split=args.val_split, output_dir=OUTPUT_DIR)

    logger.info("Final dataset saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
