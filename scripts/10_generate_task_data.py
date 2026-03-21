#!/usr/bin/env python3
"""Step 10: Generate extraction, checklist, and redline training data.

Uses GPT-4o to label EDGAR contracts and clauses for task-specific training.
Supports resuming — skips contracts/clauses already in existing output files.

Requires OPENAI_API_KEY.
Cost: ~$15-20 for 200 contracts + 200 clauses

Usage:
    # Generate up to 400 total examples per task (appends to existing)
    OPENAI_API_KEY=sk-... python scripts/10_generate_task_data.py --max-contracts 400 --max-clauses 400

    # Run only specific tasks
    OPENAI_API_KEY=sk-... python scripts/10_generate_task_data.py --tasks extraction,checklist
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path

import jsonlines

from src.data.schema import ContractType, RawContract
from src.distill.task_labeler import (
    generate_extraction_examples,
    generate_checklist_examples,
    generate_redline_examples,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/edgar")
CLAUSES_PATH = Path("data/processed/clauses/edgar_clauses.jsonl")
OUTPUT_DIR = Path("data/processed/v3_tasks")


def load_existing_examples(path: Path) -> list[dict]:
    """Load existing examples from a JSONL file."""
    if not path.exists():
        return []
    examples = []
    with jsonlines.open(str(path)) as reader:
        for row in reader:
            examples.append(row)
    return examples


def extract_source_ids(examples: list[dict], field: str = "instruction") -> set[str]:
    """Extract contract/clause identifiers from existing examples to skip them.

    For extraction/checklist: look for entity names in the instruction text.
    For redline: look for clause text fingerprints (first 100 chars).
    """
    fingerprints = set()
    for ex in examples:
        text = ex.get(field, "")
        # Use first 100 chars of the contract/clause text as fingerprint
        lines = text.split("\n\n", 1)
        if len(lines) > 1:
            fingerprints.add(lines[1][:100])
    return fingerprints


def load_all_contracts() -> list[RawContract]:
    """Load ALL EDGAR contracts, shuffled for diverse contract type coverage."""
    contracts = []
    for type_dir in sorted(RAW_DIR.iterdir()):
        if not type_dir.is_dir():
            continue
        contract_type = ContractType[type_dir.name] if type_dir.name in ContractType.__members__ else ContractType.OTHER
        for txt_file in sorted(type_dir.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8")
            if len(text) < 1000:
                continue
            contracts.append(RawContract(
                source="edgar",
                source_id=txt_file.stem,
                filename=txt_file.name,
                entity_name=txt_file.stem.split("_")[1] if "_" in txt_file.stem else "",
                contract_type=contract_type,
                text=text,
            ))
    # Shuffle to ensure diverse contract type coverage when processing a subset
    random.seed(42)
    random.shuffle(contracts)
    return contracts


def load_all_clauses() -> list[dict]:
    """Load ALL classified EDGAR clauses (no limit)."""
    clauses = []
    with jsonlines.open(str(CLAUSES_PATH)) as reader:
        for row in reader:
            if row["clause_type"] == "UNKNOWN":
                continue
            clauses.append(row)
    return clauses


def filter_new_contracts(
    contracts: list[RawContract],
    existing_fingerprints: set[str],
) -> list[RawContract]:
    """Remove contracts whose text already appears in existing examples."""
    new = []
    for c in contracts:
        fp = c.text[:4000].split("\n\n", 1)
        text_start = fp[0][:100] if len(fp) == 1 else fp[0][:100]
        # Check against fingerprints from instruction text
        if c.text[:100] not in existing_fingerprints:
            new.append(c)
    return new


def filter_new_clauses(
    clauses: list[dict],
    existing_fingerprints: set[str],
) -> list[dict]:
    """Remove clauses whose text already appears in existing examples."""
    new = []
    for c in clauses:
        text = c.get("text", "")
        if text[:100] not in existing_fingerprints:
            new.append(c)
    return new


def write_example(row: dict) -> dict:
    """Convert TrainingExample fields to serializable dict."""
    return {
        "instruction": row.instruction,
        "response": row.response,
        "clause_type": row.clause_type.value,
        "contract_type": row.contract_type.value,
        "jurisdiction": row.jurisdiction.value,
        "source": row.source,
        "quality_score": row.quality_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate task-specific training data")
    parser.add_argument("--max-contracts", type=int, default=400,
                        help="Target total examples per contract-based task (extraction, checklist)")
    parser.add_argument("--max-clauses", type=int, default=400,
                        help="Target total examples for redline task")
    parser.add_argument("--tasks", type=str, default="extraction,checklist,redline",
                        help="Comma-separated tasks to run (extraction,checklist,redline)")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all available data
    all_contracts = load_all_contracts()
    logger.info("Total EDGAR contracts available: %d", len(all_contracts))

    if "extraction" in tasks:
        # ── Extraction ────────────────────────────────────────────────
        ext_path = OUTPUT_DIR / "extraction_examples.jsonl"
        existing_ext = load_existing_examples(ext_path)
        logger.info("Existing extraction examples: %d", len(existing_ext))

        needed = args.max_contracts - len(existing_ext)
        if needed > 0:
            fps = extract_source_ids(existing_ext)
            new_contracts = filter_new_contracts(all_contracts, fps)
            logger.info("New contracts available for extraction: %d (need %d more)", len(new_contracts), needed)

            new_examples = generate_extraction_examples(new_contracts, needed)
            # Append to existing file
            with jsonlines.open(str(ext_path), mode="a") as w:
                for ex in new_examples:
                    w.write(write_example(ex))
            logger.info("Extraction: %d existing + %d new = %d total",
                        len(existing_ext), len(new_examples), len(existing_ext) + len(new_examples))
        else:
            logger.info("Extraction: already at target (%d >= %d)", len(existing_ext), args.max_contracts)

    if "checklist" in tasks:
        # ── Checklist ─────────────────────────────────────────────────
        chk_path = OUTPUT_DIR / "checklist_examples.jsonl"
        existing_chk = load_existing_examples(chk_path)
        logger.info("Existing checklist examples: %d", len(existing_chk))

        needed = args.max_contracts - len(existing_chk)
        if needed > 0:
            fps = extract_source_ids(existing_chk)
            new_contracts = filter_new_contracts(all_contracts, fps)
            logger.info("New contracts available for checklist: %d (need %d more)", len(new_contracts), needed)

            new_examples = generate_checklist_examples(new_contracts, needed)
            with jsonlines.open(str(chk_path), mode="a") as w:
                for ex in new_examples:
                    w.write(write_example(ex))
            logger.info("Checklist: %d existing + %d new = %d total",
                        len(existing_chk), len(new_examples), len(existing_chk) + len(new_examples))
        else:
            logger.info("Checklist: already at target (%d >= %d)", len(existing_chk), args.max_contracts)

    if "redline" in tasks:
        # ── Redline ───────────────────────────────────────────────────
        red_path = OUTPUT_DIR / "redline_examples.jsonl"
        existing_red = load_existing_examples(red_path)
        logger.info("Existing redline examples: %d", len(existing_red))

        needed = args.max_clauses - len(existing_red)
        if needed > 0:
            all_clauses = load_all_clauses()
            logger.info("Total classified clauses available: %d", len(all_clauses))

            fps = extract_source_ids(existing_red)
            new_clauses = filter_new_clauses(all_clauses, fps)
            logger.info("New clauses available for redline: %d (need %d more)", len(new_clauses), needed)

            new_examples = generate_redline_examples(new_clauses, needed)
            with jsonlines.open(str(red_path), mode="a") as w:
                for ex in new_examples:
                    w.write(write_example(ex))
            logger.info("Redline: %d existing + %d new = %d total",
                        len(existing_red), len(new_examples), len(existing_red) + len(new_examples))
        else:
            logger.info("Redline: already at target (%d >= %d)", len(existing_red), args.max_clauses)

    # Summary
    for task_name, filename in [("extraction", "extraction_examples.jsonl"),
                                ("checklist", "checklist_examples.jsonl"),
                                ("redline", "redline_examples.jsonl")]:
        path = OUTPUT_DIR / filename
        if path.exists():
            count = sum(1 for _ in jsonlines.open(str(path)))
            logger.info("Final %s count: %d", task_name, count)


if __name__ == "__main__":
    main()
