#!/usr/bin/env python3
"""Step 10: Generate extraction, checklist, and redline training data.

Uses GPT-4o to label EDGAR contracts and clauses for task-specific training.

Requires OPENAI_API_KEY.
Cost: ~$15-20 for 200 contracts + 200 clauses

Usage:
    OPENAI_API_KEY=sk-... python scripts/10_generate_task_data.py --max-contracts 200 --max-clauses 200
"""

import argparse
import json
import logging
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


def load_contracts(max_contracts: int) -> list[RawContract]:
    contracts = []
    for type_dir in RAW_DIR.iterdir():
        if not type_dir.is_dir():
            continue
        contract_type = ContractType[type_dir.name] if type_dir.name in ContractType.__members__ else ContractType.OTHER
        for txt_file in type_dir.glob("*.txt"):
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
            if len(contracts) >= max_contracts:
                return contracts
    return contracts


def load_clauses(max_clauses: int) -> list[dict]:
    clauses = []
    with jsonlines.open(str(CLAUSES_PATH)) as reader:
        for row in reader:
            if row["clause_type"] == "UNKNOWN":
                continue
            clauses.append(row)
            if len(clauses) >= max_clauses:
                break
    return clauses


def main():
    parser = argparse.ArgumentParser(description="Generate task-specific training data")
    parser.add_argument("--max-contracts", type=int, default=200)
    parser.add_argument("--max-clauses", type=int, default=200)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Key terms extraction
    contracts = load_contracts(args.max_contracts)
    logger.info("Loaded %d contracts for extraction + checklist", len(contracts))

    extraction_examples = generate_extraction_examples(contracts, args.max_contracts)
    with jsonlines.open(str(OUTPUT_DIR / "extraction_examples.jsonl"), mode="w") as w:
        for ex in extraction_examples:
            w.write({"instruction": ex.instruction, "response": ex.response,
                     "clause_type": ex.clause_type.value, "contract_type": ex.contract_type.value,
                     "jurisdiction": ex.jurisdiction.value, "source": ex.source,
                     "quality_score": ex.quality_score})

    # 2. Clause checklist
    checklist_examples = generate_checklist_examples(contracts, args.max_contracts)
    with jsonlines.open(str(OUTPUT_DIR / "checklist_examples.jsonl"), mode="w") as w:
        for ex in checklist_examples:
            w.write({"instruction": ex.instruction, "response": ex.response,
                     "clause_type": ex.clause_type.value, "contract_type": ex.contract_type.value,
                     "jurisdiction": ex.jurisdiction.value, "source": ex.source,
                     "quality_score": ex.quality_score})

    # 3. Redline suggestions
    clauses = load_clauses(args.max_clauses)
    logger.info("Loaded %d clauses for redline generation", len(clauses))

    redline_examples = generate_redline_examples(clauses, args.max_clauses)
    with jsonlines.open(str(OUTPUT_DIR / "redline_examples.jsonl"), mode="w") as w:
        for ex in redline_examples:
            w.write({"instruction": ex.instruction, "response": ex.response,
                     "clause_type": ex.clause_type.value, "contract_type": ex.contract_type.value,
                     "jurisdiction": ex.jurisdiction.value, "source": ex.source,
                     "quality_score": ex.quality_score})

    total = len(extraction_examples) + len(checklist_examples) + len(redline_examples)
    logger.info("Total v3 task examples: %d (extraction: %d, checklist: %d, redline: %d)",
                total, len(extraction_examples), len(checklist_examples), len(redline_examples))


if __name__ == "__main__":
    main()
