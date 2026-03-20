#!/usr/bin/env python3
"""Step 9: Generate balanced risk assessment training data.

Takes EDGAR clauses, creates MEDIUM and HIGH risk versions via controlled
degradation, labels all three versions. Produces balanced LOW/MEDIUM/HIGH data.

Requires OPENAI_API_KEY.
Cost: ~$15-20 for 200 clauses (3 API calls each × 4 labels = ~2400 calls)

Usage:
    OPENAI_API_KEY=sk-... python scripts/09_generate_risk_data.py --max-clauses 200
"""

import argparse
import logging
from pathlib import Path

import jsonlines

from src.data.schema import ClauseType, ContractType, ExtractedClause
from src.distill.degrader import generate_risk_training_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CLAUSES_PATH = Path("data/processed/clauses/edgar_clauses.jsonl")
OUTPUT_DIR = Path("data/processed/v3_tasks")


def main():
    parser = argparse.ArgumentParser(description="Generate risk assessment training data")
    parser.add_argument("--max-clauses", type=int, default=200,
                        help="Max clauses to process (default: 200)")
    args = parser.parse_args()

    # Load classified clauses (skip UNKNOWN)
    clauses = []
    with jsonlines.open(str(CLAUSES_PATH)) as reader:
        for row in reader:
            if row["clause_type"] == "UNKNOWN":
                continue
            clauses.append(ExtractedClause(
                contract_source_id=row["contract_source_id"],
                contract_type=ContractType(row["contract_type"]),
                clause_type=ClauseType(row["clause_type"]),
                heading=row["heading"],
                text=row["text"],
                quality_score=row.get("quality_score"),
            ))

    logger.info("Loaded %d classified clauses", len(clauses))

    examples = generate_risk_training_data(clauses, max_clauses=args.max_clauses)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "risk_examples.jsonl"
    with jsonlines.open(str(output_path), mode="w") as writer:
        for ex in examples:
            writer.write({
                "instruction": ex.instruction,
                "response": ex.response,
                "clause_type": ex.clause_type.value,
                "contract_type": ex.contract_type.value,
                "jurisdiction": ex.jurisdiction.value,
                "source": ex.source,
                "quality_score": ex.quality_score,
            })

    logger.info("Saved %d risk examples to %s", len(examples), output_path)


if __name__ == "__main__":
    main()
