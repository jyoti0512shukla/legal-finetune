#!/usr/bin/env python3
"""Step 3: Load CUAD dataset and extract annotated clauses.

Downloads CUAD from HuggingFace and saves extracted clauses to data/processed/cuad/.
"""

import logging
from pathlib import Path

import jsonlines

from src.data.cuad import load_cuad
from src.distill.quality_filter import filter_clauses

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/processed/cuad")


def main():
    clauses = load_cuad()
    clauses = filter_clauses(clauses, min_score=2.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "cuad_clauses.jsonl"
    with jsonlines.open(str(output_path), mode="w") as writer:
        for c in clauses:
            writer.write({
                "contract_source_id": c.contract_source_id,
                "contract_type": c.contract_type.value,
                "clause_type": c.clause_type.value,
                "heading": c.heading,
                "text": c.text,
                "char_count": c.char_count,
                "quality_score": c.quality_score,
            })

    logger.info("Saved %d CUAD clauses to %s", len(clauses), output_path)


if __name__ == "__main__":
    main()
