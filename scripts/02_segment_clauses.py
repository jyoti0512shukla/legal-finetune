#!/usr/bin/env python3
"""Step 2: Segment raw contracts into individual clauses.

Reads raw text files from data/raw/edgar/ and outputs clause JSONL to data/processed/clauses/.
"""

import json
import logging
from pathlib import Path

import jsonlines

from src.data.schema import ContractType, RawContract
from src.data.segmenter import segment_all
from src.distill.quality_filter import filter_clauses

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/edgar")
OUTPUT_DIR = Path("data/processed/clauses")


def main():
    # Load all raw contracts from text files
    contracts = []
    for type_dir in RAW_DIR.iterdir():
        if not type_dir.is_dir():
            continue
        contract_type = ContractType[type_dir.name] if type_dir.name in ContractType.__members__ else ContractType.OTHER
        for txt_file in type_dir.glob("*.txt"):
            text = txt_file.read_text(encoding="utf-8")
            contracts.append(RawContract(
                source="edgar",
                source_id=txt_file.stem,
                filename=txt_file.name,
                entity_name="",
                contract_type=contract_type,
                text=text,
            ))

    logger.info("Loaded %d raw contracts", len(contracts))

    # Segment
    clauses = segment_all(contracts)
    logger.info("Segmented into %d clauses", len(clauses))

    # Quality filter (rule-based)
    clauses = filter_clauses(clauses, min_score=3.0)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "edgar_clauses.jsonl"
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

    logger.info("Saved %d clauses to %s", len(clauses), output_path)

    # Summary
    from collections import Counter
    type_counts = Counter(c.clause_type.value for c in clauses)
    logger.info("Clause distribution: %s", dict(type_counts))


if __name__ == "__main__":
    main()
