#!/usr/bin/env python3
"""Step 4 (alt): Generate gold-standard training examples using GPT-4o.

Requires OPENAI_API_KEY environment variable.

Usage:
    OPENAI_API_KEY=sk-... python scripts/04_distill_openai.py --examples-per-type 100
"""

import argparse
import logging
from pathlib import Path

import jsonlines

from src.distill.openai_teacher import generate_clause_examples

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/processed/distilled")


def main():
    parser = argparse.ArgumentParser(description="Distill training data from GPT-4o")
    parser.add_argument("--examples-per-type", type=int, default=10,
                        help="Number of examples per clause type (default: 10)")
    args = parser.parse_args()

    examples = generate_clause_examples(examples_per_type=args.examples_per_type)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "claude_examples.jsonl"  # same filename so build_dataset picks it up
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

    logger.info("Saved %d distilled examples to %s", len(examples), output_path)


if __name__ == "__main__":
    main()
