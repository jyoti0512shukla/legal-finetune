#!/usr/bin/env python3
"""Step 7: Evaluate fine-tuned model against held-out validation set.

Usage:
    python scripts/07_eval.py --config configs/saul_7b.yaml
"""

import argparse
import logging
from pathlib import Path

from datasets import DatasetDict

from src.train.eval import evaluate_model
from src.train.qlora import TrainConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_DIR = Path("data/training")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--config", type=str, default="configs/saul_7b.yaml")
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    adapter_path = f"{config.output_dir}/final_adapter"

    dataset = DatasetDict.load_from_disk(str(DATASET_DIR))

    results = evaluate_model(
        base_model=config.base_model,
        adapter_path=adapter_path,
        eval_dataset=dataset["validation"],
        max_samples=args.max_samples,
        output_path=Path(config.output_dir) / "eval" / "results.json",
    )

    logger.info("Evaluation complete: %d samples scored", len(results))


if __name__ == "__main__":
    main()
