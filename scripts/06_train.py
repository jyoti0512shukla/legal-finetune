#!/usr/bin/env python3
"""Step 6: QLoRA fine-tune Saul-7B on the prepared dataset.

Run on Colab A100 or a machine with >= 24GB VRAM.

Usage:
    python scripts/06_train.py --config configs/saul_7b.yaml
"""

import argparse
import logging
from pathlib import Path

from datasets import DatasetDict

from src.train.qlora import TrainConfig, train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_DIR = Path("data/training")


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune")
    parser.add_argument("--config", type=str, default="configs/saul_7b.yaml")
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    logger.info("Config: %s", vars(config))

    dataset = DatasetDict.load_from_disk(str(DATASET_DIR))
    logger.info("Loaded dataset: %d train, %d validation",
                len(dataset["train"]), len(dataset["validation"]))

    adapter_path = train(config, dataset)
    logger.info("Training complete. Adapter: %s", adapter_path)


if __name__ == "__main__":
    main()
