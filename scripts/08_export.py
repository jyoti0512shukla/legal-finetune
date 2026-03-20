#!/usr/bin/env python3
"""Step 8: Merge LoRA adapter with base model for vLLM deployment.

Usage:
    python scripts/08_export.py --config configs/saul_7b.yaml
"""

import argparse
import logging

from src.export.merge import merge_and_save
from src.train.qlora import TrainConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export merged model")
    parser.add_argument("--config", type=str, default="configs/saul_7b.yaml")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo", type=str, default="")
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    adapter_path = f"{config.output_dir}/final_adapter"
    merged_path = f"{config.output_dir}/merged"

    merge_and_save(
        base_model=config.base_model,
        adapter_path=adapter_path,
        output_dir=merged_path,
        push_to_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
    )

    logger.info("Deploy with:")
    logger.info("  vllm serve %s --dtype half --chat-template /tmp/mistral.jinja", merged_path)


if __name__ == "__main__":
    main()
