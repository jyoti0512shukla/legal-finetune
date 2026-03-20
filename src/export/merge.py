"""Merge LoRA adapter with base model for deployment via vLLM.

After QLoRA training, the adapter weights are separate from the base model.
For production deployment with vLLM, we merge them into a single model
that can be served directly.
"""

import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def merge_and_save(
    base_model: str,
    adapter_path: str,
    output_dir: str,
    push_to_hub: bool = False,
    hub_repo: str = "",
) -> str:
    """Merge LoRA adapter into base model and save for vLLM deployment.

    The merged model can be served with:
        vllm serve <output_dir> --dtype half --chat-template /tmp/mistral.jinja
    """
    logger.info("Loading base model: %s", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("Loading adapter from: %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging adapter weights...")
    model = model.merge_and_unload()

    logger.info("Saving merged model to: %s", output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    if push_to_hub and hub_repo:
        logger.info("Pushing to HuggingFace Hub: %s", hub_repo)
        model.push_to_hub(hub_repo)
        tokenizer.push_to_hub(hub_repo)

    logger.info("Merged model saved. Deploy with:")
    logger.info("  vllm serve %s --dtype half --chat-template /tmp/mistral.jinja", output_dir)

    return output_dir
