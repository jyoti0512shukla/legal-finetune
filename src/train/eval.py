"""Evaluation — measures fine-tuned model quality against held-out test set.

Metrics:
1. Structural completeness: does output have expected sub-clause count?
2. Placeholder-free: no [brackets], TBD, etc.
3. Legal terminology density: presence of legal terms
4. ROUGE-L against gold-standard Claude examples
5. Per-clause-type breakdown
"""

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import PeftModel
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

PLACEHOLDER_PATTERN = re.compile(
    r"\[(?!\d)[^\]]{2,60}\]"
    r"|(\(insert[^)]{0,50}\))"
    r"|%[A-Z][A-Z_]{2,}%"
    r"|\bTBC\b|\bTBD\b",
    re.IGNORECASE,
)

LEGAL_TERMS = [
    "shall", "party", "agreement", "pursuant", "herein", "liability",
    "obligation", "terminate", "breach", "notice", "indemnify", "warrant",
    "notwithstanding", "covenant", "enforceable", "jurisdiction",
]


@dataclass
class EvalResult:
    clause_type: str
    instruction: str
    generated: str
    reference: str
    structural_score: float      # 0-1: sub-clause count correctness
    placeholder_free: bool       # no placeholders found
    legal_density: float         # fraction of legal terms present
    rouge_l: float               # ROUGE-L F1 against reference
    overall_score: float         # weighted average


def evaluate_model(
    base_model: str,
    adapter_path: str,
    eval_dataset: Dataset,
    max_samples: int = 100,
    output_path: Optional[Path] = None,
) -> list[EvalResult]:
    """Evaluate fine-tuned model on held-out examples."""

    logger.info("Loading model %s with adapter %s", base_model, adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = []

    samples = eval_dataset.select(range(min(max_samples, len(eval_dataset))))

    for row in tqdm(samples, desc="Evaluating"):
        text = row["text"]
        # Extract instruction from Mistral template
        inst_match = re.search(r"\[INST\]\s*(.*?)\s*\[/INST\]", text, re.DOTALL)
        if not inst_match:
            continue
        instruction = inst_match.group(1)
        reference = text.split("[/INST]")[-1].replace("</s>", "").strip()

        # Generate
        prompt = f"<s>[INST] {instruction} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=1024,
                temperature=0.1, do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()

        # Score
        result = score_generation(instruction, generated, reference,
                                  row.get("clause_type", "UNKNOWN"), scorer)
        results.append(result)

    # Summary
    log_summary(results)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([vars(r) for r in results], f, indent=2)
        logger.info("Eval results saved to %s", output_path)

    return results


def score_generation(
    instruction: str,
    generated: str,
    reference: str,
    clause_type: str,
    scorer: rouge_scorer.RougeScorer,
) -> EvalResult:
    """Score a single generated clause."""
    # Structural: count sub-clauses (numbered lines)
    gen_subclauses = len(re.findall(r"^\d+[.)]\s", generated, re.MULTILINE))
    ref_subclauses = len(re.findall(r"^\d+[.)]\s", reference, re.MULTILINE))
    structural = min(gen_subclauses, ref_subclauses) / max(ref_subclauses, 1)

    # Placeholder-free
    placeholders = PLACEHOLDER_PATTERN.findall(generated)
    placeholder_free = len(placeholders) == 0

    # Legal density
    gen_lower = generated.lower()
    found = sum(1 for t in LEGAL_TERMS if t in gen_lower)
    legal_density = found / len(LEGAL_TERMS)

    # ROUGE-L
    rouge = scorer.score(reference, generated)
    rouge_l = rouge["rougeL"].fmeasure

    # Overall weighted score
    overall = (
        0.25 * structural
        + 0.25 * (1.0 if placeholder_free else 0.0)
        + 0.15 * legal_density
        + 0.35 * rouge_l
    )

    return EvalResult(
        clause_type=clause_type,
        instruction=instruction[:200],
        generated=generated[:500],
        reference=reference[:500],
        structural_score=structural,
        placeholder_free=placeholder_free,
        legal_density=legal_density,
        rouge_l=rouge_l,
        overall_score=overall,
    )


def log_summary(results: list[EvalResult]):
    """Log evaluation summary with per-type breakdown."""
    if not results:
        logger.warning("No eval results")
        return

    avg_overall = sum(r.overall_score for r in results) / len(results)
    avg_rouge = sum(r.rouge_l for r in results) / len(results)
    pct_clean = sum(1 for r in results if r.placeholder_free) / len(results) * 100

    logger.info("=== EVALUATION SUMMARY ===")
    logger.info("Samples: %d", len(results))
    logger.info("Overall score: %.3f", avg_overall)
    logger.info("ROUGE-L: %.3f", avg_rouge)
    logger.info("Placeholder-free: %.1f%%", pct_clean)

    # Per-type breakdown
    by_type = defaultdict(list)
    for r in results:
        by_type[r.clause_type].append(r)

    logger.info("Per-type breakdown:")
    for ct, type_results in sorted(by_type.items()):
        avg = sum(r.overall_score for r in type_results) / len(type_results)
        logger.info("  %s: %.3f (%d samples)", ct, avg, len(type_results))
