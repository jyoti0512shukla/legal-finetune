#!/usr/bin/env python3
"""Build the v2 training dataset for Gemma 4 fine-tuning.

Two example sources:
  1. Per-clause drafting examples already produced by
     v2_generate_drafting_examples.py — 220 examples covering IP, Liability,
     Termination, etc. across 55 contracts.
  2. Full-contract drafting examples — for each of the 55 placeholder-first
     contracts, one example whose output is the entire substituted contract.
     This teaches the model to commit to a drafting style across an entire
     contract, not just a single clause.

Both sources are converted to Gemma-style chat format:
    {"conversations": [{"role": "user", "content": "..."},
                       {"role": "assistant", "content": "..."}]}

The combined dataset is shuffled, split 90/10, and written to:
    data/v2/training/v2_train.jsonl
    data/v2/training/v2_val.jsonl
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

# Reuse generator helpers — this script lives in scripts/, so add it to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_generate_drafting_examples import (
    make_constrained_variation,
    apply_substitutions,
    AGREEMENT_LABEL_BY_TYPE,
    DRAFTING_STYLE_LABELS,
    _money_short,
    _format_term,
    _format_months,
    _format_days,
    _bias_label,
    _a_or_an,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

V2_ROOT = Path("/Users/jyotimishra/legal-finetune/data/v2")
TOKENIZED_ROOT = V2_ROOT / "tokenized"
META_ROOT = V2_ROOT / "raw"
TRAINING_ROOT = V2_ROOT / "training"
DRAFTING_ROOT = TRAINING_ROOT / "drafting"


# ── 1. Load per-clause examples ─────────────────────────────────────────

def load_clause_examples(types: list[str]) -> list[dict]:
    """Load all per-clause drafting examples and convert to Gemma chat format."""
    out = []
    for t in types:
        p = DRAFTING_ROOT / f"{t}_drafting_examples.jsonl"
        if not p.exists():
            logger.warning("Skipping missing clause file: %s", p)
            continue
        with p.open() as f:
            for line in f:
                ex = json.loads(line)
                out.append({
                    "conversations": [
                        {"role": "user", "content": ex["instruction"]},
                        {"role": "assistant", "content": ex["output"]},
                    ],
                    "task": "drafting_clause",
                    "contract_type": ex["contract_type"],
                    "clause_type": ex["clause_type"],
                    "drafting_style": ex["metadata"].get("drafting_style"),
                    "industry": ex["metadata"].get("industry"),
                })
    return out


# ── 2. Build full-contract examples ─────────────────────────────────────

def build_full_contract_brief(contract_type: str, sub_map: dict, metadata: dict) -> str:
    """Build a brief asking the model to draft an entire contract.

    The brief is intentionally rich: it gives the model the full party block,
    the commercial terms, the regulatory frameworks, the negotiation bias and
    drafting style, the customer-type-specific edge cases, and an explicit
    instruction to produce the COMPLETE agreement (all 15 articles).
    """
    rng = random.Random(abs(hash(metadata.get("industry", "") + sub_map.get("CUSTOMER_NAME", ""))) % (2**32))

    agreement_label = AGREEMENT_LABEL_BY_TYPE.get(contract_type, "commercial agreement")

    opener = rng.choice([
        f"Draft a complete {agreement_label}",
        f"Generate the full text of a {agreement_label}",
        f"Prepare a complete, end-to-end {agreement_label}",
        f"Write a full {agreement_label} with all 15 articles",
    ])

    provider = sub_map.get("PROVIDER_NAME", "")
    customer = sub_map.get("CUSTOMER_NAME", "")
    p_state = sub_map.get("PROVIDER_STATE", "")
    p_entity = sub_map.get("PROVIDER_ENTITY_TYPE", "")
    c_state = sub_map.get("CUSTOMER_STATE", "")
    c_entity = sub_map.get("CUSTOMER_ENTITY_TYPE", "")
    p_article = _a_or_an(p_state) if p_state else "a"
    c_article = _a_or_an(c_state) if c_state else "a"
    party_block = (
        f"between {provider} ({p_article} {p_state} {p_entity}, the \"Provider\") "
        f"and {customer} ({c_article} {c_state} {c_entity}, the \"Customer\")"
    )

    fee = _money_short(sub_map.get("ANNUAL_FEE_AMOUNT", ""))
    term = _format_term(sub_map.get("TERM_YEARS", ""))
    cap = _format_months(sub_map.get("LIABILITY_CAP_MONTHS", ""))
    notice = _format_days(sub_map.get("NOTICE_DAYS", ""))
    gov = sub_map.get("GOVERNING_LAW_STATE", "")
    venue = sub_map.get("VENUE_COUNTY", "")

    fee_label = {
        "SAAS": "annual subscription fee",
        "MSA": "annual aggregate fees across all Statements of Work",
    }.get(contract_type, "annual fee")

    bits = []
    if fee: bits.append(f"{fee_label} of {fee}")
    if term: bits.append(f"initial term of {term}")
    if cap: bits.append(f"liability cap equal to {cap} of fees")
    if notice: bits.append(f"notice period of {notice}")
    commercial = "Key commercial terms: " + "; ".join(bits) + "." if bits else ""

    juris = ""
    if gov:
        juris = f"Governing law: {gov}, with venue in {venue}." if venue else f"Governing law: {gov}."

    industry_label = metadata.get("industry_label") or metadata.get("industry", "")
    deal_size = metadata.get("deal_size", "")
    bias = _bias_label(metadata.get("negotiation_bias", "balanced"))
    ctx_bits = []
    if industry_label: ctx_bits.append(f"Industry: {industry_label}")
    if deal_size: ctx_bits.append(f"Deal size: {deal_size.replace('_', '-')}")
    if bias: ctx_bits.append(f"Negotiation posture: {bias}")
    context = ". ".join(ctx_bits) + "." if ctx_bits else ""

    regs = metadata.get("regulatory_frameworks", []) or []
    reg_block = "Regulatory frameworks to address: " + ", ".join(regs) + "." if regs else ""

    special = metadata.get("special_features", []) or []
    special_block = ""
    if special:
        # For full contracts, include up to 4 special features so the model
        # has the full picture of what makes this contract distinctive.
        chosen = special[:4]
        special_block = "Distinctive features to include: " + "; ".join(chosen) + "."

    style = metadata.get("drafting_style")
    style_label = DRAFTING_STYLE_LABELS.get(style) if style else None
    style_block = f"Drafting style: {style_label}." if style_label else ""

    closing = (
        "Produce the complete agreement with all 15 articles in the standard "
        "ARTICLE 1 — DEFINITIONS through ARTICLE 15 — GENERAL PROVISIONS structure. "
        "Do not summarize, do not abbreviate sections, and do not skip any clause."
    )

    parts = [
        f"{opener} {party_block}.",
        context,
        commercial,
        juris,
        reg_block,
        special_block,
        style_block,
        closing,
    ]
    return " ".join(p for p in parts if p)


def load_full_contract_examples(contract_types: list[str]) -> list[dict]:
    """For each tokenized template, build one full-contract drafting example."""
    out = []
    for ctype in contract_types:
        tokenized_dir = TOKENIZED_ROOT / ctype / "synthetic"
        meta_path = META_ROOT / ctype / "synthetic" / "contract_metadata.json"
        em_path = META_ROOT / ctype / "synthetic" / "entity_maps.json"
        if not tokenized_dir.exists() or not meta_path.exists() or not em_path.exists():
            logger.warning("Skipping %s — missing files", ctype)
            continue
        meta_all = json.loads(meta_path.read_text())
        em_all = json.loads(em_path.read_text())

        for tpl_path in sorted(tokenized_dir.glob("*.txt")):
            base = tpl_path.stem
            if base.startswith("_") or base.upper() == "MANIFEST":
                continue
            meta = meta_all.get(base)
            ent = em_all.get(base)
            if not meta or not ent:
                continue

            template = tpl_path.read_text(encoding="utf-8")
            placeholders_used = set(re.findall(r"\{\{([A-Z_]+)\}\}", template))

            # Use a deterministic seed based on the contract base name
            seed = abs(hash(f"{base}::full")) % (2**32)
            sub_map = make_constrained_variation(metadata=meta, entity_map=ent, seed=seed)
            active_subs = {k: v for k, v in sub_map.items() if k in placeholders_used}
            full_contract = apply_substitutions(template, active_subs)

            # Sanity: any leftover placeholders?
            leftover = re.findall(r"\{\{([A-Z_]+)\}\}", full_contract)
            if leftover:
                logger.debug("Leftover placeholders in %s: %s", base, set(leftover))

            brief = build_full_contract_brief(ctype, sub_map, meta)
            out.append({
                "conversations": [
                    {"role": "user", "content": brief},
                    {"role": "assistant", "content": full_contract},
                ],
                "task": "drafting_full_contract",
                "contract_type": ctype,
                "drafting_style": meta.get("drafting_style"),
                "industry": meta.get("industry"),
                "source_template": base,
            })
    return out


# ── 3. Combine and split ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", nargs="+", default=["SAAS", "MSA"])
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("Loading per-clause examples...")
    clause_examples = load_clause_examples(args.types)
    logger.info("  → %d clause-level examples", len(clause_examples))

    logger.info("Building full-contract examples...")
    full_examples = load_full_contract_examples(args.types)
    logger.info("  → %d full-contract examples", len(full_examples))

    all_examples = clause_examples + full_examples
    logger.info("Total: %d examples", len(all_examples))

    rng = random.Random(args.seed)
    rng.shuffle(all_examples)

    n_val = max(1, int(len(all_examples) * args.val_fraction))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    train_path = TRAINING_ROOT / "v2_train.jsonl"
    val_path = TRAINING_ROOT / "v2_val.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Manifest with breakdown
    from collections import Counter
    style_dist_train = Counter(e.get("drafting_style") for e in train_examples)
    type_dist_train = Counter(e["contract_type"] for e in train_examples)
    task_dist_train = Counter(e["task"] for e in train_examples)
    char_lens = [len(e["conversations"][0]["content"]) + len(e["conversations"][1]["content"]) for e in train_examples]
    char_lens.sort()

    manifest = {
        "train_count": len(train_examples),
        "val_count": len(val_examples),
        "total": len(all_examples),
        "by_task": dict(task_dist_train),
        "by_contract_type": dict(type_dist_train),
        "by_drafting_style": dict(style_dist_train),
        "char_lens": {
            "min": char_lens[0],
            "median": char_lens[len(char_lens) // 2],
            "p95": char_lens[int(len(char_lens) * 0.95)],
            "max": char_lens[-1],
        },
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    (TRAINING_ROOT / "v2_manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("\n=== V2 TRAINING DATASET ===")
    logger.info("Train: %d examples → %s", len(train_examples), train_path)
    logger.info("Val:   %d examples → %s", len(val_examples), val_path)
    logger.info("Tasks: %s", dict(task_dist_train))
    logger.info("Types: %s", dict(type_dist_train))
    logger.info("Styles: %s", dict(style_dist_train))
    logger.info("Char lens: min=%d median=%d p95=%d max=%d",
                char_lens[0], char_lens[len(char_lens) // 2],
                char_lens[int(len(char_lens) * 0.95)], char_lens[-1])


if __name__ == "__main__":
    main()
