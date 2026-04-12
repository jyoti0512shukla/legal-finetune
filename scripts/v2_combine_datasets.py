#!/usr/bin/env python3
"""Combine v2 drafting + multi-task datasets into a single shuffled train/val split.

Inputs:
  data/v2/training/v2_train.jsonl, v2_val.jsonl
  data/v2/training/v2_multitask_train.jsonl, v2_multitask_val.jsonl

Outputs:
  data/v2/training/v2_combined_train.jsonl
  data/v2/training/v2_combined_val.jsonl
  data/v2/training/v2_combined_manifest.json

Splits are kept aligned (drafting train + multitask train → combined train, etc.)
so val never leaks into train.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import median


REPO = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO / "data" / "v2" / "training"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(pct * (len(s) - 1)))))
    return s[k]


def source_of(row: dict) -> str:
    if row.get("contract_type"):
        return row["contract_type"]
    return row.get("source_template", "UNKNOWN")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    drafting_train = load_jsonl(TRAIN_DIR / "v2_train.jsonl")
    drafting_val = load_jsonl(TRAIN_DIR / "v2_val.jsonl")
    multitask_train = load_jsonl(TRAIN_DIR / "v2_multitask_train.jsonl")
    multitask_val = load_jsonl(TRAIN_DIR / "v2_multitask_val.jsonl")

    rng = random.Random(args.seed)

    train = drafting_train + multitask_train
    val = drafting_val + multitask_val
    rng.shuffle(train)
    rng.shuffle(val)

    write_jsonl(TRAIN_DIR / "v2_combined_train.jsonl", train)
    write_jsonl(TRAIN_DIR / "v2_combined_val.jsonl", val)

    all_rows = train + val
    by_task = Counter(r.get("task", "unknown") for r in all_rows)
    by_source = Counter(source_of(r) for r in all_rows)
    char_lens = [
        sum(len(turn.get("value", "")) for turn in r.get("conversations", []))
        for r in all_rows
    ]

    manifest = {
        "total": len(all_rows),
        "train_count": len(train),
        "val_count": len(val),
        "by_task": dict(by_task),
        "by_source": dict(by_source),
        "char_lens": {
            "min": min(char_lens) if char_lens else 0,
            "median": int(median(char_lens)) if char_lens else 0,
            "p75": percentile(char_lens, 0.75),
            "p95": percentile(char_lens, 0.95),
            "max": max(char_lens) if char_lens else 0,
        },
        "sources": {
            "synthetic_authored": "data/v2 (175 contracts × 7 types: SAAS, MSA, NDA, EMPLOYMENT, SOW, DPA, INDEPENDENT_CONTRACTOR)",
            "cuad_real": "data/external/cuad/CUADv1.json (510 real commercial contracts × 41 clause categories, CC BY 4.0)",
            "legal_summ_real": "data/external/legal_summarization (439 real plain-English summaries from tldrlegal/tosdr, CC BY-SA — verify license)",
            "unfair_tos_real": "data/external/unfair_tos (LexGLUE UNFAIR-ToS, 9,414 sentence-level expert annotations, CC BY 4.0)",
            "acord_real": "data/external/acord (ACORD, 114 lawyer-written queries × 3,931 real B2B clauses from EDGAR, CC BY 4.0)",
        },
    }
    (TRAIN_DIR / "v2_combined_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )

    print(f"=== V2 COMBINED DATASET ===")
    print(f"Train: {len(train)}")
    print(f"Val:   {len(val)}")
    print(f"Total: {len(all_rows)}")
    print(f"Tasks: {dict(by_task)}")
    print(f"Sources: {dict(by_source)}")


if __name__ == "__main__":
    main()
