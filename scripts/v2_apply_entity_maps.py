#!/usr/bin/env python3
"""V2 contract tokenization via hand-curated entity maps.

Reads entity_maps.json which contains hand-curated entity values for each
synthetic contract, then performs word-boundary-aware substitution to produce
tokenized templates.

This is more reliable than regex extraction because the entity values are
known a priori (we wrote the contracts).

Usage:
    python scripts/v2_apply_entity_maps.py --type SAAS
"""

import argparse
import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

V2_RAW = Path("/Users/jyotimishra/legal-finetune/data/v2/raw")
V2_TOKENIZED = Path("/Users/jyotimishra/legal-finetune/data/v2/tokenized")


def tokenize_with_map(text: str, entity_map: dict) -> tuple[str, dict]:
    """Replace each entity value with {{PLACEHOLDER}} using word-boundary regex.

    Returns (tokenized_text, replacements_dict).
    """
    out = text
    replacements = {}

    # Sort by length DESC so longer values are replaced first
    # (e.g., "California, United States" before "California")
    sorted_entries = sorted(
        ((k, v) for k, v in entity_map.items() if v and not k.startswith("_")),
        key=lambda kv: len(str(kv[1])),
        reverse=True,
    )

    for placeholder, value in sorted_entries:
        value = str(value).strip()
        if len(value) < 4:
            continue
        token = "{{" + placeholder + "}}"
        # Word-boundary lookarounds (no preceding/following alphanumeric)
        escaped = re.escape(value)
        pattern = r"(?<![A-Za-z0-9_])" + escaped + r"(?![A-Za-z0-9_])"
        try:
            count = len(re.findall(pattern, out))
        except re.error:
            count = out.count(value)
            if count > 0:
                out = out.replace(value, token)
                replacements[placeholder] = {"original": value, "occurrences": count, "method": "literal"}
            continue
        if count > 0:
            out = re.sub(pattern, token, out)
            replacements[placeholder] = {
                "original": value,
                "occurrences": count,
            }

    return out, replacements


def process(contract_type: str):
    raw_dir = V2_RAW / contract_type
    synthetic_dir = raw_dir / "synthetic"
    map_file = synthetic_dir / "entity_maps.json"

    if not map_file.exists():
        logger.error("Entity map file not found: %s", map_file)
        return

    entity_maps = json.loads(map_file.read_text())
    output_dir = V2_TOKENIZED / contract_type / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    summary = []

    for contract_name, entity_map in entity_maps.items():
        if contract_name.startswith("_"):
            continue
        contract_path = synthetic_dir / f"{contract_name}.txt"
        if not contract_path.exists():
            logger.warning("Contract not found: %s", contract_path.name)
            skipped += 1
            continue

        text = contract_path.read_text(encoding="utf-8")
        tokenized, replacements = tokenize_with_map(text, entity_map)

        out_path = output_dir / f"{contract_name}.txt"
        out_path.write_text(tokenized, encoding="utf-8")

        # Save the entity map alongside the tokenized contract
        map_out_path = output_dir / f"{contract_name}.json"
        map_out_path.write_text(json.dumps({
            "source_file": contract_path.name,
            "entity_map": entity_map,
            "replacements_made": replacements,
            "total_substitutions": sum(r["occurrences"] for r in replacements.values()),
        }, indent=2), encoding="utf-8")

        total_subs = sum(r["occurrences"] for r in replacements.values())
        unique_placeholders = len(replacements)
        logger.info("✓ %-58s %2d placeholders, %3d substitutions",
                    contract_name[:58], unique_placeholders, total_subs)
        summary.append({
            "contract": contract_name,
            "placeholders": unique_placeholders,
            "substitutions": total_subs,
        })
        processed += 1

    # Write summary manifest
    manifest = {
        "type": contract_type,
        "processed": processed,
        "skipped": skipped,
        "contracts": summary,
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    logger.info("\n=== TOKENIZATION COMPLETE ===")
    logger.info("Type:      %s", contract_type)
    logger.info("Processed: %d", processed)
    logger.info("Skipped:   %d", skipped)
    logger.info("Output:    %s", output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="SAAS")
    args = parser.parse_args()
    process(args.type)


if __name__ == "__main__":
    main()
