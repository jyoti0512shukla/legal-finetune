#!/usr/bin/env python3
"""Step 1: Fetch EDGAR EX-10.x contracts.

Usage:
    python scripts/01_fetch_edgar.py --count 50 --contract-types saas,msa,nda,employment
"""

import argparse
import json
import logging
from pathlib import Path

from src.data.edgar import fetch_contracts
from src.data.schema import ContractType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/raw/edgar")


def main():
    parser = argparse.ArgumentParser(description="Fetch EDGAR contracts")
    parser.add_argument("--count", type=int, default=50, help="Contracts per type")
    parser.add_argument("--contract-types", type=str, default="msa,saas,nda,employment,software_license",
                        help="Comma-separated contract types")
    args = parser.parse_args()

    type_map = {
        "msa": ContractType.MSA,
        "saas": ContractType.SAAS,
        "nda": ContractType.NDA,
        "employment": ContractType.EMPLOYMENT,
        "software_license": ContractType.SOFTWARE_LICENSE,
        "supply": ContractType.SUPPLY,
        "vendor": ContractType.VENDOR,
        "ip_license": ContractType.IP_LICENSE,
        "consulting": ContractType.CONSULTING,
    }

    types = [type_map[t.strip().lower()] for t in args.contract_types.split(",") if t.strip().lower() in type_map]

    all_contracts = []
    for ct in types:
        contracts = fetch_contracts(ct, count=args.count, output_dir=OUTPUT_DIR / ct.name)
        all_contracts.extend(contracts)

    # Save manifest
    manifest = [
        {"source_id": c.source_id, "type": c.contract_type.value,
         "entity": c.entity_name, "chars": len(c.text)}
        for c in all_contracts
    ]
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Done: %d contracts saved to %s", len(all_contracts), OUTPUT_DIR)


if __name__ == "__main__":
    main()
