#!/usr/bin/env python3
"""V2 variation generator — expand tokenized templates into N diverse variations.

For each tokenized template (with {{PLACEHOLDERS}}), generate N variations by
substituting different realistic values from named pools. This expands the
training corpus from 15 templates → 75-150 examples without losing structure.

The substitution pools are realistic and curated:
- Provider/customer names from real-sounding companies
- US state names (with matching counties/cities)
- Date pools spanning 2020-2025
- Fee amounts in realistic ranges
- Term years, liability caps, notice days

Output: data/v2/variations/<TYPE>/<template>__var<N>.txt
        + per-variation entity map JSON
        + summary manifest

Usage:
    python scripts/v2_generate_variations.py --type SAAS --variations 5
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

V2_TOKENIZED = Path("/Users/jyotimishra/legal-finetune/data/v2/tokenized")
V2_VARIATIONS = Path("/Users/jyotimishra/legal-finetune/data/v2/variations")


# ── Substitution pools ────────────────────────────────────────────────────

PROVIDER_NAMES = [
    "Vertex Cloud Software, Inc.",
    "Quantix Data Systems, Inc.",
    "BlueShift Analytics Inc.",
    "Lighthouse Cloud Technologies, Inc.",
    "Apex Workflow Platform, Inc.",
    "PrismLogic Software Solutions, Inc.",
    "Northwind Cloud Services, Inc.",
    "Helios Platform Inc.",
    "Riverstone Software Inc.",
    "TitanSync Technologies, Inc.",
    "Cascade Digital Solutions Inc.",
    "Meridian SaaS Holdings, Inc.",
    "Acuity Software Group, Inc.",
    "Streamline Cloud Inc.",
    "CrestPoint Technology Solutions, Inc.",
    "Brightline Software, Inc.",
    "Zenith Platform Holdings, Inc.",
    "Fortis Cloud Inc.",
    "Beacon Analytics Software, Inc.",
    "Pinnacle Digital Systems, Inc.",
]

CUSTOMER_NAMES = [
    "Westbrook Industries, LLC",
    "Highland Capital Group, Inc.",
    "Sterling Hospitality Holdings, LLC",
    "Cardinal Manufacturing Co.",
    "Evergreen Properties Inc.",
    "Atlas Logistics Group, LLC",
    "Heritage Financial Services, Inc.",
    "Beacon Health Network, Inc.",
    "Summit Retail Partners LLC",
    "Wellington Asset Management LLC",
    "Lakeside Energy Solutions, Inc.",
    "Forge Industrial Holdings Inc.",
    "Birchwood Construction Company",
    "Compass Insurance Group Inc.",
    "Northern Star Distribution LLC",
    "Granite State Banking Corp.",
    "Coastal Maritime Holdings, Inc.",
    "Liberty Pharmaceuticals Inc.",
    "Pacific Trade Corporation",
    "Lonestar Energy Holdings LLC",
]

# US states with matching jurisdictional details. Zip prefix + 3-digit suffix = 5-digit zip.
JURISDICTIONS = [
    {"state": "California", "county": "Santa Clara County", "city": "San Jose", "zip_prefix": "951", "address_template": "{num} {street_name} Boulevard, Suite {suite}, San Jose, California {zip}"},
    {"state": "California", "county": "San Francisco County", "city": "San Francisco", "zip_prefix": "941", "address_template": "{num} {street_name} Street, San Francisco, California {zip}"},
    {"state": "New York", "county": "New York County", "city": "New York", "zip_prefix": "100", "address_template": "{num} {street_name} Avenue, {floor} Floor, New York, New York {zip}"},
    {"state": "Delaware", "county": "New Castle County", "city": "Wilmington", "zip_prefix": "198", "address_template": "{num} {street_name} Road, Wilmington, Delaware {zip}"},
    {"state": "Texas", "county": "Travis County", "city": "Austin", "zip_prefix": "787", "address_template": "{num} {street_name} Boulevard, Suite {suite}, Austin, Texas {zip}"},
    {"state": "Texas", "county": "Harris County", "city": "Houston", "zip_prefix": "770", "address_template": "{num} {street_name} Drive, Houston, Texas {zip}"},
    {"state": "Illinois", "county": "Cook County", "city": "Chicago", "zip_prefix": "606", "address_template": "{num} North {street_name} Avenue, Suite {suite}, Chicago, Illinois {zip}"},
    {"state": "Florida", "county": "Miami-Dade County", "city": "Miami", "zip_prefix": "331", "address_template": "{num} {street_name} Way, Miami, Florida {zip}"},
    {"state": "Massachusetts", "county": "Suffolk County", "city": "Boston", "zip_prefix": "021", "address_template": "{num} {street_name} Street, {floor} Floor, Boston, Massachusetts {zip}"},
    {"state": "Washington", "county": "King County", "city": "Seattle", "zip_prefix": "981", "address_template": "{num} {street_name} Avenue North, Seattle, Washington {zip}"},
    {"state": "Colorado", "county": "Denver County", "city": "Denver", "zip_prefix": "802", "address_template": "{num} {street_name} Street, Denver, Colorado {zip}"},
    {"state": "Georgia", "county": "Fulton County", "city": "Atlanta", "zip_prefix": "303", "address_template": "{num} {street_name} Street NE, Atlanta, Georgia {zip}"},
    {"state": "Pennsylvania", "county": "Philadelphia County", "city": "Philadelphia", "zip_prefix": "191", "address_template": "{num} {street_name} Street, Philadelphia, Pennsylvania {zip}"},
    {"state": "Ohio", "county": "Franklin County", "city": "Columbus", "zip_prefix": "432", "address_template": "{num} {street_name} Boulevard, Columbus, Ohio {zip}"},
    {"state": "Minnesota", "county": "Hennepin County", "city": "Minneapolis", "zip_prefix": "554", "address_template": "{num} {street_name} Avenue South, Minneapolis, Minnesota {zip}"},
    {"state": "North Carolina", "county": "Wake County", "city": "Raleigh", "zip_prefix": "276", "address_template": "{num} {street_name} Drive, Raleigh, North Carolina {zip}"},
]

STREET_NAMES = [
    "Madison", "Oakwood", "Birch", "Cedar", "Mission", "Innovation",
    "Commerce", "Liberty", "Hillside", "Riverside", "Park", "Maple",
    "Lakeshore", "Highland", "Crestview", "Pinecrest", "Westwood",
    "Northgate", "Summit", "Bridgeport", "Heritage", "Stonewall",
]

PROVIDER_ENTITY_TYPES = [
    "corporation",
    "limited liability company",
]

CUSTOMER_ENTITY_TYPES = [
    "corporation",
    "limited liability company",
    "limited liability partnership",
]


def _ordinal_floor(n):
    """Convert number to ordinal floor: 1→1st, 2→2nd, 3→3rd, 22→22nd."""
    suffix = "th"
    if n % 100 not in (11, 12, 13):
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _entity_type_for_name(name: str) -> str:
    """Pick an entity type that matches the company name suffix."""
    n = name.lower()
    if "llc" in n or "l.l.c" in n:
        return "limited liability company"
    if "llp" in n:
        return "limited liability partnership"
    if "lp" in n and "llp" not in n:
        return "limited partnership"
    if "company" in n or "co." in n:
        return "corporation"
    return "corporation"  # Default for Inc., Corp., etc.

EFFECTIVE_DATES = [
    "January 15, 2024", "February 28, 2024", "March 12, 2024",
    "April 8, 2024", "May 22, 2024", "June 17, 2024", "July 1, 2024",
    "August 5, 2024", "September 9, 2024", "October 14, 2024",
    "November 20, 2024", "December 3, 2024",
    "January 7, 2025", "February 11, 2025", "March 18, 2025",
    "April 22, 2025", "May 6, 2025", "June 30, 2025",
]

ANNUAL_FEE_RANGES = [
    # SMB
    "Twelve Thousand United States Dollars ($12,000)",
    "Eighteen Thousand United States Dollars ($18,000)",
    "Twenty-Four Thousand United States Dollars ($24,000)",
    "Thirty-Six Thousand United States Dollars ($36,000)",
    "Forty-Eight Thousand United States Dollars ($48,000)",
    # Mid-market
    "Seventy-Two Thousand United States Dollars ($72,000)",
    "Ninety-Six Thousand United States Dollars ($96,000)",
    "One Hundred Twenty Thousand United States Dollars ($120,000)",
    "One Hundred Fifty Thousand United States Dollars ($150,000)",
    "One Hundred Eighty Thousand United States Dollars ($180,000)",
    "Two Hundred Forty Thousand United States Dollars ($240,000)",
    # Enterprise
    "Three Hundred Sixty Thousand United States Dollars ($360,000)",
    "Four Hundred Eighty Thousand United States Dollars ($480,000)",
    "Six Hundred Thousand United States Dollars ($600,000)",
    "Seven Hundred Twenty Thousand United States Dollars ($720,000)",
    "Nine Hundred Sixty Thousand United States Dollars ($960,000)",
    "One Million Two Hundred Thousand United States Dollars ($1,200,000)",
    "One Million Eight Hundred Thousand United States Dollars ($1,800,000)",
]

IMPLEMENTATION_FEES = [
    "Ten Thousand United States Dollars ($10,000)",
    "Twenty-Five Thousand United States Dollars ($25,000)",
    "Fifty Thousand United States Dollars ($50,000)",
    "Seventy-Five Thousand United States Dollars ($75,000)",
    "One Hundred Thousand United States Dollars ($100,000)",
    "One Hundred Fifty Thousand United States Dollars ($150,000)",
    "Two Hundred Thousand United States Dollars ($200,000)",
]

TERM_YEARS = [
    "one (1)", "two (2)", "three (3)", "four (4)", "five (5)",
]

LIABILITY_CAP_MONTHS = [
    "six (6)", "twelve (12)", "eighteen (18)", "twenty-four (24)", "thirty-six (36)",
]

NOTICE_DAYS = [
    "thirty (30)", "sixty (60)", "ninety (90)", "one hundred eighty (180)",
]


# ── Variation generator ──────────────────────────────────────────────────

def generate_variation_map(seed: int) -> dict:
    """Generate one randomized substitution map using the realistic pools."""
    rng = random.Random(seed)
    juris = rng.choice(JURISDICTIONS)

    def make_address(template):
        return template.format(
            num=rng.randint(100, 9999),
            street_name=rng.choice(STREET_NAMES),
            suite=rng.choice([100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000]),
            floor=_ordinal_floor(rng.choice([2, 3, 5, 8, 10, 12, 15, 16, 20, 22, 25, 30, 35])),
            zip=f"{rng.randint(100, 999):03d}",
        )

    # Pick provider (always Delaware corp for consistency) and customer names
    provider_name = rng.choice(PROVIDER_NAMES)
    customer_name = rng.choice(CUSTOMER_NAMES)

    return {
        "PROVIDER_NAME": provider_name,
        "PROVIDER_STATE": "Delaware",
        # Match entity type to the name suffix (Inc. → corporation, LLC → llc, etc.)
        "PROVIDER_ENTITY_TYPE": _entity_type_for_name(provider_name),
        "PROVIDER_ADDRESS": make_address(juris["address_template"]),
        "CUSTOMER_NAME": customer_name,
        "CUSTOMER_STATE": juris["state"],
        "CUSTOMER_ENTITY_TYPE": _entity_type_for_name(customer_name),
        "CUSTOMER_ADDRESS": make_address(juris["address_template"]),
        "EFFECTIVE_DATE": rng.choice(EFFECTIVE_DATES),
        "GOVERNING_LAW_STATE": juris["state"],
        "VENUE_COUNTY": juris["county"],
        "VENUE_CITY": juris["city"],
        "ANNUAL_FEE_AMOUNT": rng.choice(ANNUAL_FEE_RANGES),
        "TERM_YEARS": rng.choice(TERM_YEARS),
        "LIABILITY_CAP_MONTHS": rng.choice(LIABILITY_CAP_MONTHS),
        "NOTICE_DAYS": rng.choice(NOTICE_DAYS),
        "IMPLEMENTATION_FEE": rng.choice(IMPLEMENTATION_FEES),
    }


def apply_substitutions(template: str, sub_map: dict) -> str:
    """Replace each {{PLACEHOLDER}} with the corresponding value from sub_map."""
    out = template
    for placeholder, value in sub_map.items():
        token = "{{" + placeholder + "}}"
        out = out.replace(token, str(value))
    return out


def process_template(tokenized_path: Path, num_variations: int, output_dir: Path) -> dict:
    """Generate N variations for one tokenized template."""
    template = tokenized_path.read_text(encoding="utf-8")
    base = tokenized_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find which placeholders are actually used in this template
    placeholders_used = set(re.findall(r"\{\{([A-Z_]+)\}\}", template))

    variations_generated = []
    for i in range(1, num_variations + 1):
        seed = abs(hash(base + str(i))) % (2**32)
        sub_map = generate_variation_map(seed)

        # Only keep substitutions for placeholders that exist in this template
        active_subs = {k: v for k, v in sub_map.items() if k in placeholders_used}
        variation = apply_substitutions(template, active_subs)

        # Sanity check: any unsubstituted placeholders left?
        leftover = re.findall(r"\{\{([A-Z_]+)\}\}", variation)
        if leftover:
            logger.warning("  variation %d has leftover placeholders: %s", i, set(leftover))

        # Save variation text
        var_path = output_dir / f"{base}__var{i:02d}.txt"
        var_path.write_text(variation, encoding="utf-8")

        # Save substitution map
        map_path = output_dir / f"{base}__var{i:02d}.json"
        map_path.write_text(json.dumps({
            "source_template": tokenized_path.name,
            "variation_index": i,
            "substitutions": active_subs,
        }, indent=2), encoding="utf-8")

        variations_generated.append({
            "filename": var_path.name,
            "provider": active_subs.get("PROVIDER_NAME", ""),
            "customer": active_subs.get("CUSTOMER_NAME", ""),
            "jurisdiction": active_subs.get("GOVERNING_LAW_STATE", ""),
            "fee": active_subs.get("ANNUAL_FEE_AMOUNT", ""),
        })

    return {
        "template": base,
        "placeholders_used": sorted(placeholders_used),
        "variations": variations_generated,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="SAAS")
    parser.add_argument("--variations", type=int, default=5,
                        help="Number of variations to generate per template")
    args = parser.parse_args()

    # Find tokenized templates
    tokenized_dir = V2_TOKENIZED / args.type / "synthetic"
    if not tokenized_dir.exists():
        logger.error("Tokenized dir not found: %s", tokenized_dir)
        return

    output_dir = V2_VARIATIONS / args.type / "synthetic"
    templates = sorted(tokenized_dir.glob("*.txt"))
    if not templates:
        logger.error("No tokenized templates found in %s", tokenized_dir)
        return

    logger.info("Generating %d variations for each of %d templates",
                args.variations, len(templates))

    manifest = {
        "type": args.type,
        "variations_per_template": args.variations,
        "total_templates": len(templates),
        "total_variations": len(templates) * args.variations,
        "templates": [],
    }

    for template_path in templates:
        result = process_template(template_path, args.variations, output_dir)
        manifest["templates"].append(result)
        logger.info("✓ %-58s %d variations", template_path.name[:58], args.variations)

    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("\n=== VARIATION GENERATION COMPLETE ===")
    logger.info("Type:              %s", args.type)
    logger.info("Templates:         %d", len(templates))
    logger.info("Variations/each:   %d", args.variations)
    logger.info("Total variations:  %d", len(templates) * args.variations)
    logger.info("Output:            %s", output_dir)


if __name__ == "__main__":
    main()
