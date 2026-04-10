#!/usr/bin/env python3
"""V2 contract tokenization — replace party-specific entities with placeholders.

Reads each contract in data/v2/raw/<TYPE>/, extracts party-specific entities
(party names, addresses, dates, fees, governing law, jurisdiction, term length,
liability cap), and produces:

  1. data/v2/tokenized/<TYPE>/<filename>.txt   — tokenized template with {{placeholders}}
  2. data/v2/tokenized/<TYPE>/<filename>.json  — extracted entity values
  3. data/v2/tokenized/<TYPE>/MANIFEST.json    — index of all tokenized contracts

The tokenized templates can later be used to:
  - Generate N variations per contract by substituting different values
  - Train the model to substitute names/dates from prompts (not memorize them)

Usage:
    python scripts/v2_tokenize_contracts.py --type SAAS
"""

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

V2_RAW = Path("/Users/jyotimishra/legal-finetune/data/v2/raw")
V2_TOKENIZED = Path("/Users/jyotimishra/legal-finetune/data/v2/tokenized")


@dataclass
class ContractEntities:
    """Extracted entities from a contract."""
    provider_name: str = ""
    provider_state: str = ""
    provider_entity_type: str = ""
    provider_address: str = ""
    customer_name: str = ""
    customer_state: str = ""
    customer_entity_type: str = ""
    customer_address: str = ""
    effective_date: str = ""
    governing_law_state: str = ""
    venue_county: str = ""
    venue_city: str = ""
    annual_fee_amount: str = ""  # e.g. "$480,000" or "USD 480,000"
    term_years: str = ""
    liability_cap_months: str = ""
    notice_days: str = ""
    extraction_confidence: float = 0.0
    extraction_notes: list = field(default_factory=list)


# ── Entity extraction patterns ────────────────────────────────────────────

# Anchor on the explicit ("Provider") and ("Customer") labels to find party names.
# This is much more reliable than trying to parse the whole preamble in one regex.
PROVIDER_NAME_PATTERN = re.compile(
    r"(?:by and between|between)\s+([^()]{4,200}?)\s*\(\s*\"Provider\"\s*\)",
    re.IGNORECASE | re.DOTALL,
)
CUSTOMER_NAME_PATTERN = re.compile(
    r"\(\s*\"Provider\"\s*\)\s*,?\s*and\s+([^()]{4,300}?)\s*\(\s*\"Customer\"\s*\)",
    re.IGNORECASE | re.DOTALL,
)

# Within a captured party block, extract structured fields
ENTITY_TYPE_PATTERN = re.compile(
    r"\b(?:a|an)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+"
    r"(corporation|limited liability company|LLC|L\.L\.C\.|limited partnership|LP|"
    r"limited liability partnership|LLP|partnership|mutual insurance company|"
    r"state agency|Schedule I bank|company)",
    re.IGNORECASE,
)
ADDRESS_PATTERN = re.compile(
    r"having its (?:principal place of business|registered office|head office|principal administrative office)\s+(?:located\s+)?at\s+(.+?)(?:\s*\(|$)",
    re.IGNORECASE | re.DOTALL,
)


def _clean_party_block(block: str) -> str:
    """Clean a party block — remove leading 'between', trailing punctuation."""
    block = block.strip()
    # Remove leading "between " if it slipped through
    block = re.sub(r"^(?:by and )?between\s+", "", block, flags=re.IGNORECASE)
    return block.strip().rstrip(",")


def _extract_party_name(block: str) -> str:
    """Extract just the party name from a party block.
    The party name is everything before the first ', a' or ', an'."""
    # Strip leading whitespace and "between"
    block = _clean_party_block(block)
    # Find the comma followed by " a " / " an " (start of "a Delaware corporation...")
    m = re.search(r",\s*(?:a|an)\s+[A-Z]", block)
    if m:
        block = block[:m.start()]
    # Strip trailing punctuation and whitespace
    return block.strip().rstrip(",.;").strip()


def _extract_party_details(block: str) -> tuple[str, str, str]:
    """Extract (state, entity_type, address) from a party block."""
    state = ""
    entity_type = ""
    address = ""
    et_match = ENTITY_TYPE_PATTERN.search(block)
    if et_match:
        state = et_match.group(1).strip()
        entity_type = et_match.group(2).strip()
    addr_match = ADDRESS_PATTERN.search(block)
    if addr_match:
        address = addr_match.group(1).strip().rstrip(",.")
    return state, entity_type, address

EFFECTIVE_DATE = re.compile(
    r"as of\s+(?:the\s+)?([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)

GOVERNING_LAW = re.compile(
    r"governed by\s+(?:and construed in accordance with\s+)?the laws of\s+(?:the\s+)?(?:State of\s+|Commonwealth of\s+|Province of\s+)?([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)",
    re.IGNORECASE,
)

VENUE = re.compile(
    r"(?:exclusive jurisdiction|brought exclusively).*?in\s+(?:the\s+)?(?:state and federal courts\s+(?:located|sitting)\s+)?(?:in\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+County|City of [A-Z][a-z]+),\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    re.IGNORECASE | re.DOTALL,
)

# Match annual fees like "$480,000" or "USD 480,000" or "Four Hundred Eighty Thousand United States Dollars ($480,000)"
ANNUAL_FEE = re.compile(
    r"(?:annual\s+(?:Subscription\s+Fee|fee)|Subscription\s+Fee\s+of)\s+(?:USD\s+)?\$?([\d,]+)\s*(?:United States Dollars)?",
    re.IGNORECASE,
)

# Term length like "for a period of three (3) years" or "shall continue for a period of 3 years"
TERM_YEARS = re.compile(
    r"(?:for a period of|continue for)\s+(?:a period of\s+)?(?:(\d+)\s*\(\d+\)|(\w+)\s*\((\d+)\))\s+years?",
    re.IGNORECASE,
)

# Liability cap like "twelve (12) months" or "12 months"
LIABILITY_CAP = re.compile(
    r"(?:fees paid|amount paid).*?(?:in\s+the\s+|preceding\s+the\s+)(?:(\w+)\s*\((\d+)\)|(\d+))\s+months?",
    re.IGNORECASE,
)

NOTICE_DAYS = re.compile(
    r"(?:upon\s+|with\s+|provide\s+|providing\s+)(?:(\w+)\s*\((\d+)\)|(\d+))\s+days?'?\s+(?:prior\s+)?(?:written\s+)?notice",
    re.IGNORECASE,
)


# ── Number-to-word helpers ────────────────────────────────────────────────

WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "twenty-four": 24, "thirty": 30, "thirty-six": 36, "sixty": 60, "ninety": 90,
}


def parse_number(value):
    """Convert a parsed number (string or word) to int. Returns None if unparseable."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value.isdigit():
            return int(value)
        return WORD_TO_NUM.get(value)
    return None


# ── Extraction ────────────────────────────────────────────────────────────

US_STATES = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
    "Ontario", "Quebec", "British Columbia", "Alberta",  # Canada
}

ENTITY_TYPES = {
    "corporation", "limited liability company", "llc", "limited partnership",
    "lp", "limited liability partnership", "llp", "partnership", "company",
    "mutual insurance company", "state agency",
}


def _valid_party_name(name: str) -> bool:
    """A party name must be 4+ chars, start with a capital, contain a letter,
    and not look like garbage. 'The Board of Regents...' is valid; 'and the' is not."""
    if not name or len(name) < 4:
        return False
    if not name[0].isupper():
        return False
    if not any(c.isalpha() for c in name):
        return False
    # Reject obvious garbage but allow legitimate names that start with "The"
    # (e.g., "The Board of Regents", "The Coca-Cola Company")
    bad_starts = ("between ", "and ", "a ", "an ", "with ", "for ", "of ", "to ")
    if any(name.lower().startswith(b) for b in bad_starts):
        return False
    # Reject if it's all lowercase (regex captured wrong)
    if name == name.lower():
        return False
    # Must contain at least one upper-case letter beyond the first char OR have 'Inc/Corp/LLC/Ltd/LLP/Co'
    has_proper_capitalization = any(c.isupper() for c in name[1:])
    has_corp_suffix = any(suffix in name for suffix in ["Inc", "Corp", "LLC", "Ltd", "LLP", "L.L.C", "Company", "Co.", "Bank", "Group", "Systems", "Solutions", "Holdings", "Pharmaceuticals", "Bistro"])
    if not (has_proper_capitalization or has_corp_suffix):
        return False
    return True


def _valid_state(state: str) -> bool:
    """A state must be 4+ chars and ideally match a known US state."""
    if not state or len(state) < 4:
        return False
    return state in US_STATES or state.title() in US_STATES


def _valid_entity_type(et: str) -> bool:
    if not et or len(et) < 3:
        return False
    return et.lower() in ENTITY_TYPES or any(t in et.lower() for t in ENTITY_TYPES)


def extract_entities(text: str) -> ContractEntities:
    """Extract structured entities from a contract with strict validation."""
    entities = ContractEntities()
    confidence_points = 0
    max_points = 10

    # ── Anchored extraction: find party blocks via ("Provider") / ("Customer") labels ──
    head = text[:6000]

    prov_match = PROVIDER_NAME_PATTERN.search(head)
    if prov_match:
        prov_block = prov_match.group(1)
        prov_name = _extract_party_name(prov_block)
        if _valid_party_name(prov_name):
            entities.provider_name = prov_name
            confidence_points += 1
        state, etype, address = _extract_party_details(prov_block)
        if _valid_state(state):
            entities.provider_state = state
            confidence_points += 1
        if _valid_entity_type(etype):
            entities.provider_entity_type = etype
            confidence_points += 1
        if address and len(address) > 10:
            entities.provider_address = address.split('\n')[0].strip()
            confidence_points += 1

    cust_match = CUSTOMER_NAME_PATTERN.search(head)
    if cust_match:
        cust_block = cust_match.group(1)
        cust_name = _extract_party_name(cust_block)
        if _valid_party_name(cust_name):
            entities.customer_name = cust_name
            confidence_points += 1
        state, etype, address = _extract_party_details(cust_block)
        if _valid_state(state):
            entities.customer_state = state
            confidence_points += 1
        if _valid_entity_type(etype):
            entities.customer_entity_type = etype
            confidence_points += 1
        if address and len(address) > 10:
            entities.customer_address = address.split('\n')[0].strip()
            confidence_points += 1

    # Effective date
    date_match = EFFECTIVE_DATE.search(text[:3000])
    if date_match:
        entities.effective_date = date_match.group(1).strip()
        confidence_points += 1

    # Governing law
    gov_match = GOVERNING_LAW.search(text)
    if gov_match:
        entities.governing_law_state = gov_match.group(1).strip()
        confidence_points += 1

    # Venue
    venue_match = VENUE.search(text)
    if venue_match:
        entities.venue_county = venue_match.group(1).strip()
        entities.venue_city = venue_match.group(2).strip()
        confidence_points += 1

    # Annual fee
    fee_match = ANNUAL_FEE.search(text)
    if fee_match:
        entities.annual_fee_amount = "$" + fee_match.group(1).strip()
        confidence_points += 1

    # Term years
    term_match = TERM_YEARS.search(text)
    if term_match:
        digit_form = term_match.group(1)
        word_form = term_match.group(2)
        digit_in_paren = term_match.group(3)
        years = parse_number(digit_form) or parse_number(word_form) or parse_number(digit_in_paren)
        if years:
            entities.term_years = str(years)
            confidence_points += 1

    # Liability cap months
    cap_match = LIABILITY_CAP.search(text)
    if cap_match:
        word_form = cap_match.group(1)
        digit_in_paren = cap_match.group(2)
        digit_form = cap_match.group(3)
        months = parse_number(word_form) or parse_number(digit_in_paren) or parse_number(digit_form)
        if months:
            entities.liability_cap_months = str(months)
            confidence_points += 1

    # Notice days (take the first match — usually termination notice)
    notice_match = NOTICE_DAYS.search(text)
    if notice_match:
        word_form = notice_match.group(1)
        digit_in_paren = notice_match.group(2)
        digit_form = notice_match.group(3)
        days = parse_number(word_form) or parse_number(digit_in_paren) or parse_number(digit_form)
        if days:
            entities.notice_days = str(days)
            confidence_points += 1

    entities.extraction_confidence = round(confidence_points / max_points, 2)
    return entities


# ── Tokenization ──────────────────────────────────────────────────────────

def tokenize(text: str, entities: ContractEntities) -> tuple[str, dict]:
    """Replace entity values with {{placeholder}} tokens.

    Uses word-boundary regex (lookarounds) to prevent substring matches like
    'a' inside other words being incorrectly replaced.

    Returns the tokenized text and a dict of replacements made.
    """
    out = text
    replacements_made = {}

    # Build substitution list. Skip values shorter than 4 chars to prevent
    # accidental substring matches.
    MIN_LEN = 4
    substitutions = []

    def add(placeholder, value):
        if value and len(value.strip()) >= MIN_LEN:
            substitutions.append((placeholder, value.strip()))

    add("PROVIDER_NAME", entities.provider_name)
    add("CUSTOMER_NAME", entities.customer_name)
    add("PROVIDER_ADDRESS", entities.provider_address)
    add("CUSTOMER_ADDRESS", entities.customer_address)
    add("PROVIDER_ENTITY_TYPE", entities.provider_entity_type)
    add("CUSTOMER_ENTITY_TYPE", entities.customer_entity_type)
    add("PROVIDER_STATE", entities.provider_state)
    add("CUSTOMER_STATE", entities.customer_state)
    add("EFFECTIVE_DATE", entities.effective_date)
    add("GOVERNING_LAW_STATE", entities.governing_law_state)
    add("VENUE_COUNTY", entities.venue_county)
    add("VENUE_CITY", entities.venue_city)

    # Sort by length descending so longer strings are replaced first.
    # This prevents "California" being matched inside "California, United States"
    # before the longer string is processed.
    substitutions.sort(key=lambda s: len(s[1]), reverse=True)

    for placeholder, value in substitutions:
        token = f"{{{{{placeholder}}}}}"
        # Word-boundary lookarounds: not preceded by a word char and not followed by one.
        # This ensures "a" only matches a standalone "a" not inside "and", "have", etc.
        # We use \W (non-word) lookarounds rather than \b because some entity values
        # contain commas/periods that \b doesn't handle correctly.
        escaped = re.escape(value)
        pattern = r"(?<![A-Za-z0-9_])" + escaped + r"(?![A-Za-z0-9_])"
        try:
            count = len(re.findall(pattern, out))
        except re.error:
            # Pattern too complex — fall back to plain literal replacement
            count = out.count(value)
            if count > 0:
                out = out.replace(value, token)
                replacements_made[placeholder] = {"original": value, "occurrences": count, "method": "literal"}
            continue
        if count > 0:
            out = re.sub(pattern, token, out)
            replacements_made[placeholder] = {
                "original": value,
                "occurrences": count,
            }

    return out, replacements_made


# ── Main pipeline ─────────────────────────────────────────────────────────

def process_contract(input_path: Path, output_dir: Path) -> dict:
    """Process one contract: extract entities, tokenize, save."""
    text = input_path.read_text(encoding="utf-8")
    entities = extract_entities(text)
    tokenized_text, replacements = tokenize(text, entities)

    output_dir.mkdir(parents=True, exist_ok=True)
    base = input_path.stem

    # Save tokenized template
    tokenized_path = output_dir / f"{base}.txt"
    tokenized_path.write_text(tokenized_text, encoding="utf-8")

    # Save entity JSON
    entities_path = output_dir / f"{base}.json"
    entity_data = {
        "source_file": input_path.name,
        "entities": asdict(entities),
        "replacements_made": replacements,
        "total_substitutions": sum(r["occurrences"] for r in replacements.values()),
    }
    entities_path.write_text(json.dumps(entity_data, indent=2), encoding="utf-8")

    return entity_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="SAAS")
    parser.add_argument("--include-synthetic", action="store_true", default=True)
    args = parser.parse_args()

    input_dir = V2_RAW / args.type
    output_dir = V2_TOKENIZED / args.type
    if not input_dir.exists():
        logger.error("Input dir not found: %s", input_dir)
        return

    # Collect all contract files
    contracts = sorted(input_dir.glob("*.txt"))
    if args.include_synthetic:
        synthetic_dir = input_dir / "synthetic"
        if synthetic_dir.exists():
            contracts.extend(sorted(synthetic_dir.glob("*.txt")))

    logger.info("Processing %d contracts of type %s", len(contracts), args.type)

    manifest = {
        "type": args.type,
        "total_contracts": len(contracts),
        "contracts": [],
    }

    avg_confidence = 0.0
    for contract_path in contracts:
        try:
            # Save synthetic contracts to synthetic/ subdir to mirror input structure
            if "synthetic" in contract_path.parts:
                target_dir = output_dir / "synthetic"
            else:
                target_dir = output_dir

            entity_data = process_contract(contract_path, target_dir)
            confidence = entity_data["entities"]["extraction_confidence"]
            avg_confidence += confidence
            ent = entity_data["entities"]
            logger.info("✓ %s | conf=%.2f | %s ↔ %s | gov=%s | term=%sy",
                        contract_path.name[:50],
                        confidence,
                        ent.get("provider_name", "?")[:25],
                        ent.get("customer_name", "?")[:25],
                        ent.get("governing_law_state", "?")[:15],
                        ent.get("term_years", "?"))
            manifest["contracts"].append({
                "filename": contract_path.name,
                "confidence": confidence,
                "substitutions": entity_data["total_substitutions"],
                "provider": ent.get("provider_name", ""),
                "customer": ent.get("customer_name", ""),
            })
        except Exception as e:
            logger.error("Failed to process %s: %s", contract_path.name, e)

    if contracts:
        avg_confidence /= len(contracts)

    manifest["avg_extraction_confidence"] = round(avg_confidence, 2)
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("\n=== TOKENIZATION COMPLETE ===")
    logger.info("Type:       %s", args.type)
    logger.info("Processed:  %d contracts", len(contracts))
    logger.info("Avg confid: %.2f", avg_confidence)
    logger.info("Output:     %s", output_dir)


if __name__ == "__main__":
    main()
