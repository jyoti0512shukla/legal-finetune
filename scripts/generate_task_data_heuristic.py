#!/usr/bin/env python3
"""Generate task-specific training data using heuristic extraction from real EDGAR text.

Unlike the template-based approach, this reads actual contract text and extracts
real values using regex patterns. Produces genuinely varied training examples.
"""

import json
import re
import os
import random
from pathlib import Path
from collections import Counter

import jsonlines

random.seed(42)

RAW_DIR = Path("data/raw/edgar")
CLAUSES_PATH = Path("data/processed/clauses/edgar_clauses.jsonl")
OUTPUT_DIR = Path("data/processed/v3_tasks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Extraction patterns ──────────────────────────────────────────────────────

PARTY_PATTERNS = [
    re.compile(r'(?:by and between|between)\s+(.{5,80}?)\s*(?:\(|,\s*a\s)', re.IGNORECASE),
    re.compile(r'(?:by and between|between)\s+(.{5,80}?)\s+and\s+(.{5,80}?)(?:\s*\(|\s*,)', re.IGNORECASE),
    re.compile(r'"Company"\)\s*,?\s*and\s+(.{5,80}?)\s*\(', re.IGNORECASE),
    re.compile(r'\("([^"]{3,60})"\)', re.IGNORECASE),
]

DATE_PATTERNS = [
    re.compile(r'(?:effective|dated?)\s+(?:as of\s+)?(\w+ \d{1,2},?\s*\d{4})', re.IGNORECASE),
    re.compile(r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:day of\s+)?\w+,?\s*\d{4})', re.IGNORECASE),
    re.compile(r'(\w+ \d{1,2},?\s*\d{4})', re.IGNORECASE),
]

GOV_LAW_PATTERNS = [
    re.compile(r'(?:governed by|construed (?:in accordance with|under))\s+(?:the\s+)?(?:laws?\s+of\s+)?(?:the\s+)?(?:State\s+of\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', re.IGNORECASE),
    re.compile(r'laws?\s+of\s+(?:the\s+)?(?:State\s+of\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', re.IGNORECASE),
]

NOTICE_PATTERNS = [
    re.compile(r'(\d+)\s*(?:\([^)]+\))?\s*(?:calendar\s+|business\s+)?days?\s*(?:prior\s+)?(?:written\s+)?notice', re.IGNORECASE),
]

LIABILITY_PATTERNS = [
    re.compile(r'(?:aggregate|total|maximum)\s+liability[^.]*?(?:shall not exceed|not to exceed|limited to)\s+([^.]{10,100})', re.IGNORECASE),
    re.compile(r'(?:shall not exceed|limited to|not to exceed)\s+([^.]{10,100}?)(?:\.|\sin)', re.IGNORECASE),
]

VALUE_PATTERNS = [
    re.compile(r'\$\s*[\d,]+(?:\.\d{2})?', re.IGNORECASE),
    re.compile(r'(?:USD|EUR|GBP|INR)\s*[\d,]+(?:\.\d{2})?', re.IGNORECASE),
]

ARBITRATION_PATTERNS = [
    re.compile(r'(?:arbitrat\w+)[^.]*?(?:in|at|before)\s+([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+){0,3})', re.IGNORECASE),
]


def extract_terms(text: str) -> dict:
    """Extract key terms from contract text using regex patterns."""
    terms = {
        "party_a": None, "party_b": None, "effective_date": None,
        "expiry_date": None, "contract_value": None, "liability_cap": None,
        "governing_law": None, "notice_period_days": None, "arbitration_venue": None,
    }

    # Parties
    quoted_parties = re.findall(r'\("([A-Z][^"]{2,50})"\)', text[:3000])
    if len(quoted_parties) >= 2:
        terms["party_a"] = quoted_parties[0]
        terms["party_b"] = quoted_parties[1]
    elif len(quoted_parties) == 1:
        terms["party_a"] = quoted_parties[0]

    # Also try "by and between" pattern
    if not terms["party_a"]:
        for pat in PARTY_PATTERNS:
            m = pat.search(text[:3000])
            if m:
                terms["party_a"] = m.group(1).strip().rstrip(',')
                if m.lastindex and m.lastindex >= 2:
                    terms["party_b"] = m.group(2).strip().rstrip(',')
                break

    # Dates
    for pat in DATE_PATTERNS:
        m = pat.search(text[:2000])
        if m:
            terms["effective_date"] = m.group(1).strip()
            break

    # Governing law
    for pat in GOV_LAW_PATTERNS:
        m = pat.search(text)
        if m:
            terms["governing_law"] = m.group(1).strip()
            break

    # Notice period
    for pat in NOTICE_PATTERNS:
        m = pat.search(text)
        if m:
            terms["notice_period_days"] = m.group(1)
            break

    # Liability cap
    for pat in LIABILITY_PATTERNS:
        m = pat.search(text)
        if m:
            terms["liability_cap"] = m.group(1).strip()
            break

    # Contract value
    for pat in VALUE_PATTERNS:
        m = pat.search(text[:5000])
        if m:
            terms["contract_value"] = m.group(0).strip()
            break

    # Arbitration
    for pat in ARBITRATION_PATTERNS:
        m = pat.search(text)
        if m:
            terms["arbitration_venue"] = m.group(1).strip()
            break

    return terms


# ── Checklist heuristics ──────────────────────────────────────────────────────

CLAUSE_SIGNALS = {
    "LIABILITY_LIMIT": ["limitation of liability", "aggregate liability", "shall not exceed", "cap on liability", "maximum liability"],
    "INDEMNITY": ["indemnif", "hold harmless", "defend and indemnify"],
    "TERMINATION_CONVENIENCE": ["termination for convenience", "terminate without cause", "terminate at any time"],
    "TERMINATION_CAUSE": ["termination for cause", "material breach", "right to terminate"],
    "FORCE_MAJEURE": ["force majeure", "act of god", "beyond reasonable control"],
    "CONFIDENTIALITY": ["confidential information", "non-disclosure", "shall not disclose"],
    "GOVERNING_LAW": ["governing law", "governed by", "laws of"],
    "DISPUTE_RESOLUTION": ["arbitrat", "mediat", "dispute resolution", "exclusive jurisdiction"],
    "IP_OWNERSHIP": ["intellectual property", "work product", "ip rights", "ownership of"],
    "DATA_PROTECTION": ["data protection", "personal data", "privacy", "gdpr", "dpdpa"],
    "PAYMENT_TERMS": ["payment", "invoice", "fees", "compensation", "net 30"],
    "ASSIGNMENT": ["assignment", "shall not assign", "transfer of rights"],
}


def check_clauses(text: str) -> list[dict]:
    """Check contract text for presence of 12 standard clauses."""
    text_lower = text.lower()
    results = []

    for clause_id, signals in CLAUSE_SIGNALS.items():
        found_signals = [s for s in signals if s in text_lower]

        if len(found_signals) >= 2:
            status = "PRESENT"
            risk = "LOW"
            finding = f"Clause found with strong signals: {', '.join(found_signals[:2])}"
        elif len(found_signals) == 1:
            status = "WEAK"
            risk = "MEDIUM"
            finding = f"Partial reference found ({found_signals[0]}) but clause may be incomplete"
        else:
            status = "MISSING"
            risk = "HIGH"
            finding = f"No {clause_id.replace('_', ' ').lower()} language found in the contract"

        results.append({
            "clause_id": clause_id,
            "status": status,
            "risk_level": risk,
            "finding": finding,
        })

    return results


# ── Redline heuristics ────────────────────────────────────────────────────────

WEAKNESS_PATTERNS = {
    "PAYMENT": [
        ("No late payment interest specified", "Add: Overdue amounts shall bear interest at 2% per annum above the prevailing base rate from the due date until payment in full."),
        ("No disputed invoice procedure", "Add: Any dispute regarding an invoice must be raised in writing within 15 days of receipt, with undisputed amounts paid on time."),
        ("Payment timeline not specific", "Clarify: All undisputed invoices shall be paid within 30 (thirty) days of receipt."),
    ],
    "LIABILITY": [
        ("No aggregate liability cap", "Add: The aggregate liability of either Party shall not exceed the total fees paid or payable in the 12 months preceding the claim."),
        ("No exclusion of consequential damages", "Add: Neither Party shall be liable for any indirect, consequential, special, or punitive damages, regardless of cause."),
        ("Liability cap is one-sided", "Amend to make mutual: Each Party's aggregate liability shall be subject to the same cap."),
    ],
    "TERMINATION": [
        ("No cure period for material breach", "Add: The breaching Party shall have 30 days from written notice to cure any material breach before termination takes effect."),
        ("No convenience termination right", "Add: Either Party may terminate this Agreement for convenience upon 90 days' prior written notice."),
        ("Effects of termination not specified", "Add: Upon termination, each Party shall return or destroy Confidential Information and pay all outstanding fees."),
    ],
    "CONFIDENTIALITY": [
        ("No exceptions to confidentiality", "Add standard exceptions: information that (a) is or becomes publicly available, (b) was known prior to disclosure, (c) is independently developed, or (d) is required by law."),
        ("No survival period specified", "Add: Confidentiality obligations shall survive termination for a period of 3 (three) years."),
        ("No return/destruction obligation", "Add: Upon termination, the Receiving Party shall return or destroy all Confidential Information within 30 days."),
    ],
    "FORCE_MAJEURE": [
        ("No notification requirement", "Add: The affected Party must notify the other Party within 7 days of a Force Majeure event."),
        ("No prolonged event termination right", "Add: If a Force Majeure event continues for more than 90 consecutive days, either Party may terminate without liability."),
    ],
    "IP_RIGHTS": [
        ("No clear work product assignment", "Add: All work product created under this Agreement shall vest in and be owned by the Client upon payment."),
        ("Background IP not addressed", "Add: Each Party retains ownership of its Background IP. No Background IP is transferred under this Agreement."),
    ],
    "GOVERNING_LAW": [
        ("No dispute resolution mechanism", "Add: Any dispute shall first be referred to senior management for 30 days, failing which it shall be resolved by the courts of the specified jurisdiction."),
        ("Governing law not specified", "Add: This Agreement shall be governed by and construed in accordance with the laws of [State], without regard to conflict of laws principles."),
    ],
}


def suggest_redline(clause_text: str, clause_type: str) -> dict | None:
    """Analyze a clause and suggest improvements based on what's missing."""
    patterns = WEAKNESS_PATTERNS.get(clause_type, [])
    if not patterns:
        return None

    text_lower = clause_text.lower()

    # Find the first weakness that applies
    for issue, fix in patterns:
        # Check if the fix topic is already covered
        fix_keywords = fix.lower().split()[:5]
        if not any(kw in text_lower for kw in fix_keywords if len(kw) > 4):
            return {
                "clause_name": clause_type.replace("_", " ").title(),
                "issue": issue,
                "suggested_language": fix,
                "rationale": f"Adding this provision strengthens the {clause_type.replace('_', ' ').lower()} clause by addressing a common gap that could expose a party to risk."
            }

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating task-specific training data (heuristic)...\n")

    # Load raw contracts
    contracts = []
    for type_dir in RAW_DIR.iterdir():
        if not type_dir.is_dir():
            continue
        for txt_file in sorted(type_dir.glob("*.txt"))[:25]:
            text = txt_file.read_text(encoding="utf-8")
            if len(text) > 500:
                contracts.append({
                    "source_id": txt_file.stem,
                    "contract_type": type_dir.name,
                    "text": text,
                })
    random.shuffle(contracts)
    print(f"Loaded {len(contracts)} contracts")

    # Load classified clauses
    clauses = []
    with jsonlines.open(str(CLAUSES_PATH)) as reader:
        for row in reader:
            if row["clause_type"] != "UNKNOWN":
                clauses.append(row)
    random.shuffle(clauses)
    print(f"Loaded {len(clauses)} classified clauses")

    # ── 1. Extraction examples ──
    extraction_examples = []
    for contract in contracts[:150]:
        text = contract["text"][:4000]
        terms = extract_terms(text)

        # Only keep if at least 3 fields extracted
        non_null = sum(1 for v in terms.values() if v)
        if non_null < 3:
            continue

        extraction_examples.append({
            "instruction": f"Extract key terms from this contract:\n\n{text}",
            "response": json.dumps(terms),
            "clause_type": "DEFINITIONS",
            "contract_type": contract["contract_type"],
            "jurisdiction": "California, United States",
            "source": "extraction_labeled",
            "quality_score": 4.5,
        })

    path = OUTPUT_DIR / "extraction_examples.jsonl"
    with jsonlines.open(str(path), mode="w") as w:
        for ex in extraction_examples:
            w.write(ex)
    print(f"Extraction: {len(extraction_examples)} examples (wrote {path.name})")

    # ── 2. Checklist examples ──
    checklist_examples = []
    for contract in contracts[:100]:
        text = contract["text"][:6000]
        checklist = check_clauses(text)

        # Verify it's not all MISSING or all PRESENT (need variety)
        statuses = Counter(c["status"] for c in checklist)
        if statuses.get("MISSING", 0) == 12 or statuses.get("PRESENT", 0) == 12:
            continue

        checklist_examples.append({
            "instruction": f"Check this contract for 12 standard clauses:\n\n{text}",
            "response": json.dumps({"clauses": checklist}),
            "clause_type": "GENERAL_PROVISIONS",
            "contract_type": contract["contract_type"],
            "jurisdiction": "California, United States",
            "source": "checklist_labeled",
            "quality_score": 4.5,
        })

    path = OUTPUT_DIR / "checklist_examples.jsonl"
    with jsonlines.open(str(path), mode="w") as w:
        for ex in checklist_examples:
            w.write(ex)
    print(f"Checklist: {len(checklist_examples)} examples (wrote {path.name})")

    # Check distribution
    all_statuses = Counter()
    for ex in checklist_examples:
        for c in json.loads(ex["response"])["clauses"]:
            all_statuses[c["status"]] += 1
    print(f"  Status distribution: {dict(all_statuses)}")

    # ── 3. Redline examples ──
    redline_examples = []
    for clause in clauses[:300]:
        result = suggest_redline(clause["text"], clause["clause_type"])
        if result:
            redline_examples.append({
                "instruction": f"This {clause['clause_type']} clause needs improvement. Suggest better language:\n\n{clause['text'][:2000]}",
                "response": json.dumps(result),
                "clause_type": clause["clause_type"],
                "contract_type": clause["contract_type"],
                "jurisdiction": "California, United States",
                "source": "redline_labeled",
                "quality_score": 4.5,
            })

    path = OUTPUT_DIR / "redline_examples.jsonl"
    with jsonlines.open(str(path), mode="w") as w:
        for ex in redline_examples:
            w.write(ex)
    print(f"Redline: {len(redline_examples)} examples (wrote {path.name})")

    # Check uniqueness
    unique = len(set(ex["response"] for ex in redline_examples))
    print(f"  Unique responses: {unique}/{len(redline_examples)} ({100*unique//max(len(redline_examples),1)}%)")

    total = len(extraction_examples) + len(checklist_examples) + len(redline_examples)
    print(f"\nTotal: {total} task examples")


if __name__ == "__main__":
    main()
