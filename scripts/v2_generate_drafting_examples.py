#!/usr/bin/env python3
"""V2 drafting examples generator — produce {instruction, output} training pairs.

For each tokenized contract template + entity map + per-contract metadata, this
generator extracts the major clauses (IP, Liability, Termination, Indemnity,
Confidentiality, Data Privacy, Service Level) and emits realistic
instruction → clause training examples.

Each example:
  - reuses an existing high-quality clause as the OUTPUT (no LLM hallucination)
  - generates a fresh, deterministic variation of party names / states /
    jurisdictions / fees so the model learns to substitute placeholders
  - builds a brief that mentions the same parties, industry, deal size,
    governing law, regulatory context, negotiation bias and commercial terms

Why this is the right shape:
  - The model needs to learn the *mapping* from a brief to a fully drafted
    clause that respects the slot values in the brief. By holding the clause
    body constant (modulo placeholder substitution) and varying both the brief
    *and* the placeholder values together, every example teaches the same
    skill: read the slots → apply them → produce the clause.
  - Industry / state / placeholder handling falls out for free because the
    same variation map is applied to both sides of the example.

Usage:
    python scripts/v2_generate_drafting_examples.py --type SAAS --variations 4

Output:
    data/v2/training/drafting/SAAS_drafting_examples.jsonl
    + per-clause-type counts manifest
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path

# Reuse the substitution-map generator from the variations script so the
# pools (party names, jurisdictions, fee ranges, addresses) stay consistent.
from v2_generate_variations import (
    generate_variation_map,
    apply_substitutions,
    ANNUAL_FEE_RANGES,
    IMPLEMENTATION_FEES,
    PROVIDER_NAMES,
    CUSTOMER_NAMES,
)

# Split the master fee list into deal-size tiers so a SMB contract never
# accidentally gets an enterprise fee in its variation.
# These slice boundaries match the comments in v2_generate_variations.py.
SMB_FEES = ANNUAL_FEE_RANGES[0:5]            # $12K – $48K
MID_MARKET_FEES = ANNUAL_FEE_RANGES[5:11]    # $72K – $240K
ENTERPRISE_FEES = ANNUAL_FEE_RANGES[11:]     # $360K – $1.8M

SMB_IMPL_FEES = IMPLEMENTATION_FEES[0:3]     # $10K – $50K
MID_MARKET_IMPL_FEES = IMPLEMENTATION_FEES[3:5]  # $75K – $100K
ENTERPRISE_IMPL_FEES = IMPLEMENTATION_FEES[5:]   # $150K – $200K

# Customer types whose identity is too tied to the contract context to be
# safely swapped out by the variation generator. For these, the variation
# keeps the ORIGINAL customer name, entity type, jurisdiction and address.
SPECIAL_CUSTOMER_TYPES = {
    "state_agency",
    "regulated_bank",
    "national_bank",
    "mutual_insurance_company",
    "law_firm",
    "professional_corporation",
    "broker_dealer",
    # Phase C–E additions: customer types whose contracts include very
    # industry-specific regulatory or business language that would clash
    # with a generic substitute customer name.
    "pharmaceutical_company",        # PharmaTrack — clinical trials, 21 CFR Part 11
    "regional_health_system",        # MedPath — HIPAA EHR for hospital network
    "academic_medical_center",       # TeleVita — telemedicine + NIH grants + FDA SaMD
    "registered_investment_adviser", # KeystoneAccess — SEC IA Cybersecurity Rule
    "franchisor_holding_company",    # BrandWaffle — franchise system + FTC Franchise Rule
    # MSA bucket additions
    "law_firm_engagement_with_corporate_client",  # Northwood — outside corporate counsel
    "public_company_audit_committee",             # Capstone — PCAOB-registered financial audit
    "federal_executive_agency",                   # Patriot IT — federal IT contractor
    "regional_health_system_locums",              # Mercer — healthcare staffing for hospitals
    "regional_bank_holding",                      # Sterling — bank risk advisory
    "pharmaceutical_sponsor",                     # BioCardinal — CRO for pharma sponsor
    "prime_federal_contractor",                   # DefenseCorp — federal subcontract under prime
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

V2_ROOT = Path("/Users/jyotimishra/legal-finetune/data/v2")
V2_TOKENIZED = V2_ROOT / "tokenized"
V2_RAW = V2_ROOT / "raw"
V2_TRAINING = V2_ROOT / "training" / "drafting"


# ── Clause type taxonomy ─────────────────────────────────────────────────
#
# Each clause type maps to a set of title keywords. We do title-based
# matching against the ARTICLE headers so the extractor works regardless of
# article number (different contracts use different numbering — e.g.
# MenuMaster has ARTICLE 3 — HARDWARE which shifts everything by one).

CLAUSE_TITLE_KEYWORDS = {
    "intellectual_property": [
        "INTELLECTUAL PROPERTY",
        "WORK PRODUCT",                    # MSA: work made for hire / assignment
        "OWNERSHIP OF DELIVERABLES",       # MSA alternative
        "OWNERSHIP OF WORK PRODUCT",       # MSA alternative
    ],
    "limitation_of_liability": [
        "LIMITATION OF LIABILITY",
        "LIABILITY",  # fallback if "LIMITATION OF" missing
    ],
    "termination": [
        "TERMINATION",
        "TERM AND TERMINATION",
    ],
    "indemnification": [
        "INDEMNIFICATION",
        "INDEMNITY",
    ],
    "confidentiality": [
        "CONFIDENTIALITY",
    ],
    "data_privacy": [
        "CUSTOMER DATA",
        "DATA AND PRIVACY",
        "DATA, PRIVACY",
        "DATA PROTECTION",
    ],
    "service_level": [
        "SERVICE LEVEL",
        "SLA AND SUPPORT",
    ],
    # MSA-flavored clause types
    "services_scope": [
        "SCOPE OF SERVICES",
        "SCOPE OF WORK",
        "SERVICES AND SCOPE",
    ],
    "deliverables_acceptance": [
        "DELIVERABLES AND ACCEPTANCE",
        "DELIVERABLES",
        "ACCEPTANCE OF DELIVERABLES",
    ],
    "personnel": [
        "KEY PERSONNEL",
        "PERSONNEL AND STAFFING",
        "PERSONNEL",
    ],
    "subcontractors": [
        "SUBCONTRACTORS",
        "SUBCONTRACTING",
    ],
    "insurance": [
        "INSURANCE",
    ],
}

# Human-readable clause names for the brief text. The display name can vary
# by contract type — e.g. "Intellectual Property Rights" for SaaS but
# "Work Product and IP Ownership" for MSA — so we override per-type below.
CLAUSE_DISPLAY_NAMES = {
    "intellectual_property": "Intellectual Property Rights",
    "limitation_of_liability": "Limitation of Liability",
    "termination": "Termination",
    "indemnification": "Indemnification",
    "confidentiality": "Confidentiality",
    "data_privacy": "Customer Data and Privacy",
    "service_level": "Service Level Agreement",
    "services_scope": "Scope of Services",
    "deliverables_acceptance": "Deliverables and Acceptance",
    "personnel": "Personnel and Staffing",
    "subcontractors": "Subcontractors",
    "insurance": "Insurance",
}

# Per-contract-type overrides for the display name. If a (contract_type,
# clause_type) pair has an entry here, it wins over CLAUSE_DISPLAY_NAMES.
CLAUSE_DISPLAY_NAMES_BY_TYPE = {
    "MSA": {
        "intellectual_property": "Work Product and IP Ownership",
    },
}

# Standard provisions per clause type — used to flesh out the brief so the
# model learns which sub-clauses are expected. These are *prompts to the
# model*, not constraints on the output.
CLAUSE_STANDARD_PROVISIONS = {
    "intellectual_property": [
        "Provider ownership of the Platform and all related IP",
        "Customer ownership of Customer Data",
        "Feedback license back to Provider",
        "no implied licenses",
    ],
    "limitation_of_liability": [
        "aggregate cap tied to fees paid in the prior 12 months",
        "exclusion of indirect, consequential, special and punitive damages",
        "exceptions for confidentiality, indemnity and data breaches",
        "basis-of-the-bargain language",
    ],
    "termination": [
        "termination for material breach with cure period",
        "termination for insolvency",
        "effects of termination (cessation of access, data return)",
        "transition assistance",
    ],
    "indemnification": [
        "Provider IP infringement indemnity for Customer",
        "Customer indemnity for misuse and Customer Data claims",
        "indemnification procedure (notice, control, cooperation)",
        "mitigation options if Platform becomes subject to a claim",
    ],
    "confidentiality": [
        "definition of Confidential Information",
        "use restrictions and standard of care",
        "exclusions (publicly known, independently developed, etc.)",
        "compelled disclosure procedure",
        "return or destruction on termination",
    ],
    "data_privacy": [
        "Customer ownership of Customer Data",
        "limited license for Provider to process",
        "security safeguards (encryption, access controls)",
        "breach notification within a defined window",
        "data residency",
        "data return and deletion on termination",
    ],
    "service_level": [
        "uptime commitment with calculation methodology",
        "excluded downtime (scheduled maintenance, force majeure)",
        "service credits as the sole financial remedy",
        "support tiers and response time targets",
    ],
    # MSA-flavored standard provisions
    "services_scope": [
        "scope of services described by reference to one or more SOWs",
        "professional and workmanlike performance standard",
        "Provider responsibility for results of services",
        "Customer cooperation obligations",
    ],
    "deliverables_acceptance": [
        "objective acceptance criteria set in each SOW",
        "Customer review and acceptance period",
        "rejection notice and cure right for non-conforming deliverables",
        "deemed acceptance after a defined window",
    ],
    "personnel": [
        "key personnel commitment with no replacement without consent",
        "background checks and required training",
        "removal of unsuitable personnel on Customer request",
        "non-solicitation of personnel during the engagement",
    ],
    "subcontractors": [
        "Customer prior written consent for subcontractors",
        "flow-down of confidentiality, IP, security obligations",
        "Provider remains primarily responsible for subcontractor performance",
        "subcontractor list maintained and updated",
    ],
    "insurance": [
        "Commercial General Liability minimum coverage",
        "Professional Liability / Errors & Omissions coverage",
        "Cyber Liability coverage where Provider handles Customer data",
        "Workers' Compensation as required by state law",
        "Provider names Customer as additional insured where applicable",
    ],
}

# Per-contract-type overrides for the standard provisions list. The IP
# clause in particular is *very* different between SaaS (license-grant) and
# MSA (work-product assignment) — without the override, the brief would
# instruct the model to write SaaS-style IP language inside an MSA clause,
# which is the wrong training signal.
CLAUSE_STANDARD_PROVISIONS_BY_TYPE = {
    "MSA": {
        "intellectual_property": [
            "Provider's pre-existing IP and tools retained by Provider",
            "Work Product assigned to Customer (work made for hire to the maximum extent permitted, with express assignment as backup)",
            "license-back to Provider for any pre-existing IP embedded in Work Product",
            "Customer's pre-existing IP and Customer Materials retained by Customer",
            "no implied licenses",
        ],
        "termination": [
            "termination for material breach with cure period",
            "termination for insolvency",
            "Customer right to terminate for convenience on notice",
            "delivery and assignment of Work Product through the termination date",
            "wind-down assistance for in-progress SOWs",
        ],
    },
}

# Different brief opening phrases — keeps the training data from looking
# templated and matches the kind of variation real users produce.
BRIEF_OPENERS = [
    "Draft the {clause_display} clause for a {agreement_label}",
    "Write the {clause_display} section for a {agreement_label}",
    "Generate the {clause_display} provision for a {agreement_label}",
    "I need the {clause_display} clause for a {agreement_label}",
    "Prepare the {clause_display} article for a {agreement_label}",
]

# Drafting style → human-readable label that goes into the brief.
# The model should learn that when the brief asks for a particular style,
# the output's tone, sentence length and use of Latin should match.
DRAFTING_STYLE_LABELS = {
    "biglaw_formal": (
        "BigLaw formal — long sentences, defined-term-heavy, "
        "WHEREAS recitals, formal third-person voice"
    ),
    "plain_english_common_paper": (
        "Plain English / Common Paper — short sentences, \"we\" and \"you\", "
        "minimal Latin, defined terms used inline, no WHEREAS recitals"
    ),
    "click_through_tos": (
        "Click-through Terms of Service — terse numbered lists, "
        "no preamble, no signature blocks, accept-by-clicking style"
    ),
    "modular_order_form": (
        "Modular MSA + Order Form — bare master agreement with all "
        "commercials pushed into a separate Order Form"
    ),
    "customer_paper_negotiated": (
        "Customer paper / heavily negotiated — visible customer-side "
        "markups, audit rights, sub-processor lists, regulator pass-through"
    ),
    "modern_tech_minimalist": (
        "Modern tech minimalist — Stripe/Notion-style plain language, "
        "short sections, friendly tone, no Latin"
    ),
}


AGREEMENT_LABEL_BY_TYPE = {
    "SAAS": "Software-as-a-Service Subscription Agreement",
    "MSA": "Master Services Agreement",
    "NDA": "Mutual Non-Disclosure Agreement",
    "EMPLOYMENT": "Employment Agreement",
    "VENDOR_SUPPLY": "Vendor / Supply Agreement",
    "RESELLER": "Reseller Agreement",
    "DPA": "Data Processing Addendum",
    "SOW": "Statement of Work",
    "SOFTWARE_LICENSE": "Software License Agreement",
    "INDEPENDENT_CONTRACTOR": "Independent Contractor Agreement",
}


# ── Clause extraction ────────────────────────────────────────────────────

ARTICLE_HEADER_RE = re.compile(
    r"^ARTICLE\s+(\d+)\s*[—\-–]\s*(.+?)\s*$",
    re.MULTILINE,
)


def extract_clauses(text: str) -> dict[str, dict]:
    """Split a tokenized contract into ARTICLE-bounded sections, then map each
    section to one of the clause types in CLAUSE_TITLE_KEYWORDS.

    Returns a dict keyed by clause_type, with each value being:
        {"title": "<original article title>", "body": "<full article text>"}
    """
    headers = []
    for m in ARTICLE_HEADER_RE.finditer(text):
        headers.append({
            "number": int(m.group(1)),
            "title": m.group(2).strip(),
            "title_upper": m.group(2).strip().upper(),
            "start": m.start(),
        })

    if not headers:
        return {}

    # Determine the body of each article: from start of header to start of next
    sections = []
    for i, h in enumerate(headers):
        end = headers[i + 1]["start"] if i + 1 < len(headers) else len(text)
        body = text[h["start"]:end].rstrip()
        sections.append({**h, "body": body})

    # Map each section to a clause type, using title keyword matching.
    # Use the most specific match first by sorting keywords by length descending.
    clauses = {}
    for clause_type, keywords in CLAUSE_TITLE_KEYWORDS.items():
        for sec in sections:
            for kw in keywords:
                if kw in sec["title_upper"]:
                    # Prefer a more specific match if we already have one
                    existing = clauses.get(clause_type)
                    if existing is None or len(kw) > existing.get("_kw_len", 0):
                        clauses[clause_type] = {
                            "title": sec["title"],
                            "body": sec["body"],
                            "article_number": sec["number"],
                            "_kw_len": len(kw),
                        }
                    break

    # Drop the bookkeeping field
    for v in clauses.values():
        v.pop("_kw_len", None)

    return clauses


# ── Brief construction ──────────────────────────────────────────────────

def _money_short(amount_str: str) -> str:
    """Pull the parenthesized numeric form out of a fee string for the brief.

    e.g. 'Two Hundred Forty Thousand United States Dollars ($240,000)' → '$240,000'
    """
    m = re.search(r"\(([^)]+)\)", amount_str or "")
    if m:
        return m.group(1)
    return amount_str or ""


def _format_term(term_years: str) -> str:
    """'three (3)' → '3 years'."""
    m = re.search(r"\((\d+)\)", term_years or "")
    if m:
        n = int(m.group(1))
        return f"{n} year" if n == 1 else f"{n} years"
    return term_years or ""


def _format_months(months_str: str) -> str:
    """'twelve (12)' → '12 months'."""
    m = re.search(r"\((\d+)\)", months_str or "")
    if m:
        n = int(m.group(1))
        return f"{n} month" if n == 1 else f"{n} months"
    return months_str or ""


def _format_days(days_str: str) -> str:
    """'thirty (30)' → '30 days'."""
    m = re.search(r"\((\d+)\)", days_str or "")
    if m:
        return f"{m.group(1)} days"
    return days_str or ""


def _bias_label(bias: str) -> str:
    return {
        "provider_friendly": "provider-friendly",
        "balanced": "balanced",
        "customer_friendly": "customer-friendly",
    }.get(bias, "balanced")


def _a_or_an(word: str) -> str:
    """Return 'a' or 'an' based on the first sound of the next word.

    Plain heuristic: vowel letter → 'an', except for 'u' words that start
    with a 'yoo' sound (United, University, Utah) and silent-h words like
    'honor'. We list the common exceptions explicitly.
    """
    if not word:
        return "a"
    first = word.strip()[0].lower()
    lower = word.strip().lower()
    # 'yoo' sound exceptions — these take 'a' even though they start with U
    if lower.startswith(("united", "univers", "utah", "uniform", "unique", "user")):
        return "a"
    # Silent-h exceptions — these take 'an' even though they start with H
    if lower.startswith(("honor", "honest", "hour", "heir")):
        return "an"
    return "an" if first in "aeiou" else "a"


def build_brief(
    *,
    contract_type: str,
    clause_type: str,
    sub_map: dict,
    metadata: dict,
    style_seed: int,
) -> str:
    """Construct a realistic brief that mentions parties, industry, jurisdiction,
    deal terms and the relevant standard provisions for the clause.

    The brief embeds the SAME placeholder values that will be substituted into
    the output clause, so the model learns to honor the slots in the brief.
    """
    rng = random.Random(style_seed)

    agreement_label = AGREEMENT_LABEL_BY_TYPE.get(contract_type, "commercial agreement")
    type_display_overrides = CLAUSE_DISPLAY_NAMES_BY_TYPE.get(contract_type, {})
    clause_display = type_display_overrides.get(clause_type) or CLAUSE_DISPLAY_NAMES[clause_type]

    opener_template = rng.choice(BRIEF_OPENERS)
    opener = opener_template.format(clause_display=clause_display, agreement_label=agreement_label)

    # Party block
    provider_name = sub_map["PROVIDER_NAME"]
    provider_state = sub_map["PROVIDER_STATE"]
    provider_entity = sub_map["PROVIDER_ENTITY_TYPE"]
    customer_name = sub_map["CUSTOMER_NAME"]
    customer_state = sub_map.get("CUSTOMER_STATE", sub_map.get("GOVERNING_LAW_STATE", ""))
    customer_entity = sub_map.get("CUSTOMER_ENTITY_TYPE", "")

    provider_article = _a_or_an(provider_state)
    customer_article = _a_or_an(customer_state) if customer_state else _a_or_an(customer_entity)
    party_block = (
        f"between {provider_name} ({provider_article} {provider_state} {provider_entity}, the \"Provider\") "
        f"and {customer_name} ({customer_article} {customer_state} {customer_entity}, the \"Customer\")"
    )

    # Commercial terms block
    annual_fee = _money_short(sub_map.get("ANNUAL_FEE_AMOUNT", ""))
    term = _format_term(sub_map.get("TERM_YEARS", ""))
    cap_months = _format_months(sub_map.get("LIABILITY_CAP_MONTHS", ""))
    notice = _format_days(sub_map.get("NOTICE_DAYS", ""))
    governing_law = sub_map.get("GOVERNING_LAW_STATE", "")
    venue = sub_map.get("VENUE_COUNTY", "")

    # The label for the headline fee depends on the contract type. SaaS deals
    # talk about "annual subscription fee"; services agreements talk about
    # "annual aggregate fees across all Statements of Work" or similar.
    fee_label = {
        "SAAS": "annual subscription fee",
        "MSA": "annual aggregate fees across all Statements of Work",
        "SOW": "Statement of Work fee",
    }.get(contract_type, "annual fee")

    commercial_bits = []
    if annual_fee:
        commercial_bits.append(f"{fee_label} of {annual_fee}")
    if term:
        commercial_bits.append(f"initial term of {term}")
    if cap_months:
        commercial_bits.append(f"liability cap equal to {cap_months} of fees")
    if notice:
        commercial_bits.append(f"notice period of {notice}")

    commercial_block = ""
    if commercial_bits:
        commercial_block = "Key commercial terms: " + "; ".join(commercial_bits) + "."

    # Jurisdiction block
    jurisdiction_block = ""
    if governing_law:
        if venue:
            jurisdiction_block = f"Governing law: {governing_law}, with venue in {venue}."
        else:
            jurisdiction_block = f"Governing law: {governing_law}."

    # Industry / regulatory / bias / style context
    industry_label = metadata.get("industry_label") or metadata.get("industry", "")
    deal_size = metadata.get("deal_size", "")
    bias = _bias_label(metadata.get("negotiation_bias", "balanced"))
    regs = metadata.get("regulatory_frameworks", []) or []
    special = metadata.get("special_features", []) or []
    drafting_style = metadata.get("drafting_style")
    style_label = DRAFTING_STYLE_LABELS.get(drafting_style) if drafting_style else None

    context_bits = []
    if industry_label:
        context_bits.append(f"Industry: {industry_label}")
    if deal_size:
        context_bits.append(f"Deal size: {deal_size.replace('_', '-')}")
    if bias:
        context_bits.append(f"Negotiation posture: {bias}")
    context_block = ". ".join(context_bits) + "." if context_bits else ""

    reg_block = ""
    if regs:
        reg_block = "Regulatory frameworks to address: " + ", ".join(regs) + "."

    # Pick 1-2 special features that are most relevant for this clause type;
    # otherwise pick a random one. This keeps briefs realistic without
    # demanding every clause cover every feature.
    special_block = ""
    if special:
        # Heuristic: include a special feature if it mentions a relevant keyword
        relevant_kw = {
            "intellectual_property": ["source code", "ip", "license", "intellectual"],
            "limitation_of_liability": ["liability", "cap", "limitation", "remedies"],
            "termination": ["exit", "transition", "termination"],
            "indemnification": ["indemn", "sovereign", "regulator"],
            "data_privacy": ["data", "privacy", "ferpa", "hipaa", "pipeda", "pci", "encryption", "residency", "breach"],
            "service_level": ["sla", "uptime", "incident", "downtime", "support", "academic"],
            "confidentiality": ["confidential", "privilege", "ai training", "trade secret"],
        }.get(clause_type, [])
        relevant = [s for s in special if any(k in s.lower() for k in relevant_kw)]
        chosen = relevant[:2] if relevant else [rng.choice(special)]
        special_block = "Specific requirements: " + "; ".join(chosen) + "."

    # Standard provisions to include — per-type override wins over the
    # generic list (e.g. MSA "intellectual_property" → work-product assignment
    # language instead of SaaS license-grant language).
    type_provisions = CLAUSE_STANDARD_PROVISIONS_BY_TYPE.get(contract_type, {})
    provisions = type_provisions.get(clause_type) or CLAUSE_STANDARD_PROVISIONS.get(clause_type, [])
    provisions_block = ""
    if provisions:
        provisions_block = "Cover the standard provisions: " + ", ".join(provisions) + "."

    # Effective date
    effective_block = ""
    if sub_map.get("EFFECTIVE_DATE"):
        effective_block = f"Effective Date: {sub_map['EFFECTIVE_DATE']}."

    style_block = ""
    if style_label:
        style_block = f"Drafting style: {style_label}."

    # Stitch the brief together. Drop empty pieces so we don't get awkward gaps.
    parts = [
        f"{opener} {party_block}.",
        effective_block,
        context_block,
        commercial_block,
        jurisdiction_block,
        reg_block,
        special_block,
        style_block,
        provisions_block,
    ]
    brief = " ".join(p for p in parts if p)
    return brief


# ── Metadata-aware variation ────────────────────────────────────────────

def make_constrained_variation(
    *,
    metadata: dict,
    entity_map: dict,
    seed: int,
) -> dict:
    """Build a substitution map that respects the contract's metadata.

    The base variation is randomized from the global pools, but several
    fields are pinned to the original entity_map values when the metadata
    says they should be:

    - Non-USD contracts (e.g. Canadian banking) keep their full
      jurisdiction stack and currency-specific fee/implementation values.
    - Special customer types (state agencies, regulated banks, law firms)
      keep their customer name, entity type, address and state, since
      generic substitutes would clash with the bias / regulatory context
      embedded in the clause body.
    - Annual fee and implementation fee are drawn from the deal-size tier
      so an SMB contract never gets an enterprise fee.

    The provider state is preserved when it's non-Delaware (e.g. Canada),
    since the original contract was drafted with that jurisdiction in mind.
    """
    rng = random.Random(seed)
    base = generate_variation_map(seed)

    # 1. Currency / jurisdiction lock for non-USD contracts
    if metadata.get("currency") and metadata["currency"] != "USD":
        for k in (
            "PROVIDER_STATE",
            "PROVIDER_ADDRESS",
            "CUSTOMER_STATE",
            "CUSTOMER_ADDRESS",
            "GOVERNING_LAW_STATE",
            "VENUE_COUNTY",
            "VENUE_CITY",
            "ANNUAL_FEE_AMOUNT",
            "IMPLEMENTATION_FEE",
        ):
            if entity_map.get(k):
                base[k] = entity_map[k]

    else:
        # 2. USD contracts: constrain fee to deal-size tier
        deal_size = metadata.get("deal_size")
        if deal_size == "smb":
            base["ANNUAL_FEE_AMOUNT"] = rng.choice(SMB_FEES)
            base["IMPLEMENTATION_FEE"] = rng.choice(SMB_IMPL_FEES)
        elif deal_size == "mid_market":
            base["ANNUAL_FEE_AMOUNT"] = rng.choice(MID_MARKET_FEES)
            base["IMPLEMENTATION_FEE"] = rng.choice(MID_MARKET_IMPL_FEES)
        elif deal_size == "enterprise":
            base["ANNUAL_FEE_AMOUNT"] = rng.choice(ENTERPRISE_FEES)
            base["IMPLEMENTATION_FEE"] = rng.choice(ENTERPRISE_IMPL_FEES)

    # 3. Special customer types: keep original identity
    customer_type = metadata.get("customer_type")
    if customer_type in SPECIAL_CUSTOMER_TYPES:
        for k in (
            "CUSTOMER_NAME",
            "CUSTOMER_ENTITY_TYPE",
            "CUSTOMER_STATE",
            "CUSTOMER_ADDRESS",
            "GOVERNING_LAW_STATE",
            "VENUE_COUNTY",
            "VENUE_CITY",
        ):
            if entity_map.get(k):
                base[k] = entity_map[k]

    # 4. Honour the original effective date if metadata is silent — variation
    #    dates are otherwise random and decoupled from the contract's intent.
    if entity_map.get("EFFECTIVE_DATE"):
        # 50/50: use original or use a fresh random one (for diversity)
        if rng.random() < 0.5:
            base["EFFECTIVE_DATE"] = entity_map["EFFECTIVE_DATE"]

    return base


# ── Example generation ──────────────────────────────────────────────────

def select_clause_types(metadata: dict, available: set[str]) -> list[str]:
    """Pick 4 clause types per contract: the 3 universal ones (IP, Liability,
    Termination) plus 1 rotating clause based on the contract's primary focus.

    The rotating clause is what gives us diversity across contracts — a pharma
    contract gets a Data Privacy example, a cybersecurity contract gets an
    Indemnity example, etc.
    """
    universal = ["intellectual_property", "limitation_of_liability", "termination"]
    selected = [c for c in universal if c in available]

    # Rotating clause
    focus = metadata.get("primary_clause_focus", []) or []
    focus_to_key = {
        "intellectual property": "intellectual_property",
        "limitation of liability": "limitation_of_liability",
        "termination": "termination",
        "indemnity": "indemnification",
        "data privacy": "data_privacy",
        "service level": "service_level",
        "confidentiality": "confidentiality",
    }

    rotating_candidates = []
    for f in focus:
        key = focus_to_key.get(f.lower())
        if key and key in available and key not in selected:
            rotating_candidates.append(key)

    # Fallback rotating clauses if focus didn't yield anything
    if not rotating_candidates:
        for k in ["data_privacy", "service_level", "confidentiality", "indemnification"]:
            if k in available and k not in selected:
                rotating_candidates.append(k)
                break

    if rotating_candidates:
        selected.append(rotating_candidates[0])

    return selected


def process_contract(
    *,
    contract_type: str,
    template_path: Path,
    entity_map: dict,
    metadata: dict,
) -> list[dict]:
    """Generate drafting examples for one tokenized contract."""
    template = template_path.read_text(encoding="utf-8")
    base_name = template_path.stem

    clauses = extract_clauses(template)
    if not clauses:
        logger.warning("  no clauses extracted from %s", base_name)
        return []

    available = set(clauses.keys())
    chosen_clauses = select_clause_types(metadata, available)
    if not chosen_clauses:
        logger.warning("  no clause types selected for %s (available: %s)", base_name, available)
        return []

    # Find which placeholders this template actually uses (so we don't include
    # irrelevant slots in the substitution).
    placeholders_used = set(re.findall(r"\{\{([A-Z_]+)\}\}", template))

    examples = []
    for i, clause_type in enumerate(chosen_clauses, start=1):
        clause = clauses[clause_type]

        # Deterministic seed per (contract, clause) pair so reruns are stable.
        seed = abs(hash(f"{base_name}::{clause_type}::{i}")) % (2**32)
        sub_map = make_constrained_variation(
            metadata=metadata,
            entity_map=entity_map,
            seed=seed,
        )

        # Restrict to placeholders the template actually uses
        active_subs = {k: v for k, v in sub_map.items() if k in placeholders_used}

        # Substitute into the clause body to get the OUTPUT
        output_clause = apply_substitutions(clause["body"], active_subs)

        # Sanity check: warn on leftover placeholders, but still emit the
        # example — the few unsubstituted ones in the original template are
        # the model's failure mode we want to teach against.
        leftover = re.findall(r"\{\{([A-Z_]+)\}\}", output_clause)
        if leftover:
            logger.debug("  leftover placeholders in %s/%s: %s", base_name, clause_type, set(leftover))

        # Build the brief using the FULL sub_map (so the brief mentions all
        # the slot values, not just the ones in this particular clause).
        # This is intentional — a real user would describe the deal at the
        # contract level, not at the clause level.
        brief = build_brief(
            contract_type=contract_type,
            clause_type=clause_type,
            sub_map=sub_map,
            metadata=metadata,
            style_seed=seed,
        )

        examples.append({
            "task": "drafting",
            "contract_type": contract_type,
            "clause_type": clause_type,
            "source_template": base_name,
            "source_article_title": clause["title"],
            "source_article_number": clause["article_number"],
            "instruction": brief,
            "output": output_clause,
            "metadata": {
                "industry": metadata.get("industry"),
                "deal_size": metadata.get("deal_size"),
                "negotiation_bias": metadata.get("negotiation_bias"),
                "drafting_style": metadata.get("drafting_style"),
                "regulatory_frameworks": metadata.get("regulatory_frameworks", []),
                "provider_name": sub_map.get("PROVIDER_NAME"),
                "customer_name": sub_map.get("CUSTOMER_NAME"),
                "governing_law_state": sub_map.get("GOVERNING_LAW_STATE")
                    or sub_map.get("CUSTOMER_STATE"),
                "annual_fee": sub_map.get("ANNUAL_FEE_AMOUNT"),
            },
        })

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="SAAS",
                        help="Contract type (SAAS, MSA, NDA, ...)")
    parser.add_argument("--variations-per-contract", type=int, default=4,
                        help="Reserved for future expansion (currently 4 hard-coded based on clause selection)")
    args = parser.parse_args()

    contract_type = args.type
    tokenized_dir = V2_TOKENIZED / contract_type / "synthetic"
    metadata_file = V2_RAW / contract_type / "synthetic" / "contract_metadata.json"
    entity_map_file = V2_RAW / contract_type / "synthetic" / "entity_maps.json"

    if not tokenized_dir.exists():
        logger.error("Tokenized dir not found: %s", tokenized_dir)
        return
    if not metadata_file.exists():
        logger.error("Metadata file not found: %s", metadata_file)
        return
    if not entity_map_file.exists():
        logger.error("Entity map file not found: %s", entity_map_file)
        return

    metadata_all = json.loads(metadata_file.read_text())
    entity_maps_all = json.loads(entity_map_file.read_text())

    templates = sorted(tokenized_dir.glob("*.txt"))
    if not templates:
        logger.error("No tokenized templates found in %s", tokenized_dir)
        return

    output_dir = V2_TRAINING
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{contract_type}_drafting_examples.jsonl"

    all_examples = []
    per_contract = []
    per_clause_type = {}

    for template_path in templates:
        base = template_path.stem
        if base.startswith("_") or base.upper() == "MANIFEST":
            continue
        meta = metadata_all.get(base)
        ent = entity_maps_all.get(base)
        if not meta:
            logger.warning("No metadata for %s — skipping", base)
            continue
        if not ent:
            logger.warning("No entity map for %s — skipping", base)
            continue

        examples = process_contract(
            contract_type=contract_type,
            template_path=template_path,
            entity_map=ent,
            metadata=meta,
        )
        all_examples.extend(examples)
        per_contract.append({
            "template": base,
            "industry": meta.get("industry"),
            "deal_size": meta.get("deal_size"),
            "examples": len(examples),
            "clause_types": [e["clause_type"] for e in examples],
        })
        for e in examples:
            per_clause_type[e["clause_type"]] = per_clause_type.get(e["clause_type"], 0) + 1
        logger.info("✓ %-58s %d examples (%s)",
                    base[:58], len(examples),
                    ", ".join(e["clause_type"] for e in examples))

    # Write JSONL
    with output_path.open("w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Write a summary manifest next to the JSONL
    manifest = {
        "type": contract_type,
        "task": "drafting",
        "total_examples": len(all_examples),
        "total_contracts": len(per_contract),
        "per_clause_type_counts": per_clause_type,
        "contracts": per_contract,
    }
    manifest_path = output_dir / f"{contract_type}_drafting_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("\n=== DRAFTING EXAMPLES GENERATED ===")
    logger.info("Type:           %s", contract_type)
    logger.info("Contracts:      %d", len(per_contract))
    logger.info("Total examples: %d", len(all_examples))
    logger.info("By clause type: %s", per_clause_type)
    logger.info("Output:         %s", output_path)
    logger.info("Manifest:       %s", manifest_path)


if __name__ == "__main__":
    main()
