#!/usr/bin/env python3
"""Build multi-task training dataset from the v2 contracts.

Generates training data for 5 non-drafting task types from the existing 105
v2 contracts (SAAS + MSA + NDA + EMPLOYMENT). The 5 tasks are:

  Phase A — Extraction        (contract → structured JSON metadata)
  Phase B — Q&A               (contract + question → answer with citation)
  Phase C — Checklist QA      (clause → present/absent provisions checklist)
  Phase D — Risk flagging     (clause/contract → risk register)
  Phase E — Summarization     (contract → plain-English summary)

The same 105 contracts that power the drafting bucket are reused as input here,
with no new contract authoring. Metadata, entity_maps, and the Defense 2 filter
output provide the labels for free; only Phase D needs lightly templated
LLM-judgment risk patterns derived from negotiation_bias and special_features.

Each task uses diversity angles (multiple output formats, perspectives, lengths,
question phrasings) and includes negative examples where the answer is "not
specified" or "no significant risks identified" — so the model learns to handle
ambiguity and absence rather than hallucinating.

Output format matches the existing drafting dataset (Gemma chat conversations
with task metadata):

    {
      "conversations": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "task": "extraction" | "qa" | "checklist_qa" | "risk_flagging" | "summary",
      "contract_type": "SAAS" | ...,
      "source_template": "...",
      ...
    }

Usage:
    python3 scripts/v2_build_multitask_dataset.py
    python3 scripts/v2_build_multitask_dataset.py --phases A,B
    python3 scripts/v2_build_multitask_dataset.py --types SAAS NDA
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from collections import Counter

# Reuse helpers from the drafting generator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_generate_drafting_examples import (
    extract_clauses,
    apply_substitutions,
    make_constrained_variation,
    filter_provisions_by_coverage,
    select_clause_types,
    CLAUSE_STANDARD_PROVISIONS,
    CLAUSE_STANDARD_PROVISIONS_BY_TYPE,
    CLAUSE_DISPLAY_NAMES,
    CLAUSE_DISPLAY_NAMES_BY_TYPE,
    AGREEMENT_LABEL_BY_TYPE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

V2_ROOT = Path("/Users/jyotimishra/legal-finetune/data/v2")
CUAD_PATH = Path("/Users/jyotimishra/legal-finetune/data/external/cuad/CUADv1.json")
LEGAL_SUMM_DIR = Path("/Users/jyotimishra/legal-finetune/data/external/legal_summarization")
UNFAIR_TOS_PATH = Path("/Users/jyotimishra/legal-finetune/data/external/unfair_tos/unfair_tos.jsonl")
ACORD_DIR = Path("/Users/jyotimishra/legal-finetune/data/external/acord")
TOKENIZED_ROOT = V2_ROOT / "tokenized"
META_ROOT = V2_ROOT / "raw"
TRAINING_ROOT = V2_ROOT / "training"

# ── Loading helpers ─────────────────────────────────────────────────────────


def load_contracts(types: list[str]) -> list[dict]:
    """Load every v2 contract for the given types, returning a list of dicts:
        {
          "contract_type": "SAAS" | "MSA" | "NDA" | "EMPLOYMENT",
          "base": "CLAUDE_BrandWaffle_...",
          "metadata": {...},
          "entity_map": {...},
          "template": "<raw template text with placeholders>",
          "sub_map": {...},          # one realistic substitution
          "full_contract": "<substituted text>",
          "clauses": {...},          # extract_clauses output on the substituted contract
        }
    """
    contracts = []
    for ct in types:
        tokenized_dir = TOKENIZED_ROOT / ct / "synthetic"
        meta_path = META_ROOT / ct / "synthetic" / "contract_metadata.json"
        em_path = META_ROOT / ct / "synthetic" / "entity_maps.json"

        if not tokenized_dir.exists() or not meta_path.exists() or not em_path.exists():
            logger.warning("Skipping %s — missing files", ct)
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
            seed = abs(hash(f"{base}::multitask")) % (2**32)
            sub_map = make_constrained_variation(metadata=meta, entity_map=ent, seed=seed)
            placeholders_used = set(re.findall(r"\{\{([A-Z_]+)\}\}", template))
            active_subs = {k: v for k, v in sub_map.items() if k in placeholders_used}
            full = apply_substitutions(template, active_subs)

            # Extract clauses on the SUBSTITUTED contract so the clause bodies
            # contain real party names / values rather than placeholders.
            clauses = extract_clauses(full)

            contracts.append({
                "contract_type": ct,
                "base": base,
                "metadata": meta,
                "entity_map": ent,
                "template": template,
                "sub_map": sub_map,
                "full_contract": full,
                "clauses": clauses,
            })
    return contracts


def _short_money(s: str) -> str:
    """'Five Hundred Thousand United States Dollars (US$500,000)' → 'US$500,000'."""
    if not s:
        return ""
    m = re.search(r"\(([^)]+)\)", s)
    return m.group(1) if m else s


def _format_term(t: str) -> str:
    """'three (3)' → '3 years'; falls back to original."""
    if not t:
        return ""
    m = re.search(r"\((\d+)\)", t)
    return f"{m.group(1)} years" if m else t


# ── Phase A — Extraction ────────────────────────────────────────────────────


# Different "extraction request" phrasings for diversity
EXTRACTION_OPENERS_FULL = [
    "Extract the key terms from this contract and return them as JSON:",
    "Read the following agreement and extract its key terms in structured form:",
    "Pull the standard contract metadata from this agreement:",
    "Abstract the following contract — return parties, term, key commercial terms, governing law, and distinctive provisions:",
    "Please read this contract and produce a structured summary of its key terms:",
]

EXTRACTION_OPENERS_SINGLE_FIELD = [
    "What is the {field} in the following contract?",
    "Read this contract and tell me the {field}:",
    "Extract the {field} from this agreement:",
    "What does this contract say about {field}?",
]

EXTRACTION_OPENERS_PARTIAL = [
    "From this contract, extract only the following fields and return as JSON: {fields}.",
    "I just need {fields} from this contract — return them as a JSON object.",
    "Read this contract and pull out: {fields}. Return the result as JSON.",
]

# Friendly field labels for single-field extractions
FIELD_LABELS = {
    "parties": "parties to the agreement",
    "effective_date": "effective date",
    "agreement_type": "type of agreement",
    "governing_law": "governing law",
    "venue": "exclusive venue / forum for disputes",
    "term": "term of the agreement",
    "termination_notice": "termination notice period",
    "annual_fee": "annual fee",
    "liability_cap": "limitation of liability cap",
    "regulatory_frameworks": "applicable regulatory frameworks",
    "industry": "industry context",
    "negotiation_posture": "negotiation posture",
    "drafting_style": "drafting style",
    "special_features": "distinctive features",
    "base_salary": "base salary",
    "target_bonus": "target bonus",
    "equity_grant": "initial equity grant",
}


def _build_full_extraction(contract: dict) -> dict:
    """Build the structured extraction (the 'ground truth' label)."""
    meta = contract["metadata"]
    sub = contract["sub_map"]
    ct = contract["contract_type"]
    parties = {
        "provider": {
            "name": sub.get("PROVIDER_NAME"),
            "state": sub.get("PROVIDER_STATE"),
            "entity_type": sub.get("PROVIDER_ENTITY_TYPE"),
            "address": sub.get("PROVIDER_ADDRESS"),
            "role_label": (meta.get("party_roles") or {}).get("provider", "Provider"),
        },
        "customer": {
            "name": sub.get("CUSTOMER_NAME"),
            "state": sub.get("CUSTOMER_STATE"),
            "entity_type": sub.get("CUSTOMER_ENTITY_TYPE"),
            "address": sub.get("CUSTOMER_ADDRESS"),
            "role_label": (meta.get("party_roles") or {}).get("customer", "Customer"),
        },
    }

    extraction = {
        "agreement_type": meta.get("agreement_label_override") or AGREEMENT_LABEL_BY_TYPE.get(ct),
        "contract_category": ct,
        "parties": parties,
        "effective_date": sub.get("EFFECTIVE_DATE"),
        "governing_law": sub.get("GOVERNING_LAW_STATE"),
        "venue": f"{sub.get('VENUE_COUNTY', '')}, {sub.get('GOVERNING_LAW_STATE', '')}".strip(", ") or None,
        "industry": meta.get("industry_label") or meta.get("industry"),
        "deal_size": meta.get("deal_size"),
        "negotiation_posture": meta.get("negotiation_bias"),
        "drafting_style": meta.get("drafting_style"),
        "regulatory_frameworks": meta.get("regulatory_frameworks") or [],
        "distinctive_provisions": meta.get("special_features") or [],
    }

    # Commercial terms — different by contract type
    if ct == "EMPLOYMENT":
        extraction["compensation"] = {
            "base_salary": _short_money(sub.get("EMPLOYEE_BASE_SALARY", "")) or None,
            "target_bonus_pct": sub.get("EMPLOYEE_TARGET_BONUS_PCT") or None,
            "initial_equity": _short_money(sub.get("EMPLOYEE_INITIAL_EQUITY", "")) or None,
            "title": sub.get("EMPLOYEE_TITLE"),
            "reports_to": sub.get("EMPLOYEE_REPORTS_TO"),
        }
        if not meta.get("suppress_initial_term") and sub.get("TERM_YEARS"):
            extraction["initial_term"] = _format_term(sub.get("TERM_YEARS", ""))
        else:
            extraction["initial_term"] = "indefinite (at-will or indefinite term)"
    elif ct == "NDA":
        extraction["initial_term"] = _format_term(sub.get("TERM_YEARS", "")) or None
        extraction["termination_notice"] = sub.get("NOTICE_DAYS")
    else:
        # SaaS / MSA — full commercial terms
        extraction["commercial_terms"] = {
            "annual_fee": _short_money(sub.get("ANNUAL_FEE_AMOUNT", "")) or None,
            "implementation_fee": _short_money(sub.get("IMPLEMENTATION_FEE", "")) or None,
            "initial_term": _format_term(sub.get("TERM_YEARS", "")) or None,
            "liability_cap_months": sub.get("LIABILITY_CAP_MONTHS"),
            "termination_notice": sub.get("NOTICE_DAYS"),
        }

    # Drop None values for cleanliness
    return _drop_nones(extraction)


def _drop_nones(d):
    if isinstance(d, dict):
        return {k: _drop_nones(v) for k, v in d.items() if v not in (None, "", [], {})}
    if isinstance(d, list):
        return [_drop_nones(x) for x in d if x not in (None, "")]
    return d


def _format_extraction_output(extraction: dict, fmt: str = "json") -> str:
    """Format the extraction in one of several output formats for diversity."""
    if fmt == "json":
        return "```json\n" + json.dumps(extraction, indent=2, ensure_ascii=False) + "\n```"

    if fmt == "markdown_table":
        rows = []
        rows.append("| Field | Value |")
        rows.append("|---|---|")
        rows.append(f"| Agreement type | {extraction.get('agreement_type', '—')} |")
        parties = extraction.get("parties", {})
        if parties.get("provider"):
            p = parties["provider"]
            rows.append(f"| {p.get('role_label', 'Provider')} | {p.get('name')} ({p.get('state')} {p.get('entity_type')}) |")
        if parties.get("customer"):
            c = parties["customer"]
            rows.append(f"| {c.get('role_label', 'Customer')} | {c.get('name')} ({c.get('state')} {c.get('entity_type')}) |")
        if extraction.get("effective_date"):
            rows.append(f"| Effective date | {extraction['effective_date']} |")
        if extraction.get("governing_law"):
            rows.append(f"| Governing law | {extraction['governing_law']} |")
        if extraction.get("venue"):
            rows.append(f"| Venue | {extraction['venue']} |")
        if extraction.get("industry"):
            rows.append(f"| Industry | {extraction['industry']} |")
        if extraction.get("negotiation_posture"):
            rows.append(f"| Negotiation posture | {extraction['negotiation_posture']} |")
        if extraction.get("drafting_style"):
            rows.append(f"| Drafting style | {extraction['drafting_style']} |")
        if "commercial_terms" in extraction:
            ct = extraction["commercial_terms"]
            for k, v in ct.items():
                rows.append(f"| {k.replace('_', ' ').title()} | {v} |")
        if "compensation" in extraction:
            comp = extraction["compensation"]
            for k, v in comp.items():
                rows.append(f"| {k.replace('_', ' ').title()} | {v} |")
        if extraction.get("regulatory_frameworks"):
            rows.append(f"| Regulatory frameworks | {', '.join(extraction['regulatory_frameworks'])} |")
        if extraction.get("distinctive_provisions"):
            joined = "; ".join(extraction["distinctive_provisions"][:3])
            rows.append(f"| Distinctive provisions | {joined} |")
        return "\n".join(rows)

    if fmt == "narrative":
        parts = []
        agt = extraction.get("agreement_type", "agreement")
        p = extraction.get("parties", {}).get("provider", {})
        c = extraction.get("parties", {}).get("customer", {})
        parts.append(f"This is a **{agt}** between {p.get('name', '?')} ({p.get('state', '?')} {p.get('entity_type', '?')}) and {c.get('name', '?')} ({c.get('state', '?')} {c.get('entity_type', '?')}).")
        if extraction.get("effective_date"):
            parts.append(f"It is effective as of {extraction['effective_date']}.")
        if extraction.get("governing_law"):
            parts.append(f"It is governed by the laws of {extraction['governing_law']}, with venue in {extraction.get('venue', extraction['governing_law'])}.")
        if extraction.get("industry"):
            parts.append(f"The contract context is {extraction['industry']}.")
        if "commercial_terms" in extraction:
            ct = extraction["commercial_terms"]
            bits = []
            if ct.get("annual_fee"):
                bits.append(f"an annual fee of {ct['annual_fee']}")
            if ct.get("initial_term"):
                bits.append(f"an initial term of {ct['initial_term']}")
            if ct.get("liability_cap_months"):
                bits.append(f"a liability cap of {ct['liability_cap_months']} months of fees")
            if bits:
                parts.append("Key commercial terms include " + ", ".join(bits) + ".")
        if "compensation" in extraction:
            comp = extraction["compensation"]
            bits = []
            if comp.get("base_salary"):
                bits.append(f"base salary of {comp['base_salary']}")
            if comp.get("target_bonus_pct"):
                bits.append(f"target bonus of {comp['target_bonus_pct']}% of base")
            if comp.get("initial_equity"):
                bits.append(f"initial equity grant of {comp['initial_equity']}")
            if bits:
                parts.append("Compensation includes " + ", ".join(bits) + ".")
        if extraction.get("regulatory_frameworks"):
            parts.append(f"Applicable regulatory frameworks: {', '.join(extraction['regulatory_frameworks'])}.")
        if extraction.get("distinctive_provisions"):
            parts.append("Distinctive provisions include: " + "; ".join(extraction["distinctive_provisions"][:3]) + ".")
        return " ".join(parts)

    raise ValueError(f"unknown extraction output format: {fmt}")


def gen_phase_a_extraction(contracts: list[dict], rng: random.Random) -> list[dict]:
    """Phase A — Extraction examples."""
    examples = []

    for c in contracts:
        full = c["full_contract"]
        if not full:
            continue
        full_extraction = _build_full_extraction(c)

        # 1. Full extraction in JSON (the most common case)
        opener = rng.choice(EXTRACTION_OPENERS_FULL)
        prompt = f"{opener}\n\n{full}"
        examples.append({
            "conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _format_extraction_output(full_extraction, "json")},
            ],
            "task": "extraction",
            "subtask": "full_json",
            "contract_type": c["contract_type"],
            "source_template": c["base"],
        })

        # 2. Full extraction in markdown table (diversity)
        opener = rng.choice(EXTRACTION_OPENERS_FULL)
        prompt = f"{opener}\n\nReturn the result as a markdown table.\n\n{full}"
        examples.append({
            "conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _format_extraction_output(full_extraction, "markdown_table")},
            ],
            "task": "extraction",
            "subtask": "full_markdown",
            "contract_type": c["contract_type"],
            "source_template": c["base"],
        })

        # 3. Full extraction as narrative summary (diversity)
        prompt = f"Read this contract and write a short structured narrative summary of its key terms.\n\n{full}"
        examples.append({
            "conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _format_extraction_output(full_extraction, "narrative")},
            ],
            "task": "extraction",
            "subtask": "full_narrative",
            "contract_type": c["contract_type"],
            "source_template": c["base"],
        })

        # 4. Single-field extractions (multiple per contract for variety)
        possible_fields = ["governing_law", "venue", "effective_date", "agreement_type",
                           "industry", "regulatory_frameworks", "negotiation_posture"]
        if c["contract_type"] in ("SAAS", "MSA"):
            possible_fields += ["annual_fee", "term", "termination_notice", "liability_cap"]
        elif c["contract_type"] == "EMPLOYMENT":
            possible_fields += ["base_salary", "target_bonus", "equity_grant"]
        elif c["contract_type"] == "NDA":
            possible_fields += ["term", "termination_notice"]

        chosen_fields = rng.sample(possible_fields, min(3, len(possible_fields)))
        for field in chosen_fields:
            ex = _build_single_field_example(c, full, field, full_extraction, rng)
            if ex:
                examples.append(ex)

        # 5. Partial extraction — pick 3-5 fields
        partial_fields = rng.sample(possible_fields, min(rng.randint(3, 5), len(possible_fields)))
        ex = _build_partial_extraction_example(c, full, partial_fields, full_extraction, rng)
        if ex:
            examples.append(ex)

        # 6. NEGATIVE EXAMPLE: ask for a field that doesn't exist
        ex = _build_negative_extraction_example(c, full, rng)
        if ex:
            examples.append(ex)

    return examples


def _build_single_field_example(c: dict, full: str, field: str, extraction: dict, rng: random.Random) -> dict | None:
    """Build a single-field extraction example."""
    label = FIELD_LABELS.get(field, field.replace("_", " "))
    opener = rng.choice(EXTRACTION_OPENERS_SINGLE_FIELD).format(field=label)

    answer = _extract_single_field(field, extraction, c)
    if answer is None:
        return None

    return {
        "conversations": [
            {"role": "user", "content": f"{opener}\n\n{full}"},
            {"role": "assistant", "content": answer},
        ],
        "task": "extraction",
        "subtask": f"single_field:{field}",
        "contract_type": c["contract_type"],
        "source_template": c["base"],
    }


def _extract_single_field(field: str, extraction: dict, contract: dict) -> str | None:
    """Translate a field name into a natural-language answer string."""
    sub = contract["sub_map"]
    if field == "governing_law":
        gl = extraction.get("governing_law")
        return f"The governing law is **{gl}**." if gl else None
    if field == "venue":
        v = extraction.get("venue")
        return f"The exclusive venue for disputes is **{v}**." if v else None
    if field == "effective_date":
        d = extraction.get("effective_date")
        return f"The effective date is **{d}**." if d else None
    if field == "agreement_type":
        t = extraction.get("agreement_type")
        return f"This is **{t}**." if t else None
    if field == "industry":
        i = extraction.get("industry")
        return f"The industry context is **{i}**." if i else None
    if field == "regulatory_frameworks":
        rf = extraction.get("regulatory_frameworks") or []
        if not rf:
            return "No specific regulatory frameworks are referenced in the contract."
        return "The contract references the following regulatory frameworks: " + ", ".join(f"**{r}**" for r in rf) + "."
    if field == "negotiation_posture":
        np = extraction.get("negotiation_posture")
        return f"The negotiation posture of the contract is **{np}**." if np else None
    if field == "annual_fee":
        ct = extraction.get("commercial_terms", {})
        af = ct.get("annual_fee")
        return f"The annual fee is **{af}**." if af else "The contract does not specify an annual fee."
    if field == "term":
        ct = extraction.get("commercial_terms") or {}
        t = ct.get("initial_term") or extraction.get("initial_term")
        return f"The initial term is **{t}**." if t else None
    if field == "termination_notice":
        n = sub.get("NOTICE_DAYS")
        return f"The termination notice period is **{n} days**." if n else None
    if field == "liability_cap":
        ct = extraction.get("commercial_terms", {})
        cap = ct.get("liability_cap_months")
        return f"The liability cap is **{cap} months of fees**." if cap else None
    if field == "base_salary":
        comp = extraction.get("compensation", {})
        s = comp.get("base_salary")
        return f"The annual base salary is **{s}**." if s else None
    if field == "target_bonus":
        comp = extraction.get("compensation", {})
        b = comp.get("target_bonus_pct")
        return f"The target annual bonus is **{b}% of base salary**." if b else None
    if field == "equity_grant":
        comp = extraction.get("compensation", {})
        e = comp.get("initial_equity")
        return f"The initial equity grant has a value of approximately **{e}**." if e else None
    return None


def _build_partial_extraction_example(c: dict, full: str, fields: list[str], extraction: dict, rng: random.Random) -> dict | None:
    """Build a partial JSON extraction with only the requested fields."""
    pretty_fields = ", ".join(FIELD_LABELS.get(f, f.replace("_", " ")) for f in fields)
    opener = rng.choice(EXTRACTION_OPENERS_PARTIAL).format(fields=pretty_fields)
    prompt = f"{opener}\n\n{full}"

    # Build a JSON object with just the requested fields
    out = {}
    for field in fields:
        v = _extract_single_field_raw(field, extraction, c)
        out[field] = v

    out_text = "```json\n" + json.dumps(out, indent=2, ensure_ascii=False) + "\n```"
    return {
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": out_text},
        ],
        "task": "extraction",
        "subtask": "partial_json",
        "contract_type": c["contract_type"],
        "source_template": c["base"],
    }


def _extract_single_field_raw(field: str, extraction: dict, contract: dict):
    """Like _extract_single_field but returns raw values for JSON inclusion."""
    sub = contract["sub_map"]
    if field == "governing_law":
        return extraction.get("governing_law")
    if field == "venue":
        return extraction.get("venue")
    if field == "effective_date":
        return extraction.get("effective_date")
    if field == "agreement_type":
        return extraction.get("agreement_type")
    if field == "industry":
        return extraction.get("industry")
    if field == "regulatory_frameworks":
        return extraction.get("regulatory_frameworks") or []
    if field == "negotiation_posture":
        return extraction.get("negotiation_posture")
    if field == "annual_fee":
        return extraction.get("commercial_terms", {}).get("annual_fee")
    if field == "term":
        return (extraction.get("commercial_terms") or {}).get("initial_term") or extraction.get("initial_term")
    if field == "termination_notice":
        return sub.get("NOTICE_DAYS")
    if field == "liability_cap":
        return extraction.get("commercial_terms", {}).get("liability_cap_months")
    if field == "base_salary":
        return extraction.get("compensation", {}).get("base_salary")
    if field == "target_bonus":
        return extraction.get("compensation", {}).get("target_bonus_pct")
    if field == "equity_grant":
        return extraction.get("compensation", {}).get("initial_equity")
    return None


# Negative-example questions: ask for fields that NDAs / Employment contracts
# don't have. The model should learn to say "not specified" rather than
# hallucinate a value.
NEGATIVE_FIELD_PER_TYPE = {
    "NDA": [
        ("annual fee", "the annual fee"),
        ("liability cap", "the limitation of liability cap"),
        ("base salary", "the base salary"),
        ("equity grant", "the equity grant"),
    ],
    "EMPLOYMENT": [
        ("annual fee", "the annual subscription fee"),
        ("liability cap", "the limitation of liability cap"),
        ("statement of work", "the SOW deliverables"),
    ],
    "SAAS": [
        ("base salary", "the executive's base salary"),
        ("target bonus", "the executive's target bonus"),
        ("union", "the collective bargaining unit"),
    ],
    "MSA": [
        ("base salary", "the executive's base salary"),
        ("equity grant", "the equity grant"),
        ("FINRA", "the FINRA registration requirements"),
    ],
}


def _build_negative_extraction_example(c: dict, full: str, rng: random.Random) -> dict | None:
    """Generate a 'this field doesn't exist' negative example."""
    candidates = NEGATIVE_FIELD_PER_TYPE.get(c["contract_type"], [])
    if not candidates:
        return None
    field, phrase = rng.choice(candidates)
    prompt = f"What is {phrase} in this contract?\n\n{full}"
    answer = (
        f"This contract does not specify {phrase}. "
        f"This is a {AGREEMENT_LABEL_BY_TYPE.get(c['contract_type'], 'commercial agreement')}, "
        f"and the concept of '{field}' does not apply to this type of contract."
    )
    return {
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "task": "extraction",
        "subtask": "negative",
        "contract_type": c["contract_type"],
        "source_template": c["base"],
    }


# ── Phase B — Q&A ───────────────────────────────────────────────────────────
#
# Q&A pairs are templated against the structured metadata. For each contract
# we generate a mix of: factual questions, yes/no questions, multi-part
# questions, and "answer not in contract" negative examples. The model learns
# both to answer accurately AND to refuse hallucination when the answer is
# absent.

# Question templates: each template returns (question, answer) pairs.
# Multiple phrasings of the same question give the model question-paraphrase
# robustness without inventing facts.


def _qa_governing_law(c, ext, rng):
    gl = ext.get("governing_law")
    if not gl:
        return []
    venue = ext.get("venue")
    answer_short = f"The agreement is governed by **{gl}** law."
    answer_long = f"The agreement is governed by the laws of **{gl}**" + (f", with venue in **{venue}**." if venue else ".")
    questions = [
        ("What is the governing law of this agreement?", answer_long),
        ("Which state's law governs this contract?", answer_short),
        ("Where would a dispute under this agreement be litigated?", answer_long),
        (f"Is this agreement governed by {gl} law?", f"Yes — the agreement is governed by **{gl}** law."),
    ]
    return [rng.choice(questions)]


def _qa_parties(c, ext, rng):
    parties = ext.get("parties", {})
    p = parties.get("provider", {}).get("name")
    cu = parties.get("customer", {}).get("name")
    if not p or not cu:
        return []
    pl = parties.get("provider", {}).get("role_label", "Provider")
    cl = parties.get("customer", {}).get("role_label", "Customer")
    return [
        (
            "Who are the parties to this agreement?",
            f"The parties are **{p}** (the {pl}) and **{cu}** (the {cl})."
        )
    ]


def _qa_effective_date(c, ext, rng):
    d = ext.get("effective_date")
    if not d:
        return []
    return [(rng.choice([
        "What is the effective date of this contract?",
        "When does this agreement take effect?",
        "On what date did this agreement become effective?",
    ]), f"The effective date is **{d}**.")]


def _qa_term(c, ext, rng):
    ct = ext.get("commercial_terms") or {}
    t = ct.get("initial_term") or ext.get("initial_term")
    if not t:
        return []
    return [(rng.choice([
        "What is the initial term of this agreement?",
        "How long does this contract run?",
        "What's the term of this agreement?",
    ]), f"The initial term is **{t}**.")]


def _qa_termination(c, ext, rng):
    sub = c["sub_map"]
    notice = sub.get("NOTICE_DAYS")
    if not notice:
        return []
    return [(rng.choice([
        "What is the termination notice period under this agreement?",
        "How much notice is required to terminate this contract?",
        "How many days' notice is required for termination?",
    ]), f"The termination notice period is **{notice} days**.")]


def _qa_fee(c, ext, rng):
    if c["contract_type"] not in ("SAAS", "MSA"):
        return []
    ct = ext.get("commercial_terms", {})
    fee = ct.get("annual_fee")
    if not fee:
        return []
    return [(rng.choice([
        "What is the annual fee under this agreement?",
        "How much does the customer pay annually?",
        "What's the total annual contract value?",
    ]), f"The annual fee is **{fee}**.")]


def _qa_liability_cap(c, ext, rng):
    if c["contract_type"] not in ("SAAS", "MSA"):
        return []
    ct = ext.get("commercial_terms", {})
    cap = ct.get("liability_cap_months")
    if not cap:
        return []
    return [(rng.choice([
        "What is the limitation of liability cap?",
        "How much is the cap on liability?",
        "What's the maximum amount each party can be liable for?",
    ]), f"The liability cap is **{cap} months of fees** paid in the prior 12 months. Note that this cap typically has carve-outs for confidentiality breaches, indemnity, and gross negligence — check the actual Limitation of Liability article for the full list.")]


def _qa_regulatory(c, ext, rng):
    rf = ext.get("regulatory_frameworks") or []
    if not rf:
        return [(
            "What regulatory frameworks does this contract reference?",
            "This contract does not explicitly reference any specific regulatory frameworks."
        )]
    return [(rng.choice([
        "What regulatory frameworks does this contract reference?",
        "Which laws and regulations apply to this agreement?",
        "What's the regulatory context of this contract?",
    ]), "The contract references the following regulatory frameworks: " + ", ".join(f"**{r}**" for r in rf) + ".")]


def _qa_industry(c, ext, rng):
    i = ext.get("industry")
    if not i:
        return []
    return [(rng.choice([
        "What industry context is this contract for?",
        "What industry does this agreement involve?",
        "What's the business context of this contract?",
    ]), f"The contract is in the context of **{i}**.")]


def _qa_negotiation_bias(c, ext, rng):
    np = ext.get("negotiation_posture")
    if not np:
        return []
    label = {
        "balanced": "balanced",
        "provider_friendly": "provider-friendly (favorable to the supplier)",
        "customer_friendly": "customer-friendly (favorable to the buyer/customer)",
    }.get(np, np)
    return [(rng.choice([
        "Is this contract provider-friendly or customer-friendly?",
        "Which side does this contract favor?",
        "What's the negotiation posture of this agreement?",
    ]), f"This contract has a **{label}** negotiation posture.")]


def _qa_special_features(c, ext, rng):
    sf = ext.get("distinctive_provisions") or []
    if not sf:
        return []
    bullets = "\n".join(f"- {s}" for s in sf[:4])
    return [(rng.choice([
        "What are the most distinctive or unusual provisions in this contract?",
        "What edge cases or special features does this contract include?",
        "What makes this contract different from a standard template?",
    ]), f"This contract includes the following distinctive provisions:\n\n{bullets}")]


def _qa_employment_compensation(c, ext, rng):
    if c["contract_type"] != "EMPLOYMENT":
        return []
    comp = ext.get("compensation", {})
    salary = comp.get("base_salary")
    bonus = comp.get("target_bonus_pct")
    equity = comp.get("initial_equity")
    title = comp.get("title")
    examples = []
    if salary:
        examples.append((
            rng.choice(["What is the base salary?", "How much is the base compensation?", "What does the executive earn in base salary?"]),
            f"The annual base salary is **{salary}**." + (f" The Executive's title is **{title}**." if title else "")
        ))
    if bonus:
        examples.append((
            "What is the target annual bonus?",
            f"The target annual bonus is **{bonus}% of base salary**."
        ))
    if equity:
        examples.append((
            "What is the initial equity grant?",
            f"The Executive receives an initial equity grant valued at approximately **{equity}**."
        ))
    return examples


def _qa_employment_atwill(c, ext, rng):
    if c["contract_type"] != "EMPLOYMENT":
        return []
    suppress = c["metadata"].get("suppress_initial_term")
    if suppress:
        return [(
            rng.choice([
                "Is this employment at-will?",
                "Does the executive have a fixed term of employment?",
                "Can the company terminate this executive at any time?",
            ]),
            "This employment is **at-will** (or, for Canadian and similar non-US jurisdictions, indefinite-term). Either party may terminate the employment relationship at any time, subject to the notice and severance terms in the agreement."
        )]
    t = c["sub_map"].get("TERM_YEARS", "")
    term = _format_term(t)
    return [(
        rng.choice([
            "Is this employment at-will or fixed-term?",
            "Does the executive have a fixed term of employment?",
        ]),
        f"This is a **fixed-term** employment agreement with an initial term of **{term}**, with auto-renewal subject to non-renewal notice."
    )]


def _qa_nda_purpose(c, ext, rng):
    if c["contract_type"] != "NDA":
        return []
    industry = ext.get("industry") or ""
    return [(
        rng.choice([
            "What is the Purpose of this NDA?",
            "Why are the parties entering into this NDA?",
            "What can the receiving party use the confidential information for?",
        ]),
        f"The Purpose of this NDA is **{industry}** — the receiving party may use the confidential information solely for that defined purpose and for no other reason."
    )]


# Negative-example questions: things the contract DOESN'T say. The model
# should learn to refuse to hallucinate.
def _qa_negative_questions(c, ext, rng):
    questions = []
    ct = c["contract_type"]
    if ct == "NDA":
        questions += [
            ("What is the annual fee under this NDA?",
             "This is an NDA, which does not have an annual fee. NDAs are typically signed without monetary consideration; the consideration is the mutual exchange of confidential information."),
            ("What is the executive's base salary?",
             "I cannot find this in the contract. This is a Non-Disclosure Agreement, not an employment agreement, so it does not address executive compensation."),
        ]
    elif ct == "EMPLOYMENT":
        questions += [
            ("What is the annual subscription fee?",
             "I cannot find this in the contract. This is an employment agreement, not a SaaS subscription, so it does not address subscription fees."),
            ("What is the SLA uptime commitment?",
             "I cannot find this in the contract. This is an employment agreement and does not include any SaaS service-level commitments."),
        ]
    elif ct == "SAAS":
        questions += [
            ("What is the executive's base salary?",
             "I cannot find this in the contract. This is a SaaS subscription agreement, not an employment agreement, so it does not address executive compensation."),
            ("What is the post-employment non-compete period?",
             "I cannot find this in the contract. This is a SaaS agreement, not an employment agreement, so it does not address post-employment restrictive covenants."),
        ]
    elif ct == "MSA":
        questions += [
            ("What is the executive's equity grant?",
             "I cannot find this in the contract. This is a Master Services Agreement, not an employment agreement, so it does not address equity compensation."),
            ("Where is the merchandise FOB point?",
             "I cannot find this in the contract. This is a Master Services Agreement for the provision of services, not a goods supply agreement, so it does not address shipment or delivery terms."),
        ]
    if not questions:
        return []
    return [rng.choice(questions)]


def _qa_yes_no(c, ext, rng):
    """Yes/no questions for diversity."""
    rf = ext.get("regulatory_frameworks") or []
    examples = []
    if rf:
        framework = rng.choice(rf)
        examples.append((
            f"Does this contract address {framework}?",
            f"Yes — the contract explicitly references **{framework}** as part of its regulatory context."
        ))
    np = ext.get("negotiation_posture")
    if np == "provider_friendly":
        examples.append((
            "Is this contract favorable to the customer?",
            "No — this contract has a **provider-friendly** negotiation posture, meaning it is drafted to favor the supplier/provider rather than the customer. Specific clauses you should look out for from a customer perspective include the limitation of liability, indemnity scope, and termination rights."
        ))
    elif np == "customer_friendly":
        examples.append((
            "Is this contract favorable to the customer?",
            "Yes — this contract has a **customer-friendly** negotiation posture, meaning the customer has negotiated meaningful protections beyond a vendor-paper baseline. Common customer-side wins in such contracts include enhanced termination rights, audit rights, and uncapped liability for data breaches."
        ))
    if not examples:
        return []
    return [rng.choice(examples)]


def gen_phase_b_qa(contracts: list[dict], rng: random.Random) -> list[dict]:
    """Phase B — Q&A examples."""
    examples = []
    qa_generators = [
        _qa_governing_law,
        _qa_parties,
        _qa_effective_date,
        _qa_term,
        _qa_termination,
        _qa_fee,
        _qa_liability_cap,
        _qa_regulatory,
        _qa_industry,
        _qa_negotiation_bias,
        _qa_special_features,
        _qa_employment_compensation,
        _qa_employment_atwill,
        _qa_nda_purpose,
        _qa_yes_no,
        _qa_negative_questions,
    ]

    for c in contracts:
        full = c["full_contract"]
        if not full:
            continue
        ext = _build_full_extraction(c)

        for gen in qa_generators:
            qa_pairs = gen(c, ext, rng)
            for q, a in qa_pairs:
                prompt = f"{full}\n\nQuestion: {q}"
                examples.append({
                    "conversations": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": a},
                    ],
                    "task": "qa",
                    "subtask": gen.__name__.lstrip("_"),
                    "contract_type": c["contract_type"],
                    "source_template": c["base"],
                })

    return examples


# ── Phase C — Checklist QA ──────────────────────────────────────────────────
#
# Given a clause body, return a checklist of standard provisions present /
# absent. Defense 2's filter_provisions_by_coverage() is the ground truth:
# the same logic that builds drafting briefs gives us authoritative
# present/absent labels for free.

CHECKLIST_OPENERS = [
    "Audit this {clause} clause against the standard provisions checklist for {agreement_type}. For each provision, mark whether it is present or absent in the body.",
    "I'm reviewing a {clause} clause from {agreement_type}. Please check it against the standard provisions checklist and tell me what's present and what's missing.",
    "Walk through this {clause} clause and identify which standard provisions are covered and which are missing.",
    "Review this {clause} clause and produce a present/absent checklist of standard provisions for {agreement_type}.",
]

WHATS_MISSING_OPENERS = [
    "What standard provisions are MISSING from this {clause} clause? List only the absent provisions.",
    "I need to identify gaps in this {clause} clause. What standard provisions for {agreement_type} are not addressed?",
    "What's missing from this {clause} clause that should normally be there in {agreement_type}?",
]

WHATS_PRESENT_OPENERS = [
    "List the standard provisions that ARE present in this {clause} clause.",
    "I need to confirm which standard provisions are addressed in this {clause} clause. Just list the ones that are present.",
    "What standard provisions does this {clause} clause cover?",
]


def _format_checklist_markdown(present: list[str], absent: list[str]) -> str:
    rows = []
    for p in present:
        rows.append(f"- ✅ {p}")
    for p in absent:
        rows.append(f"- ❌ {p}")
    return "\n".join(rows)


def _format_checklist_json(present: list[str], absent: list[str]) -> str:
    obj = {
        "present": present,
        "absent": absent,
        "coverage_pct": round(len(present) / max(1, len(present) + len(absent)) * 100, 1),
    }
    return "```json\n" + json.dumps(obj, indent=2, ensure_ascii=False) + "\n```"


def _format_checklist_narrative(present: list[str], absent: list[str], clause_display: str) -> str:
    parts = [f"Reviewing this {clause_display} clause against the standard provisions checklist:"]
    if present:
        parts.append("**Present:**")
        for p in present:
            parts.append(f"- {p}")
    if absent:
        parts.append("")
        parts.append("**Missing:**")
        for p in absent:
            parts.append(f"- {p}")
        parts.append("")
        parts.append(f"**Recommendation:** Consider adding the missing provisions above to strengthen the clause. The missing items are typical of biglaw-formal {clause_display} clauses and would be expected in any contract of comparable size and sophistication.")
    else:
        parts.append("All standard provisions for this clause type are present. No gaps identified.")
    return "\n".join(parts)


def gen_phase_c_checklist(contracts: list[dict], rng: random.Random) -> list[dict]:
    """Phase C — Checklist QA examples."""
    examples = []

    for c in contracts:
        ct = c["contract_type"]
        clauses = c["clauses"]
        agreement_label = c["metadata"].get("agreement_label_override") or AGREEMENT_LABEL_BY_TYPE.get(ct, "commercial agreement")

        # Pick the same clause types we'd select for drafting examples — this
        # ensures the checklist is meaningful for the most-used clauses
        chosen = select_clause_types(c["metadata"], set(clauses.keys()), contract_type=ct)

        for clause_type in chosen:
            clause = clauses.get(clause_type)
            if not clause:
                continue
            body = clause["body"]
            if not body:
                continue

            # Get the full standard provisions list for this clause type
            type_provs = CLAUSE_STANDARD_PROVISIONS_BY_TYPE.get(ct, {})
            all_provs = type_provs.get(clause_type) or CLAUSE_STANDARD_PROVISIONS.get(clause_type, [])
            if not all_provs:
                continue

            # Defense 2 tells us which provisions are present in the body
            present = filter_provisions_by_coverage(all_provs, body)
            absent = [p for p in all_provs if p not in present]

            display = (CLAUSE_DISPLAY_NAMES_BY_TYPE.get(ct, {}).get(clause_type)
                       or CLAUSE_DISPLAY_NAMES.get(clause_type, clause_type))

            # Pick one of three framings randomly: full checklist, what's missing, what's present
            framing = rng.choice(["full", "missing", "present"])

            if framing == "full":
                opener = rng.choice(CHECKLIST_OPENERS).format(clause=display, agreement_type=agreement_label)
                fmt = rng.choice(["markdown", "json", "narrative"])
                if fmt == "markdown":
                    output = _format_checklist_markdown(present, absent)
                elif fmt == "json":
                    output = _format_checklist_json(present, absent)
                else:
                    output = _format_checklist_narrative(present, absent, display)
                examples.append({
                    "conversations": [
                        {"role": "user", "content": f"{opener}\n\n{body}"},
                        {"role": "assistant", "content": output},
                    ],
                    "task": "checklist_qa",
                    "subtask": f"full_{fmt}",
                    "contract_type": ct,
                    "clause_type": clause_type,
                    "source_template": c["base"],
                })

            elif framing == "missing":
                opener = rng.choice(WHATS_MISSING_OPENERS).format(clause=display, agreement_type=agreement_label)
                if absent:
                    output = "**Missing from this clause:**\n\n" + "\n".join(f"- {p}" for p in absent)
                else:
                    output = "Nothing significant is missing from this clause. All standard provisions for this clause type are addressed."
                examples.append({
                    "conversations": [
                        {"role": "user", "content": f"{opener}\n\n{body}"},
                        {"role": "assistant", "content": output},
                    ],
                    "task": "checklist_qa",
                    "subtask": "what_is_missing",
                    "contract_type": ct,
                    "clause_type": clause_type,
                    "source_template": c["base"],
                })

            else:  # present
                opener = rng.choice(WHATS_PRESENT_OPENERS).format(clause=display)
                if present:
                    output = "**Present in this clause:**\n\n" + "\n".join(f"- {p}" for p in present)
                else:
                    output = "None of the typical standard provisions for this clause type are addressed in the body provided. This clause may be a stub, or the body may be too short to contain meaningful substantive obligations."
                examples.append({
                    "conversations": [
                        {"role": "user", "content": f"{opener}\n\n{body}"},
                        {"role": "assistant", "content": output},
                    ],
                    "task": "checklist_qa",
                    "subtask": "what_is_present",
                    "contract_type": ct,
                    "clause_type": clause_type,
                    "source_template": c["base"],
                })

    return examples


# ── Phase D — Risk flagging ─────────────────────────────────────────────────
#
# Given a clause body (or full contract) and a perspective (provider or
# customer), return a risk register with severity and rationale. The risk
# patterns below are derived from the metadata's negotiation_bias,
# special_features, and from common asymmetries that appear in the contracts
# we authored.

# Risk pattern: trigger condition → (severity, who_bears_risk, message)
# The trigger checks whether a particular keyword is in the clause body or
# whether a metadata flag is set. The message is the risk explanation.

RISK_OPENERS = [
    "Review this {clause} clause from a {perspective} perspective and flag any risks. Output a risk register with severity (HIGH / MEDIUM / LOW) and a brief rationale for each item.",
    "I'm representing the {perspective} side of this contract. What risks do I need to be aware of in this {clause} clause?",
    "From a {perspective}'s perspective, what should I be worried about in this {clause} clause? Flag the issues with severity and explanation.",
    "Identify the legal and commercial risks in this {clause} clause from the {perspective}'s perspective.",
]


def _emoji_for_severity(s: str) -> str:
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(s, "•")


def _format_risk_register(risks: list[dict], clause_display: str, perspective: str) -> str:
    if not risks:
        return (
            f"**No significant risks identified** in this {clause_display} clause from a {perspective} perspective. "
            f"The clause appears to be drafted in a balanced manner consistent with industry-standard {clause_display} provisions, "
            f"and does not contain any clauses that would create asymmetric exposure or unusual obligations for the {perspective}."
        )
    parts = [f"**Risk register** ({clause_display} clause, {perspective} perspective):"]
    parts.append("")
    for r in risks:
        parts.append(f"{_emoji_for_severity(r['severity'])} **{r['severity']}** — {r['title']}")
        parts.append(f"   _Why this matters:_ {r['reason']}")
        parts.append("")
    return "\n".join(parts).rstrip()


def _risks_for_clause(c: dict, clause_type: str, body: str, perspective: str) -> list[dict]:
    """Generate a risk list for a specific clause body, given a perspective."""
    risks = []
    meta = c["metadata"]
    bias = meta.get("negotiation_bias")
    sf = meta.get("special_features") or []
    body_low = body.lower()

    # ── Limitation of liability ────────────────────────
    if clause_type == "limitation_of_liability" and perspective == "customer":
        if "six (6)" in body_low or "6 months" in body_low:
            risks.append({
                "severity": "HIGH",
                "title": "Liability cap is only 6 months of fees — well below industry standard.",
                "reason": "Most enterprise SaaS deals cap liability at 12-24 months of fees. A 6-month cap leaves the customer materially under-protected against vendor breach. Push for at least 12 months or carve-outs for data breaches and confidentiality.",
            })
        if "twelve (12)" in body_low and "twenty-four (24)" not in body_low:
            risks.append({
                "severity": "MEDIUM",
                "title": "Liability cap is 12 months — at the lower end of industry standard.",
                "reason": "12 months is the negotiated minimum for most enterprise deals. For high-risk data flows, push for 24 months or uncapped exposure for confidentiality and indemnity carve-outs.",
            })
        if "consequential" in body_low and "exclud" in body_low:
            risks.append({
                "severity": "MEDIUM",
                "title": "Indirect / consequential damages are excluded.",
                "reason": "This standard exclusion materially limits the customer's recovery in the event of vendor breach (e.g., loss of business, loss of data, loss of profit). Negotiate carve-outs for confidentiality breaches and gross negligence.",
            })
        if "essential element" not in body_low and "basis of the bargain" not in body_low:
            risks.append({
                "severity": "LOW",
                "title": "Missing 'essential element of the bargain' acknowledgment.",
                "reason": "Without this language, the LoL clause is more vulnerable to attack as unconscionable in some jurisdictions. The vendor would normally include it; its absence is unusual.",
            })

    if clause_type == "limitation_of_liability" and perspective == "provider":
        if "uncapped" in body_low or "unlimited" in body_low:
            risks.append({
                "severity": "HIGH",
                "title": "Uncapped liability exposure for one or more carve-outs.",
                "reason": "Uncapped liability for confidentiality, indemnity, or data breach is a non-trivial financial exposure for the provider. Push to cap these at 2-3x the annual fees or at fixed dollar amounts.",
            })

    # ── Termination ────────────────────────────────────
    if clause_type == "termination" and perspective == "customer":
        if "convenience" not in body_low:
            risks.append({
                "severity": "MEDIUM",
                "title": "No customer right to terminate for convenience.",
                "reason": "Without a convenience termination right, the customer is locked in for the full term. Negotiate at least a 90-day convenience right with a modest early-termination fee.",
            })
        if "material breach" in body_low and "thirty (30)" not in body_low and "30 days" not in body_low:
            risks.append({
                "severity": "LOW",
                "title": "Cure period for material breach may be longer than 30 days.",
                "reason": "30-day cure is standard. Longer cure periods give the breaching party too much rope; negotiate down to 30 days or less for material breaches.",
            })
        if "exit" not in body_low and "transition" not in body_low:
            risks.append({
                "severity": "MEDIUM",
                "title": "No exit / transition assistance provisions.",
                "reason": "On termination, the customer needs structured transition assistance to migrate data and services to a successor vendor. Without it, the customer faces a 'cliff' on termination day.",
            })

    if clause_type == "termination" and perspective == "provider":
        if "convenience" in body_low and ("customer" in body_low or "client" in body_low):
            risks.append({
                "severity": "MEDIUM",
                "title": "Customer has a termination-for-convenience right.",
                "reason": "Convenience termination shifts the customer-retention risk entirely to the provider. Negotiate to require an early-termination fee or a notice period long enough to wind down delivery.",
            })

    # ── Restrictive covenants ──────────────────────────
    if clause_type == "restrictive_covenants" and perspective == "customer":
        # Customer here = the employee in employment contracts
        if "non-competition" in body_low or "non-compete" in body_low:
            if "12" in body_low or "twelve" in body_low:
                risks.append({
                    "severity": "MEDIUM",
                    "title": "12-month post-employment non-compete restriction.",
                    "reason": "A 12-month non-compete may significantly restrict the executive's ability to work in the same industry. Verify enforceability under the governing state law (e.g., Cal. B&P § 16600 voids these in California).",
                })
            elif "two (2)" in body_low or "24" in body_low or "twenty-four" in body_low:
                risks.append({
                    "severity": "HIGH",
                    "title": "24-month post-employment non-compete is aggressive.",
                    "reason": "Two-year non-competes are at the outer edge of enforceability and may be struck or narrowed by courts. Negotiate down to 12 months or carve out specific industry segments.",
                })
        if "non-solicit" in body_low and "two (2)" in body_low:
            risks.append({
                "severity": "MEDIUM",
                "title": "Two-year non-solicitation restriction.",
                "reason": "Two-year non-solicits are enforceable in most states but should be limited to the specific employees the executive worked with. Push for a narrower geographic and substantive scope.",
            })

    # ── Severance ──────────────────────────────────────
    if clause_type == "severance" and perspective == "customer":  # employee
        if "twelve (12) months" in body_low and "eighteen (18) months" not in body_low:
            risks.append({
                "severity": "LOW",
                "title": "12-month severance is below market for senior executives.",
                "reason": "Senior executives typically negotiate 18-24 months of severance. 12 months is the minimum acceptable level for a CEO or C-suite role.",
            })
        if "release" in body_low and "general release" in body_low:
            risks.append({
                "severity": "MEDIUM",
                "title": "Severance is conditioned on a general release of claims.",
                "reason": "The release waives all claims against the employer, including unknown claims. Read the release carefully and negotiate carve-outs for indemnification, vested equity, and accrued PTO.",
            })

    # ── Confidentiality ────────────────────────────────
    if clause_type == "confidentiality" and perspective == "customer":
        if "no transfer" not in body_low and "no license" not in body_low:
            risks.append({
                "severity": "LOW",
                "title": "No explicit 'no IP license' acknowledgment.",
                "reason": "While not strictly necessary, an explicit no-license clause prevents accidental implied license arguments. Standard practice is to include one.",
            })

    # ── Change in control ──────────────────────────────
    if clause_type == "change_in_control" and perspective == "customer":  # executive
        if "best-net" in body_low or "best net" in body_low:
            risks.append({
                "severity": "MEDIUM",
                "title": "Section 280G best-net cutback (no gross-up).",
                "reason": "The best-net cutback reduces parachute payments to avoid the 20% Section 4999 excise tax, but the executive bears the tax risk if the cutback is wrong. Negotiate for a gross-up if your seniority justifies it.",
            })
        if "gross-up" in body_low or "gross up" in body_low:
            risks.append({
                "severity": "LOW",
                "title": "Section 280G gross-up payment.",
                "reason": "Gross-ups are increasingly rare due to shareholder advisory votes. They are favorable for the executive but may attract criticism in say-on-pay votes.",
            })

    # ── Compensation ───────────────────────────────────
    if clause_type == "compensation" and perspective == "customer":  # employee
        if bias == "provider_friendly":
            risks.append({
                "severity": "MEDIUM",
                "title": "Compensation is on a provider-friendly framework.",
                "reason": "The employer has retained significant discretion over bonus amounts, equity grants, and compensation adjustments. Negotiate to convert discretionary elements to formula-based metrics where possible.",
            })

    # ── Special features-driven risks ──────────────────
    for feature in sf[:5]:
        f = feature.lower()
        if "no residuals" in f and perspective == "customer":
            risks.append({
                "severity": "LOW",
                "title": "No-residuals clause prohibits unaided memory use of confidential information.",
                "reason": "The receiving party cannot rely on its representatives' unaided memory to use information learned during the engagement. This is unusually strict; most NDAs allow residuals as a practical matter.",
            })
        if "section 280g gross-up" in f and perspective == "customer":
            risks.append({
                "severity": "LOW",
                "title": "Section 280G gross-up is a 'legacy' provision.",
                "reason": "Gross-ups are uncommon in modern executive contracts due to shareholder pressure. Verify that this is acceptable to the company's compensation committee before relying on it.",
            })
        if "across-the-board reduction" in f and perspective == "customer":
            risks.append({
                "severity": "MEDIUM",
                "title": "Across-the-board reduction clause permits compensation cuts.",
                "reason": "The employer can cut your compensation as part of a company-wide reduction without triggering Good Reason. Verify the cap on the reduction percentage.",
            })

    # Limit to 4 risks per clause
    return risks[:4]


def gen_phase_d_risk(contracts: list[dict], rng: random.Random) -> list[dict]:
    """Phase D — Risk flagging examples."""
    examples = []

    for c in contracts:
        ct = c["contract_type"]
        clauses = c["clauses"]
        chosen = select_clause_types(c["metadata"], set(clauses.keys()), contract_type=ct)

        for clause_type in chosen:
            clause = clauses.get(clause_type)
            if not clause:
                continue
            body = clause["body"]
            if not body or len(body) < 200:
                continue

            display = (CLAUSE_DISPLAY_NAMES_BY_TYPE.get(ct, {}).get(clause_type)
                       or CLAUSE_DISPLAY_NAMES.get(clause_type, clause_type))

            # Generate one perspective per clause (alternate based on hash for balance)
            perspective = "customer" if hash(c["base"] + clause_type) % 2 == 0 else "provider"
            risks = _risks_for_clause(c, clause_type, body, perspective)

            opener = rng.choice(RISK_OPENERS).format(clause=display, perspective=perspective)
            output = _format_risk_register(risks, display, perspective)

            examples.append({
                "conversations": [
                    {"role": "user", "content": f"{opener}\n\n{body}"},
                    {"role": "assistant", "content": output},
                ],
                "task": "risk_flagging",
                "subtask": f"clause_{perspective}_perspective",
                "contract_type": ct,
                "clause_type": clause_type,
                "source_template": c["base"],
            })

    return examples


# ── Phase E — Summarization ─────────────────────────────────────────────────
#
# Plain-English summaries of full contracts at varying lengths and audiences.

SUMMARY_OPENERS = {
    "one_sentence": [
        "Summarize this contract in one sentence.",
        "Give me a one-sentence description of what this contract is.",
        "What is this contract, in one line?",
    ],
    "executive": [
        "Write a 100-word executive summary of this contract suitable for the CFO.",
        "Summarize this contract for our CFO in roughly 100 words.",
        "Brief executive summary please — about 100 words, focus on the commercial terms and any major risks.",
    ],
    "detailed": [
        "Write a detailed 300-word summary of this contract for the General Counsel.",
        "Give me a thorough 300-word summary of this agreement, focusing on legal risks and obligations.",
        "Walk me through this contract — about 300 words, the kind of summary you'd write for a GC review.",
    ],
    "business_owner": [
        "Explain this contract in plain English to a non-lawyer business owner.",
        "Summarize this contract for someone who isn't a lawyer.",
        "Write a plain-English summary of this contract for the founder/CEO who's not a lawyer.",
    ],
}


def _build_one_sentence_summary(c: dict, ext: dict) -> str:
    parties = ext.get("parties", {})
    p_name = parties.get("provider", {}).get("name", "?")
    c_name = parties.get("customer", {}).get("name", "?")
    agt = ext.get("agreement_type", "agreement")
    industry = ext.get("industry") or "general"
    return f"This is {agt} between {p_name} and {c_name} in the {industry} context."


def _build_executive_summary(c: dict, ext: dict) -> str:
    parts = []
    parties = ext.get("parties", {})
    p = parties.get("provider", {}).get("name", "?")
    cu = parties.get("customer", {}).get("name", "?")
    pl = parties.get("provider", {}).get("role_label", "Provider")
    cl = parties.get("customer", {}).get("role_label", "Customer")
    agt = ext.get("agreement_type", "commercial agreement")
    parts.append(f"**{agt}** between **{p}** ({pl}) and **{cu}** ({cl}), effective **{ext.get('effective_date', 'TBD')}**.")
    industry = ext.get("industry")
    if industry:
        parts.append(f"_Context:_ {industry}.")
    if "commercial_terms" in ext:
        ct = ext["commercial_terms"]
        bits = []
        if ct.get("annual_fee"):
            bits.append(f"annual fee {ct['annual_fee']}")
        if ct.get("initial_term"):
            bits.append(f"{ct['initial_term']} initial term")
        if ct.get("liability_cap_months"):
            bits.append(f"liability cap of {ct['liability_cap_months']} months of fees")
        if bits:
            parts.append("_Commercial:_ " + ", ".join(bits) + ".")
    if "compensation" in ext:
        comp = ext["compensation"]
        bits = []
        if comp.get("base_salary"):
            bits.append(f"base salary {comp['base_salary']}")
        if comp.get("target_bonus_pct"):
            bits.append(f"target bonus {comp['target_bonus_pct']}% of base")
        if comp.get("initial_equity"):
            bits.append(f"equity grant {comp['initial_equity']}")
        if bits:
            parts.append("_Compensation:_ " + ", ".join(bits) + ".")
    gl = ext.get("governing_law")
    if gl:
        parts.append(f"_Governing law:_ {gl}.")
    rf = ext.get("regulatory_frameworks") or []
    if rf:
        parts.append(f"_Regulatory:_ {', '.join(rf[:3])}{'...' if len(rf) > 3 else ''}.")
    sf = ext.get("distinctive_provisions") or []
    if sf:
        parts.append(f"_Distinctive:_ {sf[0]}.")
    return " ".join(parts)


def _build_detailed_summary(c: dict, ext: dict) -> str:
    parts = []
    parties = ext.get("parties", {})
    p = parties.get("provider", {})
    cu = parties.get("customer", {})
    agt = ext.get("agreement_type", "commercial agreement")

    parts.append(
        f"This is {agt}, dated {ext.get('effective_date', '[effective date]')}, between "
        f"**{p.get('name', '?')}** ({p.get('state', '?')} {p.get('entity_type', '?')}) as the {p.get('role_label', 'Provider')}, "
        f"and **{cu.get('name', '?')}** ({cu.get('state', '?')} {cu.get('entity_type', '?')}) as the {cu.get('role_label', 'Customer')}."
    )

    industry = ext.get("industry")
    if industry:
        parts.append(f"\n\n**Industry context.** {industry.capitalize()}. The {ext.get('deal_size', 'mid-market').replace('_', '-')} deal size shapes the commercial terms below.")

    if "commercial_terms" in ext:
        ct = ext["commercial_terms"]
        bits = []
        if ct.get("annual_fee"):
            bits.append(f"annual fee of **{ct['annual_fee']}**")
        if ct.get("initial_term"):
            bits.append(f"initial term of **{ct['initial_term']}**")
        if ct.get("liability_cap_months"):
            bits.append(f"liability cap of **{ct['liability_cap_months']} months** of fees")
        if ct.get("termination_notice"):
            bits.append(f"termination notice period of **{ct['termination_notice']} days**")
        if bits:
            parts.append(f"\n\n**Commercial terms.** Key commercial terms include " + ", ".join(bits) + ".")

    if "compensation" in ext:
        comp = ext["compensation"]
        bits = []
        if comp.get("base_salary"):
            bits.append(f"base salary of **{comp['base_salary']}**")
        if comp.get("target_bonus_pct"):
            bits.append(f"target bonus of **{comp['target_bonus_pct']}%** of base")
        if comp.get("initial_equity"):
            bits.append(f"initial equity grant valued at **{comp['initial_equity']}**")
        if bits:
            parts.append(f"\n\n**Compensation.** The Executive receives " + ", ".join(bits) + ".")

    gl = ext.get("governing_law")
    venue = ext.get("venue")
    if gl:
        parts.append(f"\n\n**Governing law and venue.** Governed by {gl} law, with venue in {venue or gl}.")

    rf = ext.get("regulatory_frameworks") or []
    if rf:
        parts.append(f"\n\n**Regulatory context.** The contract is subject to or references the following frameworks: {', '.join(rf)}.")

    sf = ext.get("distinctive_provisions") or []
    if sf:
        parts.append(f"\n\n**Distinctive provisions.** This contract includes several non-standard provisions that warrant attention:")
        for s in sf[:4]:
            parts.append(f"\n- {s}")

    np = ext.get("negotiation_posture")
    if np:
        parts.append(f"\n\n**Negotiation posture.** This contract is drafted with a **{np.replace('_', '-')}** posture, meaning the terms are positioned to favor that side and counterparty negotiation should focus on the asymmetric provisions.")

    return "".join(parts)


def _build_business_owner_summary(c: dict, ext: dict) -> str:
    parts = []
    parties = ext.get("parties", {})
    p = parties.get("provider", {}).get("name", "the other party")
    cu = parties.get("customer", {}).get("name", "you")
    agt = ext.get("agreement_type", "agreement")

    parts.append(f"Here's what this {agt} actually says, in plain English:")
    parts.append("")
    parts.append(f"**Who:** {p} and {cu} are the two parties.")

    if ext.get("effective_date"):
        parts.append(f"**When:** It starts on {ext['effective_date']}.")

    if "commercial_terms" in ext:
        ct = ext["commercial_terms"]
        if ct.get("annual_fee"):
            parts.append(f"**How much:** The fee is {ct['annual_fee']} per year.")
        if ct.get("initial_term"):
            parts.append(f"**How long:** The initial commitment is {ct['initial_term']}.")
        if ct.get("liability_cap_months"):
            parts.append(f"**If something goes wrong:** Either side's maximum financial exposure is roughly {ct['liability_cap_months']} months of fees — beyond that, the other side eats the loss.")

    if "compensation" in ext:
        comp = ext["compensation"]
        if comp.get("base_salary"):
            parts.append(f"**Pay:** Base salary is {comp['base_salary']}.")
        if comp.get("target_bonus_pct"):
            parts.append(f"**Bonus:** Target bonus is {comp['target_bonus_pct']}% of salary if performance goals are hit.")
        if comp.get("initial_equity"):
            parts.append(f"**Equity:** There's an initial equity grant of about {comp['initial_equity']}, vesting over time.")

    np = ext.get("negotiation_posture")
    if np == "provider_friendly":
        parts.append("")
        parts.append("**Watch out for:** This contract is drafted to favor the supplier. The biggest things to push back on are the limitation of liability cap, the termination rights, and the indemnity.")
    elif np == "customer_friendly":
        parts.append("")
        parts.append("**The good news:** This contract favors you (the customer). The supplier has accepted enhanced termination rights, audit rights, and broader indemnity than typical vendor-paper.")
    else:
        parts.append("")
        parts.append("**Tone:** Pretty balanced overall — neither side has clearly outmaneuvered the other.")

    sf = ext.get("distinctive_provisions") or []
    if sf:
        parts.append("")
        parts.append(f"**One thing to double-check:** {sf[0]}")

    return "\n".join(parts)


def gen_phase_e_summary(contracts: list[dict], rng: random.Random) -> list[dict]:
    """Phase E — Summarization examples."""
    examples = []
    for c in contracts:
        full = c["full_contract"]
        if not full:
            continue
        ext = _build_full_extraction(c)

        # Generate one of each summary type per contract for variety
        for kind in ("one_sentence", "executive", "detailed", "business_owner"):
            opener = rng.choice(SUMMARY_OPENERS[kind])
            if kind == "one_sentence":
                output = _build_one_sentence_summary(c, ext)
            elif kind == "executive":
                output = _build_executive_summary(c, ext)
            elif kind == "detailed":
                output = _build_detailed_summary(c, ext)
            else:
                output = _build_business_owner_summary(c, ext)

            examples.append({
                "conversations": [
                    {"role": "user", "content": f"{opener}\n\n{full}"},
                    {"role": "assistant", "content": output},
                ],
                "task": "summary",
                "subtask": kind,
                "contract_type": c["contract_type"],
                "source_template": c["base"],
            })
    return examples


# ── CUAD AUGMENTATION ───────────────────────────────────────────────────────
#
# CUAD (Contract Understanding Atticus Dataset) is 510 real commercial
# contracts annotated by lawyers across 41 clause categories. License:
# CC BY 4.0 (commercial use OK with attribution).
#
# We use CUAD as direct augmentation for Phase A (extraction), Phase B (Q&A),
# and Phase C (checklist QA), where it's a strong fit. Skipped for drafting
# (use our 105 authored contracts), summarization (no summary annotations),
# and risk flagging (CUAD doesn't include severity annotations — we'd need
# to author the risk-pattern overlay separately).
#
# Diversity discipline: CUAD's question phrasings are templated and
# repetitive ("Highlight the parts (if any) of this contract related to
# 'X'..."). We REPHRASE every question into 2-4 natural-language variants
# so the model doesn't learn the templated form. We also vary output formats
# and cap CUAD's contribution to keep the synthetic↔real balance reasonable.

# Map from CUAD category name → (friendly_field_label, [question_paraphrases])
# Each CUAD contract has all 41 categories asked, with answers being either
# spans from the contract (positive) or empty (negative — clause not present).
CUAD_CATEGORIES = {
    "Document Name": (
        "document name / contract title",
        [
            "What is this document called?",
            "What kind of contract is this?",
            "What is the title of this agreement?",
        ],
    ),
    "Parties": (
        "parties to the agreement",
        [
            "Who are the parties to this agreement?",
            "Identify the parties to this contract.",
            "Who is this contract between?",
        ],
    ),
    "Agreement Date": (
        "agreement signing date",
        [
            "When was this contract signed?",
            "What is the agreement date?",
            "On what date was this agreement entered into?",
        ],
    ),
    "Effective Date": (
        "effective date",
        [
            "When does this contract take effect?",
            "What is the effective date of this agreement?",
            "When did the agreement become effective?",
        ],
    ),
    "Expiration Date": (
        "expiration date",
        [
            "When does this contract expire?",
            "What is the end date of this agreement?",
            "When does the agreement terminate?",
        ],
    ),
    "Renewal Term": (
        "renewal term",
        [
            "What is the renewal term?",
            "How does this contract renew?",
            "What are the renewal provisions?",
        ],
    ),
    "Notice Period To Terminate Renewal": (
        "non-renewal notice period",
        [
            "How much notice is needed to prevent automatic renewal?",
            "What is the non-renewal notice period?",
            "When does the non-renewal notice need to be given?",
        ],
    ),
    "Governing Law": (
        "governing law",
        [
            "What is the governing law of this agreement?",
            "Which state's or country's law governs this contract?",
            "Where would a dispute under this agreement be heard?",
        ],
    ),
    "Most Favored Nation": (
        "most-favored-nation clause",
        [
            "Is there a most-favored-nation clause in this contract?",
            "Does this contract include MFN provisions?",
            "Are there any MFN protections?",
        ],
    ),
    "Non-Compete": (
        "non-compete clause",
        [
            "Is there a non-compete clause in this contract?",
            "Does this contract restrict either party from competing?",
            "Are there any non-competition restrictions?",
        ],
    ),
    "Exclusivity": (
        "exclusivity provisions",
        [
            "Is this an exclusive arrangement?",
            "Does this contract contain exclusivity provisions?",
            "Is one party granted exclusive rights?",
        ],
    ),
    "No-Solicit Of Customers": (
        "customer non-solicitation clause",
        [
            "Is there a customer non-solicit?",
            "Does this contract restrict solicitation of the other party's customers?",
        ],
    ),
    "Competitive Restriction Exception": (
        "competitive restriction exceptions",
        [
            "Are there any carve-outs from the competitive restrictions?",
            "What are the exceptions to the non-compete or exclusivity provisions?",
        ],
    ),
    "No-Solicit Of Employees": (
        "employee non-solicitation clause",
        [
            "Is there an employee non-solicit?",
            "Does this contract restrict hiring the other party's employees?",
        ],
    ),
    "Non-Disparagement": (
        "non-disparagement clause",
        [
            "Is there a non-disparagement clause?",
            "Does this contract prohibit disparaging statements?",
        ],
    ),
    "Termination For Convenience": (
        "termination for convenience right",
        [
            "Can either party terminate this contract for convenience?",
            "Is there a termination-for-convenience right?",
            "Does either party have a no-fault termination right?",
        ],
    ),
    "Rofr/Rofo/Rofn": (
        "right of first refusal, offer, or negotiation",
        [
            "Is there a right of first refusal?",
            "Does this contract include a right of first offer or first negotiation?",
            "Are there any first-refusal rights?",
        ],
    ),
    "Change Of Control": (
        "change-of-control clause",
        [
            "What happens on a change of control?",
            "Is there a change-in-control provision in this contract?",
            "Does a change of control trigger any rights or obligations?",
        ],
    ),
    "Anti-Assignment": (
        "anti-assignment clause",
        [
            "Can this contract be assigned?",
            "Are there any restrictions on assignment?",
            "Is there an anti-assignment clause?",
        ],
    ),
    "Revenue/Profit Sharing": (
        "revenue or profit sharing",
        [
            "Is there a revenue-sharing or profit-sharing arrangement?",
            "Does this contract include any revenue or profit splits?",
        ],
    ),
    "Price Restrictions": (
        "price restrictions",
        [
            "Are there any pricing restrictions?",
            "Can the prices in this contract change?",
            "Does this contract restrict price increases?",
        ],
    ),
    "Minimum Commitment": (
        "minimum commitment",
        [
            "What is the minimum commitment under this contract?",
            "Is there a minimum purchase obligation?",
            "Does the contract require any minimum spend or volume?",
        ],
    ),
    "Volume Restriction": (
        "volume restrictions",
        [
            "Are there any volume restrictions?",
            "Does this contract cap or floor the volume of transactions?",
        ],
    ),
    "Ip Ownership Assignment": (
        "IP ownership / assignment",
        [
            "Who owns the intellectual property?",
            "Is there an IP assignment clause?",
            "What does this contract say about IP ownership?",
        ],
    ),
    "Joint Ip Ownership": (
        "joint IP ownership",
        [
            "Is there joint IP ownership?",
            "Does this contract create any jointly-owned intellectual property?",
        ],
    ),
    "License Grant": (
        "license grant",
        [
            "What license is granted?",
            "What are the license terms?",
            "What does the licensee get?",
        ],
    ),
    "Non-Transferable License": (
        "license transferability",
        [
            "Is the license transferable?",
            "Can the licensee transfer or sublicense?",
        ],
    ),
    "Affiliate License-Licensor": (
        "licensor's affiliate license rights",
        [
            "Can the licensor's affiliates use the licensed IP?",
            "Does the license extend to the licensor's affiliates?",
        ],
    ),
    "Affiliate License-Licensee": (
        "licensee's affiliate license rights",
        [
            "Can the licensee's affiliates use the license?",
            "Does the license extend to the licensee's affiliates?",
        ],
    ),
    "Unlimited/All-You-Can-Eat-License": (
        "unlimited / all-you-can-eat license",
        [
            "Is the license unlimited?",
            "Is this an all-you-can-eat license?",
        ],
    ),
    "Irrevocable Or Perpetual License": (
        "irrevocable or perpetual license",
        [
            "Is the license irrevocable or perpetual?",
            "Can the license be revoked?",
        ],
    ),
    "Source Code Escrow": (
        "source code escrow",
        [
            "Is there a source code escrow arrangement?",
            "Does this contract require source code to be placed in escrow?",
        ],
    ),
    "Post-Termination Services": (
        "post-termination services / wind-down",
        [
            "What happens after termination?",
            "Are there any post-termination obligations?",
            "Is there a transition or wind-down period?",
        ],
    ),
    "Audit Rights": (
        "audit rights",
        [
            "Are there audit rights in this contract?",
            "Can one party audit the other?",
            "What are the audit provisions?",
        ],
    ),
    "Uncapped Liability": (
        "uncapped liability carve-outs",
        [
            "Is there any uncapped liability under this contract?",
            "Are there carve-outs from the limitation of liability cap?",
            "What liability is unlimited?",
        ],
    ),
    "Cap On Liability": (
        "limitation of liability cap",
        [
            "What is the cap on liability?",
            "What is the limitation of liability amount?",
            "How much is the maximum exposure?",
        ],
    ),
    "Liquidated Damages": (
        "liquidated damages",
        [
            "Are there liquidated damages in this contract?",
            "Does this contract specify a liquidated damages amount?",
        ],
    ),
    "Warranty Duration": (
        "warranty duration",
        [
            "How long does the warranty last?",
            "What is the duration of the warranty?",
        ],
    ),
    "Insurance": (
        "insurance requirements",
        [
            "What insurance is required under this contract?",
            "What are the insurance requirements?",
        ],
    ),
    "Covenant Not To Sue": (
        "covenant not to sue",
        [
            "Is there a covenant not to sue?",
            "Does this contract prohibit either party from suing?",
        ],
    ),
    "Third Party Beneficiary": (
        "third-party beneficiaries",
        [
            "Are there any third-party beneficiaries?",
            "Does this contract grant rights to non-parties?",
        ],
    ),
}

# Clauses that map to the standard contract metadata fields used in our
# Phase A extraction format. Used for the full-extraction CUAD output.
CUAD_TO_EXTRACTION_FIELD = {
    "Document Name": "document_name",
    "Parties": "parties",
    "Agreement Date": "agreement_date",
    "Effective Date": "effective_date",
    "Expiration Date": "expiration_date",
    "Renewal Term": "renewal_term",
    "Notice Period To Terminate Renewal": "non_renewal_notice",
    "Governing Law": "governing_law",
}


def load_cuad(max_context_chars: int = 60000):
    """Load the CUAD raw JSON. Returns a list of dicts:
        {
          "title": "...",
          "context": "<full contract text>",
          "qas": [{"category": "...", "question": "...", "answers": [...], "is_impossible": bool}, ...]
        }
    Each contract has 41 QAs (one per CUAD category).

    Contracts longer than max_context_chars are skipped to fit within a
    reasonable training context window. Default 60K chars (~15K tokens)
    keeps ~72% of CUAD and fits in a 16K-token seq_length training config.
    """
    if not CUAD_PATH.exists():
        logger.warning("CUAD data not found at %s — skipping CUAD augmentation", CUAD_PATH)
        return []

    with CUAD_PATH.open() as f:
        raw = json.load(f)

    contracts = []
    skipped_too_long = 0
    for c in raw["data"]:
        # Each contract has one paragraph (the full contract is the context)
        para = c["paragraphs"][0]
        context = para["context"]
        if len(context) > max_context_chars:
            skipped_too_long += 1
            continue
        qas = []
        for qa in para["qas"]:
            # Extract the category from the templated question
            m = re.search(r'related to "([^"]+)"', qa["question"])
            if not m:
                continue
            category = m.group(1)
            # Some categories appear with different casing in CUAD; normalize
            if category not in CUAD_CATEGORIES:
                # Try case-insensitive match
                for key in CUAD_CATEGORIES:
                    if key.lower() == category.lower():
                        category = key
                        break
            answers = qa.get("answers", [])
            is_impossible = not bool(answers) or all(not a.get("text") for a in answers)
            qas.append({
                "category": category,
                "question": qa["question"],
                "answers": answers,
                "is_impossible": is_impossible,
            })
        contracts.append({
            "title": c["title"],
            "context": context,
            "qas": qas,
        })
    logger.info(
        "Loaded %d CUAD contracts (%d total QAs); skipped %d contracts longer than %d chars",
        len(contracts), sum(len(c["qas"]) for c in contracts), skipped_too_long, max_context_chars
    )
    return contracts


def _format_cuad_positive_answer(category: str, answers: list, paraphrase: str, rng: random.Random) -> str:
    """Format a positive (clause-present) CUAD answer."""
    spans = [a.get("text", "").strip() for a in answers if a.get("text")]
    spans = [s for s in spans if s][:3]  # cap at 3 spans
    if not spans:
        return ""

    label, _ = CUAD_CATEGORIES.get(category, (category, []))

    # Pick from a few output styles
    style = rng.choice(["yes_with_quote", "direct_quote", "summary"])

    if style == "yes_with_quote":
        primary = spans[0]
        # Truncate very long spans
        if len(primary) > 600:
            primary = primary[:600] + "..."
        return f"**Yes** — this contract addresses {label}. The relevant text is:\n\n> {primary}"

    if style == "direct_quote":
        if len(spans) == 1:
            return spans[0] if len(spans[0]) < 800 else spans[0][:800] + "..."
        return "\n\n".join(f"- {s if len(s) < 400 else s[:400] + '...'}" for s in spans)

    # summary
    primary = spans[0]
    if len(primary) > 400:
        primary = primary[:400] + "..."
    return f"This contract does address {label}. Specifically: {primary}"


def _format_cuad_negative_answer(category: str, rng: random.Random) -> str:
    """Format a negative (clause-not-present) CUAD answer. These are gold
    natural negatives — pre-labeled by lawyers."""
    label, _ = CUAD_CATEGORIES.get(category, (category, []))
    options = [
        f"**No** — this contract does not contain {label}.",
        f"This contract does not address {label}.",
        f"I cannot find {label} in this contract.",
        f"No, there is no {label} in this agreement.",
    ]
    return rng.choice(options)


def gen_cuad_qa(cuad_contracts: list, rng: random.Random, max_examples: int = 5000) -> list[dict]:
    """Phase B (Q&A) augmentation from CUAD. Generates rephrased Q&A pairs
    using the natural-language paraphrases from CUAD_CATEGORIES instead of
    CUAD's templated question form."""
    examples = []
    for c in cuad_contracts:
        for qa in c["qas"]:
            category = qa["category"]
            if category not in CUAD_CATEGORIES:
                continue
            label, paraphrases = CUAD_CATEGORIES[category]

            # Pick one paraphrased question for variety
            question = rng.choice(paraphrases)

            if qa["is_impossible"]:
                answer = _format_cuad_negative_answer(category, rng)
            else:
                answer = _format_cuad_positive_answer(category, qa["answers"], question, rng)
                if not answer:
                    continue

            prompt = f"{c['context']}\n\nQuestion: {question}"
            examples.append({
                "conversations": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "task": "qa",
                "subtask": f"cuad:{category.lower().replace(' ', '_').replace('/', '_')}",
                "contract_type": "CUAD_REAL",
                "source_template": c["title"][:60],
                "is_negative": qa["is_impossible"],
            })

    # Sample down if we have too many — but make sure we keep proportional
    # negative-positive balance and category coverage.
    if len(examples) > max_examples:
        rng.shuffle(examples)
        examples = examples[:max_examples]
    return examples


def gen_cuad_extraction(cuad_contracts: list, rng: random.Random, max_examples: int = 2000) -> list[dict]:
    """Phase A (Extraction) augmentation from CUAD. Generates contract-level
    structured extractions covering the 8 standard metadata fields plus a
    'distinctive_provisions' field listing the present optional clauses."""
    examples = []
    for c in cuad_contracts:
        # Build a structured extraction from the QA labels
        extraction = {}
        present_optional = []
        for qa in c["qas"]:
            category = qa["category"]
            if category not in CUAD_CATEGORIES:
                continue
            label, _ = CUAD_CATEGORIES[category]

            if qa["is_impossible"] or not qa["answers"]:
                continue
            spans = [a.get("text", "").strip() for a in qa["answers"] if a.get("text")]
            if not spans:
                continue
            primary = spans[0]
            if len(primary) > 200:
                primary = primary[:200] + "..."

            # Map to standard extraction field if possible
            if category in CUAD_TO_EXTRACTION_FIELD:
                extraction[CUAD_TO_EXTRACTION_FIELD[category]] = primary
            else:
                # All other categories are "optional / distinctive provisions"
                present_optional.append({
                    "category": label,
                    "excerpt": primary,
                })

        if present_optional:
            extraction["distinctive_provisions"] = present_optional[:8]  # cap at 8

        # Build the example
        opener = rng.choice([
            "Extract the key terms from this contract and return them as JSON. Include the standard metadata fields and a list of any distinctive optional clauses present.",
            "Read the following contract and produce a structured extraction with parties, dates, governing law, and any unusual provisions present in the body.",
            "Pull all the standard contract metadata from this agreement, plus any non-standard or distinctive clauses, and return as JSON.",
        ])
        output = "```json\n" + json.dumps(extraction, indent=2, ensure_ascii=False) + "\n```"
        examples.append({
            "conversations": [
                {"role": "user", "content": f"{opener}\n\n{c['context']}"},
                {"role": "assistant", "content": output},
            ],
            "task": "extraction",
            "subtask": "cuad_full",
            "contract_type": "CUAD_REAL",
            "source_template": c["title"][:60],
        })

        # Also generate ~3 single-field CUAD extractions per contract for variety
        present_qas = [qa for qa in c["qas"] if not qa["is_impossible"] and qa.get("answers") and qa["category"] in CUAD_CATEGORIES]
        if present_qas:
            for qa in rng.sample(present_qas, min(3, len(present_qas))):
                category = qa["category"]
                label, paraphrases = CUAD_CATEGORIES[category]
                spans = [a.get("text", "").strip() for a in qa["answers"] if a.get("text")][:1]
                if not spans:
                    continue
                primary = spans[0]
                if len(primary) > 400:
                    primary = primary[:400] + "..."
                question = f"What does this contract say about {label}? Return just the relevant text."
                examples.append({
                    "conversations": [
                        {"role": "user", "content": f"{question}\n\n{c['context']}"},
                        {"role": "assistant", "content": primary},
                    ],
                    "task": "extraction",
                    "subtask": f"cuad_single:{category.lower().replace(' ', '_')}",
                    "contract_type": "CUAD_REAL",
                    "source_template": c["title"][:60],
                })

    if len(examples) > max_examples:
        rng.shuffle(examples)
        examples = examples[:max_examples]
    return examples


def gen_cuad_checklist(cuad_contracts: list, rng: random.Random, max_examples: int = 1500) -> list[dict]:
    """Phase C (Checklist QA) augmentation from CUAD. Generates per-contract
    41-clause checklists marking each category as present or absent."""
    examples = []
    for c in cuad_contracts:
        present = []
        absent = []
        for qa in c["qas"]:
            category = qa["category"]
            if category not in CUAD_CATEGORIES:
                continue
            label, _ = CUAD_CATEGORIES[category]
            if qa["is_impossible"] or not qa["answers"]:
                absent.append(label)
            elif not any(a.get("text") for a in qa["answers"]):
                absent.append(label)
            else:
                present.append(label)

        if not present and not absent:
            continue

        # Output format diversity
        fmt = rng.choice(["markdown", "json", "narrative"])
        opener = rng.choice([
            "Audit this commercial contract against the standard CUAD 41-clause checklist. For each category, mark present or absent.",
            "Review this contract and produce a present/absent checklist of standard clause types that lawyers look for in commercial contracts.",
            "Run a 41-point commercial contract review on this agreement and produce a coverage checklist.",
        ])

        if fmt == "markdown":
            lines = []
            lines.append(f"**Coverage:** {len(present)}/{len(present) + len(absent)} ({round(len(present)/(len(present)+len(absent))*100, 1)}%)")
            lines.append("")
            lines.append("**Present:**")
            for p in present:
                lines.append(f"- ✅ {p}")
            lines.append("")
            lines.append("**Absent:**")
            for p in absent:
                lines.append(f"- ❌ {p}")
            output = "\n".join(lines)
        elif fmt == "json":
            obj = {
                "present": present,
                "absent": absent,
                "coverage_pct": round(len(present)/(len(present)+len(absent))*100, 1),
            }
            output = "```json\n" + json.dumps(obj, indent=2, ensure_ascii=False) + "\n```"
        else:
            output = (
                f"This contract addresses **{len(present)} of {len(present) + len(absent)}** standard commercial clauses. "
                f"The clauses that ARE present are: {', '.join(present)}. "
                f"The clauses that are NOT present (and that you may want to verify with the other party) are: {', '.join(absent[:8])}{'...' if len(absent) > 8 else ''}."
            )

        examples.append({
            "conversations": [
                {"role": "user", "content": f"{opener}\n\n{c['context']}"},
                {"role": "assistant", "content": output},
            ],
            "task": "checklist_qa",
            "subtask": f"cuad_full_{fmt}",
            "contract_type": "CUAD_REAL",
            "source_template": c["title"][:60],
        })

    if len(examples) > max_examples:
        rng.shuffle(examples)
        examples = examples[:max_examples]
    return examples


# ── LEGAL_SUMMARIZATION (mteb) AUGMENTATION ─────────────────────────────────
#
# mteb/legal_summarization is 439 (clause, plain-English summary) pairs
# sourced from tldrlegal.com and tosdr.org — real human-written summaries
# of consumer-facing contract clauses (Terms of Service, Privacy Policies,
# EULAs). Average clause length 605 chars; average summary 94 chars.
#
# Quality is high: substantively accurate, plain English, casual voice.
# These are exactly the kind of plain-English summaries our Phase E
# synthetic templates can't produce because they're real human writing.
#
# LICENSE NOTE: The MTEB wrapper is Apache 2.0, but the underlying
# tldrlegal.com content is CC BY-SA 3.0 and the tosdr.org content is
# CC BY-SA 4.0. The "ShareAlike" clause may have implications for
# commercial models trained on this data — verify with counsel before
# any commercial deployment. For research/evaluation/internal use, no
# concerns.

LEGAL_SUMM_OPENERS = [
    "Summarize this contract clause in plain English. Keep it short — one or two sentences.",
    "TL;DR this clause for me — what does it actually mean?",
    "Translate this clause into plain English. Keep it short.",
    "What does this clause actually say in plain English? Keep it brief.",
    "I'm not a lawyer. What does this clause mean?",
    "Plain-English summary please — one sentence or two.",
    "What's the upshot of this clause for a regular person?",
    "Distill this clause to its key point in plain language.",
]


def load_legal_summarization():
    """Load mteb/legal_summarization and return a list of (clause, summary) pairs.

    Reconstructs the pairs from corpus.jsonl + queries.jsonl + qrels.jsonl.
    Returns: list of dicts {"clause": str, "summary": str, "corpus_id": str, "query_id": str}
    """
    if not LEGAL_SUMM_DIR.exists():
        logger.warning("legal_summarization data not found at %s — skipping", LEGAL_SUMM_DIR)
        return []

    corpus_path = LEGAL_SUMM_DIR / "corpus.jsonl"
    queries_path = LEGAL_SUMM_DIR / "queries.jsonl"
    qrels_path = LEGAL_SUMM_DIR / "qrels.jsonl"

    if not (corpus_path.exists() and queries_path.exists() and qrels_path.exists()):
        logger.warning("legal_summarization missing files in %s — skipping", LEGAL_SUMM_DIR)
        return []

    corpus = {}
    with corpus_path.open() as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = d["text"]

    queries = {}
    with queries_path.open() as f:
        for line in f:
            d = json.loads(line)
            queries[d["_id"]] = d["text"]

    pairs = []
    with qrels_path.open() as f:
        for line in f:
            d = json.loads(line)
            clause = corpus.get(d["corpus-id"])
            summary = queries.get(d["query-id"])
            if not clause or not summary:
                continue
            # Skip pairs where the clause is too short to be meaningful
            if len(clause) < 50:
                continue
            pairs.append({
                "clause": clause,
                "summary": summary,
                "corpus_id": d["corpus-id"],
                "query_id": d["query-id"],
            })
    logger.info("Loaded %d (clause, summary) pairs from legal_summarization", len(pairs))
    return pairs


def gen_legal_summarization_e(pairs: list, rng: random.Random, max_examples: int = 600) -> list[dict]:
    """Phase E (Summarization) augmentation from mteb/legal_summarization.

    Generates clause-level plain-English summary examples. Distinct from our
    existing Phase E sub-tasks (which do FULL CONTRACT → multi-sentence
    summary) — this adds CLAUSE → 1-2 sentence plain-English summary, which
    is what users want when they highlight a confusing clause.
    """
    examples = []
    for p in pairs:
        opener = rng.choice(LEGAL_SUMM_OPENERS)
        prompt = f"{opener}\n\n{p['clause']}"
        examples.append({
            "conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": p["summary"]},
            ],
            "task": "summary",
            "subtask": "clause_plain_english",
            "contract_type": "LEGAL_SUMM_REAL",
            "source_template": f"tldrlegal_tosdr:{p['corpus_id']}",
        })

    if len(examples) > max_examples:
        rng.shuffle(examples)
        examples = examples[:max_examples]
    return examples


# ── UNFAIR-ToS augmentation (Phase D — risk flagging) ──────────────────────
#
# LexGLUE UNFAIR-ToS provides 9,414 sentence-level annotations across 8
# unfairness categories drawn from real consumer ToS documents (CC BY 4.0).
# We use it as a real-world counterweight to our 403 synthetic Phase D
# examples. The 89% negative ratio (clauses with no labels) is also gold for
# anti-hallucination training — it teaches the model to say "no significant
# risks identified" when a clause is genuinely benign.
#
# Caveats:
#   - These are CONSUMER terms-of-service, not B2B commercial contracts.
#     The risk taxonomy still maps cleanly (limitation of liability,
#     unilateral termination, unilateral change, jurisdiction, arbitration,
#     etc. all appear in B2B paper). The clause text style is more
#     conversational than B2B, which adds register diversity.
#   - We do NOT use perspective framing for UNFAIR-ToS (consumer vs provider
#     is unambiguous in ToS — the consumer is always the disadvantaged
#     party). Synthetic Phase D continues to handle perspective balancing.

UNFAIR_TOS_LABEL_TO_RISK = {
    "Limitation of liability": {
        "title": "Limitation of liability clause restricts the user's recovery.",
        "severity": "HIGH",
        "reason": "The provider has limited or excluded its liability for harm caused to the user. This shifts risk asymmetrically onto the user — they may not be able to recover damages even where the provider is at fault. In a B2B context, push for liability carve-outs (data breach, indemnity, gross negligence) and a meaningful cap (12-24 months of fees).",
    },
    "Unilateral termination": {
        "title": "Provider has a unilateral right to terminate or suspend.",
        "severity": "HIGH",
        "reason": "The provider can terminate or suspend the user's access without cause, often without notice and without refund. This creates significant business continuity risk. Negotiate for termination only on material breach with a cure period, or at least a reasonable notice period and pro-rata refund.",
    },
    "Unilateral change": {
        "title": "Provider can unilaterally change the terms of the contract.",
        "severity": "HIGH",
        "reason": "The provider has reserved the right to change the agreement at any time, sometimes without notifying the user. This makes the contract a moving target — the user could be bound to materially different terms after signing. Push for advance notice of changes and a right to terminate without penalty if material changes are unacceptable.",
    },
    "Content removal": {
        "title": "Provider can remove the user's content at its discretion.",
        "severity": "MEDIUM",
        "reason": "The provider has the right to remove, modify, or refuse to display user-generated content without notice or appeal. This is a particular concern where the user has paid for or relied on hosted content. Negotiate for notice-and-takedown procedures, an appeal mechanism, and exemptions for content the user has paid to host.",
    },
    "Contract by using": {
        "title": "Contract is formed merely by using the service (browsewrap).",
        "severity": "MEDIUM",
        "reason": "The user is bound to the agreement simply by using the service, without any explicit acceptance. Browsewrap agreements are vulnerable to enforceability challenges in many jurisdictions, and they create constructive-notice problems. For important agreements, insist on explicit clickwrap acceptance with a record of when and how the user accepted.",
    },
    "Choice of law": {
        "title": "Choice-of-law clause favors the provider's home jurisdiction.",
        "severity": "MEDIUM",
        "reason": "The agreement applies the law of a jurisdiction chosen by the provider, often one that is favorable to it (e.g., Delaware, California, or the provider's home country). This may strip the user of consumer-protection rights they would otherwise have. In a B2B context, evaluate whether the chosen law is acceptable; for cross-border deals, consider whether neutral law (e.g., New York or English law) is preferable.",
    },
    "Jurisdiction": {
        "title": "Forum-selection clause requires litigation in the provider's preferred venue.",
        "severity": "MEDIUM",
        "reason": "Disputes must be brought in a court chosen by the provider, often far from the user. This makes it expensive and impractical for the user to enforce its rights or defend against claims. Negotiate for a neutral venue, or at minimum a venue where the user has a meaningful presence.",
    },
    "Arbitration": {
        "title": "Mandatory arbitration clause limits the user's procedural rights.",
        "severity": "MEDIUM",
        "reason": "Disputes must be resolved by binding arbitration (often combined with a class-action waiver). Arbitration limits discovery, appeal rights, and the availability of public remedies. For high-value B2B disputes this can be acceptable in exchange for speed and confidentiality, but evaluate the chosen arbitral forum, the cost-allocation rules, and whether the class-action waiver is enforceable in the relevant jurisdiction.",
    },
}


UNFAIR_TOS_OPENERS = [
    "Review this clause and flag any risks. Output a risk register with severity (HIGH / MEDIUM / LOW) and a brief rationale for each item.",
    "What risks should I be aware of in this clause?",
    "Identify the legal and commercial risks in this clause. Use a HIGH/MEDIUM/LOW severity scale.",
    "I'm reviewing this clause as part of a contract review. Flag any concerning provisions with severity and explanation.",
    "Spot any one-sided or asymmetric provisions in this clause. Rate severity HIGH / MEDIUM / LOW.",
    "Does this clause contain anything I should push back on? Flag with severity and reasoning.",
    "Risk-flag this clause for me. HIGH / MEDIUM / LOW severity, with a brief explanation for each finding.",
    "Read this clause and tell me what's risky about it. Use a severity scale.",
]


UNFAIR_TOS_CLEAN_RESPONSES = [
    "**No significant risks identified** in this clause. The provision appears to be drafted in a balanced manner and does not contain unusual asymmetries, broad provider discretions, or limitations on the counterparty's rights that would warrant flagging.",
    "**No significant risks identified.** This clause is consistent with standard market practice and does not contain provider-favorable asymmetries, unilateral rights, or liability limitations that would require negotiation.",
    "**No significant risks identified** in this clause. It does not contain any of the typical risk patterns (limitation of liability, unilateral termination, unilateral change, choice-of-law overreach, mandatory arbitration, content removal rights, or browsewrap formation) that would warrant a flag.",
    "**No significant risks identified.** The clause appears benign on its face — no asymmetric obligations, no broad provider discretion, no carve-outs from the counterparty's normal remedies. Nothing here that requires negotiation.",
]


def load_unfair_tos() -> list[dict]:
    """Load LexGLUE UNFAIR-ToS sentences from JSONL.

    Returns: list of dicts {"text": str, "labels": list[str], "split": str}
    """
    if not UNFAIR_TOS_PATH.exists():
        logger.warning("unfair_tos data not found at %s — skipping", UNFAIR_TOS_PATH)
        return []
    rows = []
    with UNFAIR_TOS_PATH.open() as f:
        for line in f:
            d = json.loads(line)
            text = (d.get("text") or "").strip()
            if len(text) < 30:
                continue
            rows.append({
                "text": text,
                "labels": d.get("labels") or [],
                "split": d.get("split") or "train",
            })
    logger.info("Loaded %d UNFAIR-ToS sentences", len(rows))
    return rows


def _format_unfair_tos_register(labels: list[str], rng: random.Random) -> str:
    """Format an UNFAIR-ToS row as a risk register matching the synthetic Phase D format."""
    if not labels:
        return rng.choice(UNFAIR_TOS_CLEAN_RESPONSES)
    parts = ["**Risk register:**", ""]
    for label in labels:
        risk = UNFAIR_TOS_LABEL_TO_RISK.get(label)
        if not risk:
            continue
        parts.append(f"{_emoji_for_severity(risk['severity'])} **{risk['severity']}** — {risk['title']}")
        parts.append(f"   _Why this matters:_ {risk['reason']}")
        parts.append("")
    return "\n".join(parts).rstrip()


def gen_unfair_tos_d(rows: list[dict], rng: random.Random,
                     max_examples: int = 4000,
                     negative_ratio: float = 0.5) -> list[dict]:
    """Phase D (Risk flagging) augmentation from LexGLUE UNFAIR-ToS.

    Generates real-world clause-level risk-flagging examples. The dataset is
    naturally 89% negative (clauses with no risk labels) — we downsample
    negatives to `negative_ratio` of the final mix to keep training balance
    sensible while still providing strong signal that benign clauses should
    return "no significant risks identified".
    """
    if not rows:
        return []

    positives = [r for r in rows if r["labels"]]
    negatives = [r for r in rows if not r["labels"]]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    # Cap positives first (they're the rarer signal)
    if max_examples is not None:
        # Target: roughly (1 - negative_ratio) positives, negative_ratio negatives
        max_pos = min(len(positives), int(max_examples * (1 - negative_ratio)))
        max_neg = min(len(negatives), max_examples - max_pos)
        positives = positives[:max_pos]
        negatives = negatives[:max_neg]

    examples = []
    for r in positives + negatives:
        opener = rng.choice(UNFAIR_TOS_OPENERS)
        prompt = f"{opener}\n\n{r['text']}"
        output = _format_unfair_tos_register(r["labels"], rng)
        subtask = "clause_risk_positive" if r["labels"] else "clause_risk_negative"
        examples.append({
            "conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output},
            ],
            "task": "risk_flagging",
            "subtask": subtask,
            "contract_type": "UNFAIR_TOS_REAL",
            "source_template": f"lexglue_unfair_tos:{r['split']}",
            "labels": r["labels"],
        })

    rng.shuffle(examples)
    return examples


# ── ACORD augmentation (Phase B — Q&A on B2B clauses) ──────────────────────
#
# ACORD (Atticus Clause Retrieval Dataset, CC BY 4.0) provides 114
# lawyer-written queries and 3,931 real B2B clauses from EDGAR filings, with
# 126K query-clause pairs rated 0-4 for relevance. The queries are exactly
# the kind of granular, lawyer-style descriptions a contract reviewer uses
# when scanning a contract — "buyer-favorable cap on liability clauses",
# "indemnification carveout to cap on liability", "uncapped IP infringement
# exception", etc.
#
# We frame this as a binary Q&A task:
#   "Does this clause match the following description: [query]?"
#   Score 3-4 → "Yes, ..."
#   Score 0   → "No, ..."
#
# Score 1-2 are skipped as ambiguous (1 = "non-relevant but helpful, same
# category as query"; 2 = "partially relevant"). Score 0 is the explicit
# negative label assigned by the annotators (not a default zero).
#
# This adds B2B clause-pattern recognition to Phase B without requiring
# per-example reasoning text we don't have.

ACORD_POSITIVE_OPENERS = [
    "Does this clause match the following description: \"{query}\"? Answer yes or no with a brief explanation.",
    "I'm scanning a contract and looking for clauses that match this pattern: \"{query}\". Does this clause match? Answer yes/no with a one-sentence explanation.",
    "Review this clause. Does it qualify as: \"{query}\"?",
    "Does this clause contain a \"{query}\"? Answer yes or no with reasoning.",
    "I'm looking for: {query}. Does this clause fit that description?",
    "Pattern check — is this an example of \"{query}\"?",
    "From a contract-review perspective, would you classify this clause as \"{query}\"?",
    "Does the following clause match this description from my deal review checklist: \"{query}\"?",
]


ACORD_POSITIVE_TEMPLATES = [
    "Yes — this clause matches the description \"{query}\". The provision falls in the {category} category and fits that pattern.",
    "Yes. This clause is in the {category} category and qualifies as \"{query}\".",
    "Yes, this clause matches. The provision falls within the {category} category and exhibits the features described in the query (\"{query}\").",
    "Yes — this is an example of \"{query}\". The clause is in the {category} category and has the structural features called out in the query.",
    "Yes. This {category}-category clause matches the \"{query}\" pattern.",
]


ACORD_NEGATIVE_TEMPLATES = [
    "No — this clause does not match the description \"{query}\". It does not fit the {category} pattern in the form described.",
    "No, this clause does not qualify as \"{query}\". The clause does not exhibit the features called out in the query.",
    "No. This clause does not match the \"{query}\" pattern.",
    "No — the clause is not \"{query}\". It addresses a different subject matter or lacks the structural features described.",
    "No, this is not \"{query}\". The clause does not contain the elements the query is asking about.",
]


def load_acord(max_clause_chars: int = 6000):
    """Load ACORD queries, corpus, and qrels.

    Returns: dict with keys:
        queries: dict[query_id] -> {"text": str, "category": str, "split": str}
        corpus: dict[clause_id] -> str (clause text)
        qrels: list of (query_id, clause_id, score) tuples
    """
    if not ACORD_DIR.exists():
        logger.warning("ACORD data not found at %s — skipping", ACORD_DIR)
        return None

    queries_path = ACORD_DIR / "queries.jsonl"
    corpus_path = ACORD_DIR / "corpus.jsonl"
    qrels_dir = ACORD_DIR / "qrels"

    if not (queries_path.exists() and corpus_path.exists() and qrels_dir.exists()):
        logger.warning("ACORD missing files in %s — skipping", ACORD_DIR)
        return None

    queries = {}
    with queries_path.open() as f:
        for line in f:
            d = json.loads(line)
            queries[d["_id"]] = {
                "text": d["text"],
                "category": (d.get("metadata") or {}).get("category", ""),
                "split": (d.get("metadata") or {}).get("split", ""),
            }

    corpus = {}
    skipped_long = 0
    with corpus_path.open() as f:
        for line in f:
            d = json.loads(line)
            text = (d.get("text") or "").strip()
            if len(text) < 80:
                continue
            if len(text) > max_clause_chars:
                skipped_long += 1
                continue
            corpus[d["_id"]] = text

    qrels = []
    import csv as _csv
    for split in ("train", "test", "valid"):
        p = qrels_dir / f"{split}.tsv"
        if not p.exists():
            continue
        with p.open() as f:
            reader = _csv.reader(f, delimiter="\t")
            next(reader)  # header
            for row in reader:
                if len(row) < 3:
                    continue
                q_id, c_id, score = row[0], row[1], row[2]
                try:
                    qrels.append((q_id, c_id, int(score)))
                except ValueError:
                    continue

    logger.info(
        "Loaded ACORD: %d queries, %d clauses (skipped %d long), %d qrels",
        len(queries), len(corpus), skipped_long, len(qrels),
    )
    return {"queries": queries, "corpus": corpus, "qrels": qrels}


def gen_acord_qa(acord: dict, rng: random.Random,
                 max_examples: int = 2500,
                 negatives_per_positive: float = 2.0) -> list[dict]:
    """Phase B (Q&A) augmentation from ACORD.

    Generates binary "does this clause match the description?" examples
    using only the strong positive ratings (3-4) and the explicit negative
    ratings (0). Score 1-2 are skipped as ambiguous.

    Note: ACORD has only ~771 strong positives across all splits, heavily
    skewed toward Limitation of Liability and Indemnification. We use ALL
    available positives by default and pair them with `negatives_per_positive`
    negatives sampled randomly. The `max_examples` cap is a safety ceiling.
    """
    if not acord:
        return []
    queries = acord["queries"]
    corpus = acord["corpus"]
    qrels = acord["qrels"]

    seen_pairs = set()
    positives = []  # (query_id, clause_id) where score >= 3
    negatives = []  # (query_id, clause_id) where score == 0
    for q_id, c_id, score in qrels:
        if q_id not in queries or c_id not in corpus:
            continue
        key = (q_id, c_id)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        if score >= 3:
            positives.append(key)
        elif score == 0:
            negatives.append(key)

    rng.shuffle(positives)
    rng.shuffle(negatives)

    # Use all positives, sample negatives proportionally
    target_negatives = min(len(negatives), int(len(positives) * negatives_per_positive))
    negatives = negatives[:target_negatives]

    # Apply safety cap if needed
    total = len(positives) + len(negatives)
    if total > max_examples:
        scale = max_examples / total
        positives = positives[: max(1, int(len(positives) * scale))]
        negatives = negatives[: max_examples - len(positives)]

    examples = []
    for q_id, c_id in positives:
        q = queries[q_id]
        clause = corpus[c_id]
        opener = rng.choice(ACORD_POSITIVE_OPENERS).format(query=q["text"])
        answer = rng.choice(ACORD_POSITIVE_TEMPLATES).format(
            query=q["text"], category=q["category"] or "contract",
        )
        examples.append({
            "conversations": [
                {"role": "user", "content": f"{opener}\n\n{clause}"},
                {"role": "assistant", "content": answer},
            ],
            "task": "qa",
            "subtask": "acord_clause_match_positive",
            "contract_type": "ACORD_REAL",
            "source_template": f"acord:{q_id}:{c_id}",
            "acord_score": "positive",
            "acord_category": q["category"],
        })

    for q_id, c_id in negatives:
        q = queries[q_id]
        clause = corpus[c_id]
        opener = rng.choice(ACORD_POSITIVE_OPENERS).format(query=q["text"])
        answer = rng.choice(ACORD_NEGATIVE_TEMPLATES).format(
            query=q["text"], category=q["category"] or "contract",
        )
        examples.append({
            "conversations": [
                {"role": "user", "content": f"{opener}\n\n{clause}"},
                {"role": "assistant", "content": answer},
            ],
            "task": "qa",
            "subtask": "acord_clause_match_negative",
            "contract_type": "ACORD_REAL",
            "source_template": f"acord:{q_id}:{c_id}",
            "acord_score": "negative",
            "acord_category": q["category"],
        })

    rng.shuffle(examples)
    return examples


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", nargs="+", default=["SAAS", "MSA", "NDA", "EMPLOYMENT", "SOW", "DPA", "INDEPENDENT_CONTRACTOR", "LICENSE"])
    parser.add_argument("--phases", default="A,B,C,D,E",
                        help="Comma-separated phase letters to generate (default: all)")
    parser.add_argument("--include-cuad", action="store_true",
                        help="Augment Phases A/B/C with CUAD (510 real commercial contracts × 41 clause types)")
    parser.add_argument("--cuad-max-context-chars", type=int, default=60000,
                        help="Skip CUAD contracts longer than this (default: 60000 chars ≈ 15K tokens, keeps ~72%% of CUAD)")
    parser.add_argument("--cuad-qa-cap", type=int, default=5000,
                        help="Maximum CUAD Q&A examples to include (default: 5000)")
    parser.add_argument("--cuad-extraction-cap", type=int, default=2000,
                        help="Maximum CUAD extraction examples to include (default: 2000)")
    parser.add_argument("--cuad-checklist-cap", type=int, default=1500,
                        help="Maximum CUAD checklist examples to include (default: 1500)")
    parser.add_argument("--include-legal-summarization", action="store_true",
                        help="Augment Phase E with mteb/legal_summarization (439 real human-written plain-English clause summaries; CC BY-SA underlying content — verify license for commercial use)")
    parser.add_argument("--legal-summ-cap", type=int, default=600,
                        help="Maximum legal_summarization examples to include (default: 600)")
    parser.add_argument("--include-unfair-tos", action="store_true",
                        help="Augment Phase D with LexGLUE UNFAIR-ToS (9,414 real sentence-level expert-annotated clauses across 8 unfairness categories; CC BY 4.0)")
    parser.add_argument("--unfair-tos-cap", type=int, default=4000,
                        help="Maximum UNFAIR-ToS examples to include (default: 4000)")
    parser.add_argument("--unfair-tos-negative-ratio", type=float, default=0.5,
                        help="Fraction of UNFAIR-ToS examples that should be 'no risk' negatives (default: 0.5)")
    parser.add_argument("--include-acord", action="store_true",
                        help="Augment Phase B with ACORD (114 lawyer-written queries × 3,931 real B2B clauses from EDGAR, 126K relevance-rated pairs; CC BY 4.0). Uses all 771 strong positives + balanced negatives.")
    parser.add_argument("--acord-cap", type=int, default=2500,
                        help="Safety ceiling on ACORD examples (default: 2500). Actual count is min(cap, all_positives × (1 + negatives_per_positive)).")
    parser.add_argument("--acord-negatives-per-positive", type=float, default=2.0,
                        help="Number of 'no match' negatives sampled per positive (default: 2.0)")
    parser.add_argument("--acord-max-clause-chars", type=int, default=6000,
                        help="Skip ACORD clauses longer than this (default: 6000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    contracts = load_contracts(args.types)
    logger.info("Loaded %d contracts across %d types", len(contracts), len(args.types))

    cuad_contracts = []
    if args.include_cuad:
        cuad_contracts = load_cuad(max_context_chars=args.cuad_max_context_chars)

    legal_summ_pairs = []
    if args.include_legal_summarization:
        legal_summ_pairs = load_legal_summarization()

    unfair_tos_rows = []
    if args.include_unfair_tos:
        unfair_tos_rows = load_unfair_tos()

    acord = None
    if args.include_acord:
        acord = load_acord(max_clause_chars=args.acord_max_clause_chars)

    phases = [p.strip().upper() for p in args.phases.split(",") if p.strip()]
    examples = []

    if "A" in phases:
        a = gen_phase_a_extraction(contracts, rng)
        logger.info("Phase A (extraction):           %d examples (synthetic)", len(a))
        examples.extend(a)
        if cuad_contracts:
            ca = gen_cuad_extraction(cuad_contracts, rng, max_examples=args.cuad_extraction_cap)
            logger.info("Phase A (extraction) + CUAD:   +%d examples (real)", len(ca))
            examples.extend(ca)
    if "B" in phases:
        b = gen_phase_b_qa(contracts, rng)
        logger.info("Phase B (Q&A):                  %d examples (synthetic)", len(b))
        examples.extend(b)
        if cuad_contracts:
            cb = gen_cuad_qa(cuad_contracts, rng, max_examples=args.cuad_qa_cap)
            logger.info("Phase B (Q&A) + CUAD:          +%d examples (real)", len(cb))
            examples.extend(cb)
        if acord:
            ab = gen_acord_qa(
                acord, rng,
                max_examples=args.acord_cap,
                negatives_per_positive=args.acord_negatives_per_positive,
            )
            logger.info("Phase B (Q&A) + ACORD:         +%d examples (real, CC BY 4.0)", len(ab))
            examples.extend(ab)
    if "C" in phases:
        c = gen_phase_c_checklist(contracts, rng)
        logger.info("Phase C (checklist):            %d examples (synthetic)", len(c))
        examples.extend(c)
        if cuad_contracts:
            cc = gen_cuad_checklist(cuad_contracts, rng, max_examples=args.cuad_checklist_cap)
            logger.info("Phase C (checklist) + CUAD:    +%d examples (real)", len(cc))
            examples.extend(cc)
    if "D" in phases:
        d = gen_phase_d_risk(contracts, rng)
        logger.info("Phase D (risk):                 %d examples (synthetic)", len(d))
        examples.extend(d)
        if unfair_tos_rows:
            ud = gen_unfair_tos_d(
                unfair_tos_rows, rng,
                max_examples=args.unfair_tos_cap,
                negative_ratio=args.unfair_tos_negative_ratio,
            )
            logger.info("Phase D (risk) + UNFAIR-ToS:   +%d examples (real, CC BY 4.0)", len(ud))
            examples.extend(ud)
    if "E" in phases:
        e = gen_phase_e_summary(contracts, rng)
        logger.info("Phase E (summary):              %d examples (synthetic)", len(e))
        examples.extend(e)
        if legal_summ_pairs:
            le = gen_legal_summarization_e(legal_summ_pairs, rng, max_examples=args.legal_summ_cap)
            logger.info("Phase E (summary) + legal_summ: +%d examples (real human-written, CC BY-SA)", len(le))
            examples.extend(le)

    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_fraction))
    val = examples[:n_val]
    train = examples[n_val:]

    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    train_path = TRAINING_ROOT / "v2_multitask_train.jsonl"
    val_path = TRAINING_ROOT / "v2_multitask_val.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    by_task = Counter(e["task"] for e in examples)
    by_type = Counter(e["contract_type"] for e in examples)
    by_subtask = Counter(e.get("subtask", "default") for e in examples)
    char_lens = sorted(len(e["conversations"][0]["content"]) + len(e["conversations"][1]["content"]) for e in examples)

    manifest = {
        "train_count": len(train),
        "val_count": len(val),
        "total": len(examples),
        "by_task": dict(by_task),
        "by_subtask": dict(by_subtask),
        "by_contract_type": dict(by_type),
        "char_lens": {
            "min": char_lens[0],
            "median": char_lens[len(char_lens) // 2],
            "p95": char_lens[int(len(char_lens) * 0.95)],
            "max": char_lens[-1],
        },
    }
    (TRAINING_ROOT / "v2_multitask_manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("=== V2 MULTI-TASK DATASET ===")
    logger.info("Train: %d → %s", len(train), train_path)
    logger.info("Val:   %d → %s", len(val), val_path)
    logger.info("Tasks: %s", dict(by_task))
    logger.info("Types: %s", dict(by_type))
    logger.info("Char lens: min=%d median=%d p95=%d max=%d",
                char_lens[0], char_lens[len(char_lens) // 2],
                char_lens[int(len(char_lens) * 0.95)], char_lens[-1])


if __name__ == "__main__":
    main()
