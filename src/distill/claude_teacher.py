"""Claude distillation — uses Claude API to generate gold-standard training examples.

This is the key differentiator: Claude generates perfect clause examples that
Saul-7B learns from. Claude is used ONCE offline to produce training data.
No Claude calls happen in production.

Requires ANTHROPIC_API_KEY environment variable.
"""

import json
import logging
import os
import random
import time
from typing import Optional

import anthropic
from tqdm import tqdm

from src.data.schema import (
    ClauseType,
    ContractType,
    Jurisdiction,
    TrainingExample,
)

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 2048


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return anthropic.Anthropic(api_key=api_key)


# ── Clause generation prompts ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior legal draftsman with 20+ years of experience drafting
commercial contracts across US, UK, Indian, and international jurisdictions.

You produce production-ready contract clauses that are:
- Legally precise and enforceable
- Commercially balanced
- Free of any placeholders, brackets, or TBD markers
- Structured with numbered sub-clauses
- Specific (contains actual numbers, periods, rates — not vague language)

Output ONLY the clause text. No preamble, no explanation, no markdown formatting."""

CLAUSE_INSTRUCTIONS: dict[ClauseType, str] = {
    ClauseType.PAYMENT: (
        "Draft a PAYMENT TERMS clause with exactly 4 sub-clauses covering: "
        "(1) payment schedule with specific days and currency, "
        "(2) invoicing procedure with acceptance timeline, "
        "(3) late payment interest with a specific rate, "
        "(4) disputed invoice process with specific days for dispute notification."
    ),
    ClauseType.LIABILITY: (
        "Draft a LIABILITY AND INDEMNITY clause with exactly 5 sub-clauses covering: "
        "(1) mutual cap on aggregate liability at 12 months' fees, "
        "(2) exclusion of indirect/consequential/punitive damages, "
        "(3) mutual indemnity for breach, negligence, or wilful misconduct, "
        "(4) indemnification procedure with 30-day notice, "
        "(5) survival of obligations."
    ),
    ClauseType.TERMINATION: (
        "Draft a TERMINATION clause with exactly 4 sub-clauses covering: "
        "(1) termination for cause with cure period, "
        "(2) termination for convenience with notice period, "
        "(3) effects of termination (return materials, pay outstanding), "
        "(4) survival of specific provisions."
    ),
    ClauseType.CONFIDENTIALITY: (
        "Draft a CONFIDENTIALITY clause with exactly 5 sub-clauses covering: "
        "(1) definition of confidential information, "
        "(2) non-disclosure obligations, "
        "(3) exceptions (public domain, prior knowledge, compelled disclosure), "
        "(4) return or destruction on termination, "
        "(5) survival period."
    ),
    ClauseType.IP_RIGHTS: (
        "Draft an INTELLECTUAL PROPERTY RIGHTS clause with exactly 4 sub-clauses: "
        "(1) work product ownership assigned to client upon payment, "
        "(2) background IP retained by each party, "
        "(3) licence grant for background IP in deliverables, "
        "(4) IP indemnification with notice and defence procedures."
    ),
    ClauseType.FORCE_MAJEURE: (
        "Draft a FORCE MAJEURE clause with exactly 5 sub-clauses: "
        "(1) broad definition including pandemic, war, cyberattack, government action, "
        "(2) 7-day notification obligation, "
        "(3) suspension of obligations during the event, "
        "(4) mitigation duty, "
        "(5) right to terminate after 90 days of continued force majeure."
    ),
    ClauseType.GOVERNING_LAW: (
        "Draft a GOVERNING LAW AND DISPUTE RESOLUTION clause with exactly 3 sub-clauses: "
        "(1) governing law statement, "
        "(2) negotiation — 30-day good-faith discussion, "
        "(3) dispute resolution (courts or arbitration as appropriate for jurisdiction)."
    ),
    ClauseType.DEFINITIONS: (
        "Draft a DEFINITIONS clause with exactly 7 definitions appropriate for the contract type. "
        "Each definition must follow: '\"DefinedTerm\" means [full definition].' format. "
        "Always include: Confidential Information, Affiliate, Intellectual Property, Effective Date. "
        "Choose 3 more relevant to the stated contract type."
    ),
    ClauseType.GENERAL_PROVISIONS: (
        "Draft a GENERAL PROVISIONS clause with exactly 8 sub-clauses: "
        "(1) entire agreement, (2) amendments in writing, (3) severability, "
        "(4) waiver, (5) notices, (6) assignment, (7) counterparts, "
        "(8) relationship of parties (independent contractors)."
    ),
    ClauseType.REPRESENTATIONS_WARRANTIES: (
        "Draft a REPRESENTATIONS AND WARRANTIES clause with exactly 5 sub-clauses: "
        "(1) authority and due incorporation, (2) no conflict, "
        "(3) compliance with law, (4) no pending litigation, (5) survival."
    ),
    ClauseType.DATA_PROTECTION: (
        "Draft a DATA PROTECTION clause with exactly 5 sub-clauses: "
        "(1) compliance with applicable data protection laws, "
        "(2) data processed only for stated purposes, "
        "(3) appropriate security measures, "
        "(4) 72-hour breach notification, "
        "(5) data subject rights cooperation."
    ),
    ClauseType.SERVICES: (
        "Draft a SERVICES clause with exactly 4 sub-clauses: "
        "(1) scope of services per Statement of Work, "
        "(2) change request procedure, "
        "(3) service standards and acceptance criteria, "
        "(4) subcontracting restrictions."
    ),
}


def _build_user_prompt(
    clause_type: ClauseType,
    contract_type: ContractType,
    jurisdiction: Jurisdiction,
) -> str:
    """Build the user prompt for generating a specific clause."""
    instruction = CLAUSE_INSTRUCTIONS.get(clause_type, f"Draft a {clause_type.value} clause.")
    return (
        f"{instruction}\n\n"
        f"Contract type: {contract_type.value}\n"
        f"Jurisdiction: {jurisdiction.value}\n"
        f"Use party names appropriate for this contract type "
        f"(e.g., 'the Service Provider' and 'the Client' for MSA, "
        f"'the Licensor' and 'the Licensee' for license agreements).\n"
        f"Begin immediately with sub-clause 1. Output ONLY the clause text."
    )


def _build_instruction(
    clause_type: ClauseType,
    contract_type: ContractType,
    jurisdiction: Jurisdiction,
) -> str:
    """Build the instruction string that will be used as the training input."""
    return (
        f"Draft a {clause_type.value.replace('_', ' ').title()} clause "
        f"for a {contract_type.value} governed by {jurisdiction.value} law."
    )


# ── Core generation ────────────────────────────────────────────────────────

def generate_clause_examples(
    examples_per_type: int = 10,
    clause_types: Optional[list[ClauseType]] = None,
    contract_types: Optional[list[ContractType]] = None,
    jurisdictions: Optional[list[Jurisdiction]] = None,
) -> list[TrainingExample]:
    """Generate gold-standard clause examples using Claude.

    Iterates over clause_type × contract_type × jurisdiction combinations
    and generates training pairs.
    """
    client = _get_client()

    if clause_types is None:
        clause_types = [ct for ct in ClauseType if ct not in (ClauseType.UNKNOWN, ClauseType.INDEMNITY)]
    if contract_types is None:
        contract_types = [ContractType.MSA, ContractType.SAAS, ContractType.NDA,
                          ContractType.EMPLOYMENT, ContractType.SOFTWARE_LICENSE]
    if jurisdictions is None:
        jurisdictions = [Jurisdiction.CALIFORNIA, Jurisdiction.NEW_YORK,
                         Jurisdiction.DELAWARE, Jurisdiction.INDIA, Jurisdiction.ENGLAND]

    # Build all combinations and sample
    combos = [
        (ct, ctype, j)
        for ct in clause_types
        for ctype in contract_types
        for j in jurisdictions
    ]
    random.shuffle(combos)

    target = examples_per_type * len(clause_types)
    type_counts: dict[ClauseType, int] = {ct: 0 for ct in clause_types}
    examples: list[TrainingExample] = []

    for clause_type, contract_type, jurisdiction in tqdm(combos, desc="Distilling with Claude"):
        if type_counts[clause_type] >= examples_per_type:
            continue
        if len(examples) >= target:
            break

        user_prompt = _build_user_prompt(clause_type, contract_type, jurisdiction)

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            clause_text = response.content[0].text.strip()
        except Exception as e:
            logger.warning("Claude call failed for %s/%s/%s: %s",
                           clause_type, contract_type, jurisdiction, e)
            time.sleep(2)
            continue

        if len(clause_text) < 100:
            logger.debug("Response too short (%d chars), skipping", len(clause_text))
            continue

        instruction = _build_instruction(clause_type, contract_type, jurisdiction)

        example = TrainingExample(
            instruction=instruction,
            response=clause_text,
            clause_type=clause_type,
            contract_type=contract_type,
            jurisdiction=jurisdiction,
            source="distilled",
            quality_score=5.0,  # Claude output is our gold standard
            metadata={
                "model": MODEL,
                "prompt_version": "v1",
            },
        )
        examples.append(example)
        type_counts[clause_type] += 1

        # Rate limit: ~50 req/min for Sonnet
        time.sleep(1.2)

    logger.info(
        "Generated %d distilled examples across %d clause types",
        len(examples), len([ct for ct, n in type_counts.items() if n > 0]),
    )
    return examples


# ── Quality grading ────────────────────────────────────────────────────────

QUALITY_SYSTEM = """You are a senior legal quality reviewer. Grade the following contract clause
on a scale of 1-5:

5 = Production-ready: legally precise, complete, specific numbers/dates, no placeholders
4 = Good: minor style issues but legally sound and complete
3 = Acceptable: some vagueness or missing specifics but structurally correct
2 = Poor: significant issues — missing sub-clauses, placeholders, wrong contract type
1 = Unusable: garbage, wrong topic, incoherent

Output ONLY a JSON object: {"score": N, "reason": "one sentence"}"""


def grade_clause(client: anthropic.Anthropic, clause_text: str, clause_type: str) -> tuple[float, str]:
    """Grade a clause for training data quality using Claude."""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=100,
            system=QUALITY_SYSTEM,
            messages=[{"role": "user", "content": f"Clause type: {clause_type}\n\n{clause_text}"}],
        )
        text = response.content[0].text.strip()
        data = json.loads(text)
        return float(data["score"]), data.get("reason", "")
    except Exception as e:
        logger.warning("Grading failed: %s", e)
        return 3.0, "grading failed"
