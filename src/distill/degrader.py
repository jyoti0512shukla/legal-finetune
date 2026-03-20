"""Controlled clause degradation — generates MEDIUM and HIGH risk versions
from real LOW-risk EDGAR clauses.

Takes a well-drafted clause and asks GPT-4o to produce weakened (MEDIUM)
and broken (HIGH) versions. This creates balanced training data for
risk assessment without needing real high-risk contracts.

Each real clause yields 3 training examples:
  - Original    → LOW risk
  - Weakened    → MEDIUM risk  (vague, missing specifics)
  - Broken      → HIGH risk    (missing key protections)
"""

import json
import logging
import os
import time
from typing import Optional

from openai import OpenAI
from tqdm import tqdm

from src.data.schema import ClauseType, ContractType, Jurisdiction, ExtractedClause, TrainingExample

logger = logging.getLogger(__name__)

MODEL = "gpt-4o"

DEGRADE_MEDIUM_SYSTEM = """You are a legal training data generator. Given a well-drafted contract clause,
produce a WEAKENED version that a legal reviewer would rate as MEDIUM risk.

Rules for MEDIUM risk:
- Keep the clause structure intact but make it vague
- Remove specific numbers (days, percentages, amounts) and replace with vague language ("reasonable", "as agreed", "promptly")
- Remove statutory references
- Make obligations one-sided where they were mutual
- Keep the clause recognizable — it should still look like a real clause, just a poor one

Output ONLY the weakened clause text. No explanation."""

DEGRADE_HIGH_SYSTEM = """You are a legal training data generator. Given a well-drafted contract clause,
produce a BROKEN version that a legal reviewer would rate as HIGH risk.

Rules for HIGH risk:
- Remove entire critical sub-clauses (e.g., remove the liability cap, remove the indemnity, remove breach notification)
- Strip out all protections for one party
- Leave dangerous gaps (no termination rights, no dispute resolution, no data breach notification)
- Make it obviously deficient — a junior lawyer should spot the problems
- Keep it short — HIGH risk clauses are often incomplete

Output ONLY the broken clause text. No explanation."""

RISK_LABEL_SYSTEM = """You are a senior legal risk analyst. Rate this contract clause and explain why.

Output EXACTLY this JSON format:
{"risk": "HIGH|MEDIUM|LOW", "issues": ["issue 1", "issue 2"], "summary": "one sentence overall assessment"}

HIGH = clause missing, dangerously one-sided, or has critical gaps
MEDIUM = clause present but vague, incomplete, or missing specifics
LOW = clause clear, balanced, specific, and enforceable"""


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def degrade_clause(
    client: OpenAI,
    clause: ExtractedClause,
) -> tuple[Optional[str], Optional[str]]:
    """Generate MEDIUM and HIGH risk versions of a clause.

    Returns (medium_text, high_text). Either can be None on failure.
    """
    medium_text = None
    high_text = None

    # Generate MEDIUM version
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=1500,
            temperature=0.7,
            messages=[
                {"role": "system", "content": DEGRADE_MEDIUM_SYSTEM},
                {"role": "user", "content": f"Clause type: {clause.clause_type.value}\n\n{clause.text[:3000]}"},
            ],
        )
        medium_text = resp.choices[0].message.content.strip()
        if len(medium_text) < 50:
            medium_text = None
    except Exception as e:
        logger.warning("MEDIUM degradation failed: %s", e)

    time.sleep(0.5)

    # Generate HIGH version
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=1000,
            temperature=0.8,
            messages=[
                {"role": "system", "content": DEGRADE_HIGH_SYSTEM},
                {"role": "user", "content": f"Clause type: {clause.clause_type.value}\n\n{clause.text[:3000]}"},
            ],
        )
        high_text = resp.choices[0].message.content.strip()
        if len(high_text) < 30:
            high_text = None
    except Exception as e:
        logger.warning("HIGH degradation failed: %s", e)

    return medium_text, high_text


def label_clause_risk(
    client: OpenAI,
    clause_text: str,
    clause_type: str,
) -> Optional[dict]:
    """Ask GPT-4o to rate and explain a clause's risk level.

    Returns {"risk": "HIGH|MEDIUM|LOW", "issues": [...], "summary": "..."}
    """
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=300,
            temperature=0.2,
            messages=[
                {"role": "system", "content": RISK_LABEL_SYSTEM},
                {"role": "user", "content": f"Clause type: {clause_type}\n\n{clause_text[:3000]}"},
            ],
        )
        text = resp.choices[0].message.content.strip()
        # Extract JSON
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception as e:
        logger.warning("Risk labeling failed: %s", e)
        return None


def generate_risk_training_data(
    clauses: list[ExtractedClause],
    max_clauses: Optional[int] = None,
) -> list[TrainingExample]:
    """Generate balanced risk assessment training data from EDGAR clauses.

    For each clause:
    1. Original → labeled as LOW (verified by GPT-4o)
    2. Weakened version → labeled as MEDIUM
    3. Broken version → labeled as HIGH

    Returns instruction/response pairs for risk assessment training.
    """
    client = _get_client()

    if max_clauses:
        clauses = clauses[:max_clauses]

    examples: list[TrainingExample] = []

    for clause in tqdm(clauses, desc="Generating risk training data"):
        # 1. Label the original clause
        original_label = label_clause_risk(client, clause.text, clause.clause_type.value)
        time.sleep(0.5)

        if original_label:
            examples.append(TrainingExample(
                instruction=f"Assess the risk level of this {clause.clause_type.value} clause:\n\n{clause.text[:2000]}",
                response=json.dumps(original_label),
                clause_type=clause.clause_type,
                contract_type=clause.contract_type,
                jurisdiction=Jurisdiction.CALIFORNIA,
                source="risk_original",
                quality_score=5.0,
            ))

        # 2. Generate degraded versions
        medium_text, high_text = degrade_clause(client, clause)
        time.sleep(0.5)

        if medium_text:
            medium_label = label_clause_risk(client, medium_text, clause.clause_type.value)
            time.sleep(0.5)
            if medium_label:
                # Force the label to MEDIUM if GPT-4o disagrees (we know it was deliberately weakened)
                medium_label["risk"] = "MEDIUM"
                examples.append(TrainingExample(
                    instruction=f"Assess the risk level of this {clause.clause_type.value} clause:\n\n{medium_text[:2000]}",
                    response=json.dumps(medium_label),
                    clause_type=clause.clause_type,
                    contract_type=clause.contract_type,
                    jurisdiction=Jurisdiction.CALIFORNIA,
                    source="risk_degraded_medium",
                    quality_score=5.0,
                ))

        if high_text:
            high_label = label_clause_risk(client, high_text, clause.clause_type.value)
            time.sleep(0.5)
            if high_label:
                high_label["risk"] = "HIGH"
                examples.append(TrainingExample(
                    instruction=f"Assess the risk level of this {clause.clause_type.value} clause:\n\n{high_text[:2000]}",
                    response=json.dumps(high_label),
                    clause_type=clause.clause_type,
                    contract_type=clause.contract_type,
                    jurisdiction=Jurisdiction.CALIFORNIA,
                    source="risk_degraded_high",
                    quality_score=5.0,
                ))

    logger.info(
        "Generated %d risk training examples (LOW: %d, MEDIUM: %d, HIGH: %d)",
        len(examples),
        sum(1 for e in examples if e.source == "risk_original"),
        sum(1 for e in examples if e.source == "risk_degraded_medium"),
        sum(1 for e in examples if e.source == "risk_degraded_high"),
    )
    return examples
