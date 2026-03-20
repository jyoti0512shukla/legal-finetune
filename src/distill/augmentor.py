"""Instruction augmentation — multiplies training data by generating
diverse instruction variations for the same clause text.

Given a real clause from EDGAR/CUAD, generates multiple instruction phrasings
so the model learns to respond to varied user requests.
"""

import json
import logging
import os
import time
from typing import Optional

import anthropic

from src.data.schema import (
    ClauseType,
    ContractType,
    Jurisdiction,
    ExtractedClause,
    TrainingExample,
)

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-20250514"

AUGMENT_SYSTEM = """You are generating training data for a legal AI model.
Given a real contract clause, generate diverse instruction phrasings that would
produce this clause as output.

Rules:
- Each instruction must be a natural request a lawyer would make
- Vary the specificity: some instructions are detailed, some are brief
- Include the contract type and jurisdiction in SOME (not all) instructions
- Output ONLY a JSON array of instruction strings, nothing else

Example output:
["Draft a payment terms clause for a California SaaS agreement",
 "Write the payment section with 30-day net terms and late payment interest",
 "Generate a fees and payment clause for a subscription agreement",
 "Draft payment terms covering invoicing, late fees, and disputed invoices"]"""


def augment_clause(
    client: anthropic.Anthropic,
    clause: ExtractedClause,
    variations: int = 5,
) -> list[TrainingExample]:
    """Generate instruction variations for a single extracted clause."""
    prompt = (
        f"Clause type: {clause.clause_type.value}\n"
        f"Contract type: {clause.contract_type.value}\n\n"
        f"Clause text:\n{clause.text[:3000]}\n\n"
        f"Generate exactly {variations} diverse instruction phrasings."
    )

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            system=AUGMENT_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Extract JSON array
        start = text.index("[")
        end = text.rindex("]") + 1
        instructions = json.loads(text[start:end])
    except Exception as e:
        logger.warning("Augmentation failed for %s: %s", clause.contract_source_id, e)
        return []

    examples = []
    for instruction in instructions:
        if not isinstance(instruction, str) or len(instruction) < 10:
            continue
        examples.append(TrainingExample(
            instruction=instruction,
            response=clause.text,
            clause_type=clause.clause_type,
            contract_type=clause.contract_type,
            jurisdiction=Jurisdiction.CALIFORNIA,  # default for EDGAR/US contracts
            source="augmented",
            quality_score=clause.quality_score,
        ))

    return examples


def augment_batch(
    clauses: list[ExtractedClause],
    variations_per_clause: int = 5,
    max_clauses: Optional[int] = None,
) -> list[TrainingExample]:
    """Augment a batch of clauses with instruction variations."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    client = anthropic.Anthropic(api_key=api_key)

    if max_clauses:
        clauses = clauses[:max_clauses]

    all_examples = []
    for clause in clauses:
        examples = augment_clause(client, clause, variations_per_clause)
        all_examples.extend(examples)
        time.sleep(1.0)  # rate limit

    logger.info("Augmented %d clauses into %d training examples", len(clauses), len(all_examples))
    return all_examples
