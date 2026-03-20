"""CUAD dataset loader.

The CUAD (Contract Understanding Atticus Dataset) contains 510 real contracts
with 13,000+ annotations across 41 clause types. We map CUAD's clause labels
to our ClauseType enum and extract clause text with context.

Dataset: https://huggingface.co/datasets/cuad
Paper: https://arxiv.org/abs/2103.06268
"""

import logging
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

from src.data.schema import ClauseType, ContractType, ExtractedClause

logger = logging.getLogger(__name__)

# Map CUAD's 41 annotation categories to our ClauseType
CUAD_TO_CLAUSE_TYPE: dict[str, ClauseType] = {
    # Termination
    "Termination For Convenience": ClauseType.TERMINATION,
    "Rofr/Rofo/Rofn": ClauseType.TERMINATION,
    "Change Of Control": ClauseType.TERMINATION,
    "Anti-Assignment": ClauseType.TERMINATION,
    "Expiration Date": ClauseType.TERMINATION,
    "Renewal Term": ClauseType.TERMINATION,
    # Liability / Indemnity
    "Cap On Liability": ClauseType.LIABILITY,
    "Limitation Of Liability": ClauseType.LIABILITY,
    "Indemnification": ClauseType.INDEMNITY,
    "Insurance": ClauseType.LIABILITY,
    # Confidentiality
    "Non-Compete": ClauseType.CONFIDENTIALITY,
    "Non-Solicitation": ClauseType.CONFIDENTIALITY,
    "Non-Disclosure Agreement": ClauseType.CONFIDENTIALITY,
    # IP
    "Ip Ownership Assignment": ClauseType.IP_RIGHTS,
    "License Grant": ClauseType.IP_RIGHTS,
    "Joint Ip Ownership": ClauseType.IP_RIGHTS,
    # Payment
    "Price Restrictions": ClauseType.PAYMENT,
    "Minimum Commitment": ClauseType.PAYMENT,
    "Revenue/Profit Sharing": ClauseType.PAYMENT,
    "Audit Rights": ClauseType.PAYMENT,
    # Governing law
    "Governing Law": ClauseType.GOVERNING_LAW,
    "Competitive Restriction Exception": ClauseType.GOVERNING_LAW,
    # Force Majeure
    "Uncapped Liability": ClauseType.FORCE_MAJEURE,
    # General
    "Most Favored Nation": ClauseType.GENERAL_PROVISIONS,
    "Covenant Not To Sue": ClauseType.GENERAL_PROVISIONS,
    "Third Party Beneficiary": ClauseType.GENERAL_PROVISIONS,
    "Affiliate License-Loss Of Ip": ClauseType.GENERAL_PROVISIONS,
    "Warranty Duration": ClauseType.REPRESENTATIONS_WARRANTIES,
    "Post-Termination Services": ClauseType.GENERAL_PROVISIONS,
    "Volume Restriction": ClauseType.GENERAL_PROVISIONS,
    "Exclusivity": ClauseType.GENERAL_PROVISIONS,
    "No-Solicit Of Employees": ClauseType.GENERAL_PROVISIONS,
    "No-Solicit Of Customers": ClauseType.GENERAL_PROVISIONS,
    "Irrevocable Or Perpetual License": ClauseType.IP_RIGHTS,
    "Source Code Escrow": ClauseType.IP_RIGHTS,
}


def load_cuad(max_contracts: Optional[int] = None) -> list[ExtractedClause]:
    """Load CUAD dataset from HuggingFace and extract annotated clause spans.

    Returns a list of ExtractedClause objects with the annotated text and type.
    """
    logger.info("Loading CUAD dataset from HuggingFace...")
    ds = load_dataset("cuad", split="test")

    clauses: list[ExtractedClause] = []
    seen_ids: set[str] = set()

    for row in tqdm(ds, desc="Processing CUAD"):
        contract_id = row.get("id", "")

        # CUAD has one row per question — group by contract
        if max_contracts and len(seen_ids) >= max_contracts:
            if contract_id not in seen_ids:
                continue
        seen_ids.add(contract_id)

        title = row.get("title", "")
        context = row.get("context", "")
        question = row.get("question", "")
        answers = row.get("answers", {})
        answer_texts = answers.get("text", [])

        if not answer_texts or not answer_texts[0]:
            continue

        # Map the CUAD question category to our clause type
        # CUAD questions follow format: "Highlight the parts..."
        # The category is in the question or we match by title
        clause_type = ClauseType.UNKNOWN
        for cuad_label, ct in CUAD_TO_CLAUSE_TYPE.items():
            if cuad_label.lower() in question.lower():
                clause_type = ct
                break

        if clause_type == ClauseType.UNKNOWN:
            continue

        for answer_text in answer_texts:
            if len(answer_text.strip()) < 50:
                continue

            clause = ExtractedClause(
                contract_source_id=f"cuad_{contract_id}",
                contract_type=ContractType.OTHER,  # CUAD doesn't classify contract type
                clause_type=clause_type,
                heading=question[:100],
                text=answer_text.strip(),
            )
            clauses.append(clause)

    logger.info("Loaded %d annotated clauses from CUAD (%d contracts)", len(clauses), len(seen_ids))
    return clauses
