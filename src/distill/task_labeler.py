"""Task-specific labeling — generates training data for extraction, checklist,
and redline tasks using GPT-4o to label real EDGAR contracts.

Each function takes raw contract text and produces instruction/response
training pairs for a specific downstream task.
"""

import json
import logging
import os
import time
from typing import Optional

from openai import OpenAI
from tqdm import tqdm

from src.data.schema import ClauseType, ContractType, Jurisdiction, RawContract, TrainingExample

logger = logging.getLogger(__name__)

MODEL = "gpt-4o"


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


# ── Key Terms Extraction ───────────────────────────────────────────────────

EXTRACTION_SYSTEM = """You are a legal data extraction specialist. Extract key terms from this contract.

Output ONLY valid JSON:
{
  "party_a": "full legal name or null",
  "party_b": "full legal name or null",
  "effective_date": "YYYY-MM-DD or null",
  "expiry_date": "YYYY-MM-DD or null",
  "contract_value": "amount with currency or null",
  "liability_cap": "cap description or null",
  "governing_law": "jurisdiction or null",
  "notice_period_days": "number or null",
  "arbitration_venue": "venue or null"
}"""


def generate_extraction_examples(
    contracts: list[RawContract],
    max_contracts: Optional[int] = None,
) -> list[TrainingExample]:
    """Generate key terms extraction training pairs from EDGAR contracts."""
    client = _get_client()
    if max_contracts:
        contracts = contracts[:max_contracts]

    examples = []
    for contract in tqdm(contracts, desc="Generating extraction examples"):
        # Truncate to first 4000 chars (key terms are usually in the first pages)
        text = contract.text[:4000]
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=400,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {"role": "user", "content": f"Contract text:\n{text}"},
                ],
            )
            result = resp.choices[0].message.content.strip()
            # Validate it's JSON
            start = result.index("{")
            end = result.rindex("}") + 1
            parsed = json.loads(result[start:end])

            # Skip if too few fields extracted
            non_null = sum(1 for v in parsed.values() if v and v != "null")
            if non_null < 3:
                continue

            examples.append(TrainingExample(
                instruction=f"Extract key terms from this contract:\n\n{text}",
                response=json.dumps(parsed, indent=2),
                clause_type=ClauseType.DEFINITIONS,
                contract_type=contract.contract_type,
                jurisdiction=Jurisdiction.CALIFORNIA,
                source="extraction_labeled",
                quality_score=4.5,
            ))
        except Exception as e:
            logger.debug("Extraction labeling failed for %s: %s", contract.source_id, e)

        time.sleep(0.8)

    logger.info("Generated %d extraction training examples", len(examples))
    return examples


# ── Clause Checklist ───────────────────────────────────────────────────────

CHECKLIST_SYSTEM = """You are a senior legal reviewer. Check this contract for the presence and quality of 12 standard clauses.

For each clause, provide:
- clause_id: one of LIABILITY_LIMIT, INDEMNITY, TERMINATION_CONVENIENCE, TERMINATION_CAUSE, FORCE_MAJEURE, CONFIDENTIALITY, GOVERNING_LAW, DISPUTE_RESOLUTION, IP_OWNERSHIP, DATA_PROTECTION, PAYMENT_TERMS, ASSIGNMENT
- status: PRESENT (clearly present), WEAK (present but incomplete), MISSING (not found)
- risk_level: HIGH (missing/dangerous), MEDIUM (improvable), LOW (clear/balanced)
- finding: one sentence

Output ONLY valid JSON: {"clauses": [...]}"""


def generate_checklist_examples(
    contracts: list[RawContract],
    max_contracts: Optional[int] = None,
) -> list[TrainingExample]:
    """Generate clause checklist training pairs from EDGAR contracts."""
    client = _get_client()
    if max_contracts:
        contracts = contracts[:max_contracts]

    examples = []
    for contract in tqdm(contracts, desc="Generating checklist examples"):
        text = contract.text[:6000]
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=1200,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": CHECKLIST_SYSTEM},
                    {"role": "user", "content": f"Contract ({contract.entity_name}):\n{text}"},
                ],
            )
            result = resp.choices[0].message.content.strip()
            start = result.index("{")
            end = result.rindex("}") + 1
            parsed = json.loads(result[start:end])

            if "clauses" not in parsed or len(parsed["clauses"]) < 8:
                continue

            examples.append(TrainingExample(
                instruction=f"Check this contract for 12 standard clauses:\n\n{text}",
                response=json.dumps(parsed, indent=2),
                clause_type=ClauseType.GENERAL_PROVISIONS,
                contract_type=contract.contract_type,
                jurisdiction=Jurisdiction.CALIFORNIA,
                source="checklist_labeled",
                quality_score=4.5,
            ))
        except Exception as e:
            logger.debug("Checklist labeling failed for %s: %s", contract.source_id, e)

        time.sleep(1.0)

    logger.info("Generated %d checklist training examples", len(examples))
    return examples


# ── Redline Suggestions ───────────────────────────────────────────────────

REDLINE_SYSTEM = """You are a senior legal drafting expert. Given a contract clause that has been
identified as WEAK or problematic, suggest improved language.

Output ONLY valid JSON:
{
  "clause_name": "name of the clause",
  "issue": "one sentence describing the problem",
  "suggested_language": "the improved clause text ready for insertion",
  "rationale": "why this change protects the client"
}"""


def generate_redline_examples(
    clauses: list[dict],  # from edgar_clauses.jsonl
    max_clauses: Optional[int] = None,
) -> list[TrainingExample]:
    """Generate redline suggestion training pairs.

    Takes EDGAR clauses and asks GPT-4o to identify weaknesses and suggest fixes.
    Only clauses that GPT-4o finds issues with become training examples.
    """
    client = _get_client()
    if max_clauses:
        clauses = clauses[:max_clauses]

    examples = []
    for clause_data in tqdm(clauses, desc="Generating redline examples"):
        text = clause_data.get("text", "")[:3000]
        clause_type = clause_data.get("clause_type", "UNKNOWN")

        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=800,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": REDLINE_SYSTEM},
                    {"role": "user", "content": (
                        f"This {clause_type} clause may need improvement. "
                        f"If it has issues, suggest better language. "
                        f"If it's already strong, output: {{\"clause_name\": \"{clause_type}\", \"issue\": \"none\", \"suggested_language\": \"\", \"rationale\": \"\"}}\n\n"
                        f"{text}"
                    )},
                ],
            )
            result = resp.choices[0].message.content.strip()
            start = result.index("{")
            end = result.rindex("}") + 1
            parsed = json.loads(result[start:end])

            # Only keep examples where GPT-4o found actual issues
            if parsed.get("issue", "none").lower() == "none":
                continue
            if not parsed.get("suggested_language"):
                continue

            examples.append(TrainingExample(
                instruction=f"This {clause_type} clause needs improvement. Suggest better language:\n\n{text}",
                response=json.dumps(parsed, indent=2),
                clause_type=ClauseType(clause_type) if clause_type in ClauseType.__members__ else ClauseType.UNKNOWN,
                contract_type=ContractType(clause_data.get("contract_type", "Other")),
                jurisdiction=Jurisdiction.CALIFORNIA,
                source="redline_labeled",
                quality_score=4.5,
            ))
        except Exception as e:
            logger.debug("Redline labeling failed: %s", e)

        time.sleep(0.8)

    logger.info("Generated %d redline training examples", len(examples))
    return examples
