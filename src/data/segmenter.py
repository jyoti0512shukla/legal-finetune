"""Contract clause segmenter.

Takes raw contract text and splits it into individual clauses by detecting
section headings. Each clause is classified by type using heading pattern matching.
"""

import logging
import re
from typing import Optional

from src.data.schema import (
    ClauseType,
    ContractType,
    CLAUSE_HEADING_PATTERNS,
    ExtractedClause,
    RawContract,
)

logger = logging.getLogger(__name__)

# Matches numbered section headings: "1.", "1.1", "ARTICLE I", "Section 3.2", "CLAUSE 5"
HEADING_PATTERN = re.compile(
    r"^(?:"
    r"(?:ARTICLE|SECTION|CLAUSE)\s+(?:[IVXLC]+|\d+)"  # ARTICLE IV, SECTION 3
    r"|\d{1,2}(?:\.\d{1,2})*\.?\s+"                    # 1. or 3.2 or 4.1.
    r")"
    r"\s*[-–—:]?\s*"
    r"(.+)",                                             # capture the heading text
    re.IGNORECASE | re.MULTILINE,
)

# Minimum clause length to keep (chars)
MIN_CLAUSE_LENGTH = 200
MAX_CLAUSE_LENGTH = 15_000


def classify_heading(heading: str) -> ClauseType:
    """Classify a section heading into a ClauseType by pattern matching."""
    heading_lower = heading.lower().strip().rstrip(".:;")

    for clause_type, patterns in CLAUSE_HEADING_PATTERNS.items():
        for pattern in patterns:
            if pattern in heading_lower:
                return clause_type

    return ClauseType.UNKNOWN


def segment_contract(contract: RawContract) -> list[ExtractedClause]:
    """Split a raw contract into individual clauses.

    Strategy:
    1. Find all section headings using regex
    2. Text between consecutive headings = one clause
    3. Classify each clause by its heading
    4. Filter out too-short or unclassified clauses
    """
    text = contract.text
    lines = text.split("\n")

    # Find heading positions
    headings: list[tuple[int, str, str]] = []  # (line_idx, full_match, heading_text)
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped or len(line_stripped) > 200:
            continue
        match = HEADING_PATTERN.match(line_stripped)
        if match:
            headings.append((i, line_stripped, match.group(1).strip()))

    if not headings:
        # Fallback: try splitting on all-caps lines as headings
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped and 5 < len(stripped) < 80
                    and stripped == stripped.upper()
                    and stripped[0].isalpha()):
                headings.append((i, stripped, stripped))

    if not headings:
        logger.debug("No headings found in %s", contract.source_id)
        return []

    # Extract text between consecutive headings
    clauses = []
    for idx, (line_idx, full_match, heading_text) in enumerate(headings):
        # End of this clause = start of next heading (or end of document)
        end_idx = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        body_lines = lines[line_idx + 1 : end_idx]
        body = "\n".join(body_lines).strip()

        if len(body) < MIN_CLAUSE_LENGTH:
            continue
        if len(body) > MAX_CLAUSE_LENGTH:
            body = body[:MAX_CLAUSE_LENGTH]

        clause_type = classify_heading(heading_text)

        clause = ExtractedClause(
            contract_source_id=contract.source_id,
            contract_type=contract.contract_type,
            clause_type=clause_type,
            heading=heading_text,
            text=body,
        )
        clauses.append(clause)

    known = [c for c in clauses if c.clause_type != ClauseType.UNKNOWN]
    logger.info(
        "Segmented %s: %d total clauses, %d classified (%s)",
        contract.source_id,
        len(clauses),
        len(known),
        ", ".join(c.clause_type.value for c in known),
    )
    return clauses


def segment_all(contracts: list[RawContract]) -> list[ExtractedClause]:
    """Segment a list of contracts into clauses."""
    all_clauses = []
    for contract in contracts:
        all_clauses.extend(segment_contract(contract))
    logger.info("Total: %d clauses from %d contracts", len(all_clauses), len(contracts))
    return all_clauses
