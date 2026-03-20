"""Quality filter — grades extracted clauses and removes low-quality ones.

Uses a combination of rule-based checks and optional Claude grading
to ensure only high-quality clauses enter the training set.
"""

import logging
import os
import re
import time
from typing import Optional

from src.data.schema import ExtractedClause, TrainingExample

logger = logging.getLogger(__name__)

# Patterns that indicate low-quality or incomplete text
DISQUALIFYING_PATTERNS = [
    re.compile(r"\[INSERT", re.IGNORECASE),
    re.compile(r"\[PARTY", re.IGNORECASE),
    re.compile(r"\[DATE\]", re.IGNORECASE),
    re.compile(r"\[___+\]"),
    re.compile(r"\*{3,}"),
    re.compile(r"\bTBD\b"),
    re.compile(r"\bTBC\b"),
    re.compile(r"PAGE \d+ OF \d+", re.IGNORECASE),
    re.compile(r"EXHIBIT [A-Z]", re.IGNORECASE),  # exhibit references without content
]

# Minimum quality thresholds
MIN_CHARS = 200
MAX_CHARS = 10_000
MIN_SENTENCES = 2
MIN_UNIQUE_WORDS = 30


def rule_based_score(text: str) -> tuple[float, list[str]]:
    """Score a clause 0-5 using rule-based heuristics. Returns (score, issues)."""
    issues = []
    score = 5.0

    # Length checks
    if len(text) < MIN_CHARS:
        issues.append(f"too short ({len(text)} chars)")
        score -= 2.0
    if len(text) > MAX_CHARS:
        issues.append(f"too long ({len(text)} chars)")
        score -= 0.5

    # Sentence count
    sentences = re.split(r"[.!?]+\s", text)
    if len(sentences) < MIN_SENTENCES:
        issues.append(f"too few sentences ({len(sentences)})")
        score -= 1.5

    # Unique word count (catches repetitive/degenerate text)
    words = set(text.lower().split())
    if len(words) < MIN_UNIQUE_WORDS:
        issues.append(f"too few unique words ({len(words)})")
        score -= 1.5

    # Disqualifying patterns
    for pattern in DISQUALIFYING_PATTERNS:
        if pattern.search(text):
            issues.append(f"contains disqualifying pattern: {pattern.pattern}")
            score -= 1.0

    # Check for actual legal content (not just boilerplate/headers)
    legal_terms = ["shall", "party", "agreement", "pursuant", "herein",
                   "liability", "obligation", "terminate", "breach", "notice"]
    found = sum(1 for t in legal_terms if t in text.lower())
    if found < 2:
        issues.append("insufficient legal terminology")
        score -= 1.0

    return max(0.0, min(5.0, score)), issues


def filter_clauses(
    clauses: list[ExtractedClause],
    min_score: float = 3.0,
) -> list[ExtractedClause]:
    """Filter clauses by rule-based quality score."""
    kept = []
    for clause in clauses:
        score, issues = rule_based_score(clause.text)
        clause.quality_score = score
        if score >= min_score:
            kept.append(clause)
        else:
            logger.debug("Filtered out %s clause (%s): %.1f — %s",
                         clause.clause_type.value, clause.contract_source_id,
                         score, "; ".join(issues))

    logger.info("Quality filter: kept %d / %d clauses (min_score=%.1f)",
                len(kept), len(clauses), min_score)
    return kept


def filter_examples(
    examples: list[TrainingExample],
    min_score: float = 3.0,
) -> list[TrainingExample]:
    """Filter training examples by quality score."""
    kept = []
    for ex in examples:
        if ex.quality_score is None:
            score, _ = rule_based_score(ex.response)
            ex.quality_score = score
        if ex.quality_score >= min_score:
            kept.append(ex)

    logger.info("Quality filter: kept %d / %d examples (min_score=%.1f)",
                len(kept), len(examples), min_score)
    return kept


def grade_with_claude(
    clauses: list[ExtractedClause],
    max_clauses: Optional[int] = None,
) -> list[ExtractedClause]:
    """Grade clauses using Claude for higher-precision quality scoring.

    This is optional and costs ~$3 for 1000 clauses. Use rule_based_score
    for free filtering first, then Claude-grade the survivors.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — skipping Claude grading")
        return clauses

    from src.distill.claude_teacher import grade_clause
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    to_grade = clauses[:max_clauses] if max_clauses else clauses

    for clause in to_grade:
        score, reason = grade_clause(client, clause.text, clause.clause_type.value)
        clause.quality_score = score
        time.sleep(0.5)

    logger.info("Claude-graded %d clauses", len(to_grade))
    return clauses
