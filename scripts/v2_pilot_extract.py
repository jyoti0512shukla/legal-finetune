#!/usr/bin/env python3
"""V2 pilot extraction — pull a small set of contracts and validate the v2 quality pipeline.

Pulls 5 SaaS contracts from EDGAR, applies:
  - Quality scoring (reject < 3)
  - Type detection (reject UNKNOWN)
  - Year filter (reject pre-2020)
  - Deduplication
  - Diversity selection (max 2 per company, max 4 globally)
  - PII redaction
  - Length distribution check
  - Negotiation bias detection

Outputs:
  data/v2_pilot/<filer>_<year>_<accession>.txt  — passing contracts
  data/v2_pilot/diversity_report.json           — sanity check report
  data/v2_pilot/rejection_log.json              — what was rejected and why

Usage:
    python scripts/v2_pilot_extract.py --type SAAS --target 5
"""

import argparse
import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from src.data.edgar import search_edgar, download_exhibit
from src.data.schema import ContractType, EDGAR_QUERIES
from src.data.cik_whitelist import (
    SAAS_FILERS_TIERED, SAAS_TIER_TARGETS, all_saas_ciks, get_filer_tier,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/v2_pilot")

# ── Quality scoring ────────────────────────────────────────────────────────

REDACTION_RE = re.compile(r"\[(?:\*+|REDACTED|OMITTED)\]", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")

STANDARD_CLAUSES_REGEX = [
    r"(?i)governing\s+law",
    r"(?i)indemnif",
    r"(?i)limit.{0,20}liability",
    r"(?i)terminat",
    r"(?i)confidential",
]


@dataclass
class ContractCandidate:
    accession: str
    filer: str
    cik: str
    filing_year: int
    detected_type: str
    text: str
    url: str
    word_count: int = 0
    quality_score: int = 0
    quality_issues: list = field(default_factory=list)
    fingerprint: str = ""
    bias: str = "unknown"
    length_bucket: str = "unknown"
    tier: str = "unknown"  # mega/large/mid/growth/unknown (for whitelist sourcing)


def score_contract(text: str) -> tuple[int, list[str]]:
    """Score a contract 1-5 for quality. Returns (score, issues_list)."""
    score = 5
    issues = []

    word_count = len(text.split())

    if word_count < 1500:
        score -= 2
        issues.append(f"too_short_{word_count}_words")
    if word_count > 30000:
        score -= 1
        issues.append("too_long_likely_full_doc")

    has_articles = bool(re.search(r"(?im)^\s*(?:ARTICLE|Article)\s+\d", text))
    has_sections = bool(re.search(r"(?im)^\s*(?:Section|SECTION)\s+\d", text))
    has_numbered = bool(re.search(r"(?m)^\s*\d+\.\s+[A-Z]", text))
    if not (has_articles or has_sections or has_numbered):
        score -= 2
        issues.append("no_clause_structure")

    missing = sum(1 for p in STANDARD_CLAUSES_REGEX if not re.search(p, text))
    score -= missing  # -1 per missing standard clause
    if missing > 0:
        issues.append(f"missing_{missing}_standard_clauses")

    redaction_count = len(REDACTION_RE.findall(text))
    if redaction_count > 20:
        score -= 2
        issues.append(f"heavy_redaction_{redaction_count}")
    elif redaction_count > 5:
        score -= 1
        issues.append(f"some_redaction_{redaction_count}")

    # OCR garbage indicators (high non-ASCII ratio)
    if len(text) > 0:
        weird_ratio = len([c for c in text if ord(c) > 127]) / len(text)
        if weird_ratio > 0.05:
            score -= 2
            issues.append(f"ocr_artifacts_{weird_ratio:.2%}")

    # Schedule/exhibit only
    if re.match(r"^\s*Schedule\s+[A-Z]", text[:200]) and not has_articles:
        score -= 3
        issues.append("schedule_only")

    # Has parties
    has_parties = bool(re.search(r"(?i)(?:by and between|made.{0,20}between|the\s+\"\w+\"\s*\)\s*,?\s*and)", text[:3000]))
    if not has_parties:
        score -= 1
        issues.append("no_clear_parties")

    return max(0, score), issues


def detect_contract_type(text: str) -> tuple[str, dict]:
    """Detect the contract type from content. Returns (type, scores_dict)."""
    text_lower = text[:8000].lower()  # widened from 6000
    title_area = text[:1500].lower()  # for title-based detection

    # Title-based strong signals (these are nearly definitive)
    title_signals = {
        "SAAS": [
            "software as a service",
            "saas agreement",
            "subscription services agreement",
            "master subscription agreement",
            "cloud services agreement",
            "platform services agreement",
            "saas subscription",
        ],
        "MSA": [
            "master services agreement",
            "master service agreement",
            "professional services agreement",
        ],
        "NDA": [
            "non-disclosure agreement",
            "nondisclosure agreement",
            "confidentiality agreement",
            "mutual nda",
        ],
        "EMPLOYMENT": [
            "employment agreement",
        ],
        "SOFTWARE_LICENSE": [
            "software license agreement",
            "end user license agreement",
            "perpetual license agreement",
        ],
    }

    # Title match = score 10 (nearly definitive)
    scores = {k: 0 for k in title_signals}
    for type_key, phrases in title_signals.items():
        for phrase in phrases:
            if phrase in title_area:
                scores[type_key] += 10

    # Body indicators (lower weight)
    scores["SAAS"] += sum([
        2 if ("subscription" in text_lower and "fee" in text_lower) else 0,
        1 if "saas" in text_lower or "software-as-a-service" in text_lower else 0,
        1 if "uptime" in text_lower or "service level" in text_lower or "sla" in text_lower else 0,
        1 if "authorized users" in text_lower or "permitted users" in text_lower else 0,
        1 if "hosted" in text_lower else 0,
        1 if "platform" in text_lower and "access" in text_lower else 0,
        1 if "subscription term" in text_lower else 0,
        2 if "subscription fee" in text_lower else 0,
    ])
    scores["MSA"] += sum([
        2 if "statement of work" in text_lower else 0,
        1 if "deliverables" in text_lower and "saas" not in text_lower else 0,
        1 if "professional services" in text_lower else 0,
        1 if "scope of work" in text_lower else 0,
    ])
    scores["NDA"] += sum([
        1 if "non-disclosure" in text_lower or "nondisclosure" in text_lower else 0,
        1 if "confidential information" in text_lower and len(text) < 10000 else 0,
        1 if "purpose of the disclosure" in text_lower else 0,
        1 if "receiving party" in text_lower and "disclosing party" in text_lower else 0,
    ])
    scores["EMPLOYMENT"] += sum([
        1 if "employee" in text_lower and "employer" in text_lower else 0,
        2 if "annual base salary" in text_lower or "annual salary" in text_lower else 0,
        1 if "termination of employment" in text_lower else 0,
        1 if "executive" in text_lower and "compensation" in text_lower else 0,
    ])
    scores["SOFTWARE_LICENSE"] += sum([
        1 if "software license" in text_lower else 0,
        2 if "perpetual license" in text_lower else 0,
        1 if "object code" in text_lower or "source code" in text_lower else 0,
        1 if "licensee" in text_lower and "licensor" in text_lower else 0,
    ])

    best_type = max(scores, key=scores.get)
    # Threshold lowered: title match (10) OR strong body signals (>=4) qualifies
    if scores[best_type] < 4:
        return "UNKNOWN", scores
    return best_type, scores


def detect_bias(text: str) -> str:
    """Detect contract negotiation bias: pro_vendor / pro_customer / balanced."""
    pro_vendor = sum(text.lower().count(p.lower()) for p in [
        "vendor's sole discretion",
        "as determined by vendor",
        "vendor reserves the right",
        "as is basis",
        "without warranty",
        "provider's sole discretion",
    ])
    pro_customer = sum(text.lower().count(p.lower()) for p in [
        "customer's sole discretion",
        "subject to customer approval",
        "customer may terminate at any time",
        "service credits",
        "service level credits",
    ])
    if abs(pro_vendor - pro_customer) <= 2:
        return "balanced"
    return "pro_vendor" if pro_vendor > pro_customer else "pro_customer"


def length_bucket(word_count: int) -> str:
    if word_count < 2000:
        return "short"
    if word_count < 6000:
        return "medium"
    if word_count < 12000:
        return "long"
    return "very_long"


def fingerprint_text(text: str) -> str:
    """Stable fingerprint for dedup.
    Skips EDGAR exhibit header noise (filename, exhibit number) by anchoring on
    the first AGREEMENT title and hashing 2K chars from there. Falls back to a
    normalized hash of the first 2K chars if no AGREEMENT title is found.
    """
    # Find the first AGREEMENT title to anchor the fingerprint
    title_match = re.search(r"(?im)^[A-Z][A-Z &/'\-]{3,80}AGREEMENT", text)
    start = title_match.start() if title_match else 0
    # Take 2K chars from the anchor and normalize
    snippet = text[start:start + 2000]
    normalized = re.sub(r"\s+", " ", snippet).lower().strip()
    # Strip remaining EDGAR noise (exhibit refs like "EX-10.18 25 filename23.htm")
    normalized = re.sub(r"ex-?\s*\d+\.?\d*\s*\d*\s*[a-z0-9_]+\.htm", "", normalized)
    return hashlib.md5(normalized.encode()).hexdigest()


def redact_pii(text: str) -> tuple[str, dict]:
    """Redact emails and phones. Returns (redacted_text, counts_dict)."""
    counts = {"emails": 0, "phones": 0}
    counts["emails"] = len(EMAIL_RE.findall(text))
    counts["phones"] = len(PHONE_RE.findall(text))
    text = EMAIL_RE.sub("[Email]", text)
    text = PHONE_RE.sub("[Phone]", text)
    return text, counts


# ── Diversity selection ───────────────────────────────────────────────────

MAX_PER_COMPANY_PER_TYPE = 2
GLOBAL_MAX_PER_COMPANY = 4
MIN_YEAR = 2020


def select_diverse(
    candidates: list[ContractCandidate],
    target_count: int,
    global_company_counts: dict,
) -> tuple[list[ContractCandidate], list[dict]]:
    """Select up to target_count contracts maximizing company diversity."""
    selected: list[ContractCandidate] = []
    rejected: list[dict] = []
    company_counts: dict[str, int] = defaultdict(int)

    # Sort by quality (highest first)
    sorted_candidates = sorted(candidates, key=lambda c: c.quality_score, reverse=True)

    for c in sorted_candidates:
        if len(selected) >= target_count:
            break

        if company_counts[c.filer] >= MAX_PER_COMPANY_PER_TYPE:
            rejected.append({
                "accession": c.accession,
                "filer": c.filer,
                "reason": "max_per_company_per_type_reached",
            })
            continue

        if global_company_counts.get(c.filer, 0) + company_counts[c.filer] >= GLOBAL_MAX_PER_COMPANY:
            rejected.append({
                "accession": c.accession,
                "filer": c.filer,
                "reason": "global_max_per_company_reached",
            })
            continue

        selected.append(c)
        company_counts[c.filer] += 1

    return selected, rejected


def select_tier_balanced(
    candidates: list[ContractCandidate],
    tier_targets: dict[str, int],
    target_count: int,
) -> tuple[list[ContractCandidate], list[dict], dict]:
    """Select contracts to hit tier distribution targets, with backfill from
    adjacent tiers if a tier is short.

    Returns (selected, rejected, tier_counts).
    """
    selected: list[ContractCandidate] = []
    rejected: list[dict] = []
    company_counts: dict[str, int] = defaultdict(int)
    tier_counts: dict[str, int] = {t: 0 for t in tier_targets}

    # Group candidates by tier, sorted by quality
    by_tier: dict[str, list[ContractCandidate]] = defaultdict(list)
    for c in candidates:
        by_tier[c.tier].append(c)
    for tier in by_tier:
        by_tier[tier].sort(key=lambda c: (c.quality_score, c.word_count), reverse=True)

    # Pass 1: try to hit each tier's target
    for tier, target in tier_targets.items():
        for c in by_tier.get(tier, []):
            if tier_counts[tier] >= target:
                break
            if company_counts[c.filer] >= MAX_PER_COMPANY_PER_TYPE:
                continue
            selected.append(c)
            tier_counts[tier] += 1
            company_counts[c.filer] += 1

    # Pass 2: backfill from adjacent tiers if any underfilled
    backfill_order = ["large", "mega", "mid", "growth", "unknown"]
    while len(selected) < target_count:
        added = False
        for tier in backfill_order:
            for c in by_tier.get(tier, []):
                if c in selected:
                    continue
                if company_counts[c.filer] >= MAX_PER_COMPANY_PER_TYPE:
                    continue
                selected.append(c)
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
                company_counts[c.filer] += 1
                added = True
                break
            if added:
                break
        if not added:
            break  # No more candidates available

    # Build rejection log for unselected candidates
    selected_ids = {c.accession for c in selected}
    for c in candidates:
        if c.accession not in selected_ids:
            reason = "below_quality_cutoff_for_tier"
            if company_counts[c.filer] >= MAX_PER_COMPANY_PER_TYPE:
                reason = f"company_cap_reached_{c.filer}"
            rejected.append({
                "accession": c.accession,
                "filer": c.filer,
                "tier": c.tier,
                "quality_score": c.quality_score,
                "reason": reason,
            })

    return selected, rejected, tier_counts


# ── Pipeline ──────────────────────────────────────────────────────────────

def extract_year(date_str: str) -> int:
    """Parse YYYY from a date string."""
    if not date_str:
        return 0
    m = re.match(r"(\d{4})", date_str)
    return int(m.group(1)) if m else 0


def fetch_hits_from_whitelist(contract_type: ContractType, tiered_filers: dict) -> list[dict]:
    """Fetch EX-10.x hits from the CIK whitelist (one search per company)."""
    query = EDGAR_QUERIES.get(contract_type)
    all_hits = []
    flat = []
    for tier, filers in tiered_filers.items():
        for name, cik in filers.items():
            flat.append((name, cik, tier))

    logger.info("Fetching from %d whitelisted filers (CIK-based)", len(flat))
    for name, cik, tier in flat:
        try:
            hits = search_edgar(
                query=query,
                start_date="2020-01-01",
                end_date="2025-12-31",
                max_results=20,  # Per company — gives diversity within filer
                cik=cik,
            )
            for h in hits:
                h["_tier"] = tier
                h["_whitelist_filer"] = name
            all_hits.extend(hits)
            logger.info("  %s [%s]: %d hits", name, tier, len(hits))
        except Exception as e:
            logger.warning("  %s: failed - %s", name, e)
    return all_hits


def fetch_hits_broad_fallback(contract_type: ContractType, max_results: int) -> list[dict]:
    """Fallback: broad EDGAR search (used to fill gaps if whitelist is short)."""
    query = EDGAR_QUERIES.get(contract_type)
    return search_edgar(query, start_date="2020-01-01", end_date="2025-12-31",
                        max_results=max_results)


SAAS_SUBQUERIES = [
    '"software as a service"',
    '"saas agreement"',
    '"master subscription agreement"',
    '"subscription services agreement"',
    '"cloud services agreement"',
    '"platform services agreement"',
]


def run_pilot(contract_type: ContractType, target_count: int):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run multiple sub-queries and merge — single OR query hits SEC server errors at high offset
    sub_queries = SAAS_SUBQUERIES if contract_type == ContractType.SAAS else [EDGAR_QUERIES.get(contract_type)]
    raw_hits = []
    seen_accessions = set()
    for sq in sub_queries:
        logger.info("Searching: %s", sq)
        try:
            hits = search_edgar(
                sq, start_date="2020-01-01", end_date="2025-12-31",
                max_results=500,  # per sub-query
            )
            new = [h for h in hits if h["accession"] not in seen_accessions]
            raw_hits.extend(new)
            seen_accessions.update(h["accession"] for h in new)
            logger.info("  → %d new (total %d)", len(new), len(raw_hits))
        except Exception as e:
            logger.warning("Sub-query failed: %s — %s", sq, e)
        if len(raw_hits) >= target_count * 30:
            break

    logger.info("Total raw hits: %d", len(raw_hits))

    tier_targets = None  # Disabled — using broad search only

    candidates: list[ContractCandidate] = []
    rejection_log: list[dict] = []

    seen_fingerprints: set[str] = set()

    for hit in raw_hits:
        # Year filter
        year = extract_year(hit.get("date", ""))
        if year < MIN_YEAR:
            rejection_log.append({"accession": hit["accession"], "reason": f"year_too_old_{year}"})
            continue

        text = download_exhibit(hit["url"])
        if not text:
            rejection_log.append({"accession": hit["accession"], "reason": "download_failed"})
            continue

        # Deduplication
        fp = fingerprint_text(text)
        if fp in seen_fingerprints:
            rejection_log.append({"accession": hit["accession"], "filer": hit["entity"], "reason": "duplicate"})
            continue
        seen_fingerprints.add(fp)

        # Quality scoring
        score, issues = score_contract(text)
        if score < 3:
            rejection_log.append({
                "accession": hit["accession"],
                "filer": hit["entity"],
                "reason": f"quality_score_{score}",
                "issues": issues,
            })
            continue

        # Type detection
        detected, type_scores = detect_contract_type(text)
        if detected == "UNKNOWN":
            rejection_log.append({
                "accession": hit["accession"],
                "filer": hit["entity"],
                "reason": "type_unknown",
                "type_scores": type_scores,
            })
            continue

        # Don't accept contracts that don't match the requested type
        if detected != contract_type.name:
            rejection_log.append({
                "accession": hit["accession"],
                "filer": hit["entity"],
                "reason": f"type_mismatch_{detected}_expected_{contract_type.name}",
            })
            continue

        # Build candidate
        word_count = len(text.split())
        tier = hit.get("_tier", "unknown")

        candidate = ContractCandidate(
            accession=hit["accession"],
            filer=hit["entity"],
            cik=hit["cik"],
            filing_year=year,
            detected_type=detected,
            text=text,
            url=hit["url"],
            word_count=word_count,
            quality_score=score,
            quality_issues=issues,
            fingerprint=fp,
            bias=detect_bias(text),
            length_bucket=length_bucket(word_count),
            tier=tier,
        )
        candidates.append(candidate)
        logger.info("ACCEPTED %s [%s] (%s, %d, score=%d, %s)",
                    candidate.filer, tier, year, word_count, score, candidate.length_bucket)

        # Stop if we have enough quality candidates to select from
        if len(candidates) >= target_count * 4:
            break

    logger.info("After quality + type filter: %d candidates from %d hits",
                len(candidates), len(raw_hits))

    # Selection: tier-balanced if we have a whitelist, else basic diversity
    tier_counts: dict[str, int] = {}
    if tier_targets:
        # Scale tier targets if user wants fewer/more than 25
        scaled_targets = {t: max(1, round(c * target_count / 25)) for t, c in tier_targets.items()}
        # Make sure scaled total matches target
        diff = target_count - sum(scaled_targets.values())
        if diff != 0:
            biggest_tier = max(scaled_targets, key=scaled_targets.get)
            scaled_targets[biggest_tier] += diff
        logger.info("Tier targets (scaled to %d): %s", target_count, scaled_targets)
        selected, diversity_rejections, tier_counts = select_tier_balanced(
            candidates, scaled_targets, target_count
        )
    else:
        selected, diversity_rejections = select_diverse(candidates, target_count, global_company_counts={})
    rejection_log.extend(diversity_rejections)

    # PII redaction + save
    saved_files = []
    pii_total = {"emails": 0, "phones": 0}
    for c in selected:
        redacted_text, pii_counts = redact_pii(c.text)
        pii_total["emails"] += pii_counts["emails"]
        pii_total["phones"] += pii_counts["phones"]

        safe_filer = re.sub(r"[^\w-]", "_", c.filer)[:50]
        filename = f"{contract_type.name}_{safe_filer}_{c.filing_year}_{c.accession}.txt"
        path = OUTPUT_DIR / filename
        path.write_text(redacted_text, encoding="utf-8")
        saved_files.append(filename)

    # Build diversity report
    company_distribution = defaultdict(int)
    year_distribution = defaultdict(int)
    bias_distribution = defaultdict(int)
    length_distribution = defaultdict(int)
    tier_distribution = defaultdict(int)
    for c in selected:
        company_distribution[c.filer] += 1
        year_distribution[c.filing_year] += 1
        bias_distribution[c.bias] += 1
        length_distribution[c.length_bucket] += 1
        tier_distribution[c.tier] += 1

    report = {
        "contract_type": contract_type.name,
        "target_count": target_count,
        "raw_hits": len(raw_hits),
        "after_quality_filter": len(candidates),
        "after_diversity_filter": len(selected),
        "rejected_total": len(rejection_log),
        "selected": [{
            "filer": c.filer,
            "tier": c.tier,
            "year": c.filing_year,
            "accession": c.accession,
            "word_count": c.word_count,
            "quality_score": c.quality_score,
            "quality_issues": c.quality_issues,
            "bias": c.bias,
            "length_bucket": c.length_bucket,
        } for c in selected],
        "distribution": {
            "tiers": dict(tier_distribution),
            "tier_targets": tier_counts if tier_targets else None,
            "companies": dict(company_distribution),
            "unique_companies": len(company_distribution),
            "years": dict(year_distribution),
            "bias": dict(bias_distribution),
            "length_bucket": dict(length_distribution),
        },
        "pii_redacted": pii_total,
    }

    report_path = OUTPUT_DIR / "diversity_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    rejection_path = OUTPUT_DIR / "rejection_log.json"
    rejection_path.write_text(json.dumps(rejection_log[:50], indent=2))  # Top 50 rejections

    logger.info("\n=== PILOT SUMMARY ===")
    logger.info("Contract type:        %s", contract_type.name)
    logger.info("Raw hits:             %d", len(raw_hits))
    logger.info("After quality filter: %d", len(candidates))
    logger.info("Selected:             %d / %d", len(selected), target_count)
    logger.info("Unique companies:     %d", len(company_distribution))
    logger.info("Tier distribution:    %s", dict(tier_distribution))
    logger.info("Year spread:          %s", dict(year_distribution))
    logger.info("Bias:                 %s", dict(bias_distribution))
    logger.info("Length buckets:       %s", dict(length_distribution))
    logger.info("PII redacted:         %s", pii_total)
    logger.info("Saved %d files to %s", len(saved_files), OUTPUT_DIR)
    logger.info("Diversity report:     %s", report_path)
    logger.info("Rejection log:        %s", rejection_path)


def main():
    parser = argparse.ArgumentParser(description="V2 pilot corpus extraction")
    parser.add_argument("--type", default="SAAS",
                        choices=[t.name for t in ContractType])
    parser.add_argument("--target", type=int, default=5, help="Number of contracts to select")
    args = parser.parse_args()

    contract_type = ContractType[args.type]
    run_pilot(contract_type, args.target)


if __name__ == "__main__":
    main()
