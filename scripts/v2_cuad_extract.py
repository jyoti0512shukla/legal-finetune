#!/usr/bin/env python3
"""V2 CUAD extraction — mine CUAD's 510 contracts for type-tagged training data.

CUAD is organized into Part_I/II/III subfolders by contract type. The
full_contract_txt/ folder is flat but filenames carry the type. Per-type:

  Type folder name → maps to our 10 contract types

This script:
  1. Reads CUAD's PDF folder structure to learn type for each contract
  2. Reads matching .txt file from full_contract_txt/
  3. Applies quality scoring (reuse from v2_pilot_extract)
  4. Applies diversity selection (max 2 per company per type)
  5. Outputs per-type buckets to data/v2_pilot/cuad/<TYPE>/

Usage:
    python scripts/v2_cuad_extract.py --target 25
"""

import argparse
import hashlib
import json
import logging
import re
import shutil
from collections import defaultdict, Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CUAD_ROOT = Path("/Users/jyotimishra/legal-finetune/data/raw/cuad/CUAD_v1")
PDF_ROOT = CUAD_ROOT / "full_contract_pdf"
TXT_ROOT = CUAD_ROOT / "full_contract_txt"
OUTPUT_ROOT = Path("/Users/jyotimishra/legal-finetune/data/v2_pilot/cuad")

# Map CUAD folder names → our v2 contract types
# Some folders are merged (e.g. Distributor + Reseller → RESELLER)
CUAD_FOLDER_TO_TYPE = {
    "Hosting": "SAAS",                         # Hosting agreements (some are SaaS-adjacent)
    "License_Agreements": "SOFTWARE_LICENSE",  # Direct match
    "Service": "MSA",                          # Service agreements
    "Supply": "VENDOR_SUPPLY",                 # Supply agreements
    "Manufacturing": "VENDOR_SUPPLY",          # Manufacturing also vendor-side
    "Distributor": "RESELLER",                 # Distributor = reseller
    "Reseller": "RESELLER",                    # Direct
    "Maintenance": "MAINTENANCE",              # Software maintenance
    "Outsourcing": "OUTSOURCING",              # MSA-adjacent
    "Consulting Agreements": "CONSULTING",     # Independent contractor / consulting
    "IP": "IP_LICENSE",                        # IP licensing
    "Development": "DEVELOPMENT",              # Software dev
}

# Contract types we care about for v2 (subset of all CUAD folders)
TARGET_TYPES = {
    "SOFTWARE_LICENSE",
    "MSA",
    "VENDOR_SUPPLY",
    "RESELLER",
    "MAINTENANCE",
    "OUTSOURCING",
    "CONSULTING",
    "IP_LICENSE",
    "SAAS",  # Try CUAD for SaaS even though most are old
}

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
class CuadCandidate:
    filename: str
    txt_path: Path
    contract_type: str       # Mapped to our v2 type
    cuad_folder: str         # Original CUAD folder
    filer: str               # Extracted from filename
    filing_year: int = 0
    text: str = ""
    word_count: int = 0
    quality_score: int = 0
    quality_issues: list = field(default_factory=list)
    fingerprint: str = ""


def extract_filer_from_filename(filename: str) -> tuple[str, int]:
    """Extract company name and year from CUAD filename.
    Examples:
      Freecook_20180605_S-1_EX-10.3_..._Hosting Agreement.pdf  → ("Freecook", 2018)
      ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT.txt     → ("ABILITYINC", 2020)
      LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.txt  → ("LIMEENERGYCO", 1999)
    """
    name = filename.replace(".txt", "").replace(".pdf", "")
    # Pattern A: NAME_YYYYMMDD_FORM_EX-X.X_...
    m = re.match(r"^([A-Za-z][A-Za-z0-9]+?)_(\d{4})\d{4}_", name)
    if m:
        return m.group(1), int(m.group(2))
    # Pattern B: NAME_MM_DD_YYYY-EX-...
    m = re.match(r"^([A-Z][A-Z0-9]*)_\d{2}_\d{2}_(\d{4})", name)
    if m:
        return m.group(1), int(m.group(2))
    # Fallback: just first segment
    m = re.match(r"^([A-Za-z][A-Za-z0-9]+)", name)
    return (m.group(1) if m else "Unknown", 0)


def score_contract(text: str) -> tuple[int, list[str]]:
    """Reused from v2_pilot_extract — score 1-5."""
    score = 5
    issues = []
    word_count = len(text.split())

    if word_count < 1500:
        score -= 2
        issues.append(f"too_short_{word_count}_words")
    if word_count > 30000:
        score -= 1
        issues.append("too_long")

    has_articles = bool(re.search(r"(?im)^\s*(?:ARTICLE|Article)\s+\d", text))
    has_sections = bool(re.search(r"(?im)^\s*(?:Section|SECTION)\s+\d", text))
    has_numbered = bool(re.search(r"(?m)^\s*\d+\.\s+[A-Z]", text))
    if not (has_articles or has_sections or has_numbered):
        score -= 2
        issues.append("no_clause_structure")

    missing = sum(1 for p in STANDARD_CLAUSES_REGEX if not re.search(p, text))
    score -= missing
    if missing > 0:
        issues.append(f"missing_{missing}_standard_clauses")

    redaction_count = len(REDACTION_RE.findall(text))
    if redaction_count > 20:
        score -= 2
        issues.append(f"heavy_redaction_{redaction_count}")
    elif redaction_count > 5:
        score -= 1

    if len(text) > 0:
        weird = len([c for c in text if ord(c) > 127]) / len(text)
        if weird > 0.05:
            score -= 2
            issues.append(f"ocr_artifacts_{weird:.2%}")

    has_parties = bool(re.search(r"(?i)(?:by and between|made.{0,20}between|the\s+\"\w+\"\s*\)\s*,?\s*and)", text[:3000]))
    if not has_parties:
        score -= 1
        issues.append("no_clear_parties")

    return max(0, score), issues


def fingerprint_text(text: str) -> str:
    title_match = re.search(r"(?im)^[A-Z][A-Z &/'\-]{3,80}AGREEMENT", text)
    start = title_match.start() if title_match else 0
    snippet = text[start:start + 2000]
    normalized = re.sub(r"\s+", " ", snippet).lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()


def redact_pii(text: str) -> tuple[str, dict]:
    counts = {"emails": len(EMAIL_RE.findall(text)), "phones": len(PHONE_RE.findall(text))}
    text = EMAIL_RE.sub("[Email]", text)
    text = PHONE_RE.sub("[Phone]", text)
    return text, counts


def refine_type_from_filename(filename: str, default_type: str) -> str:
    """CUAD's License_Agreements is mostly content licensing.
    Refine the type by reading the filename for stronger signals.
    Returns the refined type or 'SKIP' if it should be excluded.
    """
    fn = filename.lower()

    # Software vs content license disambiguation
    if default_type == "SOFTWARE_LICENSE":
        # Pure content license — exclude from software license bucket
        if "content license" in fn:
            return "CONTENT_LICENSE"  # Tracked separately, not in target
        if "trademark" in fn and "license" in fn:
            return "TRADEMARK_LICENSE"
        if "music" in fn or "publishing" in fn or "media" in fn:
            return "CONTENT_LICENSE"
        # Real software signals
        if any(k in fn for k in ["software", "technology", "saas", "patent", "platform"]):
            return "SOFTWARE_LICENSE"
        # Generic "License Agreement" without other signals — conservative keep
        if "license agreement" in fn:
            return "SOFTWARE_LICENSE"
        return "CONTENT_LICENSE"

    # Hosting agreements: only 2018+ count as SaaS-adjacent
    if default_type == "SAAS":
        # Check filename year — pre-2015 hosting is NOT modern SaaS
        m = re.search(r"_(\d{4})\d{4}_|_\d{2}_\d{2}_(\d{4})", filename)
        year = 0
        if m:
            year = int(m.group(1) or m.group(2) or 0)
        if year < 2015:
            return "LEGACY_HOSTING"  # Track but not in target
        return "SAAS"

    return default_type


def discover_cuad_contracts() -> list[CuadCandidate]:
    """Walk CUAD's PDF folders to learn contract types, then load matching .txt files."""
    candidates = []

    for part_dir in PDF_ROOT.iterdir():
        if not part_dir.is_dir():
            continue
        for type_dir in part_dir.iterdir():
            if not type_dir.is_dir():
                continue
            cuad_folder = type_dir.name
            default_type = CUAD_FOLDER_TO_TYPE.get(cuad_folder)
            if default_type not in TARGET_TYPES:
                continue

            # Find all PDFs in this type folder, then map to .txt
            for pdf in type_dir.glob("*.pdf"):
                txt_name = pdf.stem + ".txt"
                txt_path = TXT_ROOT / txt_name
                if not txt_path.exists():
                    continue

                # Refine type based on filename signals
                refined_type = refine_type_from_filename(pdf.name, default_type)
                if refined_type not in TARGET_TYPES:
                    continue  # Skipped (e.g. content license, legacy hosting)

                filer, year = extract_filer_from_filename(pdf.name)
                candidates.append(CuadCandidate(
                    filename=pdf.name,
                    txt_path=txt_path,
                    contract_type=refined_type,
                    cuad_folder=cuad_folder,
                    filer=filer,
                    filing_year=year,
                ))

    return candidates


def process_candidates(candidates: list[CuadCandidate]) -> list[CuadCandidate]:
    """Read text, score quality, dedup. Returns processed (still-passing) candidates."""
    passing = []
    seen_fp = set()

    for c in candidates:
        try:
            c.text = c.txt_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning("Failed to read %s: %s", c.txt_path, e)
            continue

        c.word_count = len(c.text.split())
        c.fingerprint = fingerprint_text(c.text)
        if c.fingerprint in seen_fp:
            continue
        seen_fp.add(c.fingerprint)

        c.quality_score, c.quality_issues = score_contract(c.text)
        if c.quality_score < 3:
            continue

        passing.append(c)

    return passing


def select_diverse_per_type(
    passing: list[CuadCandidate],
    target_per_type: int,
    max_per_company_per_type: int = 2,
    global_max_per_company: int = 4,
    min_year: int = 0,  # Set to 0 for CUAD since most are pre-2020
) -> dict[str, list[CuadCandidate]]:
    """Group by type, sort by quality, pick top with diversity."""
    by_type = defaultdict(list)
    for c in passing:
        if c.filing_year and c.filing_year < min_year:
            continue
        by_type[c.contract_type].append(c)

    # Sort each type by quality (highest first), then by year (newest first)
    for t in by_type:
        by_type[t].sort(key=lambda c: (c.quality_score, c.filing_year, c.word_count), reverse=True)

    selected: dict[str, list[CuadCandidate]] = defaultdict(list)
    global_company_counts: dict[str, int] = defaultdict(int)

    for type_name, candidates in by_type.items():
        per_type_company_counts: dict[str, int] = defaultdict(int)
        for c in candidates:
            if len(selected[type_name]) >= target_per_type:
                break
            if per_type_company_counts[c.filer] >= max_per_company_per_type:
                continue
            if global_company_counts[c.filer] >= global_max_per_company:
                continue
            selected[type_name].append(c)
            per_type_company_counts[c.filer] += 1
            global_company_counts[c.filer] += 1

    return dict(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=25, help="Target contracts per type")
    parser.add_argument("--min-year", type=int, default=0,
                        help="Min filing year (CUAD has many pre-2010 contracts; default 0 = no filter)")
    args = parser.parse_args()

    if not PDF_ROOT.exists():
        logger.error("CUAD not found at %s — run download first", CUAD_ROOT)
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    logger.info("=== Phase 1: Discover CUAD contracts ===")
    candidates = discover_cuad_contracts()
    logger.info("Discovered %d candidates across target types", len(candidates))

    type_counts = Counter(c.contract_type for c in candidates)
    logger.info("Per-type discovery:")
    for t, n in type_counts.most_common():
        logger.info("  %-20s %d", t, n)

    logger.info("\n=== Phase 2: Quality filtering ===")
    passing = process_candidates(candidates)
    logger.info("Passing quality filter: %d / %d", len(passing), len(candidates))

    pass_counts = Counter(c.contract_type for c in passing)
    logger.info("Per-type after quality:")
    for t, n in pass_counts.most_common():
        logger.info("  %-20s %d", t, n)

    logger.info("\n=== Phase 3: Diversity selection (target %d per type) ===", args.target)
    selected = select_diverse_per_type(
        passing, target_per_type=args.target, min_year=args.min_year
    )

    # Save and report
    full_report = {"target_per_type": args.target, "min_year": args.min_year, "types": {}}
    pii_grand_total = {"emails": 0, "phones": 0}

    for type_name, contracts in sorted(selected.items()):
        out_dir = OUTPUT_ROOT / type_name
        out_dir.mkdir(parents=True, exist_ok=True)

        type_report = {
            "selected": [],
            "company_distribution": {},
            "year_distribution": {},
            "quality_distribution": {},
        }
        company_dist = Counter()
        year_dist = Counter()
        quality_dist = Counter()
        type_pii = {"emails": 0, "phones": 0}

        for c in contracts:
            redacted, pii = redact_pii(c.text)
            type_pii["emails"] += pii["emails"]
            type_pii["phones"] += pii["phones"]

            safe_filer = re.sub(r"[^\w-]", "_", c.filer)[:50]
            year_str = str(c.filing_year) if c.filing_year else "unknown"
            out_path = out_dir / f"{type_name}_{safe_filer}_{year_str}_{c.filename.replace('.pdf', '.txt').replace('/', '_')}"
            out_path.write_text(redacted, encoding="utf-8")

            company_dist[c.filer] += 1
            year_dist[c.filing_year] += 1
            quality_dist[c.quality_score] += 1

            type_report["selected"].append({
                "filer": c.filer,
                "year": c.filing_year,
                "word_count": c.word_count,
                "quality_score": c.quality_score,
                "quality_issues": c.quality_issues,
                "cuad_folder": c.cuad_folder,
                "filename": c.filename,
            })

        type_report["company_distribution"] = dict(company_dist)
        type_report["unique_companies"] = len(company_dist)
        type_report["year_distribution"] = dict(sorted(year_dist.items()))
        type_report["quality_distribution"] = dict(quality_dist)
        type_report["count"] = len(contracts)
        type_report["pii_redacted"] = type_pii
        pii_grand_total["emails"] += type_pii["emails"]
        pii_grand_total["phones"] += type_pii["phones"]
        full_report["types"][type_name] = type_report

    full_report["pii_grand_total"] = pii_grand_total

    report_path = OUTPUT_ROOT / "diversity_report.json"
    report_path.write_text(json.dumps(full_report, indent=2, default=str))

    logger.info("\n=== CUAD EXTRACTION SUMMARY ===")
    for type_name, contracts in sorted(selected.items()):
        type_report = full_report["types"][type_name]
        logger.info("  %-20s %2d contracts, %2d unique companies, years %s",
                    type_name,
                    type_report["count"],
                    type_report["unique_companies"],
                    list(type_report["year_distribution"].keys())[:3] if type_report["year_distribution"] else "?")
    logger.info("PII redacted: %s", pii_grand_total)
    logger.info("Output dir:   %s", OUTPUT_ROOT)
    logger.info("Report:       %s", report_path)


if __name__ == "__main__":
    main()
