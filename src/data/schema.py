"""Data models for the fine-tuning pipeline.

These mirror the clause/contract types from legal-partner but are
kept independent so this repo has no Java dependency.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ClauseType(str, Enum):
    DEFINITIONS = "DEFINITIONS"
    SERVICES = "SERVICES"
    PAYMENT = "PAYMENT"
    CONFIDENTIALITY = "CONFIDENTIALITY"
    IP_RIGHTS = "IP_RIGHTS"
    LIABILITY = "LIABILITY"
    TERMINATION = "TERMINATION"
    FORCE_MAJEURE = "FORCE_MAJEURE"
    GOVERNING_LAW = "GOVERNING_LAW"
    GENERAL_PROVISIONS = "GENERAL_PROVISIONS"
    REPRESENTATIONS_WARRANTIES = "REPRESENTATIONS_WARRANTIES"
    DATA_PROTECTION = "DATA_PROTECTION"
    INDEMNITY = "INDEMNITY"
    UNKNOWN = "UNKNOWN"


class ContractType(str, Enum):
    MSA = "Master Services Agreement"
    NDA = "Non-Disclosure Agreement"
    SAAS = "SaaS Subscription Agreement"
    EMPLOYMENT = "Employment Agreement"
    SOFTWARE_LICENSE = "Software License Agreement"
    SUPPLY = "Supply Agreement"
    VENDOR = "Vendor Agreement"
    IP_LICENSE = "IP License Agreement"
    CONSULTING = "Consulting Agreement"
    OTHER = "Other"


class Jurisdiction(str, Enum):
    CALIFORNIA = "California, United States"
    NEW_YORK = "New York, United States"
    DELAWARE = "Delaware, United States"
    TEXAS = "Texas, United States"
    ENGLAND = "England and Wales"
    INDIA = "India"
    SINGAPORE = "Singapore"
    AUSTRALIA = "Australia"
    GERMANY = "Germany"


# EDGAR search presets — maps contract type to SEC EFTS search query
EDGAR_QUERIES = {
    ContractType.MSA: '"master services agreement" OR "master service agreement" OR "professional services agreement"',
    ContractType.SAAS: (
        '"software as a service" OR "saas agreement" '
        'OR "master subscription agreement" '
        'OR "subscription services agreement" '
        'OR "cloud services agreement" '
        'OR "platform services agreement"'
    ),
    ContractType.NDA: '"non-disclosure agreement" OR "confidentiality agreement" OR "mutual nda"',
    ContractType.EMPLOYMENT: '"employment agreement" "annual base salary"',
    ContractType.SOFTWARE_LICENSE: '"software license agreement" OR "end user license agreement"',
    ContractType.SUPPLY: '"supply agreement" OR "manufacturing and supply"',
    ContractType.VENDOR: '"vendor agreement" OR "vendor services agreement"',
    ContractType.IP_LICENSE: '"license agreement" "royalty" OR "patent license"',
    ContractType.CONSULTING: '"consulting agreement" OR "independent contractor agreement"',
}

# Clause heading patterns — used by the segmenter to identify clause boundaries
CLAUSE_HEADING_PATTERNS: dict[ClauseType, list[str]] = {
    ClauseType.DEFINITIONS: [
        "definitions", "defined terms", "interpretation",
    ],
    ClauseType.SERVICES: [
        "services", "scope of services", "scope of work", "statement of work",
        "description of services",
    ],
    ClauseType.PAYMENT: [
        "payment", "fees", "compensation", "pricing", "invoicing",
        "fees and payment", "payment terms",
    ],
    ClauseType.CONFIDENTIALITY: [
        "confidentiality", "confidential information", "non-disclosure",
        "nondisclosure", "proprietary information",
    ],
    ClauseType.IP_RIGHTS: [
        "intellectual property", "ip rights", "ip ownership",
        "ownership of work product", "proprietary rights",
    ],
    ClauseType.LIABILITY: [
        "limitation of liability", "liability", "limitation",
        "cap on liability", "aggregate liability",
    ],
    ClauseType.INDEMNITY: [
        "indemnification", "indemnity", "hold harmless",
    ],
    ClauseType.TERMINATION: [
        "termination", "term and termination", "duration and termination",
    ],
    ClauseType.FORCE_MAJEURE: [
        "force majeure", "act of god", "excused performance",
    ],
    ClauseType.GOVERNING_LAW: [
        "governing law", "applicable law", "choice of law",
        "dispute resolution", "arbitration", "jurisdiction",
    ],
    ClauseType.GENERAL_PROVISIONS: [
        "general provisions", "miscellaneous", "boilerplate",
        "entire agreement", "general",
    ],
    ClauseType.REPRESENTATIONS_WARRANTIES: [
        "representations", "warranties", "representations and warranties",
    ],
    ClauseType.DATA_PROTECTION: [
        "data protection", "data privacy", "personal data",
        "data processing", "gdpr", "dpdpa",
    ],
}


@dataclass
class RawContract:
    """A raw contract document fetched from EDGAR or other source."""
    source: str                         # "edgar", "cuad", "upload"
    source_id: str                      # EDGAR accession number, CUAD contract ID
    filename: str
    entity_name: str                    # filing company
    contract_type: ContractType
    text: str
    url: Optional[str] = None
    filing_date: Optional[str] = None


@dataclass
class ExtractedClause:
    """A single clause segmented from a contract."""
    contract_source_id: str
    contract_type: ContractType
    clause_type: ClauseType
    heading: str
    text: str
    char_count: int = 0
    quality_score: Optional[float] = None   # 1-5, set by quality filter

    def __post_init__(self):
        self.char_count = len(self.text)


@dataclass
class TrainingExample:
    """An instruction/response pair ready for fine-tuning."""
    instruction: str
    response: str
    clause_type: ClauseType
    contract_type: ContractType
    jurisdiction: Jurisdiction
    source: str                         # "edgar", "cuad", "distilled", "augmented"
    quality_score: Optional[float] = None
    metadata: dict = field(default_factory=dict)
