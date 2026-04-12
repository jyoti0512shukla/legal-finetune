#!/usr/bin/env python3
"""V2 drafting examples generator — produce {instruction, output} training pairs.

For each tokenized contract template + entity map + per-contract metadata, this
generator extracts the major clauses (IP, Liability, Termination, Indemnity,
Confidentiality, Data Privacy, Service Level) and emits realistic
instruction → clause training examples.

Each example:
  - reuses an existing high-quality clause as the OUTPUT (no LLM hallucination)
  - generates a fresh, deterministic variation of party names / states /
    jurisdictions / fees so the model learns to substitute placeholders
  - builds a brief that mentions the same parties, industry, deal size,
    governing law, regulatory context, negotiation bias and commercial terms

Why this is the right shape:
  - The model needs to learn the *mapping* from a brief to a fully drafted
    clause that respects the slot values in the brief. By holding the clause
    body constant (modulo placeholder substitution) and varying both the brief
    *and* the placeholder values together, every example teaches the same
    skill: read the slots → apply them → produce the clause.
  - Industry / state / placeholder handling falls out for free because the
    same variation map is applied to both sides of the example.

Usage:
    python scripts/v2_generate_drafting_examples.py --type SAAS --variations 4

Output:
    data/v2/training/drafting/SAAS_drafting_examples.jsonl
    + per-clause-type counts manifest
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path

# Reuse the substitution-map generator from the variations script so the
# pools (party names, jurisdictions, fee ranges, addresses) stay consistent.
from v2_generate_variations import (
    generate_variation_map,
    apply_substitutions,
    ANNUAL_FEE_RANGES,
    IMPLEMENTATION_FEES,
    PROVIDER_NAMES,
    CUSTOMER_NAMES,
)

# Split the master fee list into deal-size tiers so a SMB contract never
# accidentally gets an enterprise fee in its variation.
# These slice boundaries match the comments in v2_generate_variations.py.
SMB_FEES = ANNUAL_FEE_RANGES[0:5]            # $12K – $48K
MID_MARKET_FEES = ANNUAL_FEE_RANGES[5:11]    # $72K – $240K
ENTERPRISE_FEES = ANNUAL_FEE_RANGES[11:]     # $360K – $1.8M

SMB_IMPL_FEES = IMPLEMENTATION_FEES[0:3]     # $10K – $50K
MID_MARKET_IMPL_FEES = IMPLEMENTATION_FEES[3:5]  # $75K – $100K
ENTERPRISE_IMPL_FEES = IMPLEMENTATION_FEES[5:]   # $150K – $200K

# Customer types whose identity is too tied to the contract context to be
# safely swapped out by the variation generator. For these, the variation
# keeps the ORIGINAL customer name, entity type, jurisdiction and address.
SPECIAL_CUSTOMER_TYPES = {
    "state_agency",
    "regulated_bank",
    "national_bank",
    "mutual_insurance_company",
    "law_firm",
    "professional_corporation",
    "broker_dealer",
    # Phase C–E additions: customer types whose contracts include very
    # industry-specific regulatory or business language that would clash
    # with a generic substitute customer name.
    "pharmaceutical_company",        # PharmaTrack — clinical trials, 21 CFR Part 11
    "regional_health_system",        # MedPath — HIPAA EHR for hospital network
    "academic_medical_center",       # TeleVita — telemedicine + NIH grants + FDA SaMD
    "registered_investment_adviser", # KeystoneAccess — SEC IA Cybersecurity Rule
    "franchisor_holding_company",    # BrandWaffle — franchise system + FTC Franchise Rule
    # MSA bucket additions
    "law_firm_engagement_with_corporate_client",  # Northwood — outside corporate counsel
    "public_company_audit_committee",             # Capstone — PCAOB-registered financial audit
    "federal_executive_agency",                   # Patriot IT — federal IT contractor
    "regional_health_system_locums",              # Mercer — healthcare staffing for hospitals
    "regional_bank_holding",                      # Sterling — bank risk advisory
    "pharmaceutical_sponsor",                     # BioCardinal — CRO for pharma sponsor
    "prime_federal_contractor",                   # DefenseCorp — federal subcontract under prime
    # NDA bucket — these customer types carry identity constraints that
    # generic name substitution would violate (e.g. an individual employee
    # can't be replaced with a random corporation).
    "individual_employee",                        # Atlas — employee confidentiality/invention assignment
    "public_company_acquirer",                    # StrataCore — M&A due diligence strategic acquirer
    "public_company_both_parties",                # NovaBio — two-public-company JV exploration
    "fortune_500_retailer",                       # Meridian — Fortune 500 customer-paper vendor NDA
    # EMPLOYMENT bucket — every employment agreement has an individual on
    # the customer (employee) side, plus various employer types where the
    # employer's identity is fixed by industry (regulated banking, federal
    # contractor, public company executive, etc.). The variation generator
    # must preserve these so the contracts stay coherent on every run.
    "individual_executive",                       # senior executive (CEO/CFO/GC) employed by a public company
    "individual_engineer",                        # senior engineer at a tech employer
    "individual_offer_letter",                   # at-will offer letter to a non-executive
    "individual_consultant_classified_as_employee",  # rare W-2 vs 1099 classification disputes
    "regulated_banking_executive_employer",      # Cathay-style two-party (holding co + bank) employment
    "public_company_executive_employer",         # MKS-style at-will public company executive
    "canadian_executive_employer",               # Aurinia-style BC ESA / federal Canadian
    "aviation_executive_employer",               # Mesa-style airline industry
    "biotech_executive_employer",                # Neurobo / CrestWell-style biotech executive
    "oil_gas_executive_employer",                # Vivakor-style midstream petroleum / Texas
    "startup_founder_employer",                  # founder hire / pre-IPO
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

V2_ROOT = Path("/Users/jyotimishra/legal-finetune/data/v2")
V2_TOKENIZED = V2_ROOT / "tokenized"
V2_RAW = V2_ROOT / "raw"
V2_TRAINING = V2_ROOT / "training" / "drafting"


# ── Clause type taxonomy ─────────────────────────────────────────────────
#
# Each clause type maps to a set of title keywords. We do title-based
# matching against the ARTICLE headers so the extractor works regardless of
# article number (different contracts use different numbering — e.g.
# MenuMaster has ARTICLE 3 — HARDWARE which shifts everything by one).

CLAUSE_TITLE_KEYWORDS = {
    "intellectual_property": [
        "INTELLECTUAL PROPERTY",
        "WORK PRODUCT",                    # MSA: work made for hire / assignment
        "OWNERSHIP OF DELIVERABLES",       # MSA alternative
        "OWNERSHIP OF WORK PRODUCT",       # MSA alternative
    ],
    "limitation_of_liability": [
        "LIMITATION OF LIABILITY",
        "LIABILITY",  # fallback if "LIMITATION OF" missing
    ],
    "termination": [
        "TERMINATION",
        "TERM AND TERMINATION",
    ],
    "indemnification": [
        "INDEMNIFICATION",
        "INDEMNITY",
    ],
    "confidentiality": [
        "CONFIDENTIALITY OBLIGATIONS",
        "CONFIDENTIAL INFORMATION",
        "OBLIGATIONS OF THE RECEIVING PARTY",
        "CONFIDENTIALITY",
    ],
    "data_privacy": [
        "CUSTOMER DATA",
        "DATA AND PRIVACY",
        "DATA, PRIVACY",
        "DATA PROTECTION",
    ],
    "service_level": [
        "SERVICE LEVEL",
        "SLA AND SUPPORT",
    ],
    # MSA-flavored clause types
    "services_scope": [
        "SCOPE OF SERVICES",
        "SCOPE OF WORK",
        "SERVICES AND SCOPE",
    ],
    "deliverables_acceptance": [
        "DELIVERABLES AND ACCEPTANCE",
        "DELIVERABLES",
        "ACCEPTANCE OF DELIVERABLES",
    ],
    "personnel": [
        "KEY PERSONNEL",
        "PERSONNEL AND STAFFING",
        "PERSONNEL",
    ],
    "subcontractors": [
        "SUBCONTRACTORS",
        "SUBCONTRACTING",
    ],
    "subprocessors": [
        "SUB-PROCESSORS",
        "SUBPROCESSORS",
        "SUB PROCESSORS",
        "USE OF SUB-PROCESSORS",
        "USE OF SUBPROCESSORS",
        "ENGAGEMENT OF SUB-PROCESSORS",
        "ONWARD SUB-PROCESSING",
        "ONWARD SUBPROCESSING",
        "SUB-OPERATORS",
        "SUBOPERATORS",
    ],
    "insurance": [
        "INSURANCE",
    ],
    # NDA-flavored clause types
    "purpose": [
        "PURPOSE",
        "PERMITTED PURPOSE",
        "PERMITTED USE",
    ],
    "exclusions": [
        "EXCLUSIONS FROM CONFIDENTIALITY",
        "EXCLUSIONS",
        "EXCEPTIONS TO CONFIDENTIALITY",
    ],
    "compelled_disclosure": [
        "COMPELLED DISCLOSURE",
        "REQUIRED DISCLOSURE",
        "DISCLOSURE REQUIRED BY LAW",
    ],
    "return_or_destruction": [
        "RETURN OR DESTRUCTION",
        "RETURN AND DESTRUCTION",
        "RETURN OF CONFIDENTIAL INFORMATION",
    ],
    "term_survival": [
        "TERM AND SURVIVAL",
        "SURVIVAL",
    ],
    "no_license": [
        "NO LICENSE",
        "NO IP TRANSFER",
        "NO IMPLIED LICENSE",
        "OWNERSHIP AND NO LICENSE",
    ],
    "equitable_remedies": [
        "EQUITABLE RELIEF",
        "EQUITABLE REMEDIES",
        "REMEDIES",
        "INJUNCTIVE RELIEF",
    ],
    # EMPLOYMENT-flavored clause types. The clause types here describe what
    # is uniquely needed for employment agreements: position, compensation,
    # severance, restrictive covenants, change-in-control. Some clause
    # types are reused from above (confidentiality, intellectual_property,
    # termination) — those don't need new keyword entries because the
    # existing keywords already match employment article titles.
    "position_duties": [
        "POSITION AND DUTIES",
        "DUTIES AND POSITION",
        "POSITION, DUTIES",
        "TITLE AND DUTIES",
        "EMPLOYMENT AND DUTIES",
    ],
    "compensation": [
        "COMPENSATION AND BENEFITS",
        "COMPENSATION",
        "BASE SALARY AND BONUS",
        "SALARY AND BENEFITS",
    ],
    "severance": [
        "SEVERANCE",
        "SEVERANCE BENEFITS",
        "TERMINATION BENEFITS",
        "COMPENSATION UPON TERMINATION",
        "PAYMENTS UPON TERMINATION",
    ],
    "restrictive_covenants": [
        "RESTRICTIVE COVENANTS",
        "NON-COMPETE",
        "NON-COMPETITION",
        "NON-SOLICITATION",
        "COVENANTS OF EXECUTIVE",
        "COVENANTS OF EMPLOYEE",
        "POST-EMPLOYMENT RESTRICTIONS",
    ],
    "change_in_control": [
        "CHANGE IN CONTROL",
        "CHANGE OF CONTROL",
        "CHANGE-IN-CONTROL",
    ],
    "dispute_resolution": [
        "DISPUTE RESOLUTION",
        "ARBITRATION",
        "GOVERNING LAW AND DISPUTE",
        "GOVERNING LAW AND ARBITRATION",
    ],
    # SOW-flavored clause types. SOWs are project-specific descriptions of work
    # that operate UNDER a parent MSA. Their core clauses focus on the specific
    # project: scope, milestones, fees structure, change control, assumptions,
    # and client responsibilities.
    #
    # Note: deliverables_acceptance, personnel, subcontractors, insurance
    # already exist above and are reused for SOWs. The clause types defined
    # here are the SOW-specific ones not present in MSA/SAAS/NDA/EMPLOYMENT.
    "sow_scope": [
        "PROJECT DESCRIPTION AND SCOPE",
        "PROJECT SCOPE",
        "SCOPE OF THIS STATEMENT OF WORK",
        "DESCRIPTION OF SERVICES",
        "PROJECT DESCRIPTION",
    ],
    "milestones_schedule": [
        "PROJECT SCHEDULE AND MILESTONES",
        "PROJECT SCHEDULE",
        "MILESTONES AND SCHEDULE",
        "MILESTONES",
        "SCHEDULE OF WORK",
        "PROJECT TIMELINE",
    ],
    "fees_payment_sow": [
        "FEES AND PAYMENT",
        "FEES, EXPENSES AND PAYMENT",
        "PROJECT FEES",
        "COMPENSATION AND INVOICING",
        "PRICING AND PAYMENT",
    ],
    "change_control": [
        "CHANGE CONTROL",
        "CHANGE CONTROL PROCEDURE",
        "CHANGE ORDERS",
        "CHANGE MANAGEMENT",
        "SCOPE CHANGES",
    ],
    "client_responsibilities": [
        "CLIENT RESPONSIBILITIES",
        "CUSTOMER RESPONSIBILITIES",
        "CUSTOMER OBLIGATIONS",
        "CLIENT OBLIGATIONS",
        "CUSTOMER COOPERATION",
    ],
    "assumptions_dependencies": [
        "ASSUMPTIONS AND DEPENDENCIES",
        "PROJECT ASSUMPTIONS",
        "ASSUMPTIONS",
        "DEPENDENCIES AND ASSUMPTIONS",
    ],
    # DPA-flavored clause types. DPAs (Data Processing Addendums) operate UNDER
    # a parent SaaS Agreement, MSA, or BAA. Their clauses are defined by GDPR
    # Article 28 and equivalents (UK GDPR, CCPA/CPRA service-provider terms,
    # India DPDP Act 2023, LGPD, PIPEDA). The clause types defined here are
    # the DPA-specific ones not present in other contract types.
    #
    # Note: subprocessors already exists above (added during SOW Phase 0) and
    # is reused for DPAs with a per-type override.
    "controller_processor_roles": [
        "RELATIONSHIP OF THE PARTIES",
        "ROLES OF THE PARTIES",
        "CONTROLLER AND PROCESSOR",
        "PARTIES' ROLES",
        "ROLE OF THE PARTIES",
    ],
    "processing_scope": [
        "SCOPE OF PROCESSING",
        "SUBJECT MATTER AND SCOPE",
        "PROCESSING DETAILS",
        "DETAILS OF PROCESSING",
        "NATURE AND PURPOSE OF PROCESSING",
    ],
    "security_measures": [
        "TECHNICAL AND ORGANIZATIONAL MEASURES",
        "TECHNICAL AND ORGANISATIONAL MEASURES",
        "SECURITY MEASURES",
        "SECURITY OF PROCESSING",
        "INFORMATION SECURITY",
    ],
    "breach_notification": [
        "PERSONAL DATA BREACH NOTIFICATION",
        "PERSONAL DATA BREACHES",
        "DATA BREACH NOTIFICATION",
        "BREACH NOTIFICATION",
        "INCIDENT NOTIFICATION",
    ],
    "international_transfers": [
        "INTERNATIONAL DATA TRANSFERS",
        "INTERNATIONAL TRANSFERS",
        "CROSS-BORDER TRANSFERS",
        "TRANSFERS OUT OF THE EEA",
        "RESTRICTED TRANSFERS",
    ],
    "data_subject_rights": [
        "DATA SUBJECT RIGHTS",
        "ASSISTANCE WITH DATA SUBJECT REQUESTS",
        "DATA SUBJECT REQUESTS",
        "RIGHTS OF DATA SUBJECTS",
    ],
    "dpia_cooperation": [
        "DATA PROTECTION IMPACT ASSESSMENTS",
        "DPIA COOPERATION",
        "ASSISTANCE WITH DPIAS",
        "PRIOR CONSULTATION",
    ],
    "deletion_return": [
        "DELETION OR RETURN OF PERSONAL DATA",
        "RETURN OR DELETION",
        "DELETION AND RETURN",
        "END-OF-PROCESSING DELETION",
    ],
    "audit_rights": [
        "AUDIT RIGHTS",
        "AUDITS AND INSPECTIONS",
        "CONTROLLER AUDIT RIGHTS",
        "RIGHT TO AUDIT",
    ],
    # Independent Contractor (IC) clause types. IC agreements operate as
    # standalone contracts (not addenda to a parent agreement), and the
    # core distinguishing clauses are:
    #   - ic_engagement: the engagement statement
    #   - ic_compensation: fees structure (hourly / fixed-fee / retainer /
    #     milestone / hybrid)
    #   - ic_classification: the worker-classification anchor (W-9, 1099,
    #     no benefits, no withholding) — this is the IC differentiator
    #     from EMPLOYMENT
    #   - ic_intellectual_property: work product assignment + pre-existing
    #     IP carve-out
    #   - ic_termination: termination for convenience and for cause
    #   - ic_restrictive_covenants: non-compete during, non-solicit, devote
    #     productive time
    #
    # Note: confidentiality, indemnification, dispute_resolution, insurance
    # are reused from the existing infrastructure with IC-specific overrides.
    "ic_engagement": [
        "ENGAGEMENT",
        "ENGAGEMENT OF CONTRACTOR",
        "ENGAGEMENT OF CONSULTANT",
        "RETENTION",
        "RETENTION OF CONTRACTOR",
        "ENGAGEMENT AND SERVICES",
    ],
    "ic_compensation": [
        "FEES AND EXPENSES",
        "FEES, EXPENSES",
        "COMPENSATION AND EXPENSES",
        "FEES AND PAYMENT",
        "CONSULTING FEES",
        "CONTRACTOR FEES",
        "FEES",
        "COMPENSATION OF CONTRACTOR",
        "COMPENSATION OF CONSULTANT",
    ],
    "ic_classification": [
        "INDEPENDENT CONTRACTOR RELATIONSHIP",
        "INDEPENDENT CONTRACTOR STATUS",
        "RELATIONSHIP OF THE PARTIES",
        "NO EMPLOYMENT RELATIONSHIP",
        "CONTRACTOR STATUS",
        "TAX MATTERS AND CLASSIFICATION",
    ],
    "ic_intellectual_property": [
        "WORK PRODUCT AND INTELLECTUAL PROPERTY",
        "OWNERSHIP OF WORK PRODUCT",
        "INVENTIONS AND WORK PRODUCT",
        "INTELLECTUAL PROPERTY ASSIGNMENT",
        "ASSIGNMENT OF WORK PRODUCT",
    ],
    "ic_termination": [
        "TERMINATION OF ENGAGEMENT",
        "TERMINATION AND WIND-DOWN",
        "TERM AND TERMINATION",
    ],
    "ic_restrictive_covenants": [
        "RESTRICTIVE COVENANTS AND CONFLICTS",
        "CONFLICTS OF INTEREST AND ETHICAL CONDUCT",
        "CONFLICTS AND NON-SOLICITATION",
        "NON-SOLICITATION AND NON-COMPETITION",
    ],
    # LICENSE clause types. License agreements grant rights in IP (patent,
    # trademark, copyright, software, trade secret, know-how) from a Licensor
    # to a Licensee. The core distinguishing clauses are:
    #   - license_grant: scope of the grant (exclusive/non-exclusive/sole,
    #     field of use, territory, term, sublicensing)
    #   - license_royalties_payment: upfront fee, milestone payments, running
    #     royalties, minimum annual royalties, royalty stacking, MFN
    #   - license_restrictions: licensee's restrictive covenants (no reverse
    #     engineering, no resale, no transfer, no derivative works, no
    #     out-of-field use)
    #   - license_term_termination: term, termination triggers, post-termination
    #     wind-down, sell-off period
    #   - license_warranties_indemnification: title warranty, infringement
    #     indemnification, defense and control of claims
    #   - license_audit_rights: licensor's right to audit licensee's books
    #     and royalty reports
    #   - license_improvements: ownership of improvements, grant-backs, joint
    #     inventions
    #   - license_quality_control: quality standards (especially for trademark
    #     licenses — naked license problem)
    #
    # Confidentiality is reused from the existing infrastructure with a
    # LICENSE-specific override.
    "license_grant": [
        "GRANT OF LICENSE",
        "LICENSE GRANT",
        "GRANT OF RIGHTS",
        "LICENSE",
        "SCOPE OF LICENSE",
        "RIGHTS GRANTED",
    ],
    "license_royalties_payment": [
        "ROYALTIES",
        "ROYALTIES AND PAYMENTS",
        "ROYALTIES AND PAYMENT",
        "ROYALTIES AND FEES",
        "FEES AND ROYALTIES",
        "FEES, EXPENSES, AND COMPENSATION",
        "FEES AND PAYMENT",
        "FEES, AND PAYMENT",
        "LICENSE FEES AND ROYALTIES",
        "LICENSE FEES",
        "PAYMENT AND ROYALTIES",
        "CONSIDERATION",
        "FINANCIAL TERMS",
        "FEES",
    ],
    "license_restrictions": [
        "RESTRICTIONS",
        "RESTRICTIONS ON USE",
        "USE RESTRICTIONS",
        "LICENSE RESTRICTIONS",
        "LICENSEE RESTRICTIONS",
        "PROHIBITED USES",
        "RESERVATION OF RIGHTS",
    ],
    "license_term_termination": [
        "TERM AND TERMINATION OF LICENSE",
        "LICENSE TERM AND TERMINATION",
        "TERM OF LICENSE",
        "TERMINATION OF LICENSE",
        "TERM AND TERMINATION",
    ],
    "license_warranties_indemnification": [
        "WARRANTIES AND INDEMNIFICATION",
        "REPRESENTATIONS, WARRANTIES, AND INDEMNIFICATION",
        "IP WARRANTIES AND INFRINGEMENT INDEMNITY",
        "TITLE AND INFRINGEMENT INDEMNIFICATION",
        "INFRINGEMENT INDEMNIFICATION",
    ],
    "license_audit_rights": [
        "ROYALTY AUDITS",
        "AUDIT OF ROYALTY REPORTS",
        "BOOKS AND RECORDS; AUDIT",
        "RECORDS AND AUDIT",
        "LICENSEE RECORDS AND AUDIT",
    ],
    "license_improvements": [
        "IMPROVEMENTS",
        "IMPROVEMENTS AND GRANT-BACK",
        "IMPROVEMENTS AND DERIVATIVE WORKS",
        "OWNERSHIP OF IMPROVEMENTS",
        "GRANT-BACK",
    ],
    "license_quality_control": [
        "QUALITY CONTROL",
        "QUALITY STANDARDS",
        "TRADEMARK QUALITY CONTROL",
        "BRAND STANDARDS",
        "QUALITY CONTROL AND APPROVAL",
    ],
}

# Human-readable clause names for the brief text. The display name can vary
# by contract type — e.g. "Intellectual Property Rights" for SaaS but
# "Work Product and IP Ownership" for MSA — so we override per-type below.
CLAUSE_DISPLAY_NAMES = {
    "intellectual_property": "Intellectual Property Rights",
    "limitation_of_liability": "Limitation of Liability",
    "termination": "Termination",
    "indemnification": "Indemnification",
    "confidentiality": "Confidentiality Obligations",
    "data_privacy": "Customer Data and Privacy",
    "service_level": "Service Level Agreement",
    "services_scope": "Scope of Services",
    "deliverables_acceptance": "Deliverables and Acceptance",
    "personnel": "Personnel and Staffing",
    "subcontractors": "Subcontractors",
    "subprocessors": "Sub-Processors",
    "insurance": "Insurance",
    # NDA-specific display names
    "purpose": "Permitted Purpose",
    "exclusions": "Exclusions from Confidentiality",
    "compelled_disclosure": "Compelled Disclosure",
    "return_or_destruction": "Return or Destruction of Confidential Information",
    "term_survival": "Term and Survival",
    "no_license": "No License / No IP Transfer",
    "equitable_remedies": "Equitable Remedies",
    # EMPLOYMENT-specific display names
    "position_duties": "Position and Duties",
    "compensation": "Compensation and Benefits",
    "severance": "Severance and Termination Benefits",
    "restrictive_covenants": "Restrictive Covenants",
    "change_in_control": "Change in Control",
    "dispute_resolution": "Dispute Resolution",
    # SOW-specific display names
    "sow_scope": "Project Description and Scope",
    "milestones_schedule": "Project Schedule and Milestones",
    "fees_payment_sow": "Fees and Payment",
    "change_control": "Change Control Procedure",
    "client_responsibilities": "Client Responsibilities",
    "assumptions_dependencies": "Assumptions and Dependencies",
    # DPA-specific display names
    "controller_processor_roles": "Relationship of the Parties (Controller / Processor Roles)",
    "processing_scope": "Scope of Processing",
    "security_measures": "Technical and Organizational Measures",
    "breach_notification": "Personal Data Breach Notification",
    "international_transfers": "International Data Transfers",
    "data_subject_rights": "Data Subject Rights",
    "dpia_cooperation": "Data Protection Impact Assessments",
    "deletion_return": "Deletion or Return of Personal Data",
    "audit_rights": "Audit Rights",
    # Independent Contractor (IC) display names
    "ic_engagement": "Engagement of Contractor",
    "ic_compensation": "Fees, Expenses, and Compensation",
    "ic_classification": "Independent Contractor Relationship",
    "ic_intellectual_property": "Work Product and Intellectual Property",
    "ic_termination": "Term and Termination",
    "ic_restrictive_covenants": "Restrictive Covenants and Conflicts of Interest",
    # LICENSE display names
    "license_grant": "Grant of License",
    "license_royalties_payment": "Royalties and Payments",
    "license_restrictions": "Restrictions on Use and Reservation of Rights",
    "license_term_termination": "Term and Termination of License",
    "license_warranties_indemnification": "Warranties and Infringement Indemnification",
    "license_audit_rights": "Records and Royalty Audit Rights",
    "license_improvements": "Improvements and Grant-Back",
    "license_quality_control": "Quality Control and Brand Standards",
}

# Per-contract-type overrides for the display name. If a (contract_type,
# clause_type) pair has an entry here, it wins over CLAUSE_DISPLAY_NAMES.
CLAUSE_DISPLAY_NAMES_BY_TYPE = {
    "MSA": {
        "intellectual_property": "Work Product and IP Ownership",
    },
}

# Standard provisions per clause type — used to flesh out the brief so the
# model learns which sub-clauses are expected. These are *prompts to the
# model*, not constraints on the output.
CLAUSE_STANDARD_PROVISIONS = {
    "intellectual_property": [
        "Provider ownership of the Platform and all related IP",
        "Customer ownership of Customer Data",
        "Feedback license back to Provider",
        "no implied licenses",
    ],
    "limitation_of_liability": [
        "aggregate cap tied to fees paid in the prior 12 months",
        "exclusion of indirect, consequential, special and punitive damages",
        "exceptions for confidentiality, indemnity and data breaches",
        "basis-of-the-bargain language",
    ],
    "termination": [
        "termination for material breach with cure period",
        "termination for insolvency",
        "effects of termination (cessation of access, data return)",
        "transition assistance",
    ],
    "indemnification": [
        "Provider IP infringement indemnity for Customer",
        "Customer indemnity for misuse and Customer Data claims",
        "indemnification procedure (notice, control, cooperation)",
        "mitigation options if Platform becomes subject to a claim",
    ],
    "confidentiality": [
        "definition of Confidential Information",
        "use restrictions and standard of care",
        "exclusions (publicly known, independently developed, etc.)",
        "compelled disclosure procedure",
        "return or destruction on termination",
    ],
    "data_privacy": [
        "Customer ownership of Customer Data",
        "limited license for Provider to process",
        "security safeguards (encryption, access controls)",
        "breach notification within a defined window",
        "data residency",
        "data return and deletion on termination",
    ],
    "service_level": [
        "uptime commitment with calculation methodology",
        "excluded downtime (scheduled maintenance, force majeure)",
        "service credits as the sole financial remedy",
        "support tiers and response time targets",
    ],
    # MSA-flavored standard provisions
    "services_scope": [
        "scope of services described by reference to one or more SOWs",
        "professional and workmanlike performance standard",
        "Provider responsibility for results of services",
        "Customer cooperation obligations",
    ],
    "deliverables_acceptance": [
        "objective acceptance criteria set in each SOW",
        "Customer review and acceptance period",
        "rejection notice and cure right for non-conforming deliverables",
        "deemed acceptance after a defined window",
    ],
    "personnel": [
        "key personnel commitment with no replacement without consent",
        "background checks and required training",
        "removal of unsuitable personnel on Customer request",
        "non-solicitation of personnel during the engagement",
    ],
    "subcontractors": [
        "Customer prior written consent for subcontractors",
        "flow-down of confidentiality, IP, security obligations",
        "Provider remains primarily responsible for subcontractor performance",
        "subcontractor list maintained and updated",
    ],
    "insurance": [
        "Commercial General Liability minimum coverage",
        "Professional Liability / Errors & Omissions coverage",
        "Cyber Liability coverage where Provider handles Customer data",
        "Workers' Compensation as required by state law",
        "Provider names Customer as additional insured where applicable",
    ],
    # NDA-flavored standard provisions.
    # Note: the brief asks for what the body of THIS clause is expected to
    # demonstrate. The narrow definition of "Purpose" lives in the Definitions
    # article (Article 1) of every NDA template, not in the Purpose article
    # itself, so it is intentionally NOT listed here. (LLM-as-judge audit
    # finding A: that entry was a Defense 2 false positive — the keyword
    # "purpose" matched every body trivially without the body actually
    # *defining* the Purpose.)
    "purpose": [
        "use restriction limited to the Purpose",
        "no use for competitive advantage or unrelated business",
        "no reverse engineering of any tangible Confidential Information",
    ],
    "exclusions": [
        "information already public without breach",
        "information already known to the Receiving Party prior to disclosure",
        "information independently developed without use of Confidential Information",
        "information rightfully received from a third party without duty of confidence",
    ],
    "compelled_disclosure": [
        "prompt written notice to the Disclosing Party before disclosure",
        "cooperation with the Disclosing Party's efforts to obtain a protective order",
        "disclosure limited to the portion legally required",
        "continued confidentiality over non-disclosed portions",
    ],
    "return_or_destruction": [
        "return or destruction of all Confidential Information on the Disclosing Party's request",
        "written certification of destruction on request",
        "exception for archival backups and bona fide records retention",
        "continued confidentiality obligations over any retained copies",
    ],
    "term_survival": [
        "term of the Agreement with end date or ongoing relationship",
        "survival of confidentiality obligations for a defined period after termination",
        "trade secret obligations survive as long as the information remains a trade secret",
        "termination for material breach",
    ],
    "no_license": [
        "no transfer of any ownership interest",
        "no license granted by implication, estoppel, or otherwise",
        "all rights reserved by the Disclosing Party",
        "Disclosing Party retains all patent, copyright, trademark, and trade secret rights",
    ],
    "equitable_remedies": [
        "acknowledgment that breach may cause irreparable harm",
        "right to seek injunctive relief without posting bond",
        "no obligation to prove actual damages",
        "cumulative remedies in addition to any at law or in equity",
    ],
    # EMPLOYMENT-flavored standard provisions.
    # These describe what the BODY of each clause is expected to demonstrate.
    # Following the lessons-learned section: provisions that belong in OTHER
    # articles (e.g. the at-will / fixed-term acknowledgment, which belongs
    # in the Term article) are intentionally not listed here.
    "position_duties": [
        "Executive's title, reporting line, and scope of duties",
        "full-time and exclusive services commitment",
        "limited carve-out for outside boards / charitable / passive investments",
        "principal place of employment with relocation triggers if any",
    ],
    "compensation": [
        "annual base salary with payroll cadence",
        "annual target bonus with discretion / performance metrics",
        "equity participation in the Company's incentive plan",
        "standard employee welfare benefits (medical, dental, retirement, PTO)",
        "expense reimbursement under standard Company policies",
    ],
    "severance": [
        "termination triggers (Cause / without Cause / Good Reason / death / disability)",
        "severance multiple of base salary and / or bonus",
        "continuation of group health benefits (COBRA-equivalent contribution)",
        "general release of claims as condition of severance",
        "no-mitigation clause and offset rules",
    ],
    "restrictive_covenants": [
        "non-competition restriction during employment and a defined post-employment period",
        "non-solicitation of Company employees and customers",
        "confidentiality of proprietary information surviving termination",
        "non-disparagement covenant",
        "tolling of restricted period for any breach by Executive",
    ],
    "change_in_control": [
        "definition of Change in Control event",
        "double-trigger: termination without Cause or for Good Reason within a defined window after CIC",
        "enhanced severance multiple compared to ordinary termination",
        "accelerated vesting of outstanding equity awards",
        "Section 280G best-net cutback or gross-up treatment",
    ],
    "dispute_resolution": [
        "governing law of a designated state",
        "binding arbitration or exclusive court venue for disputes",
        "carve-out for equitable relief in support of restrictive covenants",
        "waiver of jury trial",
        "fee-shifting or each-party-bears-its-own-fees rule",
    ],
    # SOW-flavored standard provisions.
    # An SOW operates UNDER a parent MSA, so its body focuses on PROJECT
    # specifics: what work is being done, by when, for how much, and how
    # changes are handled. Cross-cutting framework topics (limitation of
    # liability, indemnity, governing law, IP ownership defaults) live in the
    # parent MSA and should NOT be re-listed in SOW provision lists.
    "sow_scope": [
        "specific project description with concrete in-scope activities",
        "explicit out-of-scope items that the project will not cover",
        "objectives or success criteria the project is intended to achieve",
        "reference to the parent Master Services Agreement governing this SOW",
    ],
    "milestones_schedule": [
        "project start date and target completion date",
        "named milestones with target completion dates",
        "dependencies between milestones where applicable",
        "consequences of milestone slippage (status reporting, escalation)",
    ],
    "fees_payment_sow": [
        "pricing structure (time-and-materials, fixed-fee, milestone-based, or hybrid)",
        "rate card or fixed-fee amount with currency",
        "expense reimbursement policy with pre-approval threshold",
        "invoicing cadence and payment terms",
        "late payment interest or suspension rights",
    ],
    "change_control": [
        "definition of a Change Request with required content",
        "Provider's obligation to assess impact on scope, schedule, and fees",
        "Customer's written approval required before any change is implemented",
        "no implied or oral changes — only signed Change Orders bind the parties",
        "effect of disputed Change Requests on continuing work",
    ],
    "client_responsibilities": [
        "Customer-provided personnel and decision-makers",
        "timely Customer access to systems, data, and facilities",
        "Customer responsibility for accuracy of provided materials",
        "Customer review and approval turnaround commitment",
        "consequences if Customer delays prevent Provider from meeting milestones",
    ],
    "assumptions_dependencies": [
        "explicit assumptions on which the project pricing and schedule are based",
        "third-party dependencies (vendors, software, regulatory approvals)",
        "change control trigger if any assumption proves incorrect",
        "Customer obligation to notify Provider of any changes to assumptions",
    ],
    # DPA-flavored standard provisions.
    # A DPA operates UNDER a parent SaaS Agreement / MSA / BAA. Its body
    # focuses on the personal data processing relationship between the
    # Controller and the Processor and the obligations imposed by GDPR
    # Article 28 and analogous laws (UK GDPR, CCPA/CPRA, India DPDP Act,
    # LGPD, PIPEDA). Cross-cutting framework topics (limitation of liability,
    # indemnification, governing law, IP ownership defaults) live in the
    # parent agreement and should NOT be re-listed in DPA provision lists.
    "controller_processor_roles": [
        "designation of Controller and Processor (or sub-processor) for each processing activity",
        "Controller's instructions as the basis for the Processor's processing",
        "Processor's obligation to process Personal Data only on documented Controller instructions",
        "Processor's notification obligation if any instruction would violate applicable data protection law",
    ],
    "processing_scope": [
        "subject matter and duration of the processing",
        "nature and purpose of the processing",
        "types of Personal Data processed",
        "categories of Data Subjects",
        "Controller's lawful basis representation",
    ],
    "security_measures": [
        "implementation of appropriate technical and organizational measures (TOMs)",
        "encryption of Personal Data in transit and at rest where appropriate",
        "ongoing confidentiality, integrity, availability, and resilience of processing systems",
        "ability to restore access to Personal Data after a physical or technical incident",
        "regular testing and evaluation of the effectiveness of the TOMs",
        "personnel confidentiality and need-to-know access controls",
    ],
    "breach_notification": [
        "Processor's obligation to notify Controller without undue delay after becoming aware of a Personal Data Breach",
        "specific notification timeline (e.g., within 24, 48, or 72 hours)",
        "minimum content of the breach notification (nature, categories, approximate numbers, contact point, likely consequences, measures taken)",
        "Processor's cooperation with Controller's investigation and remediation",
        "Processor's record-keeping of all Personal Data Breaches",
    ],
    "international_transfers": [
        "permitted transfer mechanisms (adequacy decisions, Standard Contractual Clauses, UK IDTA, supplementary measures)",
        "incorporation of the EU SCCs (Commission Decision 2021/914) where applicable",
        "incorporation of the UK International Data Transfer Addendum where applicable",
        "Processor's obligation to conduct a Transfer Impact Assessment for restricted transfers",
        "Controller's right to require additional safeguards or to suspend transfers",
    ],
    "data_subject_rights": [
        "Processor's obligation to assist Controller in responding to Data Subject requests",
        "covered Data Subject rights (access, rectification, erasure, restriction, portability, objection)",
        "Processor's obligation to forward Data Subject requests received directly to Controller without delay",
        "limitation that Processor will not respond directly to Data Subjects without Controller authorization",
    ],
    "dpia_cooperation": [
        "Processor's obligation to assist Controller with Data Protection Impact Assessments under Article 35 GDPR",
        "Processor's obligation to assist with prior consultations with supervisory authorities under Article 36 GDPR",
        "scope of the assistance (information, documentation, attendance at meetings)",
        "Controller's responsibility for the DPIA itself; Processor's role is supportive",
    ],
    "deletion_return": [
        "Processor's obligation, at Controller's choice, to delete or return all Personal Data at the end of provision of the services",
        "deletion of all existing copies unless retention is required by applicable law",
        "Processor's certification of deletion on Controller's request",
        "exception for backup or archival copies retained pursuant to a documented retention schedule",
        "continued application of the DPA's confidentiality and security obligations to any retained copies",
    ],
    "audit_rights": [
        "Controller's right to audit the Processor's compliance with the DPA",
        "audit notice period and audit frequency limitations",
        "Processor's right to provide third-party audit reports (SOC 2 Type II, ISO 27001) in lieu of on-site audits",
        "scope limitations to protect other customers' confidential information and the Processor's trade secrets",
        "Controller's responsibility for the cost of audits beyond Processor's standard third-party reports",
    ],
    # Independent Contractor (IC) standard provisions.
    # IC agreements are standalone contracts between a Company and an
    # individual (or sometimes a single-person LLC) engaged on a 1099 basis.
    # The body should demonstrate the worker-classification anchor that
    # distinguishes IC from employment, plus the project / engagement
    # specifics.
    "ic_engagement": [
        "scope of services to be performed by the Contractor",
        "term of the engagement (fixed-term, project-based, or open-ended)",
        "Contractor's commitment to perform services in a professional and workmanlike manner",
        "no subcontracting or delegation without Company consent",
    ],
    "ic_compensation": [
        "fee structure (hourly rate, fixed fee, milestone payments, or monthly retainer)",
        "invoicing cadence and payment terms",
        "expense reimbursement policy with pre-approval threshold",
        "no benefits, no withholding — Contractor is responsible for own taxes",
    ],
    "ic_classification": [
        "explicit independent contractor relationship — no employer-employee, partnership, joint venture, or agency",
        "Contractor's responsibility for self-employment, Social Security, Medicare, and income taxes",
        "no Company withholding from Contractor's compensation",
        "Contractor not entitled to employee benefits (medical, retirement, vacation, workers compensation, unemployment insurance)",
        "Contractor's obligation to provide Form W-9 and Company's reporting on Form 1099-NEC",
    ],
    "ic_intellectual_property": [
        "Contractor's assignment to the Company of all Work Product created during the engagement",
        "works made for hire to the maximum extent permitted by U.S. copyright law, with explicit assignment as backup",
        "Contractor's pre-existing IP retained by the Contractor with limited license to Company as needed for the deliverables",
        "moral rights waiver to the extent permitted by applicable law",
        "Contractor's representations of original authorship and non-infringement",
    ],
    "ic_termination": [
        "termination for convenience by either party on stated notice",
        "termination by the Company for cause (material breach, misconduct, regulatory disqualification)",
        "Contractor's deliverables and final invoice obligations on termination",
        "survival of confidentiality, IP assignment, restrictive covenants, and indemnification",
    ],
    "ic_restrictive_covenants": [
        "Contractor free to perform services for other clients, except direct competitors during the engagement",
        "non-solicitation of Company employees and contractors during and after the engagement (for a defined post-engagement period)",
        "no use of Company Confidential Information in providing services to third parties",
        "Contractor's obligation to disclose conflicts of interest with Company customers, vendors, or competitors",
    ],
    # LICENSE standard provisions.
    # License agreements grant rights in IP from a Licensor to a Licensee.
    # The clause body should describe what the Licensor grants and the
    # corresponding economic, restrictive, and operational terms.
    "license_grant": [
        "identification of the Licensed IP (patent, trademark, copyright, software, trade secret, or know-how)",
        "exclusivity (exclusive, sole, or non-exclusive)",
        "field of use limitations",
        "territory in which the Licensee may exercise the rights",
        "term of the license",
        "Licensee's right (or prohibition) to grant sublicenses",
        "right to make, use, sell, import, or perform the Licensed IP",
    ],
    "license_royalties_payment": [
        "upfront license fee or signing payment",
        "running royalty rate (percentage of Net Sales or per-unit)",
        "minimum annual royalty or guaranteed payment",
        "milestone payments tied to development, regulatory, or commercial events",
        "definition of Net Sales (deductions, returns, taxes)",
        "royalty reporting cadence and royalty report contents",
        "currency, payment timing, and late payment interest",
    ],
    "license_restrictions": [
        "no reverse engineering, decompilation, or disassembly (for software)",
        "no transfer, sublicense, or assignment outside the granted scope",
        "no use outside the licensed field of use or territory",
        "no creation of derivative works without Licensor consent (where applicable)",
        "no removal of proprietary notices or markings",
        "Licensor's reservation of all rights not expressly granted (no implied licenses)",
    ],
    "license_term_termination": [
        "initial term and renewal options",
        "termination for material breach with cure period",
        "termination for insolvency or bankruptcy",
        "Licensor's termination right for failure to meet milestones or minimum royalties",
        "post-termination sell-off or wind-down period for inventory",
        "survival of payment obligations, confidentiality, and indemnification",
    ],
    "license_warranties_indemnification": [
        "Licensor's warranty of title and right to grant the license",
        "Licensor's warranty (or disclaimer) of non-infringement of third-party IP",
        "Licensor's indemnification of Licensee against third-party infringement claims arising out of the Licensed IP",
        "indemnification procedure (notice, control of defense, cooperation)",
        "Licensee's indemnification of Licensor for use outside the licensed scope",
        "limitation of liability and exclusion of consequential damages",
    ],
    "license_audit_rights": [
        "Licensee's obligation to maintain complete books and records of Net Sales and royalty calculations",
        "Licensor's right to audit Licensee's books on reasonable notice",
        "audit frequency limitation (typically once per year)",
        "use of an independent certified public accountant",
        "Licensee's payment of audit costs if the underpayment exceeds a stated threshold (e.g. 5%)",
        "Licensee's payment of any underreported royalties plus interest",
    ],
    "license_improvements": [
        "definition of Improvements to the Licensed IP",
        "ownership of Improvements made by Licensor (typically retained by Licensor and licensed to Licensee on the same terms)",
        "ownership of Improvements made by Licensee (varies — sole, joint, or grant-back to Licensor)",
        "grant-back license from Licensee to Licensor for Licensee's Improvements",
        "Licensee's obligation to disclose Improvements to Licensor",
    ],
    "license_quality_control": [
        "Licensor's quality standards for Licensee's use of the Licensed IP (especially trademarks)",
        "Licensee's obligation to submit samples to Licensor for approval prior to use",
        "Licensor's right to inspect Licensee's facilities and operations",
        "Licensor's right to require corrective action for non-conforming use",
        "naked-license avoidance language (Licensor exercises control to maintain trademark validity)",
    ],
}

# Per-contract-type overrides for the standard provisions list. The IP
# clause in particular is *very* different between SaaS (license-grant) and
# MSA (work-product assignment) — without the override, the brief would
# instruct the model to write SaaS-style IP language inside an MSA clause,
# which is the wrong training signal.
CLAUSE_STANDARD_PROVISIONS_BY_TYPE = {
    "MSA": {
        "intellectual_property": [
            "Provider's pre-existing IP and tools retained by Provider",
            "Work Product assigned to Customer (work made for hire to the maximum extent permitted, with express assignment as backup)",
            "license-back to Provider for any pre-existing IP embedded in Work Product",
            "Customer's pre-existing IP and Customer Materials retained by Customer",
            "no implied licenses",
        ],
        "termination": [
            "termination for material breach with cure period",
            "termination for insolvency",
            "Customer right to terminate for convenience on notice",
            "delivery and assignment of Work Product through the termination date",
            "wind-down assistance for in-progress SOWs",
        ],
    },
    "NDA": {
        # The NDA confidentiality clause body demonstrates the receiving
        # party's USE / DISCLOSURE / CARE / NOTIFICATION obligations. The
        # *definition* of Confidential Information lives in the Definitions
        # article, and IP-no-transfer lives in the No License article — so
        # neither belongs in this clause's brief. (LLM-as-judge audit
        # finding A: keyword filter false positives.)
        "confidentiality": [
            "use restriction — Confidential Information used solely for the defined Purpose",
            "disclosure restriction — limited to personnel with a need to know who are bound by equivalent confidentiality",
            "standard of care — at least the care the Receiving Party uses for its own confidential information, but no less than reasonable care",
            "obligation to notify the Disclosing Party of any unauthorized disclosure",
        ],
        "termination": [
            "term of the Agreement",
            "survival of confidentiality obligations for a defined period (typically 3-5 years) after the end of the term",
            "trade secret carve-out — trade secret obligations survive as long as the information remains a trade secret",
            "return or destruction of Confidential Information on termination",
        ],
        "intellectual_property": [
            "no transfer of any intellectual property rights to the Receiving Party",
            "no license granted by implication, estoppel, or otherwise",
            "all patent, copyright, trademark, and trade secret rights reserved by the Disclosing Party",
            "Feedback carve-out, if applicable, for suggestions provided during evaluation",
        ],
    },
    "INDEPENDENT_CONTRACTOR": {
        # IC-specific overrides for clauses reused from other categories.
        # Confidentiality in IC is mostly the standard NDA-style obligations,
        # but the IC body uses individual-Contractor framing rather than
        # employee or processor framing.
        "confidentiality": [
            "definition of Company Confidential Information",
            "Contractor's obligation to hold Confidential Information in strict confidence during and after the engagement",
            "no use of Confidential Information for any purpose other than performing the services",
            "return or destruction of all Company materials on termination",
            "injunctive relief for breach (no proof of actual damages required)",
        ],
        "indemnification": [
            "Contractor's indemnification of Company for any claim that Contractor is or should be classified as an employee",
            "Contractor's indemnification for any tax liability resulting from Contractor's failure to pay self-employment, Social Security, or income taxes",
            "Contractor's indemnification for breach of representations and warranties (including IP non-infringement)",
            "Contractor's professional liability / errors and omissions coverage where required",
        ],
    },
    "LICENSE": {
        # LICENSE-specific override for confidentiality. License agreements
        # use Licensor/Licensee framing and the confidential information
        # typically includes the Licensed IP itself (especially trade secrets,
        # know-how, source code, formulas) plus financial information shared
        # via royalty reports.
        "confidentiality": [
            "definition of Confidential Information including the Licensed IP itself, royalty reports, and financial information exchanged",
            "Licensee's and Licensor's mutual obligation to hold Confidential Information in strict confidence",
            "no use of Confidential Information for any purpose other than exercising rights or fulfilling obligations under the License",
            "return or destruction of Confidential Information on termination, except as needed to exercise surviving rights",
            "trade secret carve-out — trade secret protection survives as long as the information remains a trade secret",
        ],
    },
    "DPA": {
        # DPA-specific override for subprocessors. The generic version
        # focuses on services subcontractors; in a DPA the sub-processor
        # clause has very specific GDPR Article 28(2) and 28(4) obligations.
        "subprocessors": [
            "Controller's general or specific authorization for sub-processors",
            "Processor's obligation to maintain a current list of sub-processors and notify Controller of changes",
            "Controller's right to object to new sub-processors within a defined window",
            "Processor's obligation to impose data protection obligations on sub-processors equivalent to those in the DPA",
            "Processor's primary liability for the acts and omissions of its sub-processors",
        ],
    },
    "SOW": {
        # SOW-specific overrides for clause types that already exist in
        # CLAUSE_STANDARD_PROVISIONS but need SOW-flavored language. The
        # generic deliverables_acceptance provisions reference "each SOW" —
        # within an SOW that's circular, so we need project-specific language.
        "deliverables_acceptance": [
            "list of deliverables with descriptions and target dates",
            "objective acceptance criteria for each deliverable",
            "Customer review and acceptance period (typically 5-10 business days)",
            "rejection notice and Provider cure right for non-conforming deliverables",
            "deemed acceptance after the review window expires without rejection",
        ],
        "personnel": [
            "named key personnel assigned to this project",
            "minimum allocation percentage of each key person to this project",
            "no replacement of key personnel without Customer prior written consent",
            "background checks and Customer security training where required",
            "non-solicitation of project personnel during the engagement",
        ],
    },
    "EMPLOYMENT": {
        # Employment-specific overrides for clause types that already exist
        # in CLAUSE_STANDARD_PROVISIONS but need different language for the
        # employer/employee context.
        "termination": [
            "termination upon Executive's death or disability",
            "termination by the Company for Cause (with definition cross-reference)",
            "termination by the Company without Cause",
            "termination by Executive for Good Reason (with definition cross-reference)",
            "termination by Executive without Good Reason",
            "notice and effective-date mechanics for each trigger",
        ],
        "intellectual_property": [
            "Executive's assignment of all Work Product to the Company",
            "works made for hire to the maximum extent permitted by law",
            "Executive's pre-existing inventions excluded from assignment",
            "moral rights waiver to the extent permitted",
            "state-law carve-outs for inventions on Executive's own time using no Company resources (e.g. Cal. Lab. Code § 2870)",
        ],
        "confidentiality": [
            "confidential information of the Company defined to include trade secrets, customer data, financials, and personnel information",
            "Executive's duty to hold confidential information in strict confidence during and after employment",
            "no use of confidential information for any purpose other than performance of Executive's duties",
            "Defend Trade Secrets Act whistleblower immunity notice in the required statutory form",
            "return of all confidential information and Company property on termination",
        ],
    },
}

# Different brief opening phrases — keeps the training data from looking
# templated and matches the kind of variation real users produce.
BRIEF_OPENERS = [
    "Draft the {clause_display} clause for {article} {agreement_label}",
    "Write the {clause_display} section for {article} {agreement_label}",
    "Generate the {clause_display} provision for {article} {agreement_label}",
    "I need the {clause_display} clause for {article} {agreement_label}",
    "Prepare the {clause_display} article for {article} {agreement_label}",
]

# Drafting style → human-readable label that goes into the brief.
# The model should learn that when the brief asks for a particular style,
# the output's tone, sentence length and use of Latin should match.
DRAFTING_STYLE_LABELS = {
    "biglaw_formal": (
        "BigLaw formal — long sentences, defined-term-heavy, "
        "WHEREAS recitals, formal third-person voice"
    ),
    "plain_english_common_paper": (
        "Plain English / Common Paper — short sentences, \"we\" and \"you\", "
        "minimal Latin, defined terms used inline, no WHEREAS recitals"
    ),
    "click_through_tos": (
        "Click-through Terms of Service — terse numbered lists, "
        "no preamble, no signature blocks, accept-by-clicking style"
    ),
    "modular_order_form": (
        "Modular MSA + Order Form — bare master agreement with all "
        "commercials pushed into a separate Order Form"
    ),
    "customer_paper_negotiated": (
        "Customer paper / heavily negotiated — visible customer-side "
        "markups, audit rights, sub-processor lists, regulator pass-through"
    ),
    "modern_tech_minimalist": (
        "Modern tech minimalist — Stripe/Notion-style plain language, "
        "short sections, friendly tone, no Latin"
    ),
}


AGREEMENT_LABEL_BY_TYPE = {
    "SAAS": "Software-as-a-Service Subscription Agreement",
    "MSA": "Master Services Agreement",
    "NDA": "Mutual Non-Disclosure Agreement",
    "EMPLOYMENT": "Employment Agreement",
    "VENDOR_SUPPLY": "Vendor / Supply Agreement",
    "RESELLER": "Reseller Agreement",
    "DPA": "Data Processing Addendum",
    "SOW": "Statement of Work",
    "SOFTWARE_LICENSE": "Software License Agreement",
    "INDEPENDENT_CONTRACTOR": "Independent Contractor Agreement",
    "LICENSE": "License Agreement",
}


# ── Clause extraction ────────────────────────────────────────────────────

ARTICLE_HEADER_RE = re.compile(
    r"^ARTICLE\s+(\d+)\s*[—\-–]\s*(.+?)\s*$",
    re.MULTILINE,
)


def extract_clauses(text: str) -> dict[str, dict]:
    """Split a tokenized contract into ARTICLE-bounded sections, then map each
    section to one of the clause types in CLAUSE_TITLE_KEYWORDS.

    Returns a dict keyed by clause_type, with each value being:
        {"title": "<original article title>", "body": "<full article text>"}
    """
    headers = []
    for m in ARTICLE_HEADER_RE.finditer(text):
        headers.append({
            "number": int(m.group(1)),
            "title": m.group(2).strip(),
            "title_upper": m.group(2).strip().upper(),
            "start": m.start(),
        })

    if not headers:
        return {}

    # Determine the body of each article: from start of header to start of next
    sections = []
    for i, h in enumerate(headers):
        end = headers[i + 1]["start"] if i + 1 < len(headers) else len(text)
        body = text[h["start"]:end].rstrip()
        sections.append({**h, "body": body})

    # Map each section to a clause type, using title keyword matching.
    # Use the most specific match first by sorting keywords by length descending.
    clauses = {}
    for clause_type, keywords in CLAUSE_TITLE_KEYWORDS.items():
        for sec in sections:
            for kw in keywords:
                if kw in sec["title_upper"]:
                    # Prefer a more specific match if we already have one
                    existing = clauses.get(clause_type)
                    if existing is None or len(kw) > existing.get("_kw_len", 0):
                        clauses[clause_type] = {
                            "title": sec["title"],
                            "body": sec["body"],
                            "article_number": sec["number"],
                            "_kw_len": len(kw),
                        }
                    break

    # Drop the bookkeeping field
    for v in clauses.values():
        v.pop("_kw_len", None)

    return clauses


# ── Brief construction ──────────────────────────────────────────────────

def _money_short(amount_str: str) -> str:
    """Pull the parenthesized numeric form out of a fee string for the brief.

    e.g. 'Two Hundred Forty Thousand United States Dollars ($240,000)' → '$240,000'
    """
    m = re.search(r"\(([^)]+)\)", amount_str or "")
    if m:
        return m.group(1)
    return amount_str or ""


def _format_term(term_years: str) -> str:
    """'three (3)' → '3 years'."""
    m = re.search(r"\((\d+)\)", term_years or "")
    if m:
        n = int(m.group(1))
        return f"{n} year" if n == 1 else f"{n} years"
    return term_years or ""


def _format_months(months_str: str) -> str:
    """'twelve (12)' → '12 months'."""
    m = re.search(r"\((\d+)\)", months_str or "")
    if m:
        n = int(m.group(1))
        return f"{n} month" if n == 1 else f"{n} months"
    return months_str or ""


def _format_days(days_str: str) -> str:
    """'thirty (30)' → '30 days'."""
    m = re.search(r"\((\d+)\)", days_str or "")
    if m:
        return f"{m.group(1)} days"
    return days_str or ""


def _bias_label(bias: str) -> str:
    return {
        "provider_friendly": "provider-friendly",
        "balanced": "balanced",
        "customer_friendly": "customer-friendly",
    }.get(bias, "balanced")


def _a_or_an(word: str) -> str:
    """Return 'a' or 'an' based on the first sound of the next word.

    Plain heuristic: vowel letter → 'an', except for 'u' words that start
    with a 'yoo' sound (United, University, Utah) and silent-h words like
    'honor'. We list the common exceptions explicitly.
    """
    if not word:
        return "a"
    first = word.strip()[0].lower()
    lower = word.strip().lower()
    # 'yoo' sound exceptions — these take 'a' even though they start with U
    if lower.startswith(("united", "univers", "utah", "uniform", "unique", "user")):
        return "a"
    # "one-" takes "a" (pronounced "wun-")
    if lower.startswith(("one",)):
        return "a"
    # Silent-h exceptions — these take 'an' even though they start with H
    if lower.startswith(("honor", "honest", "hour", "heir")):
        return "an"
    return "an" if first in "aeiou" else "a"


def build_brief(
    *,
    contract_type: str,
    clause_type: str,
    sub_map: dict,
    metadata: dict,
    style_seed: int,
    clause_body: str = "",
) -> str:
    """Construct a realistic brief that mentions parties, industry, jurisdiction,
    deal terms and the relevant standard provisions for the clause.

    The brief embeds the SAME placeholder values that will be substituted into
    the output clause, so the model learns to honor the slots in the brief.

    If clause_body is provided, the standard-provisions list is filtered down
    to only those provisions whose keywords actually appear in the body — this
    guarantees (brief, output) coverage alignment so the model isn't trained
    to drop items from the brief.
    """
    rng = random.Random(style_seed)

    # Per-contract agreement_label override (metadata wins) — lets NDAs
    # distinguish mutual / one-way / employee flavors instead of all being
    # labeled "Mutual Non-Disclosure Agreement".
    agreement_label = (
        metadata.get("agreement_label_override")
        or AGREEMENT_LABEL_BY_TYPE.get(contract_type, "commercial agreement")
    )
    type_display_overrides = CLAUSE_DISPLAY_NAMES_BY_TYPE.get(contract_type, {})
    clause_display = type_display_overrides.get(clause_type) or CLAUSE_DISPLAY_NAMES[clause_type]

    opener_template = rng.choice(BRIEF_OPENERS)
    opener_article = _a_or_an(agreement_label)
    opener = opener_template.format(
        clause_display=clause_display,
        agreement_label=agreement_label,
        article=opener_article,
    )

    # Party block — NDAs use Disclosing/Receiving Party labels for one-way
    # and the employee flavor; mutual NDAs and SaaS/MSA use Provider/Customer.
    provider_name = sub_map["PROVIDER_NAME"]
    provider_state = sub_map["PROVIDER_STATE"]
    provider_entity = sub_map["PROVIDER_ENTITY_TYPE"]
    customer_name = sub_map["CUSTOMER_NAME"]
    customer_state = sub_map.get("CUSTOMER_STATE", sub_map.get("GOVERNING_LAW_STATE", ""))
    customer_entity = sub_map.get("CUSTOMER_ENTITY_TYPE", "")

    provider_article = _a_or_an(provider_state)
    customer_article = _a_or_an(customer_state) if customer_state else _a_or_an(customer_entity)

    # Party role labels — per-contract override wins
    party_roles = metadata.get("party_roles") or {}
    provider_role = party_roles.get("provider", "Provider")
    customer_role = party_roles.get("customer", "Customer")

    party_block = (
        f"between {provider_name} ({provider_article} {provider_state} {provider_entity}, the \"{provider_role}\") "
        f"and {customer_name} ({customer_article} {customer_state} {customer_entity}, the \"{customer_role}\")"
    )

    # Commercial terms block — skipped entirely for contract types where
    # commercial terms don't apply (NDAs, DPAs). EMPLOYMENT has its own
    # block describing base salary, target bonus, and term, since employment
    # agreements have a totally different commercial shape than SaaS/MSA.
    governing_law = sub_map.get("GOVERNING_LAW_STATE", "")
    venue = sub_map.get("VENUE_COUNTY", "")

    commercial_block = ""
    if contract_type == "EMPLOYMENT":
        # Employment agreements use the EMPLOYEE_BASE_SALARY,
        # EMPLOYEE_TARGET_BONUS, and EMPLOYEE_EQUITY_VALUE placeholders rather
        # than ANNUAL_FEE_AMOUNT / LIABILITY_CAP_MONTHS. TERM_YEARS represents
        # an explicit fixed term where applicable; for at-will agreements set
        # `suppress_initial_term: true` in the metadata to omit the term line
        # (per Lesson 4 of the rules doc — placeholder semantics must be
        # consistent within a category, and at-will employment doesn't have a
        # numeric term).
        salary = _money_short(sub_map.get("EMPLOYEE_BASE_SALARY", ""))
        bonus_pct = sub_map.get("EMPLOYEE_TARGET_BONUS_PCT", "")
        equity_value = _money_short(sub_map.get("EMPLOYEE_INITIAL_EQUITY", ""))
        term = _format_term(sub_map.get("TERM_YEARS", ""))

        comp_bits = []
        if salary:
            comp_bits.append(f"annual base salary of {salary}")
        if bonus_pct:
            comp_bits.append(f"annual target bonus of {bonus_pct}% of base salary")
        if equity_value:
            comp_bits.append(f"initial equity grant valued at {equity_value}")
        if term and not metadata.get("suppress_initial_term"):
            comp_bits.append(f"initial term of {term}")

        if comp_bits:
            commercial_block = "Key compensation terms: " + "; ".join(comp_bits) + "."
    elif contract_type == "INDEPENDENT_CONTRACTOR":
        # IC agreements have a fee structure (hourly / fixed-fee / monthly
        # retainer / milestone / hybrid), an engagement period, and a
        # notice period for termination for convenience. They are
        # standalone (no parent agreement to reference).
        fee_structure = sub_map.get("IC_FEE_STRUCTURE", "")
        hourly = _money_short(sub_map.get("IC_HOURLY_RATE", ""))
        fixed = _money_short(sub_map.get("IC_FIXED_FEE", ""))
        retainer = _money_short(sub_map.get("IC_MONTHLY_RETAINER", ""))
        term_months = sub_map.get("IC_TERM_MONTHS", "")
        notice = sub_map.get("IC_NOTICE_DAYS_CONVENIENCE", "")

        ic_bits = []
        if fee_structure:
            ic_bits.append(f"fee structure: {fee_structure}")
        if hourly:
            ic_bits.append(f"hourly rate {hourly}")
        if retainer:
            ic_bits.append(f"monthly retainer {retainer}")
        if fixed:
            ic_bits.append(f"fixed fee {fixed}")
        if term_months:
            ic_bits.append(f"engagement term {term_months} months")
        if notice:
            ic_bits.append(f"termination-for-convenience notice {notice} days")

        if ic_bits:
            commercial_block = "Key engagement terms: " + "; ".join(ic_bits) + "."
    elif contract_type == "LICENSE":
        # License agreements have a unique commercial shape: grant type,
        # field of use, territory, term, upfront fee, running royalty, and
        # minimum annual royalty. The commercial block surfaces the deal
        # economics so the model learns to honor LICENSE-specific slots.
        grant_type = sub_map.get("LICENSE_GRANT_TYPE", "")
        field_of_use = sub_map.get("LICENSE_FIELD_OF_USE", "")
        territory = sub_map.get("LICENSE_TERRITORY", "")
        license_term = sub_map.get("LICENSE_TERM", "")
        upfront = _money_short(sub_map.get("LICENSE_UPFRONT_FEE", ""))
        royalty_rate = sub_map.get("LICENSE_ROYALTY_RATE", "")
        min_royalty = _money_short(sub_map.get("LICENSE_MIN_ANNUAL_ROYALTY", ""))
        sublicense = sub_map.get("LICENSE_SUBLICENSE_RIGHTS", "")

        lic_bits = []
        if grant_type:
            lic_bits.append(f"grant: {grant_type}")
        if field_of_use:
            lic_bits.append(f"field of use: {field_of_use}")
        if territory:
            lic_bits.append(f"territory: {territory}")
        if license_term:
            lic_bits.append(f"license term: {license_term}")
        if upfront:
            lic_bits.append(f"upfront fee {upfront}")
        if royalty_rate:
            lic_bits.append(f"running royalty {royalty_rate}")
        if min_royalty:
            lic_bits.append(f"minimum annual royalty {min_royalty}")
        if sublicense:
            lic_bits.append(f"sublicensing: {sublicense}")

        if lic_bits:
            commercial_block = "Key license terms: " + "; ".join(lic_bits) + "."
    elif contract_type == "DPA":
        # DPAs don't have commercial fees — those are in the parent SaaS
        # Agreement / MSA / BAA. The DPA brief should mention the parent
        # agreement reference, the data residency region, and the breach
        # notification timeline (the most distinctive DPA commercial term).
        parent_name = sub_map.get("PARENT_AGREEMENT_NAME", "")
        parent_date = sub_map.get("PARENT_AGREEMENT_DATE", "")
        residency = sub_map.get("DATA_RESIDENCY_REGION", "")
        breach_hours = sub_map.get("BREACH_NOTIFICATION_HOURS", "")

        dpa_bits = []
        if parent_name and parent_date:
            dpa_bits.append(f"issued under the parent {parent_name} dated {parent_date}")
        elif parent_date:
            dpa_bits.append(f"issued under the parent agreement dated {parent_date}")
        if residency:
            dpa_bits.append(f"data residency: {residency}")
        if breach_hours:
            dpa_bits.append(f"personal data breach notification within {breach_hours} hours")

        if dpa_bits:
            commercial_block = "Key DPA terms: " + "; ".join(dpa_bits) + "."
    elif contract_type == "SOW":
        # SOWs have a totally different commercial frame than SaaS/MSA. They
        # have a Project Fee (fixed-fee or T&M not-to-exceed) for THIS specific
        # project, plus a project window (start/end dates) — not an annual
        # subscription fee, not a multi-year term, not a liability cap (which
        # lives in the parent MSA). The brief should reflect the SOW's actual
        # commercial shape.
        sow_fee = _money_short(sub_map.get("SOW_FEE_AMOUNT", ""))
        project_name = sub_map.get("PROJECT_NAME", "")
        start_date = sub_map.get("PROJECT_START_DATE", "")
        end_date = sub_map.get("PROJECT_END_DATE", "")
        msa_date = sub_map.get("MSA_EFFECTIVE_DATE", "")

        commercial_bits = []
        if project_name:
            commercial_bits.append(f"project name \"{project_name}\"")
        if sow_fee:
            commercial_bits.append(f"fixed Project Fee of {sow_fee}")
        if start_date and end_date:
            commercial_bits.append(f"project window from {start_date} through {end_date}")
        elif start_date:
            commercial_bits.append(f"project start date {start_date}")
        if msa_date:
            commercial_bits.append(f"issued under the parent MSA dated {msa_date}")

        if commercial_bits:
            commercial_block = "Key project terms: " + "; ".join(commercial_bits) + "."
    elif contract_type not in ("NDA",):  # INDEPENDENT_CONTRACTOR and LICENSE handled above
        annual_fee = _money_short(sub_map.get("ANNUAL_FEE_AMOUNT", ""))
        term = _format_term(sub_map.get("TERM_YEARS", ""))
        cap_months = _format_months(sub_map.get("LIABILITY_CAP_MONTHS", ""))
        notice = _format_days(sub_map.get("NOTICE_DAYS", ""))

        # The label for the headline fee depends on the contract type. SaaS
        # deals talk about "annual subscription fee"; services agreements talk
        # about "annual aggregate fees across all Statements of Work"; etc.
        fee_label = {
            "SAAS": "annual subscription fee",
            "MSA": "annual aggregate fees across all Statements of Work",
        }.get(contract_type, "annual fee")

        commercial_bits = []
        if annual_fee:
            commercial_bits.append(f"{fee_label} of {annual_fee}")
        if term:
            commercial_bits.append(f"initial term of {term}")
        if cap_months:
            commercial_bits.append(f"liability cap equal to {cap_months} of fees")
        if notice:
            commercial_bits.append(f"notice period of {notice}")

        if commercial_bits:
            commercial_block = "Key commercial terms: " + "; ".join(commercial_bits) + "."
    elif contract_type == "NDA":
        # In MOST NDA templates, TERM_YEARS is the agreement's initial term
        # and the survival period is hardcoded inside the body. The exception
        # is the employee NDA, where the agreement runs "throughout
        # employment" (no specific number) and TERM_YEARS is reused as the
        # post-employment survival period. Templates that fall in the latter
        # camp set "suppress_initial_term": true in their metadata so the
        # brief doesn't mislabel TERM_YEARS as an initial term.
        # (LLM-as-judge audit finding C.)
        if not metadata.get("suppress_initial_term"):
            term = _format_term(sub_map.get("TERM_YEARS", ""))
            if term:
                commercial_block = f"Initial term: {term}."

    # Jurisdiction block
    jurisdiction_block = ""
    if governing_law:
        if venue:
            jurisdiction_block = f"Governing law: {governing_law}, with venue in {venue}."
        else:
            jurisdiction_block = f"Governing law: {governing_law}."

    # Industry / regulatory / bias / style context
    industry_label = metadata.get("industry_label") or metadata.get("industry", "")
    deal_size = metadata.get("deal_size", "")
    bias = _bias_label(metadata.get("negotiation_bias", "balanced"))
    regs = metadata.get("regulatory_frameworks", []) or []
    special = metadata.get("special_features", []) or []
    drafting_style = metadata.get("drafting_style")
    style_label = DRAFTING_STYLE_LABELS.get(drafting_style) if drafting_style else None

    context_bits = []
    if industry_label:
        context_bits.append(f"Industry: {industry_label}")
    if deal_size:
        context_bits.append(f"Deal size: {deal_size.replace('_', '-')}")
    if bias:
        context_bits.append(f"Negotiation posture: {bias}")
    context_block = ". ".join(context_bits) + "." if context_bits else ""

    reg_block = ""
    if regs:
        reg_block = "Regulatory frameworks to address: " + ", ".join(regs) + "."

    # Pick 1-2 special features that are most relevant for this clause type.
    # If none of the contract's special features are relevant to this clause,
    # OMIT the block entirely. (LLM-as-judge audit finding E: the previous
    # behavior fell back to a random pick, which leaked contract-level
    # metadata into clause-level briefs and trained the model to expect
    # provisions in clauses where they didn't belong — e.g. asking the No
    # License clause to "address" the DTSA whistleblower notice that actually
    # belongs in the Confidentiality clause.)
    special_block = ""
    if special:
        relevant_kw = {
            # SaaS / MSA clause types
            "intellectual_property": ["source code", "ip", "license", "intellectual",
                                       "work product", "assignment", "moral rights"],
            "limitation_of_liability": ["liability", "cap", "limitation", "remedies",
                                         "uncapped", "carve-out"],
            "termination": ["exit", "transition", "termination", "wind-down",
                            "stressed exit", "regulator-directed"],
            "indemnification": ["indemn", "sovereign", "regulator pass-through",
                                 "infringement"],
            "data_privacy": ["data", "privacy", "ferpa", "hipaa", "pipeda", "pci",
                              "encryption", "residency", "breach", "phi", "byo-key",
                              "sub-processor"],
            "service_level": ["sla", "uptime", "incident", "downtime", "support",
                               "academic", "service credit"],
            # confidentiality keywords describe what belongs in the
            # Confidentiality clause body — data room handling, clean team
            # protocols, privilege preservation, regulator pass-through, etc.
            # Intentionally omitted to avoid false matches against
            # special-feature TEXT (which often mentions "Confidential
            # Information" or "trade secret" while describing a feature
            # that belongs in a different clause):
            #   - "confidential" (matches almost every NDA special feature)
            #   - "trade secret" (belongs in term_survival / no_license)
            #   - "no residuals" (belongs in purpose — governs USE)
            "confidentiality": ["data room", "watermark", "clean team",
                                 "regulator pass-through", "pipeda", "hipaa",
                                 "whistleblower", "upjohn", "attorney-client",
                                 "work-product", "privilege", "ai training",
                                 "incident report", "breach notification"],
            # NDA-specific clause types
            "purpose": ["no-shop", "no shopping", "competitive", "competing",
                         "standstill", "reg fd", "hsr", "gun-jumping",
                         "procurement integrity", "no residuals", "no use",
                         "no reverse"],
            "exclusions": ["public", "independently", "third party", "rightful",
                            "carve-out", "exemption"],
            "compelled_disclosure": ["regulator", "regulator pass-through", "reg fd",
                                       "foia", "subpoena", "outside counsel",
                                       "privilege", "anti-tipping", "8-k", "form 8-k",
                                       "fcpa", "self-disclosure"],
            "return_or_destruction": ["destruction", "sworn", "certificate",
                                        "watermark", "data room", "litigation hold",
                                        "archival", "backup"],
            "term_survival": ["survival", "trade secret", "indefinite",
                                "five-year", "ten-year", "perpetual"],
            # no_license keywords describe the ownership-reserved / no-implied-
            # license concept, which is distinct from invention assignment.
            # Section 2870 carve-outs and pre-assignment of Work Product belong
            # to the Invention Assignment article (Article 5 in the employee
            # NDA), not the No License article (Article 8). We do not extract
            # invention-assignment as a clause type, so those features simply
            # won't appear in any clause-level brief — that's correct.
            "no_license": ["no license", "no ip transfer", "implied license",
                             "estoppel", "feedback license", "moral rights",
                             "all rights reserved"],
            # Equitable remedies keywords describe REMEDY concepts —
            # irreparable harm, injunctive relief, no bond, liquidated damages.
            # Watermarking and forensic fingerprinting are content-protection
            # mechanisms (Confidentiality clause material), not remedies, so
            # they are intentionally omitted here.
            "equitable_remedies": ["irreparable", "injunction", "liquidated",
                                     "no bond", "specific performance"],
            # EMPLOYMENT-specific clause types
            "position_duties": ["title", "report", "remote", "relocat",
                                 "principal place", "outside board", "concurrent",
                                 "moonlighting", "airline pass"],
            "compensation": ["base salary", "target bonus", "equity grant",
                              "stock option", "rsu", "promotion auto-bump",
                              "promotion bump", "salary increase", "bonus opportunity",
                              "across-the-board reduction", "across the board reduction"],
            "severance": ["ordinary severance", "enhanced severance", "two-tier",
                           "two tier severance", "lump sum", "cobra contribution",
                           "general release", "release of claims", "no mitigation",
                           "no-mitigation", "irc 4980d", "section 4980d",
                           "garden leave", "garden-leave"],
            "restrictive_covenants": ["non-compete", "non-competition",
                                        "non-solicit", "non-solicitation",
                                        "non-disparagement", "tolling",
                                        "garden leave", "ma noncompetition",
                                        "massachusetts noncompetition",
                                        "16600", "section 16600",
                                        "blue pencil", "tail period"],
            "change_in_control": ["change in control", "change of control",
                                    "double-trigger", "double trigger", "single-trigger",
                                    "single trigger", "280g", "section 280g",
                                    "best-net", "best net", "gross-up", "gross up",
                                    "accelerated vesting", "acceleration",
                                    "good reason"],
            # EMPLOYMENT confidentiality / IP overrides reuse the existing
            # SaaS/MSA confidentiality keywords plus DTSA whistleblower notice
            # which is employment-specific.
            "intellectual_property_employment": ["work product", "invention assignment",
                                                   "section 2870", "labor code",
                                                   "moral rights", "prior inventions"],
            "dispute_resolution": ["arbitration", "binding arbitration",
                                    "jams", "aaa", "waiver of jury",
                                    "waive jury", "carve-out for equitable",
                                    "fee shifting", "prevailing party"],
            # SOW-specific clause types. Keywords here describe project-level
            # concerns that belong in each clause's body. Use specific phrases
            # (not generic words like "scope" or "fee") to avoid false matches
            # against features that belong in other clauses.
            "sow_scope": ["agile", "scrum", "waterfall", "discovery phase",
                          "design phase", "build phase", "ux research",
                          "data migration", "out-of-scope", "out of scope",
                          "carve-out from scope"],
            "milestones_schedule": ["go-live", "go live", "launch date",
                                     "phased rollout", "sprint", "iteration",
                                     "kickoff", "weekly status",
                                     "schedule slippage", "milestone slip"],
            "fees_payment_sow": ["t&m cap", "not-to-exceed", "not to exceed",
                                  "blended rate", "rate card", "fixed-fee",
                                  "fixed fee", "milestone payment",
                                  "completion bonus", "early-completion bonus",
                                  "expense pre-approval", "monthly invoice"],
            "change_control": ["change order", "change request", "ccb",
                                "change control board", "fast-track change",
                                "emergency change", "deemed approval"],
            "client_responsibilities": ["customer-provided", "decision-maker",
                                         "approval turnaround", "test data",
                                         "uat", "user acceptance testing",
                                         "subject matter expert", "sme",
                                         "credentials"],
            "assumptions_dependencies": ["assumption", "third-party dependency",
                                          "regulatory approval pending",
                                          "vendor delivery", "software license",
                                          "incumbent vendor"],
            "personnel": ["named team", "named personnel", "key personnel",
                           "minimum allocation", "dedicated team",
                           "background check", "security clearance",
                           "no replacement"],
            "subcontractors": ["offshore", "nearshore", "subcontractor list",
                                "approved subcontractor", "subcontractor consent",
                                "flow-down"],
            # DPA-specific clause types. Keywords here describe data
            # protection / privacy concerns that belong in each clause's
            # body. Use specific phrases (not generic words like "data" or
            # "personal") to avoid false matches.
            "controller_processor_roles": ["joint controller", "joint-controller",
                                            "co-controller", "processor on behalf",
                                            "sub-processor designation",
                                            "controller's documented instructions",
                                            "instruction limitation"],
            "processing_scope": ["categories of personal data",
                                  "categories of data subjects",
                                  "subject matter", "duration of processing",
                                  "annex i", "annex 1", "schedule 1",
                                  "lawful basis representation"],
            "security_measures": ["annex ii", "annex 2", "schedule 2",
                                   "iso 27001", "soc 2", "soc2", "encryption",
                                   "pseudonymisation", "pseudonymization",
                                   "byo-key", "bring-your-own-key",
                                   "kms", "key management"],
            "breach_notification": ["24 hours", "48 hours", "72 hours",
                                     "without undue delay", "incident response",
                                     "breach notification", "supervisory authority",
                                     "individual notification"],
            "international_transfers": ["scc", "standard contractual clauses",
                                         "2021/914", "idta", "uk addendum",
                                         "transfer impact assessment",
                                         "tia", "schrems", "adequacy",
                                         "module two", "module three",
                                         "supplementary measures",
                                         "data residency"],
            "data_subject_rights": ["dsr", "dsar", "data subject access request",
                                     "right of access", "right to erasure",
                                     "right to be forgotten", "portability",
                                     "rectification"],
            "dpia_cooperation": ["dpia", "data protection impact assessment",
                                  "article 35", "article 36",
                                  "prior consultation", "transfer impact assessment",
                                  "lia", "legitimate interests assessment"],
            "deletion_return": ["deletion", "return of data", "destruction",
                                 "end of services", "termination of services",
                                 "certification of deletion",
                                 "retention schedule"],
            "audit_rights": ["audit", "soc 2 type ii", "iso 27001",
                              "pen test report", "penetration test report",
                              "right to audit", "audit notice",
                              "in lieu of audit", "trust center"],
            # Independent Contractor (IC) clause types
            "ic_engagement": ["scope of services", "professional services",
                               "consulting services", "project-based",
                               "fixed-term", "best efforts",
                               "no subcontracting", "personal services"],
            "ic_compensation": ["hourly rate", "fixed fee", "milestone payment",
                                 "monthly retainer", "blended rate", "rate card",
                                 "expense reimbursement", "travel reimbursement",
                                 "net 30 invoicing", "net thirty"],
            "ic_classification": ["w-9", "w9", "1099", "1099-nec", "self-employment",
                                   "no withholding", "ab-5", "ab5", "abc test",
                                   "dynamex", "freelance isn't free",
                                   "freelance isnt free", "misclassification",
                                   "no benefits", "no employer-employee"],
            "ic_intellectual_property": ["work product", "work-for-hire",
                                          "work made for hire", "ip assignment",
                                          "background ip", "pre-existing ip",
                                          "moral rights", "license-back",
                                          "non-infringement"],
            "ic_termination": ["termination for convenience", "for convenience",
                                "wind-down", "transition", "for cause",
                                "material breach"],
            "ic_restrictive_covenants": ["non-solicit", "non-solicitation",
                                          "non-compete during",
                                          "no post-engagement non-compete",
                                          "moonlighting", "side hustle",
                                          "conflict disclosure", "free to perform",
                                          "best efforts"],
            # LICENSE clause types. Keywords here describe license-specific
            # concerns that belong in each clause's body. Use specific phrases
            # tied to grant scope, royalty mechanics, and IP-license patterns.
            "license_grant": ["exclusive", "non-exclusive", "sole license",
                               "field of use", "territory", "sublicens",
                               "perpetual", "irrevocable", "worldwide",
                               "have made", "have used", "have sold",
                               "right to make", "right to use", "right to sell"],
            "license_royalties_payment": ["upfront fee", "signing fee",
                                            "running royalty", "royalty rate",
                                            "minimum annual royalty",
                                            "minimum royalty", "milestone payment",
                                            "regulatory milestone", "sales milestone",
                                            "royalty stack", "royalty stacking",
                                            "net sales", "mfn", "most favored",
                                            "most-favored", "frand"],
            "license_restrictions": ["reverse engineer", "decompil", "disassembl",
                                      "no derivative", "no modification",
                                      "no transfer", "no sublicens",
                                      "out of field", "outside the field",
                                      "out of territory", "outside the territory",
                                      "reserved rights", "reservation of rights",
                                      "no implied license"],
            "license_term_termination": ["sell-off", "sell off", "wind-down",
                                          "wind down", "minimum royalty failure",
                                          "milestone failure", "sunset",
                                          "natural expiration",
                                          "termination for convenience"],
            "license_warranties_indemnification": ["title warranty",
                                                     "non-infringement warranty",
                                                     "infringement indemnit",
                                                     "third-party infringement",
                                                     "patent infringement claim",
                                                     "ip infringement",
                                                     "control of defense",
                                                     "ip indemnit"],
            "license_audit_rights": ["royalty audit", "audit of royalty",
                                      "books and records", "audit frequency",
                                      "independent cpa",
                                      "certified public accountant",
                                      "underreport", "underpayment",
                                      "audit threshold"],
            "license_improvements": ["improvements", "grant-back", "grant back",
                                       "joint inventions", "derivative works",
                                       "enhancement", "modification ownership"],
            "license_quality_control": ["quality control", "quality standard",
                                          "sample approval", "brand standard",
                                          "naked license", "trademark control",
                                          "approved use", "house style",
                                          "style guide"],
        }.get(clause_type, [])
        relevant = [s for s in special if any(k.lower() in s.lower() for k in relevant_kw)]
        chosen = relevant[:2]
        if chosen:
            special_block = "Specific requirements: " + "; ".join(chosen) + "."
        # else: no special features are relevant to this clause — omit the
        # block entirely rather than leaking an unrelated requirement.

    # Standard provisions to include — per-type override wins over the
    # generic list (e.g. MSA "intellectual_property" → work-product assignment
    # language instead of SaaS license-grant language).
    type_provisions = CLAUSE_STANDARD_PROVISIONS_BY_TYPE.get(contract_type, {})
    provisions = type_provisions.get(clause_type) or CLAUSE_STANDARD_PROVISIONS.get(clause_type, [])

    # Defense 2: filter the brief's provisions list down to only what the
    # clause body actually covers. Without this, the training pair is
    # internally inconsistent — the brief demands 6 provisions but the body
    # only demonstrates 4 — and the model learns provisions are optional.
    if clause_body and provisions:
        provisions = filter_provisions_by_coverage(provisions, clause_body)

    provisions_block = ""
    if provisions:
        provisions_block = "Cover the standard provisions: " + ", ".join(provisions) + "."

    # Effective date
    effective_block = ""
    if sub_map.get("EFFECTIVE_DATE"):
        effective_block = f"Effective Date: {sub_map['EFFECTIVE_DATE']}."

    style_block = ""
    if style_label:
        style_block = f"Drafting style: {style_label}."

    # Stitch the brief together. Drop empty pieces so we don't get awkward gaps.
    parts = [
        f"{opener} {party_block}.",
        effective_block,
        context_block,
        commercial_block,
        jurisdiction_block,
        reg_block,
        special_block,
        style_block,
        provisions_block,
    ]
    brief = " ".join(p for p in parts if p)
    return brief


# ── Defense 2: Brief-output coverage filter ─────────────────────────────

# Per-provision keyword list. When we construct the brief's "cover the standard
# provisions" list, we filter each provision against these keywords appearing
# in the actual clause body. Any provision whose keywords don't appear in the
# body is dropped from the brief — so the (brief, output) pair the model is
# trained on is always internally consistent. This is the structural fix for
# the v2.0 coverage gap (91.9% SaaS, 71.9% MSA → should go to ~100% v2.1).

_PROVISION_KEYWORDS = {
    # SaaS IP
    "Provider ownership of the Platform and all related IP":
        ["platform", "title", "intellectual property rights"],
    "Customer ownership of Customer Data":
        ["customer data", "exclusive property"],
    "Feedback license back to Provider":
        ["feedback"],
    "no implied licenses":
        ["implied", "estoppel"],
    # SaaS Limitation of Liability
    "aggregate cap tied to fees paid in the prior 12 months":
        ["aggregate", "cap", "twelve"],
    "exclusion of indirect, consequential, special and punitive damages":
        ["indirect", "consequential", "punitive"],
    "exceptions for confidentiality, indemnity and data breaches":
        ["except", "confidential", "indemn"],
    "basis-of-the-bargain language":
        ["basis of the bargain", "essential element"],
    # Termination
    "termination for material breach with cure period":
        ["material breach", "cure"],
    "termination for insolvency":
        ["insolvency", "bankruptcy", "creditors"],
    "effects of termination (cessation of access, data return)":
        ["effects of termination", "cease", "return"],
    "transition assistance":
        ["transition"],
    # Indemnification
    "Provider IP infringement indemnity for Customer":
        ["infringement", "defend", "indemnif"],
    "Customer indemnity for misuse and Customer Data claims":
        ["customer data", "indemnif"],
    "indemnification procedure (notice, control, cooperation)":
        ["procedure", "control", "cooperation"],
    "mitigation options if Platform becomes subject to a claim":
        ["mitigation", "modify", "replace"],
    # Confidentiality (generic)
    "definition of Confidential Information":
        ["confidential information"],
    "use restrictions and standard of care":
        ["use", "care"],
    "exclusions (publicly known, independently developed, etc.)":
        ["public", "independently"],
    "compelled disclosure procedure":
        ["compelled", "court order", "required by law"],
    "return or destruction on termination":
        ["return", "destroy"],
    # Data privacy
    "Customer ownership of Customer Data":
        ["customer data", "title", "retain"],
    "limited license for Provider to process":
        ["license", "process"],
    "security safeguards (encryption, access controls)":
        ["encryption", "tls", "aes"],
    "breach notification within a defined window":
        ["breach", "notif", "hours"],
    "data residency":
        ["residency", "stored", "data center"],
    "data return and deletion on termination":
        ["return", "delete", "destruction"],
    # Service level
    "uptime commitment with calculation methodology":
        ["uptime", "availab", "%"],
    "excluded downtime (scheduled maintenance, force majeure)":
        ["scheduled maintenance", "force majeure"],
    "service credits as the sole financial remedy":
        ["service credit", "sole"],
    "support tiers and response time targets":
        ["support", "response time"],
    # MSA-specific
    "Provider's pre-existing IP and tools retained by Provider":
        ["background ip", "pre-existing"],
    "Work Product assigned to Customer (work made for hire to the maximum extent permitted, with express assignment as backup)":
        ["work product", "work made for hire", "assign"],
    "license-back to Provider for any pre-existing IP embedded in Work Product":
        ["license", "background", "embedded"],
    "Customer's pre-existing IP and Customer Materials retained by Customer":
        ["customer materials", "retain"],
    "Customer right to terminate for convenience on notice":
        ["convenience"],
    "delivery and assignment of Work Product through the termination date":
        ["work product", "termination"],
    "wind-down assistance for in-progress SOWs":
        ["wind-down", "transition"],
    # Scope of services
    "scope of services described by reference to one or more SOWs":
        ["statement of work", "sow", "scope"],
    "professional and workmanlike performance standard":
        ["professional", "workmanlike"],
    "Provider responsibility for results of services":
        ["responsib"],
    "Customer cooperation obligations":
        ["cooperat"],
    # Deliverables and acceptance
    "objective acceptance criteria set in each SOW":
        ["acceptance criteria"],
    "Customer review and acceptance period":
        ["review", "acceptance period", "business days"],
    "rejection notice and cure right for non-conforming deliverables":
        ["reject", "cure", "non-conform"],
    "deemed acceptance after a defined window":
        ["deemed accept"],
    # Personnel
    "key personnel commitment with no replacement without consent":
        ["key personnel", "replace"],
    "background checks and required training":
        ["background", "training"],
    "removal of unsuitable personnel on Customer request":
        ["remov"],
    "non-solicitation of personnel during the engagement":
        ["solicit"],
    # Subcontractors
    "Customer prior written consent for subcontractors":
        ["subcontract", "consent"],
    "flow-down of confidentiality, IP, security obligations":
        ["flow-down", "flow down"],
    "Provider remains primarily responsible for subcontractor performance":
        ["subcontract", "responsible"],
    "subcontractor list maintained and updated":
        ["subcontractor", "list"],
    # Insurance
    "Commercial General Liability minimum coverage":
        ["general liability"],
    "Professional Liability / Errors & Omissions coverage":
        ["professional liability", "errors and omissions", "e&o"],
    "Cyber Liability coverage where Provider handles Customer data":
        ["cyber"],
    "Workers' Compensation as required by state law":
        ["workers", "workmen"],
    "Provider names Customer as additional insured where applicable":
        ["additional insured"],
    # NDA confidentiality
    "definition of Confidential Information (marked, identified as confidential, or information a reasonable person would treat as confidential)":
        ["confidential information", "reasonable person"],
    "use restriction — Confidential Information used solely for the defined Purpose":
        ["purpose", "use"],
    "disclosure restriction — limited to personnel with a need to know who are bound by equivalent confidentiality":
        ["need to know", "personnel", "bound"],
    "standard of care — at least the care the Receiving Party uses for its own confidential information, but no less than reasonable care":
        ["care", "reasonable"],
    "no transfer of IP or ownership to the Receiving Party":
        ["no transfer", "no license", "retain"],
    "obligation to notify the Disclosing Party of any unauthorized disclosure":
        ["notify", "unauthorized"],
    # NDA purpose
    "defined Purpose of disclosure stated narrowly":
        ["purpose"],
    "use restriction limited to the Purpose":
        ["solely", "purpose"],
    "no use for competitive advantage or unrelated business":
        ["competit", "advantage"],
    "no reverse engineering of any tangible Confidential Information":
        ["reverse engineer"],
    # NDA exclusions
    "information already public without breach":
        ["public", "breach"],
    "information already known to the Receiving Party prior to disclosure":
        ["known", "prior"],
    "information independently developed without use of Confidential Information":
        ["independently developed"],
    "information rightfully received from a third party without duty of confidence":
        ["third party", "rightfully"],
    # NDA compelled disclosure
    "prompt written notice to the Disclosing Party before disclosure":
        ["notice", "disclosing party"],
    "cooperation with the Disclosing Party's efforts to obtain a protective order":
        ["protective order"],
    "disclosure limited to the portion legally required":
        ["legally required", "portion"],
    "continued confidentiality over non-disclosed portions":
        ["continued", "confidential"],
    # NDA return or destruction
    "return or destruction of all Confidential Information on the Disclosing Party's request":
        ["return", "destruction"],
    "written certification of destruction on request":
        ["certif"],
    "exception for archival backups and bona fide records retention":
        ["archival", "backup", "retention"],
    "continued confidentiality obligations over any retained copies":
        ["retain", "confidential"],
    # NDA term and survival
    "term of the Agreement with end date or ongoing relationship":
        ["term"],
    "survival of confidentiality obligations for a defined period after termination":
        ["survive", "years", "after"],
    "trade secret obligations survive as long as the information remains a trade secret":
        ["trade secret", "survive"],
    "termination for material breach":
        ["material breach"],
    # NDA term/survival (alternate phrasing)
    "term of the Agreement":
        ["term"],
    "survival of confidentiality obligations for a defined period (typically 3-5 years) after the end of the term":
        ["survive", "years"],
    "trade secret carve-out — trade secret obligations survive as long as the information remains a trade secret":
        ["trade secret"],
    "return or destruction of Confidential Information on termination":
        ["return", "destroy", "destruction"],
    # NDA no license
    "no transfer of any ownership interest":
        ["no transfer", "title"],
    "no license granted by implication, estoppel, or otherwise":
        ["implied", "estoppel"],
    "all rights reserved by the Disclosing Party":
        ["reserved"],
    "Disclosing Party retains all patent, copyright, trademark, and trade secret rights":
        ["patent", "copyright", "trademark"],
    # NDA equitable remedies
    "acknowledgment that breach may cause irreparable harm":
        ["irreparable"],
    "right to seek injunctive relief without posting bond":
        ["injunct", "bond"],
    "no obligation to prove actual damages":
        ["actual damages"],
    "cumulative remedies in addition to any at law or in equity":
        ["cumulative", "in addition"],
    # NDA IP (NDA-specific override)
    "no transfer of any intellectual property rights to the Receiving Party":
        ["no transfer", "intellectual property"],
    "no license granted by implication, estoppel, or otherwise":
        ["implied", "estoppel"],
    "all patent, copyright, trademark, and trade secret rights reserved by the Disclosing Party":
        ["patent", "copyright", "reserved"],
    "Feedback carve-out, if applicable, for suggestions provided during evaluation":
        ["feedback"],
    # EMPLOYMENT — position and duties
    "Executive's title, reporting line, and scope of duties":
        ["title", "report"],
    "full-time and exclusive services commitment":
        ["full-time", "full time", "exclusive", "best efforts", "best effort"],
    "limited carve-out for outside boards / charitable / passive investments":
        ["board", "charitable", "civic", "personal investment", "passive"],
    "principal place of employment with relocation triggers if any":
        ["principal place", "principal location", "relocat", "work location"],
    # EMPLOYMENT — compensation
    "annual base salary with payroll cadence":
        ["base salary", "annual base"],
    "annual target bonus with discretion / performance metrics":
        ["target bonus", "annual bonus", "discretionary bonus", "incentive compensation"],
    "equity participation in the Company's incentive plan":
        ["equity", "stock option", "rsu", "restricted stock", "incentive plan"],
    "standard employee welfare benefits (medical, dental, retirement, PTO)":
        ["benefit plan", "welfare", "medical", "dental", "401(k)", "retirement", "vacation", "paid time off"],
    "expense reimbursement under standard Company policies":
        ["expense", "reimbursement"],
    # EMPLOYMENT — severance
    "termination triggers (Cause / without Cause / Good Reason / death / disability)":
        ["for cause", "without cause", "good reason", "disability", "death"],
    "severance multiple of base salary and / or bonus":
        ["severance", "lump sum", "months of base", "times annual"],
    "continuation of group health benefits (COBRA-equivalent contribution)":
        ["cobra", "group health", "health benefit"],
    "general release of claims as condition of severance":
        ["release of claims", "general release", "release agreement"],
    "no-mitigation clause and offset rules":
        ["mitigate", "mitigation", "offset"],
    # EMPLOYMENT — restrictive covenants
    "non-competition restriction during employment and a defined post-employment period":
        ["non-compete", "non-competition", "noncompete", "compete with"],
    "non-solicitation of Company employees and customers":
        ["non-solicit", "non-solicitation", "solicit"],
    "confidentiality of proprietary information surviving termination":
        ["confidential information", "proprietary information"],
    "non-disparagement covenant":
        ["disparage", "non-disparagement"],
    "tolling of restricted period for any breach by Executive":
        ["toll", "tolling"],
    # EMPLOYMENT — change in control
    "definition of Change in Control event":
        ["change in control", "change of control"],
    "double-trigger: termination without Cause or for Good Reason within a defined window after CIC":
        ["double-trigger", "double trigger", "within", "following the change", "after the change"],
    "enhanced severance multiple compared to ordinary termination":
        ["enhanced severance", "increased severance", "in lieu of"],
    "accelerated vesting of outstanding equity awards":
        ["accelerat", "vest"],
    "Section 280G best-net cutback or gross-up treatment":
        ["280g", "section 280g", "excise tax", "best net", "best-net", "gross-up", "gross up"],
    # EMPLOYMENT — dispute resolution
    "governing law of a designated state":
        ["governed by", "governing law", "construed in accordance"],
    "binding arbitration or exclusive court venue for disputes":
        ["arbitration", "exclusive jurisdiction", "exclusive venue"],
    "carve-out for equitable relief in support of restrictive covenants":
        ["equitable relief", "injunction", "injunctive"],
    "waiver of jury trial":
        ["jury trial", "waive a trial"],
    "fee-shifting or each-party-bears-its-own-fees rule":
        ["prevailing party", "attorneys' fees", "own attorney", "own fees"],
    # EMPLOYMENT — termination (employment-flavored override of generic termination keywords)
    "termination upon Executive's death or disability":
        ["death", "disability"],
    "termination by the Company for Cause (with definition cross-reference)":
        ["for cause"],
    "termination by the Company without Cause":
        ["without cause"],
    "termination by Executive for Good Reason (with definition cross-reference)":
        ["good reason"],
    "termination by Executive without Good Reason":
        ["without good reason", "voluntary resignation"],
    "notice and effective-date mechanics for each trigger":
        ["notice of termination", "effective date", "employment end date", "termination date"],
    # EMPLOYMENT — IP / invention assignment
    "Executive's assignment of all Work Product to the Company":
        ["work product", "assign", "assignment"],
    "works made for hire to the maximum extent permitted by law":
        ["work made for hire", "made for hire"],
    "Executive's pre-existing inventions excluded from assignment":
        ["prior invention", "pre-existing"],
    "moral rights waiver to the extent permitted":
        ["moral right"],
    "state-law carve-outs for inventions on Executive's own time using no Company resources (e.g. Cal. Lab. Code § 2870)":
        ["section 2870", "labor code", "own time", "own equipment"],
    # EMPLOYMENT — confidentiality (employment-flavored)
    "confidential information of the Company defined to include trade secrets, customer data, financials, and personnel information":
        ["trade secret", "customer data", "financial information", "personnel information"],
    "Executive's duty to hold confidential information in strict confidence during and after employment":
        ["strict confidence", "during and after", "during employment"],
    "no use of confidential information for any purpose other than performance of Executive's duties":
        ["other than", "in connection with", "performance of"],
    "Defend Trade Secrets Act whistleblower immunity notice in the required statutory form":
        ["1833", "defend trade secrets", "whistleblower"],
    "return of all confidential information and Company property on termination":
        ["return", "company property"],
    # SOW — sow_scope
    "specific project description with concrete in-scope activities":
        ["in-scope", "in scope", "shall perform", "will perform", "scope of"],
    "explicit out-of-scope items that the project will not cover":
        ["out-of-scope", "out of scope", "not include", "excluded from"],
    "objectives or success criteria the project is intended to achieve":
        ["objective", "success criteria", "intended to", "purpose of this"],
    "reference to the parent Master Services Agreement governing this SOW":
        ["master services agreement", "parent msa", "governed by the msa",
         "subject to the master", "incorporated by reference"],
    # SOW — milestones_schedule
    "project start date and target completion date":
        ["start date", "kickoff", "target completion", "completion date",
         "estimated completion"],
    "named milestones with target completion dates":
        ["milestone", "deliverable date"],
    "dependencies between milestones where applicable":
        ["dependent on", "predecessor", "depends on", "subject to completion"],
    "consequences of milestone slippage (status reporting, escalation)":
        ["slippage", "escalation", "status report", "weekly status",
         "delay notice"],
    # SOW — fees_payment_sow
    "pricing structure (time-and-materials, fixed-fee, milestone-based, or hybrid)":
        ["time-and-materials", "time and materials", "t&m", "fixed-fee",
         "fixed fee", "milestone-based", "milestone payment"],
    "rate card or fixed-fee amount with currency":
        ["rate card", "hourly rate", "blended rate", "us$", "usd",
         "fixed fee of"],
    "expense reimbursement policy with pre-approval threshold":
        ["expense", "reimbursement", "pre-approval", "pre approval",
         "approved expenses", "out-of-pocket"],
    "invoicing cadence and payment terms":
        ["invoice", "monthly invoice", "net 30", "net thirty", "net 45",
         "payment terms", "payable within"],
    "late payment interest or suspension rights":
        ["late payment", "interest at", "suspend services", "suspension"],
    # SOW — change_control
    "definition of a Change Request with required content":
        ["change request", "change order", "describing the change",
         "in writing"],
    "Provider's obligation to assess impact on scope, schedule, and fees":
        ["impact assessment", "impact on", "assess", "evaluation"],
    "Customer's written approval required before any change is implemented":
        ["written approval", "signed by", "executed change order",
         "authorized signatory"],
    "no implied or oral changes — only signed Change Orders bind the parties":
        ["no oral", "only by", "shall not be modified", "no change"],
    "effect of disputed Change Requests on continuing work":
        ["continue to perform", "pending resolution", "shall not stop",
         "good faith"],
    # SOW — client_responsibilities
    "Customer-provided personnel and decision-makers":
        ["customer shall provide", "designated", "point of contact",
         "decision-maker", "decision maker"],
    "timely Customer access to systems, data, and facilities":
        ["access to", "facilities", "systems", "test environment",
         "credentials"],
    "Customer responsibility for accuracy of provided materials":
        ["accuracy", "complete and accurate", "responsible for",
         "rely on"],
    "Customer review and approval turnaround commitment":
        ["business days", "review period", "approval within",
         "respond within"],
    "consequences if Customer delays prevent Provider from meeting milestones":
        ["customer delay", "delay caused", "extension of", "relief from",
         "schedule slippage"],
    # SOW — assumptions_dependencies
    "explicit assumptions on which the project pricing and schedule are based":
        ["assumption", "based on the assumption", "assume that",
         "pricing assumes"],
    "third-party dependencies (vendors, software, regulatory approvals)":
        ["third party", "third-party", "vendor", "regulatory approval",
         "dependent on"],
    "change control trigger if any assumption proves incorrect":
        ["change control", "change order", "incorrect", "invalid",
         "assumption is not"],
    "Customer obligation to notify Provider of any changes to assumptions":
        ["notify", "notice", "promptly", "in writing"],
    # SOW — deliverables_acceptance (SOW override)
    "list of deliverables with descriptions and target dates":
        ["deliverable", "described in", "table", "schedule"],
    "objective acceptance criteria for each deliverable":
        ["acceptance criteria", "criteria for"],
    "Customer review and acceptance period (typically 5-10 business days)":
        ["business days", "review period", "acceptance period"],
    "rejection notice and Provider cure right for non-conforming deliverables":
        ["reject", "non-conforming", "non conforming", "cure",
         "remedial"],
    "deemed acceptance after the review window expires without rejection":
        ["deemed accepted", "deemed acceptance", "shall be deemed",
         "if no notice"],
    # SOW — personnel (SOW override)
    "named key personnel assigned to this project":
        ["key personnel", "named", "schedule", "listed below",
         "project lead"],
    "minimum allocation percentage of each key person to this project":
        ["allocation", "% of time", "percent of", "dedicated"],
    "no replacement of key personnel without Customer prior written consent":
        ["replace", "replacement", "without", "prior written consent"],
    "background checks and Customer security training where required":
        ["background check", "security training", "training requirements"],
    "non-solicitation of project personnel during the engagement":
        ["solicit", "non-solicitation", "hire away"],
    # DPA — controller_processor_roles
    "designation of Controller and Processor (or sub-processor) for each processing activity":
        ["controller", "processor", "data exporter", "data importer"],
    "Controller's instructions as the basis for the Processor's processing":
        ["instructions", "documented instructions", "controller's instructions"],
    "Processor's obligation to process Personal Data only on documented Controller instructions":
        ["only on", "documented instructions", "in accordance with"],
    "Processor's notification obligation if any instruction would violate applicable data protection law":
        ["notify", "violate", "infringe", "applicable law"],
    # DPA — processing_scope
    "subject matter and duration of the processing":
        ["subject matter", "duration"],
    "nature and purpose of the processing":
        ["nature", "purpose of the processing"],
    "types of Personal Data processed":
        ["personal data", "types of", "categories of personal data"],
    "categories of Data Subjects":
        ["data subjects", "categories of data subjects"],
    "Controller's lawful basis representation":
        ["lawful basis", "lawful", "represents"],
    # DPA — security_measures
    "implementation of appropriate technical and organizational measures (TOMs)":
        ["technical and organizational", "technical and organisational",
         "appropriate measures", "annex ii", "annex 2"],
    "encryption of Personal Data in transit and at rest where appropriate":
        ["encryption", "in transit", "at rest", "tls", "aes"],
    "ongoing confidentiality, integrity, availability, and resilience of processing systems":
        ["confidentiality", "integrity", "availability", "resilience"],
    "ability to restore access to Personal Data after a physical or technical incident":
        ["restore", "backup", "incident", "recovery"],
    "regular testing and evaluation of the effectiveness of the TOMs":
        ["testing", "evaluation", "effectiveness", "regularly"],
    "personnel confidentiality and need-to-know access controls":
        ["confidentiality obligations", "need to know", "need-to-know",
         "access controls", "authorized personnel"],
    # DPA — breach_notification
    "Processor's obligation to notify Controller without undue delay after becoming aware of a Personal Data Breach":
        ["without undue delay", "becoming aware", "personal data breach",
         "notify"],
    "specific notification timeline (e.g., within 24, 48, or 72 hours)":
        ["hours", "within 24", "within 48", "within 72", "within twenty-four",
         "within forty-eight", "within seventy-two"],
    "minimum content of the breach notification (nature, categories, approximate numbers, contact point, likely consequences, measures taken)":
        ["nature", "categories", "consequences", "measures",
         "contact point", "likely"],
    "Processor's cooperation with Controller's investigation and remediation":
        ["cooperate", "cooperation", "investigation", "remediation",
         "assist"],
    "Processor's record-keeping of all Personal Data Breaches":
        ["record", "log", "documentation", "retain"],
    # DPA — international_transfers
    "permitted transfer mechanisms (adequacy decisions, Standard Contractual Clauses, UK IDTA, supplementary measures)":
        ["adequacy", "standard contractual clauses", "scc", "idta",
         "supplementary measures", "transfer mechanism"],
    "incorporation of the EU SCCs (Commission Decision 2021/914) where applicable":
        ["2021/914", "european commission", "module", "controller-to-processor",
         "scc"],
    "incorporation of the UK International Data Transfer Addendum where applicable":
        ["uk idta", "international data transfer addendum",
         "uk addendum", "ico"],
    "Processor's obligation to conduct a Transfer Impact Assessment for restricted transfers":
        ["transfer impact assessment", "tia", "schrems",
         "third country"],
    "Controller's right to require additional safeguards or to suspend transfers":
        ["additional safeguards", "suspend", "supplementary measures",
         "right to"],
    # DPA — data_subject_rights
    "Processor's obligation to assist Controller in responding to Data Subject requests":
        ["assist", "data subject requests", "respond", "controller in responding"],
    "covered Data Subject rights (access, rectification, erasure, restriction, portability, objection)":
        ["access", "rectification", "erasure", "restriction", "portability",
         "objection", "right to be forgotten"],
    "Processor's obligation to forward Data Subject requests received directly to Controller without delay":
        ["forward", "without delay", "directly to controller", "promptly"],
    "limitation that Processor will not respond directly to Data Subjects without Controller authorization":
        ["not respond", "without controller", "without authorization"],
    # DPA — dpia_cooperation
    "Processor's obligation to assist Controller with Data Protection Impact Assessments under Article 35 GDPR":
        ["data protection impact assessment", "dpia", "article 35",
         "assist controller"],
    "Processor's obligation to assist with prior consultations with supervisory authorities under Article 36 GDPR":
        ["prior consultation", "supervisory authority", "article 36"],
    "scope of the assistance (information, documentation, attendance at meetings)":
        ["information", "documentation", "attendance",
         "reasonable assistance"],
    "Controller's responsibility for the DPIA itself; Processor's role is supportive":
        ["controller's responsibility", "controller is responsible",
         "processor's role"],
    # DPA — deletion_return
    "Processor's obligation, at Controller's choice, to delete or return all Personal Data at the end of provision of the services":
        ["delete", "return", "end of", "termination of",
         "end of provision"],
    "deletion of all existing copies unless retention is required by applicable law":
        ["existing copies", "all copies", "unless required",
         "applicable law"],
    "Processor's certification of deletion on Controller's request":
        ["certify", "certification", "written certification",
         "confirm in writing"],
    "exception for backup or archival copies retained pursuant to a documented retention schedule":
        ["backup", "archival", "retention schedule", "retention period"],
    "continued application of the DPA's confidentiality and security obligations to any retained copies":
        ["continued", "retained", "remain subject", "shall continue"],
    # DPA — audit_rights
    "Controller's right to audit the Processor's compliance with the DPA":
        ["audit", "right to audit", "controller's right", "demonstrate compliance"],
    "audit notice period and audit frequency limitations":
        ["notice", "business days", "calendar days", "no more than",
         "once per", "annually", "twelve months"],
    "Processor's right to provide third-party audit reports (SOC 2 Type II, ISO 27001) in lieu of on-site audits":
        ["soc 2", "soc2", "iso 27001", "third-party audit", "third party audit",
         "in lieu of"],
    "scope limitations to protect other customers' confidential information and the Processor's trade secrets":
        ["other customers", "confidential information",
         "trade secret", "scope limitations"],
    "Controller's responsibility for the cost of audits beyond Processor's standard third-party reports":
        ["cost", "expense", "at controller's expense", "borne by controller"],
    # DPA — subprocessors (DPA-specific override)
    "Controller's general or specific authorization for sub-processors":
        ["general authorization", "general authorisation", "specific authorization",
         "specific authorisation", "consents", "approves"],
    "Processor's obligation to maintain a current list of sub-processors and notify Controller of changes":
        ["list", "subprocessor", "sub-processor", "notify", "update"],
    "Controller's right to object to new sub-processors within a defined window":
        ["object", "objection", "reasonable grounds", "within"],
    "Processor's obligation to impose data protection obligations on sub-processors equivalent to those in the DPA":
        ["impose", "equivalent", "flow-down", "flow down",
         "same obligations", "back-to-back"],
    "Processor's primary liability for the acts and omissions of its sub-processors":
        ["primary liability", "liable", "acts and omissions",
         "as if performed by"],
    # IC — engagement
    "scope of services to be performed by the Contractor":
        ["scope", "services", "shall perform", "engages"],
    "term of the engagement (fixed-term, project-based, or open-ended)":
        ["term", "shall commence", "shall continue", "fixed-term",
         "project", "until completion"],
    "Contractor's commitment to perform services in a professional and workmanlike manner":
        ["professional", "workmanlike", "best efforts", "commercially reasonable"],
    "no subcontracting or delegation without Company consent":
        ["no subcontract", "shall not delegate", "shall not subcontract",
         "without prior written consent", "personal services"],
    # IC — compensation
    "fee structure (hourly rate, fixed fee, milestone payments, or monthly retainer)":
        ["hourly rate", "fixed fee", "milestone", "monthly retainer",
         "per hour", "per month", "flat fee", "consulting fee"],
    "invoicing cadence and payment terms":
        ["invoice", "monthly", "net thirty", "net 30", "net forty-five",
         "net 45", "payable", "payment within"],
    "expense reimbursement policy with pre-approval threshold":
        ["expense", "reimburse", "out-of-pocket", "pre-approval",
         "approved expenses"],
    "no benefits, no withholding — Contractor is responsible for own taxes":
        ["no withholding", "no benefits", "responsible for", "own taxes",
         "self-employment", "1099"],
    # IC — classification
    "explicit independent contractor relationship — no employer-employee, partnership, joint venture, or agency":
        ["independent contractor", "no employer-employee", "no employee",
         "no partnership", "no joint venture", "no agency",
         "not an employee", "not a partner"],
    "Contractor's responsibility for self-employment, Social Security, Medicare, and income taxes":
        ["self-employment", "social security", "medicare", "income tax",
         "responsible for"],
    "no Company withholding from Contractor's compensation":
        ["no withholding", "without withholding", "shall not withhold"],
    "Contractor not entitled to employee benefits (medical, retirement, vacation, workers compensation, unemployment insurance)":
        ["medical", "retirement", "vacation", "workers compensation",
         "workers' compensation", "unemployment", "not entitled",
         "no benefits"],
    "Contractor's obligation to provide Form W-9 and Company's reporting on Form 1099-NEC":
        ["w-9", "1099", "form 1099", "1099-nec", "taxpayer identification"],
    # IC — intellectual property
    "Contractor's assignment to the Company of all Work Product created during the engagement":
        ["work product", "assigns", "assign", "assignment",
         "shall be the exclusive property"],
    "works made for hire to the maximum extent permitted by U.S. copyright law, with explicit assignment as backup":
        ["work made for hire", "works made for hire", "made for hire",
         "to the maximum extent permitted", "if and to the extent"],
    "Contractor's pre-existing IP retained by the Contractor with limited license to Company as needed for the deliverables":
        ["pre-existing", "pre existing", "background ip", "retained by",
         "limited license", "license to use"],
    "moral rights waiver to the extent permitted by applicable law":
        ["moral rights", "moral right", "waives", "waive any moral"],
    "Contractor's representations of original authorship and non-infringement":
        ["original", "non-infringement", "does not infringe",
         "represents and warrants", "infringe"],
    # IC — termination
    "termination for convenience by either party on stated notice":
        ["for convenience", "without cause", "may terminate",
         "written notice"],
    "termination by the Company for cause (material breach, misconduct, regulatory disqualification)":
        ["material breach", "misconduct", "for cause",
         "convicted", "violation"],
    "Contractor's deliverables and final invoice obligations on termination":
        ["final invoice", "deliverables", "wind-down", "transition",
         "upon termination"],
    "survival of confidentiality, IP assignment, restrictive covenants, and indemnification":
        ["survive", "survival", "shall survive"],
    # IC — restrictive covenants
    "Contractor free to perform services for other clients, except direct competitors during the engagement":
        ["free to", "may perform", "other clients", "direct competitor",
         "competitor", "competing"],
    "non-solicitation of Company employees and contractors during and after the engagement (for a defined post-engagement period)":
        ["non-solicit", "non-solicitation", "shall not solicit",
         "shall not hire", "employees and contractors", "personnel"],
    "no use of Company Confidential Information in providing services to third parties":
        ["confidential information", "third parties", "other clients",
         "for any purpose other than"],
    "Contractor's obligation to disclose conflicts of interest with Company customers, vendors, or competitors":
        ["disclose", "conflict", "conflict of interest", "customer or vendor",
         "vendors", "customers"],
    # IC — confidentiality (IC override)
    "definition of Company Confidential Information":
        ["confidential information", "trade secret", "non-public"],
    "Contractor's obligation to hold Confidential Information in strict confidence during and after the engagement":
        ["strict confidence", "during and after", "hold in confidence"],
    "no use of Confidential Information for any purpose other than performing the services":
        ["any purpose other than", "other than the services",
         "in connection with the services"],
    "return or destruction of all Company materials on termination":
        ["return", "destroy", "destruction", "company property",
         "company materials"],
    "injunctive relief for breach (no proof of actual damages required)":
        ["injunctive", "irreparable", "no proof", "without bond",
         "equitable relief"],
    # IC — indemnification (IC override)
    "Contractor's indemnification of Company for any claim that Contractor is or should be classified as an employee":
        ["misclassification", "classified as an employee", "should be classified",
         "indemnify", "tax authority"],
    "Contractor's indemnification for any tax liability resulting from Contractor's failure to pay self-employment, Social Security, or income taxes":
        ["tax liability", "self-employment", "failure to pay",
         "indemnify"],
    "Contractor's indemnification for breach of representations and warranties (including IP non-infringement)":
        ["representations", "warranties", "breach", "indemnify",
         "infringement"],
    "Contractor's professional liability / errors and omissions coverage where required":
        ["professional liability", "errors and omissions", "e&o",
         "insurance", "coverage"],
    # LICENSE — Grant
    "identification of the Licensed IP (patent, trademark, copyright, software, trade secret, or know-how)":
        ["licensed", "patent", "trademark", "copyright", "software",
         "trade secret", "know-how"],
    "exclusivity (exclusive, sole, or non-exclusive)":
        ["exclusive", "non-exclusive", "sole license"],
    "field of use limitations":
        ["field", "use"],
    "territory in which the Licensee may exercise the rights":
        ["territory"],
    "term of the license":
        ["term"],
    "Licensee's right (or prohibition) to grant sublicenses":
        ["sublicens"],
    "right to make, use, sell, import, or perform the Licensed IP":
        ["make", "use", "sell"],
    # LICENSE — Royalties / Payment
    "upfront license fee or signing payment":
        ["upfront", "signing"],
    "running royalty rate (percentage of Net Sales or per-unit)":
        ["running royalty", "royalty rate", "net sales", "per unit",
         "per-unit"],
    "minimum annual royalty or guaranteed payment":
        ["minimum annual royalty", "minimum royalty", "guaranteed"],
    "milestone payments tied to development, regulatory, or commercial events":
        ["milestone"],
    "definition of Net Sales (deductions, returns, taxes)":
        ["net sales", "deduction", "return", "tax"],
    "royalty reporting cadence and royalty report contents":
        ["royalty report"],
    "currency, payment timing, and late payment interest":
        ["currency", "interest", "late payment"],
    # LICENSE — Restrictions
    "no reverse engineering, decompilation, or disassembly (for software)":
        ["reverse engineer", "decompil", "disassembl"],
    "no transfer, sublicense, or assignment outside the granted scope":
        ["transfer", "sublicens", "assign"],
    "no use outside the licensed field of use or territory":
        ["field", "territory"],
    "no creation of derivative works without Licensor consent (where applicable)":
        ["derivative work"],
    "no removal of proprietary notices or markings":
        ["proprietary notice", "marking"],
    "Licensor's reservation of all rights not expressly granted (no implied licenses)":
        ["reservation", "no implied", "all rights"],
    # LICENSE — Term and Termination
    "initial term and renewal options":
        ["initial term", "renewal"],
    "termination for material breach with cure period":
        ["material breach", "cure"],
    "termination for insolvency or bankruptcy":
        ["insolvency", "bankruptcy"],
    "Licensor's termination right for failure to meet milestones or minimum royalties":
        ["milestone", "minimum royalty", "termination"],
    "post-termination sell-off or wind-down period for inventory":
        ["sell-off", "sell off", "wind-down", "wind down", "inventory"],
    "survival of payment obligations, confidentiality, and indemnification":
        ["survival", "survive"],
    # LICENSE — Warranties / Indemnification
    "Licensor's warranty of title and right to grant the license":
        ["title", "warrant", "right to grant"],
    "Licensor's warranty (or disclaimer) of non-infringement of third-party IP":
        ["non-infringement", "infringement", "warrant", "disclaim"],
    "Licensor's indemnification of Licensee against third-party infringement claims arising out of the Licensed IP":
        ["indemnif", "infringement", "third-party", "third party"],
    "indemnification procedure (notice, control of defense, cooperation)":
        ["procedure", "control", "defense", "cooperation"],
    "Licensee's indemnification of Licensor for use outside the licensed scope":
        ["licensee", "indemnif", "outside the scope"],
    "limitation of liability and exclusion of consequential damages":
        ["limitation of liability", "consequential"],
    # LICENSE — Audit Rights
    "Licensee's obligation to maintain complete books and records of Net Sales and royalty calculations":
        ["books and records", "maintain", "net sales"],
    "Licensor's right to audit Licensee's books on reasonable notice":
        ["audit", "notice"],
    "audit frequency limitation (typically once per year)":
        ["once per year", "calendar year", "audit frequency", "per year"],
    "use of an independent certified public accountant":
        ["independent", "certified public accountant", "cpa", "accountant"],
    "Licensee's payment of audit costs if the underpayment exceeds a stated threshold (e.g. 5%)":
        ["underpayment", "5%", "five percent", "threshold", "audit cost"],
    "Licensee's payment of any underreported royalties plus interest":
        ["underreport", "interest"],
    # LICENSE — Improvements
    "definition of Improvements to the Licensed IP":
        ["improvement"],
    "ownership of Improvements made by Licensor (typically retained by Licensor and licensed to Licensee on the same terms)":
        ["improvement", "licensor", "own"],
    "ownership of Improvements made by Licensee (varies — sole, joint, or grant-back to Licensor)":
        ["improvement", "licensee", "own"],
    "grant-back license from Licensee to Licensor for Licensee's Improvements":
        ["grant-back", "grant back"],
    "Licensee's obligation to disclose Improvements to Licensor":
        ["disclose", "improvement"],
    # LICENSE — Quality Control
    "Licensor's quality standards for Licensee's use of the Licensed IP (especially trademarks)":
        ["quality standard", "quality control"],
    "Licensee's obligation to submit samples to Licensor for approval prior to use":
        ["sample", "approval"],
    "Licensor's right to inspect Licensee's facilities and operations":
        ["inspect", "facilities"],
    "Licensor's right to require corrective action for non-conforming use":
        ["corrective action", "non-conforming"],
    "naked-license avoidance language (Licensor exercises control to maintain trademark validity)":
        ["naked license", "control", "trademark"],
}


def filter_provisions_by_coverage(provisions: list[str], clause_body: str) -> list[str]:
    """Drop any provisions whose key terms don't appear in the clause body.

    This ensures the (brief, output) training pair is internally consistent —
    the brief only asks for things the body actually demonstrates. Without
    this filter, the model learns that lists in briefs are aspirational, not
    required, and produces undercovered outputs at inference time.

    Matching is lowercased and substring-based. If a provision has no known
    keywords in the global map, it is kept by default (safer than dropping
    unknown items).
    """
    if not clause_body:
        return provisions
    body_lower = clause_body.lower()
    kept = []
    for prov in provisions:
        keywords = _PROVISION_KEYWORDS.get(prov)
        if keywords is None:
            kept.append(prov)  # unknown provision — keep by default
            continue
        if any(kw.lower() in body_lower for kw in keywords):
            kept.append(prov)
    return kept


# ── Metadata-aware variation ────────────────────────────────────────────

def make_constrained_variation(
    *,
    metadata: dict,
    entity_map: dict,
    seed: int,
) -> dict:
    """Build a substitution map that respects the contract's metadata.

    The base variation is randomized from the global pools, but several
    fields are pinned to the original entity_map values when the metadata
    says they should be:

    - Non-USD contracts (e.g. Canadian banking) keep their full
      jurisdiction stack and currency-specific fee/implementation values.
    - Special customer types (state agencies, regulated banks, law firms)
      keep their customer name, entity type, address and state, since
      generic substitutes would clash with the bias / regulatory context
      embedded in the clause body.
    - Annual fee and implementation fee are drawn from the deal-size tier
      so an SMB contract never gets an enterprise fee.

    The provider state is preserved when it's non-Delaware (e.g. Canada),
    since the original contract was drafted with that jurisdiction in mind.
    """
    rng = random.Random(seed)
    base = generate_variation_map(seed)

    # 1. Currency / jurisdiction lock for non-USD contracts
    if metadata.get("currency") and metadata["currency"] != "USD":
        for k in (
            "PROVIDER_STATE",
            "PROVIDER_ADDRESS",
            "CUSTOMER_STATE",
            "CUSTOMER_ADDRESS",
            "GOVERNING_LAW_STATE",
            "VENUE_COUNTY",
            "VENUE_CITY",
            "ANNUAL_FEE_AMOUNT",
            "IMPLEMENTATION_FEE",
        ):
            if entity_map.get(k):
                base[k] = entity_map[k]

    else:
        # 2. USD contracts: constrain fee to deal-size tier
        deal_size = metadata.get("deal_size")
        if deal_size == "smb":
            base["ANNUAL_FEE_AMOUNT"] = rng.choice(SMB_FEES)
            base["IMPLEMENTATION_FEE"] = rng.choice(SMB_IMPL_FEES)
        elif deal_size == "mid_market":
            base["ANNUAL_FEE_AMOUNT"] = rng.choice(MID_MARKET_FEES)
            base["IMPLEMENTATION_FEE"] = rng.choice(MID_MARKET_IMPL_FEES)
        elif deal_size == "enterprise":
            base["ANNUAL_FEE_AMOUNT"] = rng.choice(ENTERPRISE_FEES)
            base["IMPLEMENTATION_FEE"] = rng.choice(ENTERPRISE_IMPL_FEES)

    # 3. Special customer types: keep original identity
    customer_type = metadata.get("customer_type")
    if customer_type in SPECIAL_CUSTOMER_TYPES:
        for k in (
            "CUSTOMER_NAME",
            "CUSTOMER_ENTITY_TYPE",
            "CUSTOMER_STATE",
            "CUSTOMER_ADDRESS",
            "GOVERNING_LAW_STATE",
            "VENUE_COUNTY",
            "VENUE_CITY",
        ):
            if entity_map.get(k):
                base[k] = entity_map[k]

    # 3a. EMPLOYMENT bucket: also pin the EMPLOYER (Provider) identity and
    # all employment-specific fields. The variation generator's random pools
    # don't cover EMPLOYEE_BASE_SALARY, EMPLOYEE_TITLE, etc., so without
    # this branch the brief would skip the compensation block. We also pin
    # PROVIDER_NAME because for employment, the employer's identity is part
    # of the contract's character (a "Pacific Crest Bank CEO" agreement
    # cannot become a "Fortis Cloud" agreement on every variation —
    # Lesson 6 in CONTRACT_GENERATION_RULES.md).
    is_employment = customer_type and any(
        t in customer_type for t in ("executive", "employee", "founder", "engineer")
    )
    if is_employment:
        employment_pins = (
            "PROVIDER_NAME",
            "PROVIDER_STATE",
            "PROVIDER_ENTITY_TYPE",
            "PROVIDER_ADDRESS",
            "TERM_YEARS",
            "EMPLOYEE_TITLE",
            "EMPLOYEE_REPORTS_TO",
            "EMPLOYEE_BASE_SALARY",
            "EMPLOYEE_TARGET_BONUS_PCT",
            "EMPLOYEE_INITIAL_EQUITY",
        )
        for k in employment_pins:
            if entity_map.get(k):
                base[k] = entity_map[k]

    # 3e. LICENSE bucket: pin LICENSE-specific placeholders.
    # License agreements introduce a set of placeholders (LICENSE_GRANT_TYPE,
    # LICENSE_FIELD_OF_USE, LICENSE_TERRITORY, LICENSE_TERM, LICENSE_UPFRONT_FEE,
    # LICENSE_ROYALTY_RATE, LICENSE_MIN_ANNUAL_ROYALTY, LICENSE_AUDIT_FREQUENCY,
    # LICENSE_SUBLICENSE_RIGHTS, LICENSED_IP_DESCRIPTION) that the variation
    # pools don't generate. We also clear SaaS/MSA-flavored vars that don't
    # apply to a standalone license agreement.
    is_license = (
        metadata.get("contract_type") == "LICENSE"
        or "LICENSE_GRANT_TYPE" in entity_map
        or "LICENSE_ROYALTY_RATE" in entity_map
        or "LICENSE_UPFRONT_FEE" in entity_map
    )
    if is_license:
        license_pins = (
            "LICENSE_GRANT_TYPE",
            "LICENSE_FIELD_OF_USE",
            "LICENSE_TERRITORY",
            "LICENSE_TERM",
            "LICENSE_UPFRONT_FEE",
            "LICENSE_ROYALTY_RATE",
            "LICENSE_MIN_ANNUAL_ROYALTY",
            "LICENSE_AUDIT_FREQUENCY",
            "LICENSE_SUBLICENSE_RIGHTS",
            "LICENSED_IP_DESCRIPTION",
        )
        for k in license_pins:
            if entity_map.get(k):
                base[k] = entity_map[k]
        # Clear SaaS/MSA vars that don't apply to a standalone license
        for k in ("ANNUAL_FEE_AMOUNT", "LIABILITY_CAP_MONTHS",
                  "IMPLEMENTATION_FEE"):
            base.pop(k, None)

    # 3d. INDEPENDENT_CONTRACTOR bucket: pin IC-specific placeholders.
    # IC agreements introduce a set of placeholders (IC_FEE_STRUCTURE,
    # IC_HOURLY_RATE, IC_FIXED_FEE, IC_MONTHLY_RETAINER, IC_HOURS_PER_MONTH_CAP,
    # IC_NOTICE_DAYS_CONVENIENCE, IC_REPORT_TO, IC_TERM_MONTHS) that the
    # variation pools don't generate. We also clear SaaS/MSA-flavored vars
    # that don't apply to IC standalone agreements.
    is_ic = (
        metadata.get("contract_type") == "INDEPENDENT_CONTRACTOR"
        or "IC_FEE_STRUCTURE" in entity_map
        or "IC_HOURLY_RATE" in entity_map
        or "IC_MONTHLY_RETAINER" in entity_map
    )
    if is_ic:
        ic_pins = (
            "IC_FEE_STRUCTURE",
            "IC_HOURLY_RATE",
            "IC_FIXED_FEE",
            "IC_MONTHLY_RETAINER",
            "IC_HOURS_PER_MONTH_CAP",
            "IC_NOTICE_DAYS_CONVENIENCE",
            "IC_REPORT_TO",
            "IC_TERM_MONTHS",
            "PROJECT_NAME",
            "PROJECT_START_DATE",
            "PROJECT_END_DATE",
        )
        for k in ic_pins:
            if entity_map.get(k):
                base[k] = entity_map[k]
        # Clear SaaS/MSA vars that don't apply to a standalone IC
        for k in ("ANNUAL_FEE_AMOUNT", "LIABILITY_CAP_MONTHS",
                  "IMPLEMENTATION_FEE"):
            base.pop(k, None)
        # Pin Contractor identity (the individual / single-person LLC) —
        # the variation generator otherwise renames the Contractor to a
        # generic SaaS company name on every variation, which destroys the
        # individual-Contractor framing.
        for k in ("CUSTOMER_NAME", "CUSTOMER_ENTITY_TYPE", "CUSTOMER_STATE",
                  "CUSTOMER_ADDRESS"):
            if entity_map.get(k):
                base[k] = entity_map[k]

    # 3c. DPA bucket: pin all DPA-specific placeholders from the entity_map.
    # DPAs introduce a set of placeholders (PARENT_AGREEMENT_DATE,
    # PARENT_AGREEMENT_NAME, DATA_RESIDENCY_REGION, BREACH_NOTIFICATION_HOURS,
    # AUDIT_NOTICE_DAYS, AUDIT_FREQUENCY) that the variation pools don't
    # generate. Without this branch the template would emit literal
    # {{PARENT_AGREEMENT_DATE}} placeholders.
    #
    # We also intentionally CLEAR the SaaS/MSA-flavored vars
    # (ANNUAL_FEE_AMOUNT, LIABILITY_CAP_MONTHS, IMPLEMENTATION_FEE,
    # TERM_YEARS) so they don't leak into the DPA brief — those concepts
    # live in the parent SaaS Agreement / MSA / BAA, not in the DPA itself.
    is_dpa = (
        metadata.get("contract_type") == "DPA"
        or "PARENT_AGREEMENT_DATE" in entity_map
        or "DATA_RESIDENCY_REGION" in entity_map
        or "BREACH_NOTIFICATION_HOURS" in entity_map
    )
    if is_dpa:
        dpa_pins = (
            "PARENT_AGREEMENT_DATE",
            "PARENT_AGREEMENT_NAME",
            "DATA_RESIDENCY_REGION",
            "BREACH_NOTIFICATION_HOURS",
            "AUDIT_NOTICE_DAYS",
            "AUDIT_FREQUENCY",
            "PROCESSING_PURPOSE",
            "DATA_CATEGORIES",
            "DATA_SUBJECT_CATEGORIES",
        )
        for k in dpa_pins:
            if entity_map.get(k):
                base[k] = entity_map[k]
        # Clear SaaS/MSA vars that don't apply to a DPA
        for k in ("ANNUAL_FEE_AMOUNT", "LIABILITY_CAP_MONTHS",
                  "IMPLEMENTATION_FEE", "TERM_YEARS"):
            base.pop(k, None)

    # 3b. SOW bucket: pin all SOW-specific placeholders from the entity_map.
    # SOWs introduce a set of placeholders (SOW_FEE_AMOUNT, MSA_EFFECTIVE_DATE,
    # PROJECT_NAME, PROJECT_START_DATE, PROJECT_END_DATE,
    # CHANGE_ORDER_THRESHOLD, EXPENSE_PREAPPROVAL_THRESHOLD) that the variation
    # pools (generate_variation_map) don't generate. Without this branch the
    # template would emit literal {{SOW_FEE_AMOUNT}} placeholders.
    #
    # We also intentionally CLEAR the SaaS/MSA-flavored vars (ANNUAL_FEE_AMOUNT,
    # LIABILITY_CAP_MONTHS, IMPLEMENTATION_FEE) so they don't leak into the
    # SOW brief — those concepts live in the parent MSA, not in any individual
    # SOW. TERM_YEARS and NOTICE_DAYS are also wrong frame for SOW (the SOW has
    # PROJECT_START_DATE / PROJECT_END_DATE instead of an annual term, and any
    # termination notice is governed by the parent MSA).
    is_sow = (
        metadata.get("contract_type") == "SOW"
        or "PROJECT_NAME" in entity_map
        or "SOW_FEE_AMOUNT" in entity_map
    )
    if is_sow:
        sow_pins = (
            "SOW_FEE_AMOUNT",
            "MSA_EFFECTIVE_DATE",
            "PROJECT_NAME",
            "PROJECT_START_DATE",
            "PROJECT_END_DATE",
            "CHANGE_ORDER_THRESHOLD",
            "EXPENSE_PREAPPROVAL_THRESHOLD",
        )
        for k in sow_pins:
            if entity_map.get(k):
                base[k] = entity_map[k]
        # Clear SaaS/MSA vars that don't apply to an SOW so they don't show up
        # in the commercial_block (they live in the parent MSA, not in an SOW).
        for k in ("ANNUAL_FEE_AMOUNT", "LIABILITY_CAP_MONTHS",
                  "IMPLEMENTATION_FEE", "TERM_YEARS"):
            base.pop(k, None)

    # 4. Honour the original effective date if metadata is silent — variation
    #    dates are otherwise random and decoupled from the contract's intent.
    if entity_map.get("EFFECTIVE_DATE"):
        # 50/50: use original or use a fresh random one (for diversity)
        if rng.random() < 0.5:
            base["EFFECTIVE_DATE"] = entity_map["EFFECTIVE_DATE"]

    return base


# ── Example generation ──────────────────────────────────────────────────

def select_clause_types(metadata: dict, available: set[str], contract_type: str = "SAAS") -> list[str]:
    """Pick clause types per contract, with contract-type-aware logic.

    For SaaS/MSA: 3 universal clauses (IP, Liability, Termination) + 1 rotating
    based on the primary_clause_focus metadata.

    For NDAs: NDAs rarely have an IP article or a Limitation of Liability
    article in the SaaS sense. The core NDA clauses are Confidentiality,
    Purpose, Exclusions, Return/Destruction, Term/Survival, and Equitable
    Remedies. We pick 4 of these per contract, biased by primary_clause_focus.
    """
    focus = metadata.get("primary_clause_focus", []) or []

    if contract_type == "NDA":
        # Universal NDA clauses that every NDA should teach
        nda_universal = ["confidentiality", "term_survival", "equitable_remedies"]
        selected = [c for c in nda_universal if c in available]

        # Rotating NDA clause based on primary focus
        nda_focus_to_key = {
            "confidentiality": "confidentiality",
            "purpose": "purpose",
            "exclusions": "exclusions",
            "compelled_disclosure": "compelled_disclosure",
            "return_or_destruction": "return_or_destruction",
            "term_survival": "term_survival",
            "no_license": "no_license",
            "equitable_remedies": "equitable_remedies",
        }
        rotating_candidates = []
        for f in focus:
            key = nda_focus_to_key.get(f.lower())
            if key and key in available and key not in selected:
                rotating_candidates.append(key)

        # Fallback: pick whichever NDA-specific clause is available
        if not rotating_candidates:
            for k in ["purpose", "exclusions", "return_or_destruction",
                      "compelled_disclosure", "no_license"]:
                if k in available and k not in selected:
                    rotating_candidates.append(k)
                    break

        if rotating_candidates:
            selected.append(rotating_candidates[0])

        return selected

    if contract_type == "LICENSE":
        # License agreements grant rights in IP. Every meaningful license
        # contract has the Grant clause (what's licensed and how broadly),
        # Royalties/Payments (how the licensor gets paid), Term/Termination
        # (when the rights end), and Confidentiality. The rotating slot
        # rotates among Restrictions, Warranties/Infringement Indemnification,
        # Audit Rights, Improvements, and Quality Control. (Restrictions is
        # NOT in the universal set because pharma deals typically fold
        # restriction language into the Grant article rather than carving it
        # out as a standalone article.)
        license_universal = ["license_grant", "license_royalties_payment",
                             "license_term_termination", "confidentiality"]
        selected = [c for c in license_universal if c in available]

        license_focus_to_key = {
            "grant": "license_grant",
            "license_grant": "license_grant",
            "scope": "license_grant",
            "exclusivity": "license_grant",
            "royalties": "license_royalties_payment",
            "license_royalties_payment": "license_royalties_payment",
            "payment": "license_royalties_payment",
            "fees": "license_royalties_payment",
            "restrictions": "license_restrictions",
            "license_restrictions": "license_restrictions",
            "term": "license_term_termination",
            "termination": "license_term_termination",
            "license_term_termination": "license_term_termination",
            "warranties": "license_warranties_indemnification",
            "indemnification": "license_warranties_indemnification",
            "license_warranties_indemnification": "license_warranties_indemnification",
            "infringement": "license_warranties_indemnification",
            "audit": "license_audit_rights",
            "audit rights": "license_audit_rights",
            "license_audit_rights": "license_audit_rights",
            "royalty audit": "license_audit_rights",
            "improvements": "license_improvements",
            "license_improvements": "license_improvements",
            "grant-back": "license_improvements",
            "quality control": "license_quality_control",
            "license_quality_control": "license_quality_control",
            "brand standards": "license_quality_control",
            "confidentiality": "confidentiality",
        }
        rotating_candidates = []
        for f in focus:
            key = license_focus_to_key.get(f.lower())
            if key and key in available and key not in selected:
                rotating_candidates.append(key)

        # Fallback: pick whichever LICENSE-specific clause is available
        if not rotating_candidates:
            for k in ["license_restrictions",
                      "license_warranties_indemnification",
                      "license_audit_rights", "license_improvements",
                      "license_quality_control"]:
                if k in available and k not in selected:
                    rotating_candidates.append(k)
                    break

        if rotating_candidates:
            selected.append(rotating_candidates[0])

        return selected

    if contract_type == "INDEPENDENT_CONTRACTOR":
        # IC agreements are standalone (not addenda). Every meaningful IC
        # contract has Engagement, Compensation, IC Classification (the
        # 1099/W-9/no-benefits anchor), and Confidentiality. The rotating
        # slot rotates among IC IP, IC Termination, IC Restrictive Covenants,
        # Indemnification, Insurance, and Dispute Resolution.
        ic_universal = ["ic_engagement", "ic_compensation",
                        "ic_classification", "confidentiality"]
        selected = [c for c in ic_universal if c in available]

        ic_focus_to_key = {
            "engagement": "ic_engagement",
            "ic_engagement": "ic_engagement",
            "compensation": "ic_compensation",
            "ic_compensation": "ic_compensation",
            "fees": "ic_compensation",
            "classification": "ic_classification",
            "ic_classification": "ic_classification",
            "worker classification": "ic_classification",
            "intellectual property": "ic_intellectual_property",
            "ic_intellectual_property": "ic_intellectual_property",
            "ip": "ic_intellectual_property",
            "work product": "ic_intellectual_property",
            "termination": "ic_termination",
            "ic_termination": "ic_termination",
            "restrictive covenants": "ic_restrictive_covenants",
            "ic_restrictive_covenants": "ic_restrictive_covenants",
            "non-solicit": "ic_restrictive_covenants",
            "conflicts of interest": "ic_restrictive_covenants",
            "indemnification": "indemnification",
            "indemnity": "indemnification",
            "insurance": "insurance",
            "dispute resolution": "dispute_resolution",
            "dispute_resolution": "dispute_resolution",
            "arbitration": "dispute_resolution",
            "confidentiality": "confidentiality",
        }
        rotating_candidates = []
        for f in focus:
            key = ic_focus_to_key.get(f.lower())
            if key and key in available and key not in selected:
                rotating_candidates.append(key)

        # Fallback: pick whichever IC-specific clause is available
        if not rotating_candidates:
            for k in ["ic_intellectual_property", "ic_termination",
                      "ic_restrictive_covenants", "indemnification",
                      "insurance", "dispute_resolution"]:
                if k in available and k not in selected:
                    rotating_candidates.append(k)
                    break

        if rotating_candidates:
            selected.append(rotating_candidates[0])

        return selected

    if contract_type == "DPA":
        # DPAs operate UNDER a parent SaaS Agreement / MSA / BAA. Every
        # meaningful DPA must address the controller/processor relationship,
        # the scope of processing, security measures, and sub-processors.
        # The rotating slot rotates among the GDPR Article 28-driven topics
        # that vary by deal context (breach notification, international
        # transfers, data subject rights, DPIA cooperation, deletion/return,
        # audit rights).
        dpa_universal = ["controller_processor_roles", "processing_scope",
                         "security_measures", "subprocessors"]
        selected = [c for c in dpa_universal if c in available]

        dpa_focus_to_key = {
            "controller_processor_roles": "controller_processor_roles",
            "roles": "controller_processor_roles",
            "processing scope": "processing_scope",
            "processing_scope": "processing_scope",
            "scope of processing": "processing_scope",
            "security measures": "security_measures",
            "security_measures": "security_measures",
            "toms": "security_measures",
            "subprocessors": "subprocessors",
            "sub-processors": "subprocessors",
            "breach notification": "breach_notification",
            "breach_notification": "breach_notification",
            "incident notification": "breach_notification",
            "international transfers": "international_transfers",
            "international_transfers": "international_transfers",
            "sccs": "international_transfers",
            "data subject rights": "data_subject_rights",
            "data_subject_rights": "data_subject_rights",
            "dsr": "data_subject_rights",
            "dpia": "dpia_cooperation",
            "dpia_cooperation": "dpia_cooperation",
            "impact assessment": "dpia_cooperation",
            "deletion": "deletion_return",
            "deletion_return": "deletion_return",
            "return of data": "deletion_return",
            "audit rights": "audit_rights",
            "audit_rights": "audit_rights",
            "audits": "audit_rights",
        }
        rotating_candidates = []
        for f in focus:
            key = dpa_focus_to_key.get(f.lower())
            if key and key in available and key not in selected:
                rotating_candidates.append(key)

        # Fallback: pick whichever DPA-specific clause is available
        if not rotating_candidates:
            for k in ["breach_notification", "international_transfers",
                      "data_subject_rights", "deletion_return",
                      "audit_rights", "dpia_cooperation"]:
                if k in available and k not in selected:
                    rotating_candidates.append(k)
                    break

        if rotating_candidates:
            selected.append(rotating_candidates[0])

        return selected

    if contract_type == "SOW":
        # SOWs are project-specific and operate UNDER a parent MSA. The core
        # SOW clauses every meaningful SOW must address are: scope (what work),
        # deliverables/acceptance (what artifacts and how they're approved),
        # fees/payment (how it gets paid), and milestones (when). The rotating
        # slot rotates among change control, client responsibilities,
        # assumptions/dependencies, personnel, and subcontractors based on
        # primary_clause_focus.
        sow_universal = ["sow_scope", "deliverables_acceptance",
                         "fees_payment_sow", "milestones_schedule"]
        selected = [c for c in sow_universal if c in available]

        sow_focus_to_key = {
            "scope": "sow_scope",
            "sow_scope": "sow_scope",
            "deliverables": "deliverables_acceptance",
            "deliverables_acceptance": "deliverables_acceptance",
            "acceptance": "deliverables_acceptance",
            "milestones": "milestones_schedule",
            "milestones_schedule": "milestones_schedule",
            "schedule": "milestones_schedule",
            "fees": "fees_payment_sow",
            "fees_payment": "fees_payment_sow",
            "fees_payment_sow": "fees_payment_sow",
            "payment": "fees_payment_sow",
            "change control": "change_control",
            "change_control": "change_control",
            "change orders": "change_control",
            "client responsibilities": "client_responsibilities",
            "client_responsibilities": "client_responsibilities",
            "customer responsibilities": "client_responsibilities",
            "assumptions": "assumptions_dependencies",
            "assumptions_dependencies": "assumptions_dependencies",
            "dependencies": "assumptions_dependencies",
            "personnel": "personnel",
            "key personnel": "personnel",
            "subcontractors": "subcontractors",
        }
        rotating_candidates = []
        for f in focus:
            key = sow_focus_to_key.get(f.lower())
            if key and key in available and key not in selected:
                rotating_candidates.append(key)

        # Fallback: pick whichever SOW-specific clause is available
        if not rotating_candidates:
            for k in ["change_control", "client_responsibilities",
                      "assumptions_dependencies", "personnel", "subcontractors"]:
                if k in available and k not in selected:
                    rotating_candidates.append(k)
                    break

        if rotating_candidates:
            selected.append(rotating_candidates[0])

        return selected

    if contract_type == "EMPLOYMENT":
        # Employment agreements have a different "universal" set than SaaS/MSA.
        # Every meaningful employment agreement has Compensation, Termination,
        # and Severance language; the rotating slot rotates among the
        # employment-specific topics that vary by contract focus.
        emp_universal = ["compensation", "termination", "severance"]
        selected = [c for c in emp_universal if c in available]

        emp_focus_to_key = {
            "compensation": "compensation",
            "severance": "severance",
            "restrictive covenants": "restrictive_covenants",
            "restrictive_covenants": "restrictive_covenants",
            "non-compete": "restrictive_covenants",
            "change in control": "change_in_control",
            "change_in_control": "change_in_control",
            "position": "position_duties",
            "position_duties": "position_duties",
            "duties": "position_duties",
            "confidentiality": "confidentiality",
            "intellectual property": "intellectual_property",
            "intellectual_property": "intellectual_property",
            "ip": "intellectual_property",
            "dispute resolution": "dispute_resolution",
            "dispute_resolution": "dispute_resolution",
            "arbitration": "dispute_resolution",
        }
        rotating_candidates = []
        for f in focus:
            key = emp_focus_to_key.get(f.lower())
            if key and key in available and key not in selected:
                rotating_candidates.append(key)

        # Fallback: pick whichever employment-specific clause is available
        if not rotating_candidates:
            for k in ["restrictive_covenants", "change_in_control",
                       "position_duties", "intellectual_property",
                       "confidentiality", "dispute_resolution"]:
                if k in available and k not in selected:
                    rotating_candidates.append(k)
                    break

        if rotating_candidates:
            selected.append(rotating_candidates[0])

        return selected

    # SaaS / MSA / other: keep the original 3-universal + 1-rotating logic
    universal = ["intellectual_property", "limitation_of_liability", "termination"]
    selected = [c for c in universal if c in available]

    focus_to_key = {
        "intellectual property": "intellectual_property",
        "limitation of liability": "limitation_of_liability",
        "termination": "termination",
        "indemnity": "indemnification",
        "data privacy": "data_privacy",
        "service level": "service_level",
        "confidentiality": "confidentiality",
    }

    rotating_candidates = []
    for f in focus:
        key = focus_to_key.get(f.lower())
        if key and key in available and key not in selected:
            rotating_candidates.append(key)

    if not rotating_candidates:
        for k in ["data_privacy", "service_level", "confidentiality", "indemnification"]:
            if k in available and k not in selected:
                rotating_candidates.append(k)
                break

    if rotating_candidates:
        selected.append(rotating_candidates[0])

    return selected


def process_contract(
    *,
    contract_type: str,
    template_path: Path,
    entity_map: dict,
    metadata: dict,
) -> list[dict]:
    """Generate drafting examples for one tokenized contract."""
    template = template_path.read_text(encoding="utf-8")
    base_name = template_path.stem

    clauses = extract_clauses(template)
    if not clauses:
        logger.warning("  no clauses extracted from %s", base_name)
        return []

    available = set(clauses.keys())
    chosen_clauses = select_clause_types(metadata, available, contract_type=contract_type)
    if not chosen_clauses:
        logger.warning("  no clause types selected for %s (available: %s)", base_name, available)
        return []

    # Find which placeholders this template actually uses (so we don't include
    # irrelevant slots in the substitution).
    placeholders_used = set(re.findall(r"\{\{([A-Z_]+)\}\}", template))

    examples = []
    for i, clause_type in enumerate(chosen_clauses, start=1):
        clause = clauses[clause_type]

        # Deterministic seed per (contract, clause) pair so reruns are stable.
        seed = abs(hash(f"{base_name}::{clause_type}::{i}")) % (2**32)
        sub_map = make_constrained_variation(
            metadata=metadata,
            entity_map=entity_map,
            seed=seed,
        )

        # Restrict to placeholders the template actually uses
        active_subs = {k: v for k, v in sub_map.items() if k in placeholders_used}

        # Substitute into the clause body to get the OUTPUT
        output_clause = apply_substitutions(clause["body"], active_subs)

        # Sanity check: warn on leftover placeholders, but still emit the
        # example — the few unsubstituted ones in the original template are
        # the model's failure mode we want to teach against.
        leftover = re.findall(r"\{\{([A-Z_]+)\}\}", output_clause)
        if leftover:
            logger.debug("  leftover placeholders in %s/%s: %s", base_name, clause_type, set(leftover))

        # Build the brief using the FULL sub_map (so the brief mentions all
        # the slot values, not just the ones in this particular clause).
        # This is intentional — a real user would describe the deal at the
        # contract level, not at the clause level.
        brief = build_brief(
            contract_type=contract_type,
            clause_type=clause_type,
            sub_map=sub_map,
            metadata=metadata,
            style_seed=seed,
            clause_body=output_clause,
        )

        examples.append({
            "task": "drafting",
            "contract_type": contract_type,
            "clause_type": clause_type,
            "source_template": base_name,
            "source_article_title": clause["title"],
            "source_article_number": clause["article_number"],
            "instruction": brief,
            "output": output_clause,
            "metadata": {
                "industry": metadata.get("industry"),
                "deal_size": metadata.get("deal_size"),
                "negotiation_bias": metadata.get("negotiation_bias"),
                "drafting_style": metadata.get("drafting_style"),
                "regulatory_frameworks": metadata.get("regulatory_frameworks", []),
                "provider_name": sub_map.get("PROVIDER_NAME"),
                "customer_name": sub_map.get("CUSTOMER_NAME"),
                "governing_law_state": sub_map.get("GOVERNING_LAW_STATE")
                    or sub_map.get("CUSTOMER_STATE"),
                "annual_fee": sub_map.get("ANNUAL_FEE_AMOUNT"),
            },
        })

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="SAAS",
                        help="Contract type (SAAS, MSA, NDA, ...)")
    parser.add_argument("--variations-per-contract", type=int, default=4,
                        help="Reserved for future expansion (currently 4 hard-coded based on clause selection)")
    args = parser.parse_args()

    contract_type = args.type
    tokenized_dir = V2_TOKENIZED / contract_type / "synthetic"
    metadata_file = V2_RAW / contract_type / "synthetic" / "contract_metadata.json"
    entity_map_file = V2_RAW / contract_type / "synthetic" / "entity_maps.json"

    if not tokenized_dir.exists():
        logger.error("Tokenized dir not found: %s", tokenized_dir)
        return
    if not metadata_file.exists():
        logger.error("Metadata file not found: %s", metadata_file)
        return
    if not entity_map_file.exists():
        logger.error("Entity map file not found: %s", entity_map_file)
        return

    metadata_all = json.loads(metadata_file.read_text())
    entity_maps_all = json.loads(entity_map_file.read_text())

    templates = sorted(tokenized_dir.glob("*.txt"))
    if not templates:
        logger.error("No tokenized templates found in %s", tokenized_dir)
        return

    output_dir = V2_TRAINING
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{contract_type}_drafting_examples.jsonl"

    all_examples = []
    per_contract = []
    per_clause_type = {}

    for template_path in templates:
        base = template_path.stem
        if base.startswith("_") or base.upper() == "MANIFEST":
            continue
        meta = metadata_all.get(base)
        ent = entity_maps_all.get(base)
        if not meta:
            logger.warning("No metadata for %s — skipping", base)
            continue
        if not ent:
            logger.warning("No entity map for %s — skipping", base)
            continue

        examples = process_contract(
            contract_type=contract_type,
            template_path=template_path,
            entity_map=ent,
            metadata=meta,
        )
        all_examples.extend(examples)
        per_contract.append({
            "template": base,
            "industry": meta.get("industry"),
            "deal_size": meta.get("deal_size"),
            "examples": len(examples),
            "clause_types": [e["clause_type"] for e in examples],
        })
        for e in examples:
            per_clause_type[e["clause_type"]] = per_clause_type.get(e["clause_type"], 0) + 1
        logger.info("✓ %-58s %d examples (%s)",
                    base[:58], len(examples),
                    ", ".join(e["clause_type"] for e in examples))

    # Write JSONL
    with output_path.open("w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Write a summary manifest next to the JSONL
    manifest = {
        "type": contract_type,
        "task": "drafting",
        "total_examples": len(all_examples),
        "total_contracts": len(per_contract),
        "per_clause_type_counts": per_clause_type,
        "contracts": per_contract,
    }
    manifest_path = output_dir / f"{contract_type}_drafting_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("\n=== DRAFTING EXAMPLES GENERATED ===")
    logger.info("Type:           %s", contract_type)
    logger.info("Contracts:      %d", len(per_contract))
    logger.info("Total examples: %d", len(all_examples))
    logger.info("By clause type: %s", per_clause_type)
    logger.info("Output:         %s", output_path)
    logger.info("Manifest:       %s", manifest_path)


if __name__ == "__main__":
    main()
