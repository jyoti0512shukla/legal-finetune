#!/usr/bin/env python3
"""V2 contract synthesis — generate full SaaS contracts via OpenAI GPT-4o.

Used to fill diversity gaps that EDGAR/CUAD can't satisfy:
  - Modern (2024+) contracts
  - Specific industries (fintech, healthcare, etc.)
  - Specific company sizes (enterprise vs SMB)
  - Specific commercial terms (SLA, support tiers, billing models)

Usage:
    OPENAI_API_KEY=sk-... python scripts/v2_synthesize_contracts.py --type SAAS --count 2
"""

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("/Users/jyotimishra/legal-finetune/data/v2/raw")

MODEL = "gpt-4o"


# ── Diversity specifications ──────────────────────────────────────────────

@dataclass
class ContractSpec:
    """Concrete spec for a single contract to synthesize."""
    contract_type: str
    provider_name: str
    customer_name: str
    industry: str
    deal_size: str  # enterprise | mid_market | smb
    contract_value: str
    jurisdiction: str
    governing_law: str
    bias: str  # pro_vendor | balanced | pro_customer
    extra_terms: list[str] = field(default_factory=list)


# Diverse SaaS specs — designed to cover gaps in our real-data corpus
# 15 specs spanning enterprise/mid/SMB × CA/NY/DE/TX × pro-vendor/balanced/pro-customer
# × fintech/healthcare/retail/manufacturing/edtech/devtools/security/HR/marketing
SAAS_SPECS = [
    # 1. Enterprise + Retail + CA + balanced
    ContractSpec(
        contract_type="SAAS",
        provider_name="DataPilot Analytics, Inc.",
        customer_name="MidWest Retail Group, LLC",
        industry="Retail Analytics",
        deal_size="enterprise",
        contract_value="USD 480,000 annually ($40K/month), 3-year term with 12-month auto-renewal",
        jurisdiction="San Francisco, California",
        governing_law="laws of the State of California",
        bias="balanced",
        extra_terms=[
            "99.9% monthly uptime SLA with tiered service credits (5%/10%/25%)",
            "24x7 critical support with 1-hour first response",
            "$25,000 one-time implementation fee",
            "Provider retains all platform IP; Customer receives limited license",
            "SOC 2 Type II certified, CCPA-compliant, US data residency",
            "Liability cap at 12 months fees with IP/negligence/confidentiality carve-outs",
        ],
    ),
    # 2. Mid-market + Healthcare + NY + pro-customer
    ContractSpec(
        contract_type="SAAS",
        provider_name="ClinicFlow Health Systems, Inc.",
        customer_name="Regional Medical Group, P.C.",
        industry="Healthcare SaaS (HIPAA)",
        deal_size="mid_market",
        contract_value="USD 96,000 annually, 2-year term with auto-renewal",
        jurisdiction="New York, New York",
        governing_law="laws of the State of New York",
        bias="pro_customer",
        extra_terms=[
            "99.95% uptime SLA (clinical environment)",
            "HIPAA BAA exhibit, HITECH compliance",
            "30-min critical response, 4-hour resolution target",
            "Per-user pricing: 50 users included, $200/user/month additional",
            "Annual customer audit rights",
            "Liability cap at 24 months fees",
        ],
    ),
    # 3. Enterprise + Fintech + DE + pro-vendor
    ContractSpec(
        contract_type="SAAS",
        provider_name="LedgerCore Financial Software, Inc.",
        customer_name="Apex Capital Markets LLC",
        industry="Fintech / Trading Platform",
        deal_size="enterprise",
        contract_value="USD 2,400,000 annually, 5-year term",
        jurisdiction="Wilmington, Delaware",
        governing_law="laws of the State of Delaware",
        bias="pro_vendor",
        extra_terms=[
            "99.99% uptime SLA (mission-critical financial)",
            "Provider sole discretion on feature roadmap and deprecation",
            "AS-IS warranty disclaimer except for express limited warranties",
            "FINRA/SEC compliance certifications",
            "$150K one-time implementation fee",
            "Liability cap at 12 months fees, no super-cap exceptions except for IP indemnity",
            "No termination for convenience by Customer in first 36 months",
            "Auto-escalation 5% annually after first year",
        ],
    ),
    # 4. SMB + DevTools + CA + balanced
    ContractSpec(
        contract_type="SAAS",
        provider_name="BuildKit Cloud, Inc.",
        customer_name="Stellar Software Studio, Inc.",
        industry="Developer Tools / CI-CD",
        deal_size="smb",
        contract_value="USD 24,000 annually ($2K/month)",
        jurisdiction="San Francisco, California",
        governing_law="laws of the State of California",
        bias="balanced",
        extra_terms=[
            "99.5% uptime SLA (sufficient for build tools)",
            "Email and chat support, business hours response",
            "Self-service onboarding (no implementation fee)",
            "Per-developer-seat pricing model",
            "Customer Data backup retained 30 days post-termination",
            "Liability cap at 12 months fees",
            "30-day termination for convenience either side",
        ],
    ),
    # 5. Mid-market + EdTech + TX + balanced
    ContractSpec(
        contract_type="SAAS",
        provider_name="LearnPath Academy, Inc.",
        customer_name="Texas State University System",
        industry="Education Technology / LMS",
        deal_size="mid_market",
        contract_value="USD 180,000 annually for 5,000 student licenses",
        jurisdiction="Austin, Texas",
        governing_law="laws of the State of Texas",
        bias="pro_customer",
        extra_terms=[
            "FERPA-compliant student data handling",
            "99.9% uptime during academic terms (less strict during breaks)",
            "Single sign-on integration via SAML",
            "WCAG 2.1 AA accessibility compliance warranty",
            "Texas state agency contract terms incorporated by reference",
            "Liability cap at 12 months fees",
            "Termination for convenience with 90 days notice",
        ],
    ),
    # 6. Enterprise + Manufacturing + IL + pro-vendor
    ContractSpec(
        contract_type="SAAS",
        provider_name="FactoryMind Industrial IoT, Inc.",
        customer_name="Northern Tooling & Manufacturing Co.",
        industry="Industrial IoT / Manufacturing",
        deal_size="enterprise",
        contract_value="USD 720,000 annually ($60K/month), 3-year term",
        jurisdiction="Chicago, Illinois",
        governing_law="laws of the State of Illinois",
        bias="pro_vendor",
        extra_terms=[
            "99.9% uptime SLA, scheduled maintenance excluded",
            "On-prem edge agent + cloud platform hybrid deployment",
            "$100,000 one-time implementation including site survey",
            "Provider owns all platform improvements derived from anonymized usage data",
            "Customer Data may be aggregated and anonymized for benchmarking",
            "Liability cap at 12 months fees",
            "Force majeure includes supply chain disruptions",
        ],
    ),
    # 7. SMB + Marketing + CA + balanced
    ContractSpec(
        contract_type="SAAS",
        provider_name="PixelBoost Marketing Cloud, Inc.",
        customer_name="Bright Beverages Co., LLC",
        industry="Marketing Automation",
        deal_size="smb",
        contract_value="USD 36,000 annually ($3K/month), 1-year term",
        jurisdiction="Los Angeles, California",
        governing_law="laws of the State of California",
        bias="balanced",
        extra_terms=[
            "99.5% uptime SLA",
            "Per-contact pricing tier (50K contacts included)",
            "Email and live chat support, 24-hour business response",
            "GDPR Article 28 data processing terms",
            "Customer can export data at any time",
            "Liability cap at 12 months fees",
        ],
    ),
    # 8. Enterprise + Security + NY + pro-customer
    ContractSpec(
        contract_type="SAAS",
        provider_name="ShieldNet Cyber Security, Inc.",
        customer_name="GlobalBank Holdings, N.A.",
        industry="Cybersecurity / Threat Detection",
        deal_size="enterprise",
        contract_value="USD 1,500,000 annually, 3-year term",
        jurisdiction="New York, New York",
        governing_law="laws of the State of New York",
        bias="pro_customer",
        extra_terms=[
            "99.99% uptime SLA",
            "15-minute response for critical security incidents",
            "Provider personnel must complete background checks",
            "Right to terminate immediately upon material security breach by Provider",
            "$5M cyber insurance maintained by Provider",
            "Liability cap at 36 months fees with carve-outs for security breaches and IP indemnity",
            "Customer right to perform annual penetration testing of Provider's systems",
            "Mandatory breach notification within 24 hours",
        ],
    ),
    # 9. Mid-market + HR Tech + DE + balanced
    ContractSpec(
        contract_type="SAAS",
        provider_name="PeopleHub HR Suite, Inc.",
        customer_name="Westbrook Consulting Group, LLC",
        industry="HR Tech / Workforce Management",
        deal_size="mid_market",
        contract_value="USD 60,000 annually ($5K/month), 2-year term",
        jurisdiction="Wilmington, Delaware",
        governing_law="laws of the State of Delaware",
        bias="balanced",
        extra_terms=[
            "99.9% uptime SLA",
            "Per-employee pricing: $10/employee/month, 500 employees minimum",
            "Payroll module integration with ADP and Gusto",
            "EEO-1 reporting features included",
            "Employee data treated as Customer Data — Customer is data controller",
            "Liability cap at 12 months fees",
        ],
    ),
    # 10. Enterprise + Logistics + TX + pro-vendor
    ContractSpec(
        contract_type="SAAS",
        provider_name="RouteOptima Freight Software, Inc.",
        customer_name="Lonestar Trucking Holdings, Inc.",
        industry="Logistics / Fleet Management",
        deal_size="enterprise",
        contract_value="USD 360,000 annually for 200 vehicle licenses",
        jurisdiction="Dallas, Texas",
        governing_law="laws of the State of Texas",
        bias="pro_vendor",
        extra_terms=[
            "99.5% uptime (carve-out for satellite/cellular outages outside Provider control)",
            "Per-vehicle subscription tier",
            "Hardware (telematics device) sold separately under separate purchase order",
            "Provider sole discretion on supported vehicle makes/models",
            "Liability cap at 6 months fees (lower than standard)",
            "Termination for convenience prohibited in first 24 months",
        ],
    ),
    # 11. SMB + Legal Tech + NY + balanced
    ContractSpec(
        contract_type="SAAS",
        provider_name="ContractIQ Legal Tech, Inc.",
        customer_name="Hartman & Associates LLP",
        industry="Legal Tech / Contract Management",
        deal_size="smb",
        contract_value="USD 18,000 annually ($1.5K/month) for 10 attorneys",
        jurisdiction="New York, New York",
        governing_law="laws of the State of New York",
        bias="balanced",
        extra_terms=[
            "99.5% uptime SLA",
            "Attorney work product treated as privileged and confidential",
            "Provider obligated to maintain attorney-client privilege protections",
            "No use of Customer Data for any AI training or product improvement",
            "Liability cap at 12 months fees",
            "Customer Data deleted within 7 days of termination",
        ],
    ),
    # 12. Enterprise + Real Estate + FL + balanced
    ContractSpec(
        contract_type="SAAS",
        provider_name="PropertyPulse Real Estate Cloud, Inc.",
        customer_name="Sunshine Realty Group, LLC",
        industry="Real Estate / Property Management",
        deal_size="enterprise",
        contract_value="USD 240,000 annually, 3-year term",
        jurisdiction="Miami, Florida",
        governing_law="laws of the State of Florida",
        bias="balanced",
        extra_terms=[
            "99.9% uptime SLA",
            "Multi-tenant architecture with logical data isolation",
            "Florida real estate license number required for Authorized Users",
            "$50,000 one-time implementation including data migration",
            "Liability cap at 12 months fees",
            "Force majeure explicitly includes hurricanes and named storms",
        ],
    ),
    # 13. Mid-market + Insurance Tech + IL + pro-customer
    ContractSpec(
        contract_type="SAAS",
        provider_name="ClaimsFlow Insurance Platform, Inc.",
        customer_name="Heartland Mutual Insurance Co.",
        industry="Insurance Technology / Claims Processing",
        deal_size="mid_market",
        contract_value="USD 144,000 annually, 2-year term",
        jurisdiction="Chicago, Illinois",
        governing_law="laws of the State of Illinois",
        bias="pro_customer",
        extra_terms=[
            "99.95% uptime SLA",
            "State insurance regulator audit rights pass-through to Provider",
            "PCI-DSS Level 1 compliance",
            "Customer right to terminate immediately if Provider loses regulatory certification",
            "Liability cap at 18 months fees with carve-outs for regulatory penalties",
            "Mandatory data residency in US Midwest data centers only",
        ],
    ),
    # 14. SMB + Restaurant Tech + GA + balanced
    ContractSpec(
        contract_type="SAAS",
        provider_name="MenuMaster POS Cloud, Inc.",
        customer_name="Peachtree Hospitality Group, LLC",
        industry="Restaurant POS / Hospitality",
        deal_size="smb",
        contract_value="USD 12,000 annually ($1K/month) for 8 locations",
        jurisdiction="Atlanta, Georgia",
        governing_law="laws of the State of Georgia",
        bias="balanced",
        extra_terms=[
            "99.5% uptime SLA",
            "24x7 phone support (restaurants operate during off-hours)",
            "Per-location pricing model",
            "Hardware leasing optional under separate agreement",
            "PCI-DSS compliant payment processing",
            "Liability cap at 12 months fees",
            "30-day termination for convenience",
        ],
    ),
    # 15. Enterprise + Pharma + NJ + pro-customer
    ContractSpec(
        contract_type="SAAS",
        provider_name="PharmaTrack Clinical Data Systems, Inc.",
        customer_name="Meridian Pharmaceuticals, Inc.",
        industry="Pharma / Clinical Trial Management",
        deal_size="enterprise",
        contract_value="USD 960,000 annually, 4-year term",
        jurisdiction="Newark, New Jersey",
        governing_law="laws of the State of New Jersey",
        bias="pro_customer",
        extra_terms=[
            "99.99% uptime SLA",
            "21 CFR Part 11 compliant electronic records and signatures",
            "FDA inspection support obligations",
            "Validated environment with IQ/OQ/PQ documentation",
            "GxP audit trail features",
            "Customer can audit Provider's quality management system annually",
            "Liability cap at 24 months fees",
            "Right to source code escrow with neutral third-party agent",
        ],
    ),
]


SYSTEM_PROMPT = """You are a senior US commercial lawyer with 20+ years drafting SaaS subscription agreements for technology companies. You draft production-ready contracts that:

- Use real legal language (no placeholders, brackets, [TBD], or template markers)
- Include specific commercial terms (real fees, percentages, days, dates)
- Have clear article/section structure with numbered sub-clauses
- Use real party names provided in the prompt
- Reflect the specified jurisdiction's conventions
- Output ONLY the contract text — no preamble, no explanation, no markdown headers like ## or **
- Use verbose, formal legal phrasing typical of executed contracts (whereas, hereinafter, notwithstanding the foregoing, subject to the terms herein, etc.)
- Write each sub-clause as multiple complete sentences with substantive legal detail
"""


PASS1_PROMPT = """Draft the FIRST PART of a SaaS Subscription Agreement: title, recitals, and Article 1 (Definitions).

Requirements:
- Title in ALL CAPS
- Full preamble paragraph identifying both parties with state of incorporation and addresses
- 3-5 WHEREAS recitals explaining the deal context
- "NOW, THEREFORE..." consideration paragraph
- ARTICLE 1 — DEFINITIONS with at least 18 defined terms in alphabetical order

Each definition must be 2-4 sentences long with substantive legal precision. Include defined terms for:
Affiliate, Agreement, Authorized User, Business Day, Confidential Information, Customer Data, Documentation, Effective Date, Force Majeure Event, Implementation Services, Intellectual Property Rights, Order Form, Personal Data, Platform, Privacy Laws, Professional Services, Service Level Agreement, Services, Subscription Fee, Subscription Term, Support Services, Third-Party Applications, and any others appropriate to the deal.

Output ONLY the title, preamble, recitals, and Article 1. STOP after Article 1. Do not write Article 2."""


PASS2_PROMPT = """Continuing the same SaaS Subscription Agreement, draft Articles 2 through 8.

Articles to write:
- ARTICLE 2 — SERVICES AND SUBSCRIPTION (provision of services, license grant, restrictions, customer responsibilities, suspension rights — at least 5 sub-clauses with full prose)
- ARTICLE 3 — IMPLEMENTATION AND PROFESSIONAL SERVICES (scope, fees, acceptance criteria, change requests, customer cooperation — at least 5 sub-clauses)
- ARTICLE 4 — SUBSCRIPTION FEES AND PAYMENT (fees, billing cycle, late payment interest, taxes, fee changes, audit rights, disputed invoices — at least 6 sub-clauses)
- ARTICLE 5 — TERM AND RENEWAL (initial term, automatic renewal, notice of non-renewal, fee escalation upon renewal, effects of expiration — at least 5 sub-clauses)
- ARTICLE 6 — SERVICE LEVEL AGREEMENT AND SUPPORT (uptime commitment, exclusions, measurement methodology, service credit calculation, support tiers, response times, escalation procedures, maintenance windows — at least 7 sub-clauses)
- ARTICLE 7 — CUSTOMER DATA AND PRIVACY (ownership, license to provider, data residency, security measures, breach notification, retention, return/deletion on termination, compliance with privacy laws — at least 8 sub-clauses)
- ARTICLE 8 — INTELLECTUAL PROPERTY RIGHTS (provider IP ownership of platform, customer IP ownership of customer data, feedback license to provider, no implied licenses, third-party components — at least 5 sub-clauses)

Each article must have multiple substantive sub-clauses written in verbose formal legal prose. Reference defined terms from the previous Definitions article (assume Customer, Provider, Authorized User, Customer Data, Services, etc. are all defined). Use the actual party names from the previous part.

Output ONLY Articles 2-8. STOP after Article 8. Do not write Article 9."""


PASS3_PROMPT = """Continuing the same SaaS Subscription Agreement, draft Articles 9 through 15 (the legal/risk allocation articles).

Articles to write:
- ARTICLE 9 — CONFIDENTIALITY (definition reference, obligations, exclusions, compelled disclosure, return/destruction, survival period, equitable relief — at least 6 sub-clauses)
- ARTICLE 10 — REPRESENTATIONS AND WARRANTIES (provider warranties about services, mutual authority warranty, customer warranties about right to provide data, disclaimer of implied warranties — at least 5 sub-clauses)
- ARTICLE 11 — INDEMNIFICATION (provider IP indemnity to customer, customer indemnity to provider for misuse and customer data, indemnification procedure with notice/control/cooperation requirements, exceptions and limitations, sole remedies — at least 5 sub-clauses)
- ARTICLE 12 — LIMITATION OF LIABILITY (mutual aggregate cap with specific months, exclusion of consequential and indirect damages, super-cap exceptions for IP indemnity/breach of confidentiality/gross negligence/willful misconduct, basis of the bargain disclaimer — at least 4 sub-clauses)
- ARTICLE 13 — TERM AND TERMINATION (term reference, termination for material breach with cure period, termination for insolvency/bankruptcy, termination for convenience if applicable, effects of termination, transition assistance, refund obligations, survival — at least 7 sub-clauses)
- ARTICLE 14 — GOVERNING LAW AND DISPUTE RESOLUTION (choice of law, escalation to executives, mediation, exclusive jurisdiction, waiver of jury trial, attorneys' fees — at least 5 sub-clauses)
- ARTICLE 15 — GENERAL PROVISIONS (entire agreement, amendments in writing, severability, waiver, notices with addresses, assignment with affiliate exception, force majeure, independent contractors, no third-party beneficiaries, counterparts — at least 9 sub-clauses)

Then a SIGNATURE BLOCK with both party names, "By:", "Name:", "Title:", "Date:" lines for each.

Each article must have multiple substantive sub-clauses written in verbose formal legal prose. Use the actual party names from the previous parts. Reference defined terms from the Definitions article.

Output ONLY Articles 9-15 plus the signature block."""


def build_deal_context(spec: ContractSpec) -> str:
    """Common deal context shared across all 3 passes."""
    return f"""DEAL CONTEXT (use this consistently across all parts):

PROVIDER: {spec.provider_name}
CUSTOMER: {spec.customer_name}
INDUSTRY: {spec.industry}
DEAL SIZE: {spec.deal_size}
CONTRACT VALUE: {spec.contract_value}
JURISDICTION: {spec.jurisdiction}
GOVERNING LAW: {spec.governing_law}
NEGOTIATION BIAS: {spec.bias}

REQUIRED COMMERCIAL TERMS (must appear somewhere in the contract):
{chr(10).join(f'  - {t}' for t in spec.extra_terms)}
"""


def synthesize_contract(client: OpenAI, spec: ContractSpec) -> str:
    """Generate full contract via 3 sequential passes for longer, more substantive output."""
    logger.info("Synthesizing %s contract for %s ↔ %s (3-pass)",
                spec.contract_type, spec.provider_name, spec.customer_name)

    deal_context = build_deal_context(spec)

    # Pass 1: Title + Recitals + Definitions
    logger.info("  Pass 1/3: Title + Recitals + Definitions")
    pass1 = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": deal_context + "\n\n" + PASS1_PROMPT},
        ],
        temperature=0.4,
        max_tokens=8000,
    )
    part1_text = pass1.choices[0].message.content
    logger.info("    → %d words", len(part1_text.split()))

    # Pass 2: Operational articles (2-8)
    # We pass the previous output as context so the model maintains consistency.
    logger.info("  Pass 2/3: Articles 2-8 (operational)")
    pass2 = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": deal_context + "\n\n" + PASS1_PROMPT},
            {"role": "assistant", "content": part1_text},
            {"role": "user", "content": PASS2_PROMPT},
        ],
        temperature=0.4,
        max_tokens=8000,
    )
    part2_text = pass2.choices[0].message.content
    logger.info("    → %d words", len(part2_text.split()))

    # Pass 3: Legal/risk articles (9-15) + signature
    logger.info("  Pass 3/3: Articles 9-15 + signature")
    pass3 = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": deal_context + "\n\n" + PASS1_PROMPT},
            {"role": "assistant", "content": part1_text},
            {"role": "user", "content": PASS2_PROMPT},
            {"role": "assistant", "content": part2_text},
            {"role": "user", "content": PASS3_PROMPT},
        ],
        temperature=0.4,
        max_tokens=8000,
    )
    part3_text = pass3.choices[0].message.content
    logger.info("    → %d words", len(part3_text.split()))

    # Concatenate all 3 parts
    full_contract = part1_text.rstrip() + "\n\n" + part2_text.rstrip() + "\n\n" + part3_text.rstrip()
    logger.info("  Total: %d words", len(full_contract.split()))
    return full_contract


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w-]", "_", name)[:50]


def save_contract(text: str, spec: ContractSpec, idx: int) -> Path:
    """Save synthesized contract to data/v2/raw/SAAS/synthetic/."""
    out_dir = OUTPUT_DIR / spec.contract_type / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"SYNTHETIC_{safe_filename(spec.provider_name)}_{spec.deal_size}_{idx:02d}.txt"
    path = out_dir / filename
    path.write_text(text, encoding="utf-8")
    return path


def analyze_contract(text: str) -> dict:
    """Quick stats: word count, articles, standard clauses present."""
    word_count = len(text.split())
    articles = len(re.findall(r"(?im)^\s*ARTICLE\s+\d", text))
    has_sla = bool(re.search(r"(?i)uptime|service\s+level", text))
    has_indemnity = bool(re.search(r"(?i)indemnif", text))
    has_liability_cap = bool(re.search(r"(?i)limit.{0,30}liability|aggregate\s+liability", text))
    has_governing_law = bool(re.search(r"(?i)governing\s+law", text))
    has_termination = bool(re.search(r"(?i)terminat", text))
    has_confidentiality = bool(re.search(r"(?i)confidential", text))
    has_data_protection = bool(re.search(r"(?i)data\s+(?:protection|privacy)|personal\s+data", text))
    has_ip_clause = bool(re.search(r"(?i)intellectual\s+property|ip\s+rights", text))
    has_placeholders = bool(re.search(r"\[(?:TBD|TODO|insert|placeholder|\*+)\]", text))
    return {
        "word_count": word_count,
        "articles": articles,
        "has_sla": has_sla,
        "has_indemnity": has_indemnity,
        "has_liability_cap": has_liability_cap,
        "has_governing_law": has_governing_law,
        "has_termination": has_termination,
        "has_confidentiality": has_confidentiality,
        "has_data_protection": has_data_protection,
        "has_ip_clause": has_ip_clause,
        "has_placeholders": has_placeholders,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="SAAS", choices=["SAAS"])
    parser.add_argument("--count", type=int, default=2)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY env var not set")

    client = OpenAI()

    if args.type == "SAAS":
        specs = SAAS_SPECS[: args.count]
    else:
        raise SystemExit(f"No specs defined for type {args.type}")

    results = []
    for idx, spec in enumerate(specs, 1):
        try:
            text = synthesize_contract(client, spec)
            path = save_contract(text, spec, idx)
            stats = analyze_contract(text)
            logger.info("Saved: %s", path.name)
            logger.info("  Stats: %s", stats)
            results.append({
                "spec": {
                    "provider": spec.provider_name,
                    "customer": spec.customer_name,
                    "industry": spec.industry,
                    "deal_size": spec.deal_size,
                    "bias": spec.bias,
                },
                "filename": path.name,
                "stats": stats,
            })
        except Exception as e:
            logger.error("Failed to synthesize %s: %s", spec.provider_name, e)

    # Save report
    report_path = OUTPUT_DIR / args.type / "synthetic" / "synthesis_report.json"
    report_path.write_text(json.dumps(results, indent=2))

    logger.info("\n=== SYNTHESIS COMPLETE ===")
    logger.info("Saved %d contracts to %s", len(results), OUTPUT_DIR / args.type / "synthetic")
    logger.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
