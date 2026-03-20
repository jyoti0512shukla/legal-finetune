#!/usr/bin/env python3
"""Generate v3 task-specific fine-tuning training data from EDGAR contracts."""

import json
import os
import re
import glob

BASE = "/Users/jyotimishra/legal-finetune"
EDGAR = f"{BASE}/data/raw/edgar"
CLAUSES_FILE = f"{BASE}/data/processed/clauses/edgar_clauses.jsonl"
OUT_DIR = f"{BASE}/data/processed/v3_tasks"

os.makedirs(OUT_DIR, exist_ok=True)

CONTRACT_TYPE_MAP = {
    "MSA": "Master Services Agreement",
    "NDA": "Non-Disclosure Agreement",
    "EMPLOYMENT": "Employment Agreement",
    "SAAS": "SaaS Agreement",
    "SOFTWARE_LICENSE": "Software License Agreement",
}

SUBDIRS = ["MSA", "NDA", "EMPLOYMENT", "SAAS", "SOFTWARE_LICENSE"]


def read_file_chars(path, max_chars):
    """Read up to max_chars from a file."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_chars)
    except Exception as e:
        print(f"  Error reading {path}: {e}")
        return None


def pick_files(subdir, count):
    """Pick up to `count` files from a subdirectory."""
    pattern = os.path.join(EDGAR, subdir, "*.txt")
    files = sorted(glob.glob(pattern))
    return files[:count]


# ─── Helper: extract key terms from contract text ───

def extract_party(text, label_pattern):
    """Try to find a party name near a label pattern."""
    m = re.search(label_pattern, text, re.IGNORECASE)
    if m:
        # Take next non-empty line-ish content
        rest = text[m.end():m.end()+200]
        lines = [l.strip() for l in rest.split("\n") if l.strip()]
        if lines:
            return lines[0][:120]
    return None


def find_pattern(text, pattern):
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return m.group(0).strip()
    return None


def extract_terms_from_text(text):
    """Extract key terms from contract text."""
    terms = {}

    # Party names - look for common patterns
    # "between X and Y"
    between_match = re.search(r'between\s+(.{5,80}?)\s+(?:and|&)\s+(.{5,80?}?)(?:\s*\(|\s*,|\s*hereinafter)', text, re.IGNORECASE | re.DOTALL)
    if between_match:
        terms["party_a"] = re.sub(r'\s+', ' ', between_match.group(1).strip())[:100]
        terms["party_b"] = re.sub(r'\s+', ' ', between_match.group(2).strip())[:100]
    else:
        # Try "by and between"
        m = re.search(r'by and between\s+(.{5,100}?)\s*[\("]', text, re.IGNORECASE | re.DOTALL)
        if m:
            terms["party_a"] = re.sub(r'\s+', ' ', m.group(1).strip())[:100]
        # Look for hereinafter patterns
        hm = re.findall(r'hereinafter\s*["\u201c]([^"\u201d]{2,40})["\u201d]', text, re.IGNORECASE)
        if len(hm) >= 2:
            terms.setdefault("party_a", hm[0].strip())
            terms["party_b"] = hm[1].strip()
        elif len(hm) == 1:
            terms.setdefault("party_a", hm[0].strip())

    # Effective date
    ed = re.search(r'effective\s+(?:as\s+of\s+|date\s*[:\s]+)?(\w+\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
    terms["effective_date"] = ed.group(1).strip() if ed else None

    # Expiry / termination date
    exp = re.search(r'(?:expir|terminat|end|continue\s+until)\w*\s+(?:on\s+|date\s*[:\s]+)?(\w+\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
    terms["expiry_date"] = exp.group(1).strip() if exp else None

    # Contract value / compensation
    val = re.search(r'(?:total|aggregate|not\s+(?:to\s+)?exceed|compensation|salary|fee|amount)\s*(?:of\s+)?\$[\d,]+(?:\.\d{2})?', text, re.IGNORECASE)
    terms["contract_value"] = val.group(0).strip() if val else None

    # Liability cap
    liab = re.search(r'(?:liabilit|damages?|cap)\w*\s+(?:shall\s+)?(?:not\s+)?(?:exceed|limited\s+to|cap)\s*\$?[\d,]+', text, re.IGNORECASE)
    terms["liability_cap"] = liab.group(0).strip() if liab else None

    # Governing law
    gov = re.search(r'(?:governed?\s+by|governing\s+law)\s*(?:the\s+)?(?:laws?\s+of\s+)?(?:the\s+)?(?:State\s+of\s+)?([A-Z][a-zA-Z\s,]+?)(?:\.|,\s+without)', text, re.IGNORECASE)
    terms["governing_law"] = gov.group(1).strip()[:80] if gov else None

    # Notice period
    notice = re.search(r'(\d+)\s*(?:calendar\s+|business\s+)?days?\s*(?:prior\s+)?(?:written\s+)?notice', text, re.IGNORECASE)
    terms["notice_period_days"] = notice.group(1) if notice else None

    # Arbitration venue
    arb = re.search(r'arbitrat\w+\s+(?:shall\s+)?(?:be\s+)?(?:conducted\s+)?(?:in\s+)?([A-Z][a-zA-Z\s,]+?)(?:\.|,\s+in)', text, re.IGNORECASE)
    terms["arbitration_venue"] = arb.group(1).strip()[:80] if arb else None

    # Ensure all keys present
    for k in ["party_a", "party_b", "effective_date", "expiry_date", "contract_value",
              "liability_cap", "governing_law", "notice_period_days", "arbitration_venue"]:
        terms.setdefault(k, None)

    return terms


# ─── Helper: check clauses in contract text ───

CLAUSE_IDS = [
    "LIABILITY_LIMIT", "INDEMNITY", "TERMINATION_CONVENIENCE", "TERMINATION_CAUSE",
    "FORCE_MAJEURE", "CONFIDENTIALITY", "GOVERNING_LAW", "DISPUTE_RESOLUTION",
    "IP_OWNERSHIP", "DATA_PROTECTION", "PAYMENT_TERMS", "ASSIGNMENT"
]

CLAUSE_PATTERNS = {
    "LIABILITY_LIMIT": [r'limit\w*\s+(?:of\s+)?liabilit', r'liability\s+(?:shall\s+)?(?:not\s+)?exceed', r'cap\s+on\s+(?:aggregate\s+)?liabilit', r'maximum\s+(?:aggregate\s+)?liabilit', r'(?:in\s+no\s+event|under\s+no\s+circumstances)\s+.*(?:liab|damages)'],
    "INDEMNITY": [r'indemnif', r'hold\s+harmless', r'defend\s+and\s+indemnif'],
    "TERMINATION_CONVENIENCE": [r'terminat\w+\s+(?:for\s+)?convenience', r'terminat\w+\s+without\s+cause', r'either\s+party\s+may\s+terminat', r'may\s+terminat\w+\s+(?:this\s+)?agreement\s+(?:at\s+any\s+time|upon|by\s+(?:giving|providing))'],
    "TERMINATION_CAUSE": [r'terminat\w+\s+for\s+cause', r'terminat\w+\s+(?:upon|for)\s+(?:material\s+)?breach', r'right\s+to\s+terminat\w+\s+(?:if|upon|for)'],
    "FORCE_MAJEURE": [r'force\s+majeure', r'act\s+of\s+god', r'beyond\s+(?:the\s+)?(?:reasonable\s+)?control'],
    "CONFIDENTIALITY": [r'confidential', r'non-disclosure', r'nondisclosure', r'proprietary\s+information'],
    "GOVERNING_LAW": [r'govern\w+\s+(?:by\s+)?(?:the\s+)?law', r'choice\s+of\s+law', r'applicable\s+law'],
    "DISPUTE_RESOLUTION": [r'dispute\s+resolution', r'arbitrat', r'mediat', r'(?:exclusive\s+)?jurisdiction'],
    "IP_OWNERSHIP": [r'intellectual\s+property', r'ownership\s+of\s+(?:work|deliverable|invention)', r'work\s+(?:made\s+)?for\s+hire', r'(?:all\s+)?(?:right|title|interest)\s+(?:in\s+and\s+to|to)\s+(?:any|all|the)\s+(?:work|deliverable|invention|ip)'],
    "DATA_PROTECTION": [r'data\s+(?:protection|privacy|processing)', r'personal\s+(?:data|information)', r'gdpr', r'ccpa', r'privacy\s+(?:policy|law|regulation)'],
    "PAYMENT_TERMS": [r'payment\s+(?:terms|schedule|due)', r'(?:net|within)\s+\d+\s+days', r'invoice', r'(?:fees?|compensation|consideration)\s+(?:shall|will)\s+(?:be\s+)?(?:paid|payable)'],
    "ASSIGNMENT": [r'assignment', r'(?:may\s+not|shall\s+not)\s+assign', r'(?:consent\s+to\s+)?assign\w*\s+(?:this|any\s+right)'],
}


def check_clause(text, clause_id):
    """Check whether a clause is present, weak, or missing in the text."""
    patterns = CLAUSE_PATTERNS.get(clause_id, [])
    matches = 0
    matched_text = ""
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            matches += 1
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 80)
            matched_text = text[start:end].replace("\n", " ").strip()

    if matches >= 2:
        return "PRESENT", matched_text
    elif matches == 1:
        return "WEAK", matched_text
    else:
        return "MISSING", ""


def assess_risk(clause_id, status):
    """Assess risk level based on clause importance and status."""
    high_importance = {"LIABILITY_LIMIT", "INDEMNITY", "TERMINATION_CAUSE", "CONFIDENTIALITY", "GOVERNING_LAW", "DATA_PROTECTION"}
    medium_importance = {"TERMINATION_CONVENIENCE", "FORCE_MAJEURE", "DISPUTE_RESOLUTION", "IP_OWNERSHIP", "PAYMENT_TERMS"}
    # low: ASSIGNMENT

    if status == "PRESENT":
        return "LOW"
    elif status == "WEAK":
        if clause_id in high_importance:
            return "HIGH"
        return "MEDIUM"
    else:  # MISSING
        if clause_id in high_importance:
            return "HIGH"
        elif clause_id in medium_importance:
            return "MEDIUM"
        return "LOW"


def generate_finding(clause_id, status, context):
    """Generate a finding sentence."""
    clause_names = {
        "LIABILITY_LIMIT": "Limitation of Liability",
        "INDEMNITY": "Indemnification",
        "TERMINATION_CONVENIENCE": "Termination for Convenience",
        "TERMINATION_CAUSE": "Termination for Cause",
        "FORCE_MAJEURE": "Force Majeure",
        "CONFIDENTIALITY": "Confidentiality",
        "GOVERNING_LAW": "Governing Law",
        "DISPUTE_RESOLUTION": "Dispute Resolution",
        "IP_OWNERSHIP": "Intellectual Property Ownership",
        "DATA_PROTECTION": "Data Protection",
        "PAYMENT_TERMS": "Payment Terms",
        "ASSIGNMENT": "Assignment",
    }
    name = clause_names.get(clause_id, clause_id)

    if status == "PRESENT":
        findings = {
            "LIABILITY_LIMIT": f"Contract includes a limitation of liability clause that caps exposure.",
            "INDEMNITY": f"Indemnification obligations are defined for one or both parties.",
            "TERMINATION_CONVENIENCE": f"Either party may terminate for convenience with prior notice.",
            "TERMINATION_CAUSE": f"Termination for cause is available upon material breach.",
            "FORCE_MAJEURE": f"Force majeure provisions cover events beyond reasonable control.",
            "CONFIDENTIALITY": f"Confidentiality obligations are specified for proprietary information.",
            "GOVERNING_LAW": f"Governing law is explicitly stated in the agreement.",
            "DISPUTE_RESOLUTION": f"Dispute resolution mechanism is defined in the contract.",
            "IP_OWNERSHIP": f"Intellectual property ownership is addressed in the agreement.",
            "DATA_PROTECTION": f"Data protection and privacy requirements are included.",
            "PAYMENT_TERMS": f"Payment terms and schedule are specified.",
            "ASSIGNMENT": f"Assignment restrictions or permissions are addressed.",
        }
        return findings.get(clause_id, f"{name} clause is present in the contract.")
    elif status == "WEAK":
        findings = {
            "LIABILITY_LIMIT": f"Liability limitation language exists but lacks specific caps or carve-outs.",
            "INDEMNITY": f"Indemnification is mentioned but scope and procedures are vaguely defined.",
            "TERMINATION_CONVENIENCE": f"Termination convenience language is present but notice period is unclear.",
            "TERMINATION_CAUSE": f"Termination for cause is referenced but cure period is not specified.",
            "FORCE_MAJEURE": f"Force majeure reference found but triggering events are not enumerated.",
            "CONFIDENTIALITY": f"Confidentiality is mentioned but obligations lack specificity on duration and scope.",
            "GOVERNING_LAW": f"Governing law is referenced but jurisdiction details are incomplete.",
            "DISPUTE_RESOLUTION": f"Dispute resolution is mentioned but the mechanism is not clearly defined.",
            "IP_OWNERSHIP": f"IP ownership is referenced but allocation between parties is ambiguous.",
            "DATA_PROTECTION": f"Data protection language exists but lacks compliance framework specifics.",
            "PAYMENT_TERMS": f"Payment is mentioned but timing and late payment consequences are unclear.",
            "ASSIGNMENT": f"Assignment is referenced but consent requirements are not detailed.",
        }
        return findings.get(clause_id, f"{name} clause is weak and needs strengthening.")
    else:
        findings = {
            "LIABILITY_LIMIT": f"No limitation of liability clause found, exposing parties to unlimited damages.",
            "INDEMNITY": f"No indemnification provisions found in the contract.",
            "TERMINATION_CONVENIENCE": f"No termination for convenience right is provided.",
            "TERMINATION_CAUSE": f"No termination for cause provisions are included.",
            "FORCE_MAJEURE": f"No force majeure clause found, leaving parties exposed to unforeseen events.",
            "CONFIDENTIALITY": f"No confidentiality protections are included in the agreement.",
            "GOVERNING_LAW": f"No governing law is specified, creating jurisdictional uncertainty.",
            "DISPUTE_RESOLUTION": f"No dispute resolution mechanism is defined.",
            "IP_OWNERSHIP": f"No intellectual property ownership provisions are included.",
            "DATA_PROTECTION": f"No data protection or privacy provisions found.",
            "PAYMENT_TERMS": f"No payment terms or schedule are specified.",
            "ASSIGNMENT": f"No assignment clause found in the agreement.",
        }
        return findings.get(clause_id, f"{name} clause is missing from the contract.")


# ─── Helper: generate redline suggestions ───

CLAUSE_WEAKNESSES = {
    "INDEMNITY": {
        "issue": "The indemnification clause lacks mutual obligations and does not specify defense procedures or notice requirements.",
        "suggestion_template": "Each Party (the \"Indemnifying Party\") shall indemnify, defend, and hold harmless the other Party and its officers, directors, employees, and agents (collectively, \"Indemnified Parties\") from and against any and all third-party claims, losses, damages, liabilities, costs, and expenses (including reasonable attorneys' fees) arising out of or relating to: (a) the Indemnifying Party's breach of any representation, warranty, or obligation under this Agreement; (b) the Indemnifying Party's negligence or willful misconduct; or (c) any violation of applicable law by the Indemnifying Party. The Indemnified Party shall provide prompt written notice of any claim, reasonably cooperate in the defense, and allow the Indemnifying Party to control the defense and settlement, provided that no settlement shall impose obligations on the Indemnified Party without its prior written consent.",
        "rationale": "Adding mutual indemnification with specific defense procedures and notice requirements provides balanced protection and clear processes for handling third-party claims."
    },
    "CONFIDENTIALITY": {
        "issue": "The confidentiality clause does not specify the duration of obligations or standard exceptions for publicly available information.",
        "suggestion_template": "\"Confidential Information\" means all non-public information disclosed by either Party to the other in connection with this Agreement, whether in oral, written, electronic, or other form. The Receiving Party shall: (a) hold all Confidential Information in strict confidence using at least the same degree of care it uses for its own confidential information, but no less than reasonable care; (b) not disclose Confidential Information to any third party except to its employees, agents, or contractors who have a need to know and are bound by confidentiality obligations no less restrictive than those herein; and (c) use Confidential Information solely for the purposes of this Agreement. Confidential Information shall not include information that: (i) is or becomes publicly available through no fault of the Receiving Party; (ii) was known to the Receiving Party prior to disclosure; (iii) is independently developed without use of Confidential Information; or (iv) is lawfully received from a third party without restriction. These obligations shall survive for a period of five (5) years following termination or expiration of this Agreement.",
        "rationale": "Specifying the definition of Confidential Information, standard exceptions, minimum care standards, and a survival period creates enforceable and balanced confidentiality protections."
    },
    "TERMINATION": {
        "issue": "The termination clause does not address post-termination obligations or wind-down procedures.",
        "suggestion_template": "Either Party may terminate this Agreement: (a) for convenience upon sixty (60) days' prior written notice; or (b) for cause if the other Party materially breaches this Agreement and fails to cure such breach within thirty (30) days after receipt of written notice specifying the breach in reasonable detail. Upon termination or expiration: (i) each Party shall promptly return or destroy all Confidential Information of the other Party; (ii) all outstanding payment obligations shall become immediately due and payable; (iii) the Parties shall cooperate in good faith to effect an orderly wind-down of ongoing obligations; and (iv) Sections [Confidentiality], [Indemnification], [Limitation of Liability], and [Governing Law] shall survive termination.",
        "rationale": "Including both convenience and cause termination rights with cure periods, plus explicit post-termination obligations and survival clauses, provides clear exit procedures and protects both parties' interests."
    },
    "LIABILITY": {
        "issue": "The limitation of liability clause does not specify carve-outs for gross negligence, willful misconduct, or IP infringement.",
        "suggestion_template": "EXCEPT FOR (A) A PARTY'S INDEMNIFICATION OBLIGATIONS, (B) A PARTY'S BREACH OF CONFIDENTIALITY OBLIGATIONS, (C) A PARTY'S GROSS NEGLIGENCE OR WILLFUL MISCONDUCT, AND (D) INFRINGEMENT OF THE OTHER PARTY'S INTELLECTUAL PROPERTY RIGHTS, IN NO EVENT SHALL EITHER PARTY'S AGGREGATE LIABILITY ARISING OUT OF OR RELATED TO THIS AGREEMENT EXCEED THE TOTAL AMOUNTS PAID OR PAYABLE UNDER THIS AGREEMENT DURING THE TWELVE (12) MONTH PERIOD PRECEDING THE EVENT GIVING RISE TO THE CLAIM. IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, REGARDLESS OF THE FORM OF ACTION OR THEORY OF LIABILITY.",
        "rationale": "Adding specific carve-outs for gross negligence, willful misconduct, confidentiality breaches, and IP infringement prevents the liability cap from shielding bad-faith conduct while maintaining reasonable commercial risk allocation."
    },
    "GOVERNING_LAW": {
        "issue": "The governing law clause does not exclude conflict of laws principles or specify the forum for litigation.",
        "suggestion_template": "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, United States, without regard to its conflict of laws principles. Any legal action or proceeding arising under this Agreement shall be brought exclusively in the federal or state courts located in Wilmington, Delaware, and each Party hereby irrevocably consents to personal jurisdiction and venue in such courts. The Parties expressly exclude the application of the United Nations Convention on Contracts for the International Sale of Goods.",
        "rationale": "Excluding conflict of laws principles, specifying an exclusive forum, and excluding the UN CISG removes jurisdictional ambiguity and ensures predictable dispute resolution."
    },
    "FORCE_MAJEURE": {
        "issue": "The force majeure clause does not enumerate specific triggering events or provide for termination after prolonged force majeure.",
        "suggestion_template": "Neither Party shall be liable for any failure or delay in performing its obligations under this Agreement to the extent such failure or delay results from a Force Majeure Event. \"Force Majeure Event\" means any event beyond the reasonable control of the affected Party, including but not limited to: acts of God, natural disasters, epidemics or pandemics, war, terrorism, riots, government actions or orders, labor strikes, fire, flood, earthquake, power failure, or telecommunications disruptions. The affected Party shall: (a) promptly notify the other Party in writing of the Force Majeure Event and its expected duration; (b) use commercially reasonable efforts to mitigate the impact; and (c) resume performance as soon as reasonably practicable. If a Force Majeure Event continues for more than ninety (90) consecutive days, either Party may terminate this Agreement upon thirty (30) days' written notice.",
        "rationale": "Enumerating specific triggering events, requiring mitigation efforts, and adding a termination right after prolonged force majeure provides clarity and prevents indefinite non-performance."
    },
    "DISPUTE_RESOLUTION": {
        "issue": "The dispute resolution clause does not specify escalation steps before formal proceedings.",
        "suggestion_template": "In the event of any dispute arising out of or relating to this Agreement, the Parties shall first attempt to resolve the dispute through good-faith negotiation between senior executives of each Party for a period of thirty (30) days. If the dispute is not resolved through negotiation, either Party may submit the dispute to mediation administered by the American Arbitration Association under its Commercial Mediation Procedures. If mediation is unsuccessful within sixty (60) days, either Party may submit the dispute to binding arbitration administered by the American Arbitration Association under its Commercial Arbitration Rules. The arbitration shall be conducted by a single arbitrator in [City, State]. The arbitrator's decision shall be final and binding, and judgment upon the award may be entered in any court of competent jurisdiction. Notwithstanding the foregoing, either Party may seek injunctive or other equitable relief in any court of competent jurisdiction to protect its intellectual property rights or Confidential Information.",
        "rationale": "Including escalation steps from negotiation to mediation to binding arbitration reduces litigation costs, and preserving the right to seek injunctive relief protects critical interests like IP and confidentiality."
    },
    "IP_OWNERSHIP": {
        "issue": "The intellectual property clause does not clearly delineate pre-existing IP from newly developed IP.",
        "suggestion_template": "Each Party shall retain all right, title, and interest in and to its Pre-Existing IP. \"Pre-Existing IP\" means all intellectual property owned or controlled by a Party prior to this Agreement or developed independently of this Agreement. All Work Product created by Provider specifically for Client under this Agreement shall be owned by Client, and Provider hereby assigns to Client all right, title, and interest in such Work Product. Provider grants Client a non-exclusive, perpetual, royalty-free license to use any Pre-Existing IP of Provider that is incorporated into the Work Product, solely to the extent necessary for Client to use the Work Product. Provider retains the right to use general knowledge, skills, experience, and methodologies acquired during performance.",
        "rationale": "Clearly separating pre-existing IP from work product, providing explicit assignment of work product, and licensing back embedded pre-existing IP prevents ownership disputes and ensures both parties can operate."
    },
    "DATA_PROTECTION": {
        "issue": "The data protection clause does not address breach notification timelines or data subject rights.",
        "suggestion_template": "Provider shall process Personal Data only as instructed by Client and in compliance with applicable Data Protection Laws, including the GDPR and CCPA. Provider shall implement appropriate technical and organizational measures to ensure a level of security appropriate to the risk. In the event of a Personal Data breach, Provider shall notify Client without undue delay and in any event within seventy-two (72) hours of becoming aware of the breach. Provider shall assist Client in responding to data subject requests to exercise their rights under applicable Data Protection Laws. Upon termination of this Agreement, Provider shall, at Client's election, return or securely delete all Personal Data processed under this Agreement within thirty (30) days.",
        "rationale": "Adding specific breach notification timelines, data subject rights assistance, and data return/deletion obligations aligns with GDPR/CCPA requirements and provides clear operational procedures."
    },
    "PAYMENT_TERMS": {
        "issue": "The payment terms do not specify late payment interest or dispute procedures for invoices.",
        "suggestion_template": "Client shall pay all undisputed invoices within thirty (30) days of receipt. All invoices shall include reasonable detail of the services performed or deliverables provided. If Client disputes an invoice in good faith, Client shall notify Provider in writing within fifteen (15) days of receipt, specifying the reasons for the dispute, and shall pay the undisputed portion. The Parties shall work in good faith to resolve any invoice dispute within thirty (30) days. Late payments on undisputed amounts shall accrue interest at the lesser of one and one-half percent (1.5%) per month or the maximum rate permitted by applicable law. Provider may suspend performance upon thirty (30) days' written notice if undisputed amounts remain unpaid for more than sixty (60) days past due.",
        "rationale": "Specifying payment timing, invoice detail requirements, dispute procedures, late payment interest, and suspension rights for non-payment creates balanced financial protections for both parties."
    },
    "ASSIGNMENT": {
        "issue": "The assignment clause does not address change of control scenarios or exceptions for affiliates.",
        "suggestion_template": "Neither Party may assign this Agreement or any of its rights or obligations hereunder without the prior written consent of the other Party, which consent shall not be unreasonably withheld, conditioned, or delayed; provided, however, that either Party may assign this Agreement without consent to: (a) an Affiliate of such Party; or (b) a successor entity in connection with a merger, acquisition, or sale of all or substantially all of such Party's assets to which this Agreement relates. Any assignment in violation of this Section shall be null and void. The assigning Party shall remain jointly and severally liable with the assignee for all obligations under this Agreement.",
        "rationale": "Permitting affiliate and change-of-control assignments while requiring consent for other transfers and maintaining joint liability provides practical flexibility without reducing accountability."
    },
}


def get_redline_for_clause(clause_type, clause_text):
    """Generate a redline suggestion for a given clause."""
    # Map clause types to weakness categories
    ct = clause_type.upper()
    if "INDEMNIT" in ct or "INDEMN" in ct:
        key = "INDEMNITY"
    elif "CONFIDENT" in ct or "NDA" in ct or "NON-DISCLOSURE" in ct or "NONDISCLOSURE" in ct:
        key = "CONFIDENTIALITY"
    elif "TERMINAT" in ct:
        key = "TERMINATION"
    elif "LIABILIT" in ct or "LIMIT" in ct or "DAMAGES" in ct:
        key = "LIABILITY"
    elif "GOVERN" in ct or "CHOICE OF LAW" in ct or "APPLICABLE LAW" in ct:
        key = "GOVERNING_LAW"
    elif "FORCE" in ct or "MAJEURE" in ct:
        key = "FORCE_MAJEURE"
    elif "DISPUTE" in ct or "ARBITRAT" in ct or "MEDIAT" in ct or "JURISDICT" in ct:
        key = "DISPUTE_RESOLUTION"
    elif "IP" in ct or "INTELLECT" in ct or "OWNERSHIP" in ct or "WORK PRODUCT" in ct or "PATENT" in ct or "COPYRIGHT" in ct:
        key = "IP_OWNERSHIP"
    elif "DATA" in ct or "PRIVACY" in ct or "GDPR" in ct or "PERSONAL" in ct:
        key = "DATA_PROTECTION"
    elif "PAYMENT" in ct or "FEE" in ct or "COMPENSAT" in ct or "INVOICE" in ct or "PRICE" in ct:
        key = "PAYMENT_TERMS"
    elif "ASSIGN" in ct:
        key = "ASSIGNMENT"
    elif "SERVICE" in ct or "SCOPE" in ct or "OBLIGATION" in ct:
        # Map service-related to indemnity as a common improvement area
        key = "INDEMNITY"
    elif "WARRANT" in ct or "REPRESENT" in ct:
        key = "LIABILITY"
    elif "DEFINITION" in ct or "GENERAL" in ct:
        key = "GOVERNING_LAW"
    else:
        # Default based on text content
        text_lower = clause_text.lower()
        if "indemnif" in text_lower:
            key = "INDEMNITY"
        elif "confidential" in text_lower:
            key = "CONFIDENTIALITY"
        elif "terminat" in text_lower:
            key = "TERMINATION"
        elif "liabilit" in text_lower:
            key = "LIABILITY"
        elif "govern" in text_lower and "law" in text_lower:
            key = "GOVERNING_LAW"
        elif "payment" in text_lower or "invoice" in text_lower:
            key = "PAYMENT_TERMS"
        elif "assign" in text_lower:
            key = "ASSIGNMENT"
        else:
            key = "CONFIDENTIALITY"  # fallback

    weakness = CLAUSE_WEAKNESSES[key]
    return {
        "clause_name": key.replace("_", " ").title(),
        "issue": weakness["issue"],
        "suggested_language": weakness["suggestion_template"],
        "rationale": weakness["rationale"],
    }


# ═══════════════════════════════════════════════════════════════════
# FILE 1: extraction_examples.jsonl (100 examples, 20 per type)
# ═══════════════════════════════════════════════════════════════════

def generate_extraction_examples():
    print("=== Generating extraction_examples.jsonl ===")
    output_path = os.path.join(OUT_DIR, "extraction_examples.jsonl")
    count = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for subdir in SUBDIRS:
            files = pick_files(subdir, 20)
            contract_type = CONTRACT_TYPE_MAP[subdir]
            print(f"  {subdir}: {len(files)} files")

            for fpath in files:
                text = read_file_chars(fpath, 3000)
                if not text or len(text.strip()) < 100:
                    continue

                terms = extract_terms_from_text(text)
                response = json.dumps(terms, ensure_ascii=False)

                # Determine jurisdiction from governing law or default
                jurisdiction = "California, United States"
                if terms.get("governing_law"):
                    gl = terms["governing_law"]
                    if "new york" in gl.lower():
                        jurisdiction = "New York, United States"
                    elif "delaware" in gl.lower():
                        jurisdiction = "Delaware, United States"
                    elif "texas" in gl.lower():
                        jurisdiction = "Texas, United States"
                    elif "england" in gl.lower() or "uk" in gl.lower():
                        jurisdiction = "England, United Kingdom"
                    elif "switzerland" in gl.lower() or "swiss" in gl.lower():
                        jurisdiction = "Zurich, Switzerland"

                entry = {
                    "instruction": f"Extract key terms from this contract:\n\n{text}",
                    "response": response,
                    "clause_type": "DEFINITIONS",
                    "contract_type": contract_type,
                    "jurisdiction": jurisdiction,
                    "source": "extraction_labeled",
                    "quality_score": 4.5,
                }
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

    print(f"  Wrote {count} extraction examples to {output_path}")
    return count


# ═══════════════════════════════════════════════════════════════════
# FILE 2: checklist_examples.jsonl (50 examples, 10 per type)
# ═══════════════════════════════════════════════════════════════════

def generate_checklist_examples():
    print("=== Generating checklist_examples.jsonl ===")
    output_path = os.path.join(OUT_DIR, "checklist_examples.jsonl")
    count = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for subdir in SUBDIRS:
            files = pick_files(subdir, 10)
            contract_type = CONTRACT_TYPE_MAP[subdir]
            print(f"  {subdir}: {len(files)} files")

            for fpath in files:
                text = read_file_chars(fpath, 5000)
                if not text or len(text.strip()) < 100:
                    continue

                clauses = []
                for cid in CLAUSE_IDS:
                    status, context = check_clause(text, cid)
                    risk = assess_risk(cid, status)
                    finding = generate_finding(cid, status, context)
                    clauses.append({
                        "clause_id": cid,
                        "status": status,
                        "risk_level": risk,
                        "finding": finding,
                    })

                response = json.dumps({"clauses": clauses}, ensure_ascii=False)

                entry = {
                    "instruction": f"Check this contract for 12 standard clauses:\n\n{text}",
                    "response": response,
                    "clause_type": "GENERAL_PROVISIONS",
                    "contract_type": contract_type,
                    "jurisdiction": "California, United States",
                    "source": "checklist_labeled",
                    "quality_score": 4.5,
                }
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

    print(f"  Wrote {count} checklist examples to {output_path}")
    return count


# ═══════════════════════════════════════════════════════════════════
# FILE 3: redline_examples.jsonl (100 examples from edgar_clauses)
# ═══════════════════════════════════════════════════════════════════

def generate_redline_examples():
    print("=== Generating redline_examples.jsonl ===")
    output_path = os.path.join(OUT_DIR, "redline_examples.jsonl")
    count = 0

    # Read clauses, skip UNKNOWN
    clauses = []
    with open(CLAUSES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("clause_type", "UNKNOWN") != "UNKNOWN":
                    clauses.append(obj)
            except json.JSONDecodeError:
                continue

    print(f"  Found {len(clauses)} non-UNKNOWN clauses")

    # Take first 100
    selected = clauses[:100]

    with open(output_path, "w", encoding="utf-8") as out:
        for clause in selected:
            clause_type = clause.get("clause_type", "GENERAL")
            clause_text = clause.get("text", "")
            contract_type = clause.get("contract_type", "Unknown")

            if not clause_text or len(clause_text.strip()) < 50:
                continue

            redline = get_redline_for_clause(clause_type, clause_text)
            response = json.dumps(redline, ensure_ascii=False)

            entry = {
                "instruction": f"This {clause_type} clause needs improvement. Suggest better language:\n\n{clause_text}",
                "response": response,
                "clause_type": clause_type,
                "contract_type": contract_type,
                "jurisdiction": "California, United States",
                "source": "redline_labeled",
                "quality_score": 4.5,
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Wrote {count} redline examples to {output_path}")
    return count


if __name__ == "__main__":
    print("Generating v3 task training data...\n")
    n1 = generate_extraction_examples()
    print()
    n2 = generate_checklist_examples()
    print()
    n3 = generate_redline_examples()
    print(f"\nDone. Total examples: {n1 + n2 + n3}")
    print(f"  extraction_examples.jsonl: {n1}")
    print(f"  checklist_examples.jsonl: {n2}")
    print(f"  redline_examples.jsonl: {n3}")
