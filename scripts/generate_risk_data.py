#!/usr/bin/env python3
"""Generate risk assessment training data from EDGAR clauses."""

import json
import os
import re
import random
import hashlib

INPUT_PATH = "/Users/jyotimishra/legal-finetune/data/processed/clauses/edgar_clauses.jsonl"
OUTPUT_DIR = "/Users/jyotimishra/legal-finetune/data/processed/v3_tasks"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "risk_examples.jsonl")

# Seed for reproducibility
random.seed(42)

def load_clauses(path, max_count=100):
    """Load first max_count non-UNKNOWN clauses."""
    clauses = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get('clause_type', '') != 'UNKNOWN':
                clauses.append(obj)
            if len(clauses) >= max_count:
                break
    return clauses


def truncate_text(text, max_chars=1500):
    """Truncate text to max_chars."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def assess_original_risk(clause_type, text):
    """Assess the original EDGAR clause. Most public company clauses are LOW risk."""
    text_lower = text.lower()
    issues = []

    # Check for vagueness indicators
    vague_terms = ["reasonable", "as needed", "from time to time", "may", "best efforts",
                   "commercially reasonable", "material", "substantially"]
    vague_count = sum(1 for v in vague_terms if v in text_lower)

    # Check for specificity indicators (good signs)
    specific_indicators = ["days", "percent", "%", "dollars", "$", "section", "article",
                          "pursuant to", "notwithstanding", "shall", "within"]
    specific_count = sum(1 for s in specific_indicators if s in text_lower)

    # Check for protective clauses
    protective = ["indemnif", "warrant", "represent", "covenant", "limitation",
                  "exclusion", "remedies", "governing law", "arbitrat", "jurisdict"]
    protective_count = sum(1 for p in protective if p in text_lower)

    # Most EDGAR clauses from public companies are well-drafted
    if vague_count >= 4 and specific_count < 3:
        risk = "MEDIUM"
        if "reasonable" in text_lower and "days" not in text_lower:
            issues.append("Uses vague temporal references without specific deadlines")
        if "best efforts" in text_lower:
            issues.append("Relies on 'best efforts' standard which is ambiguous")
        if "material" in text_lower and "defined" not in text_lower:
            issues.append("Uses 'material' without clear definition threshold")
    else:
        risk = "LOW"

    # Generate appropriate issues based on clause type
    if risk == "LOW":
        issues = _low_risk_issues(clause_type, text_lower)
    else:
        if not issues:
            issues = _medium_risk_issues(clause_type, text_lower)

    summary = _generate_summary(risk, clause_type, issues)
    return risk, issues, summary


def _low_risk_issues(clause_type, text_lower):
    """Generate minor issues for low-risk clauses."""
    issues = []
    ct = clause_type.upper()

    if ct in ("INDEMNIFICATION", "INDEMNITY"):
        if "cap" not in text_lower and "limit" not in text_lower:
            issues.append("No explicit indemnification cap specified")
        if "insur" not in text_lower:
            issues.append("Does not reference insurance requirements")
        if not issues:
            issues.append("Standard indemnification provisions with adequate protections")

    elif ct in ("CONFIDENTIALITY", "NON-DISCLOSURE", "NDA"):
        if "year" not in text_lower and "month" not in text_lower and "period" not in text_lower:
            issues.append("Duration of confidentiality obligation could be more explicit")
        if "return" not in text_lower and "destroy" not in text_lower:
            issues.append("Could specify return/destruction of materials more explicitly")
        if not issues:
            issues.append("Well-structured confidentiality provisions")

    elif ct == "TERMINATION":
        if "cure" not in text_lower:
            issues.append("Could benefit from explicit cure period provisions")
        if "surviv" not in text_lower:
            issues.append("Survival of obligations after termination not explicitly addressed")
        if not issues:
            issues.append("Adequate termination provisions with standard protections")

    elif ct in ("LIMITATION OF LIABILITY", "LIABILITY"):
        if "consequential" not in text_lower:
            issues.append("Could explicitly address consequential damages exclusion")
        if "gross negligence" not in text_lower and "willful" not in text_lower:
            issues.append("Does not carve out gross negligence or willful misconduct")
        if not issues:
            issues.append("Standard liability limitation with typical commercial terms")

    elif ct in ("INTELLECTUAL PROPERTY", "IP OWNERSHIP", "IP"):
        if "work for hire" not in text_lower and "work-for-hire" not in text_lower:
            issues.append("Does not explicitly reference work-for-hire doctrine")
        if "pre-existing" not in text_lower and "background" not in text_lower:
            issues.append("Could clarify treatment of pre-existing IP more explicitly")
        if not issues:
            issues.append("IP provisions adequately address ownership and licensing")

    elif ct in ("GOVERNING LAW", "CHOICE OF LAW", "JURISDICTION"):
        if "waiv" not in text_lower:
            issues.append("Does not include jury trial waiver")
        if not issues:
            issues.append("Standard governing law provision with clear jurisdiction")

    elif ct in ("NON-COMPETE", "NON-COMPETITION", "RESTRICTIVE COVENANT"):
        if "geographic" not in text_lower and "territory" not in text_lower:
            issues.append("Geographic scope could be more precisely defined")
        if not issues:
            issues.append("Restrictive covenant with reasonable scope and duration")

    elif ct in ("PAYMENT", "COMPENSATION", "FEES"):
        if "late" not in text_lower and "interest" not in text_lower:
            issues.append("Late payment penalties not explicitly specified")
        if not issues:
            issues.append("Payment terms are clearly defined")

    elif ct in ("WARRANTY", "WARRANTIES", "REPRESENTATIONS"):
        if "disclaim" not in text_lower:
            issues.append("Warranty disclaimer language could be strengthened")
        if not issues:
            issues.append("Warranty provisions include standard representations")

    elif ct == "FORCE MAJEURE":
        if "pandemic" not in text_lower and "epidemic" not in text_lower:
            issues.append("Does not explicitly cover pandemic/epidemic events")
        if "mitigat" not in text_lower:
            issues.append("Could include mitigation obligations during force majeure")
        if not issues:
            issues.append("Force majeure clause covers standard triggering events")

    elif ct in ("ASSIGNMENT", "TRANSFER"):
        if "consent" not in text_lower:
            issues.append("Assignment consent requirements could be more explicit")
        if not issues:
            issues.append("Standard assignment provisions with appropriate restrictions")

    elif ct in ("INSURANCE", "COVERAGE"):
        if "$" not in text_lower and "dollar" not in text_lower:
            issues.append("Minimum coverage amounts not numerically specified")
        if not issues:
            issues.append("Insurance requirements are adequately defined")

    elif ct in ("DISPUTE RESOLUTION", "ARBITRATION"):
        if "cost" not in text_lower and "fee" not in text_lower:
            issues.append("Cost allocation for dispute resolution not specified")
        if not issues:
            issues.append("Dispute resolution mechanism is clearly defined")

    else:
        # Generic low-risk issues
        if "shall" in text_lower:
            issues.append("Uses mandatory language appropriately for obligation clarity")
        if "notice" not in text_lower:
            issues.append("Could benefit from explicit notice requirements")
        if not issues:
            issues.append("Clause contains standard commercial terms with adequate protections")

    return issues[:2]


def _medium_risk_issues(clause_type, text_lower):
    """Generate issues for medium-risk clauses."""
    issues = []
    ct = clause_type.upper()

    if ct in ("INDEMNIFICATION", "INDEMNITY"):
        issues = ["Indemnification scope is broadly worded without clear boundaries",
                   "Missing cap on indemnification exposure"]
    elif ct in ("CONFIDENTIALITY", "NON-DISCLOSURE", "NDA"):
        issues = ["Confidentiality duration is vague or unspecified",
                   "Definition of confidential information is overly broad"]
    elif ct == "TERMINATION":
        issues = ["Termination triggers are not clearly enumerated",
                   "Post-termination obligations lack specificity"]
    elif ct in ("LIMITATION OF LIABILITY", "LIABILITY"):
        issues = ["Liability cap amount is not clearly stated",
                   "Exclusions from liability limitation are too narrow"]
    else:
        issues = ["Key terms rely on subjective standards",
                   "Obligations lack specific performance metrics or deadlines"]
    return issues


def _generate_summary(risk, clause_type, issues):
    """Generate a one-sentence summary."""
    ct = clause_type.replace("_", " ").lower()
    if risk == "LOW":
        return f"This {ct} clause contains standard commercial terms with adequate protections for both parties."
    elif risk == "MEDIUM":
        return f"This {ct} clause has some ambiguous provisions that could create uncertainty in enforcement."
    else:
        return f"This {ct} clause is critically deficient and missing essential protections that could expose a party to significant liability."


def degrade_medium(text, clause_type):
    """Create a MEDIUM-risk degraded version of the clause text."""
    degraded = text

    # Replace specific time periods with vague language
    degraded = re.sub(r'\b(\d+)\s*(days?|months?|years?|business days?)\b',
                      'a reasonable period', degraded, flags=re.IGNORECASE)

    # Replace dollar amounts with vague references
    degraded = re.sub(r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand))?',
                      'a mutually agreed amount', degraded, flags=re.IGNORECASE)

    # Replace percentage figures
    degraded = re.sub(r'\b\d+(?:\.\d+)?%', 'a reasonable percentage', degraded)
    degraded = re.sub(r'\b\d+(?:\.\d+)?\s*percent\b', 'a reasonable percentage', degraded, flags=re.IGNORECASE)

    # Remove section/article references
    degraded = re.sub(r'(?:Section|Article|Paragraph|Clause)\s*\d+(?:\.\d+)*(?:\([a-z]\))?',
                      'the applicable provision', degraded, flags=re.IGNORECASE)

    # Replace specific statute references
    degraded = re.sub(r'(?:pursuant to|under|in accordance with)\s+(?:the\s+)?[A-Z][A-Za-z\s]+Act(?:\s+of\s+\d{4})?',
                      'under applicable law', degraded, flags=re.IGNORECASE)

    # Soften mandatory language occasionally
    degraded = re.sub(r'\bshall\b', 'should', degraded, count=2)

    # Replace specific governing law references
    degraded = re.sub(r'(?:State of|Commonwealth of)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?',
                      'the applicable jurisdiction', degraded, flags=re.IGNORECASE)

    return truncate_text(degraded, 1500)


def degrade_high(text, clause_type):
    """Create a HIGH-risk severely truncated version — just 2-3 sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out very short fragments
    sentences = [s for s in sentences if len(s) > 15]

    if len(sentences) == 0:
        return text[:200]

    # Take just 2-3 sentences, preferring opening ones that set context but miss protections
    selected = sentences[:min(3, len(sentences))]
    result = ' '.join(selected)

    # Further weaken: remove protective qualifiers
    result = re.sub(r'(?:provided(?:,?\s*however),?\s*that|notwithstanding\s+(?:the\s+)?(?:foregoing|above|anything))[^.]*\.?',
                    '', result, flags=re.IGNORECASE)
    result = re.sub(r'(?:except|unless)\s+(?:as\s+)?(?:otherwise\s+)?(?:expressly\s+)?(?:set forth|provided|agreed)[^.]*\.?',
                    '', result, flags=re.IGNORECASE)

    # Clean up whitespace
    result = re.sub(r'\s+', ' ', result).strip()

    if len(result) < 30 and len(text) > 30:
        result = text[:200]

    return truncate_text(result, 1500)


def medium_risk_issues_for_degraded(clause_type):
    """Generate MEDIUM risk issues for a degraded clause."""
    ct = clause_type.upper()
    issue_sets = {
        "INDEMNIFICATION": [
            "Specific indemnification triggers replaced with vague standards",
            "No quantified cap on indemnification liability",
            "Timeline for indemnification claims process is undefined"
        ],
        "INDEMNITY": [
            "Indemnification obligations lack specific monetary thresholds",
            "Notice requirements for claims are vaguely defined"
        ],
        "CONFIDENTIALITY": [
            "Duration of confidentiality obligations is imprecise",
            "Permitted disclosures lack specific enumeration",
            "Return of materials obligation uses soft language"
        ],
        "NON-DISCLOSURE": [
            "Scope of protected information is broadly and vaguely defined",
            "Remedies for breach rely on subjective standards"
        ],
        "TERMINATION": [
            "Cure periods use vague temporal references instead of specific days",
            "Post-termination wind-down obligations lack detail",
            "Termination for convenience lacks adequate notice period specificity"
        ],
        "LIMITATION OF LIABILITY": [
            "Liability cap expressed as vague amount rather than specific figure",
            "Consequential damages exclusion uses permissive rather than mandatory language"
        ],
        "LIABILITY": [
            "Liability allocation relies on subjective reasonableness standards",
            "Missing specific monetary thresholds for liability triggers"
        ],
        "INTELLECTUAL PROPERTY": [
            "IP assignment language uses 'should' instead of 'shall'",
            "Background IP carve-out is vaguely defined",
            "License scope lacks specific field-of-use restrictions"
        ],
        "GOVERNING LAW": [
            "Jurisdiction selection uses vague 'applicable jurisdiction' language",
            "Choice of law provision lacks specificity"
        ],
        "PAYMENT": [
            "Payment deadlines replaced with vague 'reasonable period' language",
            "Late payment interest rate is unspecified"
        ],
        "WARRANTY": [
            "Warranty scope relies on subjective quality standards",
            "Warranty period is vaguely defined"
        ],
        "FORCE MAJEURE": [
            "Force majeure triggering events are not specifically enumerated",
            "Mitigation obligations during force majeure are vague"
        ],
        "NON-COMPETE": [
            "Geographic and temporal restrictions use vague boundaries",
            "Competitive activity definition is overly broad and imprecise"
        ],
        "ASSIGNMENT": [
            "Consent requirements for assignment are softened",
            "Change of control provisions lack specific triggers"
        ],
        "INSURANCE": [
            "Minimum coverage amounts are not numerically specified",
            "Required insurance types are vaguely referenced"
        ],
        "DISPUTE RESOLUTION": [
            "Dispute resolution timeline uses vague temporal references",
            "Cost allocation for proceedings is undefined"
        ],
    }
    default = [
        "Specific numeric thresholds replaced with vague standards",
        "Mandatory obligations softened to permissive language",
        "Statutory references removed reducing enforceability clarity"
    ]
    issues = issue_sets.get(ct, default)
    return issues[:2]


def high_risk_issues_for_degraded(clause_type):
    """Generate HIGH risk issues for severely degraded clause."""
    ct = clause_type.upper()
    issue_sets = {
        "INDEMNIFICATION": [
            "Missing indemnification procedures and notice requirements entirely",
            "No cap on liability or basket threshold defined",
            "Survival period for indemnification claims omitted",
            "No defense and settlement control provisions"
        ],
        "INDEMNITY": [
            "Critical indemnification protections truncated",
            "Missing third-party claim procedures",
            "No limitations on indemnification scope"
        ],
        "CONFIDENTIALITY": [
            "Missing definition of what constitutes confidential information",
            "No duration limit on confidentiality obligations",
            "Permitted disclosure exceptions completely absent",
            "No remedies specified for breach of confidentiality"
        ],
        "NON-DISCLOSURE": [
            "Scope of non-disclosure obligations is undefined",
            "Missing return/destruction of confidential materials provision",
            "No injunctive relief or specific performance remedy"
        ],
        "TERMINATION": [
            "Missing cure period provisions entirely",
            "No specification of termination-triggering events",
            "Post-termination obligations completely absent",
            "Survival clause missing"
        ],
        "LIMITATION OF LIABILITY": [
            "No liability cap amount specified",
            "Consequential damages exclusion missing entirely",
            "No carve-outs for fraud, willful misconduct, or gross negligence",
            "Limitation provisions are incomplete and unenforceable"
        ],
        "LIABILITY": [
            "Liability allocation framework is critically incomplete",
            "Missing essential limitation and exclusion provisions",
            "No risk allocation mechanism defined"
        ],
        "INTELLECTUAL PROPERTY": [
            "IP assignment is incomplete — missing present-tense assignment language",
            "No treatment of pre-existing or background IP",
            "Missing moral rights waiver",
            "License grants lack scope, duration, or territory definitions"
        ],
        "GOVERNING LAW": [
            "No specific jurisdiction identified",
            "Missing venue selection clause",
            "No jury trial waiver or consent to jurisdiction"
        ],
        "PAYMENT": [
            "Payment amount and schedule completely undefined",
            "No late payment or default remedies",
            "Missing invoicing procedures"
        ],
        "WARRANTY": [
            "Warranty scope is undefined and incomplete",
            "Missing warranty disclaimer for implied warranties",
            "No remedy for warranty breach specified"
        ],
        "FORCE MAJEURE": [
            "Force majeure events not enumerated",
            "No notice requirement for force majeure claim",
            "Missing termination right for extended force majeure",
            "No mitigation obligation"
        ],
        "NON-COMPETE": [
            "Geographic scope completely undefined — likely unenforceable",
            "Temporal restriction missing",
            "Competitive activities not defined",
            "No consideration for the restrictive covenant"
        ],
        "ASSIGNMENT": [
            "Missing consent requirements for assignment",
            "No anti-assignment restriction",
            "Change of control not addressed"
        ],
        "INSURANCE": [
            "No minimum coverage amounts specified",
            "Required insurance types not listed",
            "Missing certificate of insurance requirements"
        ],
        "DISPUTE RESOLUTION": [
            "Dispute resolution mechanism is undefined",
            "No escalation procedure before formal proceedings",
            "Venue and procedural rules not specified"
        ],
    }
    default = [
        "Critical protective provisions are missing entirely",
        "Clause is severely truncated and lacks essential terms",
        "No remedies or enforcement mechanisms specified",
        "Clause would likely be unenforceable due to missing material terms"
    ]
    issues = issue_sets.get(ct, default)
    return issues[:3]


def make_entry(instruction_text, risk, issues, summary, clause_type, contract_type, source):
    """Create a single training entry."""
    response_obj = {
        "risk": risk,
        "issues": issues,
        "summary": summary
    }
    return {
        "instruction": instruction_text,
        "response": json.dumps(response_obj),
        "clause_type": clause_type,
        "contract_type": contract_type,
        "jurisdiction": "California, United States",
        "source": source,
        "quality_score": 5.0
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    clauses = load_clauses(INPUT_PATH, 100)
    print(f"Loaded {len(clauses)} non-UNKNOWN clauses")

    entries = []
    for i, clause in enumerate(clauses):
        clause_type = clause['clause_type']
        contract_type = clause.get('contract_type', 'Commercial Agreement')
        original_text = clause['text']
        text_1500 = truncate_text(original_text, 1500)

        # 1. risk_original
        risk, issues, summary = assess_original_risk(clause_type, original_text)
        instr = f"Assess the risk level of this {clause_type} clause:\n\n{text_1500}"
        entries.append(make_entry(instr, risk, issues, summary, clause_type, contract_type, "risk_original"))

        # 2. risk_degraded_medium
        degraded_med_text = degrade_medium(original_text, clause_type)
        med_issues = medium_risk_issues_for_degraded(clause_type)
        med_summary = f"This {clause_type.replace('_', ' ').lower()} clause has been weakened by replacing specific terms with vague standards, reducing enforceability and creating ambiguity in key obligations."
        instr_med = f"Assess the risk level of this {clause_type} clause:\n\n{degraded_med_text}"
        entries.append(make_entry(instr_med, "MEDIUM", med_issues, med_summary, clause_type, contract_type, "risk_degraded_medium"))

        # 3. risk_degraded_high
        degraded_high_text = degrade_high(original_text, clause_type)
        high_issues = high_risk_issues_for_degraded(clause_type)
        high_summary = f"This {clause_type.replace('_', ' ').lower()} clause is critically deficient, with essential protections missing entirely, creating significant legal exposure."
        instr_high = f"Assess the risk level of this {clause_type} clause:\n\n{degraded_high_text}"
        entries.append(make_entry(instr_high, "HIGH", high_issues, high_summary, clause_type, contract_type, "risk_degraded_high"))

    # Write output
    with open(OUTPUT_PATH, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Wrote {len(entries)} entries to {OUTPUT_PATH}")

    # Verification
    sources = {}
    risks = {}
    for e in entries:
        s = e['source']
        sources[s] = sources.get(s, 0) + 1
        r = json.loads(e['response'])['risk']
        risks[r] = risks.get(r, 0) + 1
    print(f"By source: {sources}")
    print(f"By risk: {risks}")


if __name__ == '__main__':
    main()
