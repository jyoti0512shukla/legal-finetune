"""CIK whitelist for high-quality contract sourcing.

Each tier represents a market cap segment. Sourcing across tiers ensures
the model learns drafting patterns from the full size spectrum, not just
mega-caps (which over-protect) or growth-stage (which under-protect).

CIKs verified against https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany
"""

# ── SaaS Subscription Agreements ──────────────────────────────────────────

SAAS_FILERS_TIERED = {
    "mega": {  # $50B+ market cap — conservative legal teams, ironclad protections
        "Salesforce": "0001108524",
        "Workday": "0001327811",
        "ServiceNow": "0001373715",
        "Adobe": "0000796343",
        "Oracle": "0001341439",
        "Intuit": "0000896878",
    },
    "large": {  # $10B-$50B — established practices, balanced deals
        "Atlassian": "0001650372",
        "Snowflake": "0001640147",
        "Datadog": "0001561550",
        "MongoDB": "0001441816",
        "Cloudflare": "0001477333",
        "Okta": "0001660134",
        "DocuSign": "0001261333",
        "HubSpot": "0001404655",
        "Zscaler": "0001713683",
        "Palo Alto Networks": "0001327567",
    },
    "mid": {  # $1B-$10B — flexible, growth-stage commercial terms
        "Twilio": "0001447669",
        "Asana": "0001477720",
        "Zendesk": "0001463172",
        "Box": "0001372612",
        "Smartsheet": "0001366561",
        "Domo": "0001505952",
        "JFrog": "0001212644",
        "GitLab": "0001653482",
        "Elastic": "0001707753",
        "PagerDuty": "0001568100",
    },
    "growth": {  # $100M-$1B — modern conventions
        "Confluent": "0001699838",
        "BigCommerce": "0001626825",
        "Coursera": "0001651562",
        "Sprinklr": "0001569345",
        "Braze": "0001676238",
        "ZoomInfo": "0001794515",
    },
}

# Distribution targets per type (totals to 25)
SAAS_TIER_TARGETS = {
    "mega": 5,
    "large": 8,
    "mid": 8,
    "growth": 4,
}


def get_filer_tier(filer_name: str, tiered_dict: dict) -> str:
    """Look up which tier a filer belongs to. Returns 'unknown' if not found."""
    filer_lower = filer_name.lower()
    for tier, filers in tiered_dict.items():
        for name in filers:
            if name.lower() in filer_lower or filer_lower.startswith(name.lower()):
                return tier
    return "unknown"


def all_saas_ciks() -> dict[str, tuple[str, str]]:
    """Return flat mapping of company name → (cik, tier) for all whitelisted SaaS filers."""
    result = {}
    for tier, filers in SAAS_FILERS_TIERED.items():
        for name, cik in filers.items():
            result[name] = (cik, tier)
    return result
