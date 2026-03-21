# Indian Legal Dataset

Training data for Indian jurisdiction — contracts, judgments, and statutes
governed by Indian law (ICA 1872, Companies Act 2013, Arbitration Act 1996, etc.)

## Directory Structure

```
indian/
├── raw/
│   ├── contracts/     # Raw Indian contracts (SEBI/BSE, SEC 20-F, Govt procurement)
│   ├── judgments/     # Supreme Court / High Court judgments
│   └── statutes/     # Indian acts and statutory text
├── processed/
│   ├── clauses/      # Segmented contract clauses
│   ├── distilled/    # Claude/GPT-4o generated gold examples
│   └── tasks/        # Task-specific training data (risk, extraction, checklist, redline)
└── training/
    ├── train.jsonl
    └── validation.jsonl
```

## Data Sources

### Tier 1: HuggingFace (zero effort, already clean)

| Dataset | Size | Use Case | Load |
|---------|------|----------|------|
| **ILDC** | 35k SC cases | Judgment text + outcome labels | `load_dataset("Exploration-Lab/ILDC_multi", "multi")` |
| **IN-Abs** | 7.1k cases | Judgment + human-written summary pairs | `load_dataset("law-ai/IN-Abs")` |
| **SARA** | ~900 pairs | Indian statutory reasoning (NLI) | `load_dataset("Exploration-Lab/SARA")` |

### Tier 2: Indian Contracts (medium effort)

| Source | Type | Access | Notes |
|--------|------|--------|-------|
| **SEBI/BSE filings** | Material contracts from listed cos | `bseindia.com/corporates/ann.html` | Best Indian contract source. Filter by: Infosys, TCS, Wipro, Reliance, HDFC, ICICI, L&T, Zomato, Paytm |
| **SEC EDGAR 20-F** | Indian cos listed in US | `efts.sec.gov` query `"Indian law" "arbitration" forms=20-F` | Infosys, Wipro, WNS — governed by Indian law, in English |
| **Govt procurement** | Standard contract forms | GeM Portal (`mkp.gem.gov.in`), CPWD forms | Public procurement templates |
| **legal-partner samples** | 10 synthetic contracts | `~/legal-partner/data/*.html` | Software dev, consulting, NDA, employment, SaaS, supply, JV agreements |

### Tier 3: Statutes & Regulations

| Source | Access | Priority Acts |
|--------|--------|---------------|
| **India Code** | `indiacode.nic.in` | ICA 1872, Companies Act 2013, Arbitration Act 1996, IT Act 2000, DPDP Act 2023 |
| **SEBI circulars** | `sebi.gov.in/legal/circulars.html` | Disclosure, governance, listing obligations |
| **RBI master directions** | `rbi.org.in` | FEMA, banking regulation |

### Tier 4: Judgment Databases

| Source | Size | Access |
|--------|------|--------|
| **Indian Kanoon** | 30M+ docs | API at `indiankanoon.org/api/` (free: 1000 queries/day) |
| **SCI website** | PDFs back to 1950 | `main.sci.gov.in/judgments` (needs PDF scraping) |
| **CommonLII India** | Structured, free | `commonlii.org/in/` |

## Build Priority

1. ILDC + IN-Abs + SARA from HuggingFace (zero effort, high impact)
2. SEBI/BSE contract filings (scraping needed, high impact)
3. India Code priority statutes (XML/PDF download)
4. Indian Kanoon API for targeted case law
5. SEC EDGAR 20-F for Indian companies

## Task Coverage Needed

Same 5 tasks as US dataset, adapted for Indian jurisdiction:

- **Drafting** — clauses referencing ICA sections, Indian arbitration, DPDP Act
- **Risk assessment** — Indian regulatory compliance (SEBI, RBI, FEMA)
- **Extraction** — Indian party names, CIN numbers, Indian jurisdiction/venues
- **Checklist** — Same 12 clauses but with Indian law specifics
- **Redline** — Indian law-aware suggestions (e.g., stamp duty, arbitration seat)
