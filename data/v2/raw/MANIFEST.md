# V2 Training Corpus — Verified Raw Contracts

Verified, type-tagged contracts collected for the Gemma 4 Legal v2 fine-tuning run.
All contracts have passed quality scoring (≥3/5), deduplication, and PII redaction.

**Last updated:** 2026-04-10
**Total contracts:** 39 across 8 contract types

## Sources

| Source | Count | Notes |
|---|---|---|
| **CUAD v1 (Atticus Project)** | 30 | 510 commercial contracts from EDGAR, type-tagged folders, CC-BY 4.0 |
| **SEC EDGAR direct (v2 pilot fetch)** | 10 SaaS | Modern (2021-2025), broad search, balanced bias |
| **Total** | **40** | Verified, real, deduplicated |

## Per-type breakdown

| Type | Count | Source | Year range | Quality |
|---|---|---|---|---|
| `SAAS` | 10 | EDGAR (10 modern) + CUAD (1 legacy hosting from 2018) | 2018-2025 | 3-5 |
| `RESELLER` | 11 | CUAD (Distributor + Reseller folders) | 1999-2019 | 3-5 |
| `VENDOR_SUPPLY` | 5 | CUAD (Supply + Manufacturing) | 2019-2020 | 3-4 |
| `MSA` | 4 | CUAD (Service folder) | 2020 | 3-4 |
| `IP_LICENSE` | 3 | CUAD (IP folder) | 2018-2019 | 3-5 |
| `SOFTWARE_LICENSE` | 2 | CUAD (License_Agreements, refined) | 2020 | 3-4 |
| `OUTSOURCING` | 2 | CUAD (Outsourcing folder) | 2015-2017 | 3 |
| `MAINTENANCE` | 2 | CUAD (Maintenance folder) | 2018 | 3 |

## Filters applied

All contracts in this corpus have passed:
1. **Quality scoring** (≥3/5) — sufficient length, has clause structure, has standard clauses (governing law, indemnity, liability, termination, confidentiality), low redaction, low OCR garbage, has clear party identification
2. **Deduplication** — title-anchored fingerprint dedup catches near-duplicate filings
3. **Type detection** — title + body keyword scoring (rejects UNKNOWN types)
4. **Type refinement** — content licenses excluded from SOFTWARE_LICENSE bucket
5. **Year filter** — pre-2015 hosting excluded from SAAS bucket (not modern enough)
6. **PII redaction** — emails and phones redacted (counts in diversity_report.json)
7. **Diversity rules** — max 2 contracts per company per type

## Filename convention

Each contract is named:
```
<TYPE>_<FILER>_<YEAR>_<accession>.txt
```

## How to add more

- **EDGAR** (modern, 2020+): `python scripts/v2_pilot_extract.py --type <TYPE> --target N`
- **CUAD** (broad, 2000-2020): `python scripts/v2_cuad_extract.py --target N`
- **Common Paper templates**: Manual download from `commonpaper.com/standards/`
- **Stanford MCC** (TODO): Bulk download for Employment + extra Services/Supply
- **Claude synthesis** (TODO): Fill gaps with synthetic high-quality contracts

## Gaps to fill for v2

To hit the v2 target of 25 contracts × 10 types = 250 total:

| Type | Have | Need | Gap | Strategy |
|---|---|---|---|---|
| SAAS | 10 | 25 | 15 | Claude synthesis |
| MSA | 4 | 25 | 21 | More EDGAR + Claude |
| Software License | 2 | 25 | 23 | EDGAR EX-10 search "perpetual license" + Claude |
| NDA | 0 | 25 | 25 | Common Paper + GitHub + Claude |
| Employment | 0 | 25 | 25 | Stanford MCC Employment (486K available) |
| Vendor/Supply | 5 | 25 | 20 | Stanford MCC + Claude |
| Reseller | 11 | 25 | 14 | Claude (CUAD already best source) |
| DPA | 0 | 25 | 25 | Common Paper + GitHub + Claude |
| Statement of Work | 0 | 25 | 25 | Hand-curate + Claude |
| Independent Contractor | 0 | 25 | 25 | Common Paper + Cooley GO + Claude |
| **TOTAL** | **40** | **250** | **210** | |

## Reports

- `diversity_report.json` — CUAD extraction details (per-type counts, year/company distributions)
- `SAAS/edgar_source_report.json` — EDGAR pilot fetch details

## Provenance / License notes

- **CUAD**: CC-BY 4.0 (commercial use OK with attribution)
- **EDGAR**: US public domain (SEC filings)
- All contracts redacted for emails/phones; signatories left intact (publicly filed)
