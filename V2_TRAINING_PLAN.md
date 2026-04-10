# Gemma 4 Legal v2 — Training Plan

**Goal:** Fix the specific failure modes observed in v1 (contract-type confusion, hallucinated statutes, missing SaaS terms, coherence loss) by adding **25 contracts × 10 types = 250 contracts**, expanded into **~6,500 task-specific training examples**.

**Estimated timeline:** 5-7 days of data prep + 4 hours training + 1 day evaluation
**Estimated cost:** ~$15-20 RunPod + Claude API for data generation

---

## v1 Failure Modes We're Fixing

| Failure | Root cause | v2 fix |
|---|---|---|
| SaaS contract gets services-style IP clause | Training data dominated by EDGAR services agreements | Add 25 SaaS-specific examples with correct Provider-retains-IP language |
| MSME (Indian law) referenced in California contract | India and US examples mixed in training | Tag every example with jurisdiction; use jurisdiction-conditional retrieval |
| ERISA Section 412(p) in SaaS assignment clause | Model regurgitates memorized statute names | Add **negative examples** (BAD: irrelevant statute → GOOD: no statute) |
| Article 10 word salad after ~200 tokens | Token degradation in long generations | Train on shorter, well-formed clauses + add coherence-rated examples |
| Duplicate "Services" definitions | No internal coherence tracking | Train on examples that maintain unique definitions |
| Missing SaaS terms (uptime, support tiers, fees) | Model hasn't seen many SaaS examples with these terms | Every SaaS example must include uptime % and response times |

---

## The 10 Contract Types (25 each = 250 contracts)

| # | Type | Why | Source |
|---|---|---|---|
| 1 | **SaaS Subscription Agreement** | Top priority — biggest v1 gap | EDGAR (Atlassian, Salesforce, ServiceNow 10-K exhibits), Common Paper |
| 2 | **Master Services Agreement (MSA)** | Most common B2B contract | EDGAR (Accenture, Cognizant, Infosys exhibits) |
| 3 | **Mutual NDA** | Universal | Common Paper, YC, public templates |
| 4 | **Statement of Work (SOW)** | Pairs with MSA | EDGAR exhibits |
| 5 | **Software License Agreement** (perpetual on-prem) | Distinct from SaaS | EDGAR (Oracle, MicroStrategy) |
| 6 | **Employment Agreement** (executive level) | Needed for HR practice | EDGAR (CEO/CFO 8-K exhibits) |
| 7 | **Independent Contractor Agreement** | Different IP profile | Common Paper, public templates |
| 8 | **Vendor / Supply Agreement** | B2B procurement | EDGAR (manufacturing, retail) |
| 9 | **Reseller / Distribution Agreement** | Channel partner | EDGAR (tech vendor exhibits) |
| 10 | **Data Processing Agreement (DPA)** | Required for SaaS/data | CCPA standard form, EDGAR |

### Sourcing rules

- **70% from EDGAR** — use existing `scripts/01_fetch_edgar.py` with form type filters
- **20% from Common Paper / public templates** — already legally vetted, clean format
- **10% hand-curated from law firm websites** (publicly published model contracts from Cooley, Wilson Sonsini, etc.)

### Diversity within each type (25 contracts)

To prevent overfitting to a single style:
- 5 contracts: **enterprise** ($10M+ deal value, complex)
- 10 contracts: **mid-market** ($100K-$10M, balanced)
- 5 contracts: **startup/SMB** (<$100K, simple)
- 5 contracts: **highly negotiated** (heavy redlines, unusual terms)

---

## Data Pipeline — How to turn 250 contracts into ~6,500 training examples

### Per-contract example generation

For each contract, generate **6 types of training examples**:

| Task | Examples per contract | Total (250 contracts) |
|---|---|---|
| Full contract draft | 1 | 250 |
| Per-clause draft (8 clauses each) | 8 | 2,000 |
| Risk assessment | 1 (with 8 categories) | 250 |
| Clause checklist | 1 (with 12 standard clauses) | 250 |
| Key terms extraction | 1 | 250 |
| Redline suggestions | 5 (5 weak clauses each) | 1,250 |
| **Negative examples** (anti-patterns) | 2 per contract | 500 |
| **TOTAL NEW** | | **~4,750** |

Combined with existing 1,827: **~6,577 total examples**.

### Example structure for each task

**1. Full contract draft example**
```json
{
  "conversations": [
    {
      "role": "system",
      "content": "You are a senior US commercial lawyer. Draft a complete SaaS Subscription Agreement governed by California law. Output as numbered articles with sub-clauses."
    },
    {
      "role": "user",
      "content": "Draft a SaaS subscription agreement between [Provider] (B2B SaaS company providing workflow automation) and [Customer] (mid-market retail enterprise). Include: $50K/year subscription fee, 12-month initial term with auto-renewal, 99.9% uptime SLA, 4-hour critical support response, Provider retains platform IP with Customer receiving limited license, mutual indemnification, liability cap at 12 months fees, governed by California law."
    },
    {
      "role": "assistant",
      "content": "ARTICLE 1 — DEFINITIONS\n1.1 \"Authorized Users\" means..."
    }
  ],
  "metadata": {
    "contract_type": "SAAS_SUBSCRIPTION",
    "jurisdiction": "California, USA",
    "deal_size": "mid_market",
    "source": "edgar_atlassian_2024",
    "quality_score": 5
  }
}
```

**2. Per-clause draft example** (most valuable — teaches clause-specific patterns)
```json
{
  "conversations": [
    {
      "role": "system",
      "content": "Draft an Intellectual Property clause for a SaaS Subscription Agreement. The Provider must retain all platform rights; Customer receives limited license only."
    },
    {
      "role": "user",
      "content": "Contract type: SaaS Subscription. Provider: CloudFlow Inc. Customer: RetailCo. Draft 4 numbered sub-clauses for IP rights."
    },
    {
      "role": "assistant",
      "content": "1. Platform Ownership. CloudFlow Inc. retains all right, title, and interest in and to the Platform...\n2. License Grant. Subject to Customer's payment of fees, CloudFlow Inc. grants Customer a limited, non-exclusive, non-transferable, non-sublicensable license to access and use the Platform...\n3. Customer Data. As between the Parties, Customer retains all rights in Customer Data...\n4. Feedback. Customer agrees that any suggestions or feedback provided to CloudFlow Inc. may be used without restriction..."
    }
  ],
  "metadata": {
    "contract_type": "SAAS_SUBSCRIPTION",
    "clause_type": "IP_RIGHTS",
    "jurisdiction": "California, USA"
  }
}
```

**3. Risk assessment example** (teaches structured JSON output)
```json
{
  "conversations": [
    {
      "role": "system",
      "content": "Analyze the contract clauses and output risk assessment in pipe-delimited format: CATEGORY|RATING|JUSTIFICATION|REFERENCE"
    },
    {
      "role": "user",
      "content": "[Full contract text with clear articles]"
    },
    {
      "role": "assistant",
      "content": "LIABILITY|MEDIUM|Liability cap at 12 months fees with carve-outs for IP, gross negligence, confidentiality breach is reasonable|Article 8\nINDEMNITY|LOW|Mutual indemnity with proper notice and defense control procedures|Article 9\nIP_RIGHTS|LOW|Clear Provider ownership of Platform with limited Customer license, appropriate for SaaS|Article 7\nTERMINATION|MEDIUM|60-day convenience termination is short for enterprise SaaS|Article 11\nFORCE_MAJEURE|LOW|Standard force majeure with 30-day suspension trigger|Article 12\nDATA_PROTECTION|MEDIUM|CCPA compliance referenced but no specific breach notification timeline|Article 13"
    }
  ]
}
```

**4. Clause checklist example**
```json
{
  "conversations": [
    {
      "role": "system",
      "content": "Review the contract against standard SaaS clauses. Output one line per clause: CLAUSE_NAME|STATUS|FINDING where STATUS is PRESENT, MISSING, or WEAK"
    },
    {
      "role": "user",
      "content": "[Full contract]"
    },
    {
      "role": "assistant",
      "content": "DEFINITIONS|PRESENT|Comprehensive definitions including Authorized Users, Subscription Term, Customer Data\nSERVICES|PRESENT|Clear scope with platform features\nFEES|PRESENT|Subscription fee with billing cycle and late payment terms\nUPTIME_SLA|PRESENT|99.9% uptime with service credits\nSUPPORT|PRESENT|Tiered support with response times\nIP_RIGHTS|PRESENT|Provider retains platform IP, customer receives license\nDATA_PROTECTION|PRESENT|CCPA compliance and breach notification\nCONFIDENTIALITY|PRESENT|Mutual NDA with 5-year survival\nLIABILITY|PRESENT|12-month fee cap with appropriate carve-outs\nINDEMNITY|PRESENT|Mutual indemnification\nTERMINATION|PRESENT|For cause and convenience\nGOVERNING_LAW|PRESENT|California law, San Francisco jurisdiction"
    }
  ]
}
```

**5. Key terms extraction example**
```json
{
  "conversations": [
    {
      "role": "system",
      "content": "Extract key terms from the contract. Output as labelled lines: FIELD: value"
    },
    {
      "role": "user",
      "content": "[Contract text]"
    },
    {
      "role": "assistant",
      "content": "PARTY_A: CloudFlow Inc. (Provider)\nPARTY_B: RetailCo Inc. (Customer)\nEFFECTIVE_DATE: January 1, 2026\nTERM: 12 months with automatic 12-month renewals\nSUBSCRIPTION_FEE: $50,000 per annum, billed monthly\nIMPLEMENTATION_FEE: $10,000 one-time\nUPTIME_SLA: 99.9% monthly\nSUPPORT_RESPONSE: 4 hours critical, 1 business day non-critical\nLIABILITY_CAP: 12 months of fees paid\nGOVERNING_LAW: California, USA\nDISPUTE_RESOLUTION: Arbitration in San Francisco, CA"
    }
  ]
}
```

**6. Redline / improvement example**
```json
{
  "conversations": [
    {
      "role": "system",
      "content": "Improve the weak clause to better protect the client's interests. Output the improved clause."
    },
    {
      "role": "user",
      "content": "Weak clause: 'Vendor will use commercially reasonable efforts to maintain service availability.'\nClient position: We need a hard SLA commitment with service credits."
    },
    {
      "role": "assistant",
      "content": "Service Level Commitment. Vendor shall maintain a minimum service availability of ninety-nine and nine-tenths percent (99.9%) measured monthly, calculated as (Total Minutes - Downtime) / Total Minutes. Scheduled maintenance windows announced at least 48 hours in advance shall not count toward Downtime. If Vendor fails to meet the SLA in any calendar month, Customer shall be entitled to service credits as follows: (i) below 99.9%: 5% of monthly fee; (ii) below 99.5%: 10%; (iii) below 99%: 25%. Service credits are Customer's sole remedy for SLA breaches except in the case of three (3) consecutive months of SLA failure, which shall constitute material breach entitling Customer to terminate for cause without further notice."
    }
  ]
}
```

**7. NEGATIVE examples** (most powerful for fixing v1 failures) — show wrong → right
```json
{
  "conversations": [
    {
      "role": "system",
      "content": "You are reviewing a draft IP clause. Identify why it is wrong for a SaaS contract and rewrite it correctly."
    },
    {
      "role": "user",
      "content": "Draft IP clause for a SaaS subscription agreement: 'All deliverables, work product, documents, and materials created specifically for the Customer under this Agreement shall vest in and be owned by the Customer upon payment of the applicable fees in full.'"
    },
    {
      "role": "assistant",
      "content": "ANALYSIS: This clause is INCORRECT for a SaaS subscription agreement. It uses services/consulting language ('deliverables vest in Customer') which is appropriate for custom development work but wrong for SaaS where the Provider retains all rights to the platform.\n\nIn SaaS contracts:\n- The Platform is Provider-owned IP\n- Customer receives a limited license (not ownership)\n- Customer Data is Customer-owned\n- Improvements to the Platform belong to Provider\n\nCORRECTED CLAUSE:\n1. Platform Ownership. Provider retains all right, title, and interest in and to the Platform, including all intellectual property rights therein. No rights are granted to Customer except as expressly set forth herein.\n2. License Grant. Subject to Customer's compliance with this Agreement and payment of all fees, Provider grants Customer a limited, non-exclusive, non-transferable, non-sublicensable license during the Subscription Term to access and use the Platform solely for Customer's internal business purposes.\n3. Customer Data. As between the Parties, Customer retains all right, title, and interest in and to Customer Data. Customer grants Provider a limited license to use Customer Data solely to provide the Platform and Services."
    }
  ],
  "metadata": {
    "example_type": "negative_to_correction",
    "anti_pattern": "services_ip_in_saas",
    "contract_type": "SAAS_SUBSCRIPTION"
  }
}
```

---

## Data Generation Pipeline

### Step 1: Source raw contracts (1-2 days)
```bash
# Use existing EDGAR script with type filters
python scripts/01_fetch_edgar.py --type SAAS --count 25
python scripts/01_fetch_edgar.py --type MSA --count 25
# ... for each of the 10 types

# Output: data/raw/<type>/contract_001.txt
```

### Step 2: Clean and segment (1 day)
- Strip exhibit headers, page numbers, OCR artifacts
- Identify article boundaries
- Extract metadata (parties, jurisdiction, governing law)
- Score quality 1-5; reject scores below 3

### Step 3: Generate task examples with Claude (2-3 days)

Use **Claude Sonnet 4.5** as the data generator. For each contract, run 6 prompts:

```python
# Pseudo-code
for contract in contracts:
    examples.append(generate_full_draft_example(contract))
    examples.extend(generate_per_clause_examples(contract))  # 8 per contract
    examples.append(generate_risk_assessment(contract))
    examples.append(generate_checklist(contract))
    examples.append(generate_extraction(contract))
    examples.extend(generate_redline_examples(contract))  # 5 per contract
    examples.extend(generate_negative_examples(contract))  # 2 per contract
```

Cost estimate: 250 contracts × 6 prompts × ~5K tokens each = 7.5M tokens × Claude rates ≈ **$30-50**

### Step 4: Human review (1 day)
- Spot-check 10% of generated examples
- Reject obvious errors
- Fix systematic issues in the generation prompts and regenerate

### Step 5: Format and split
```bash
# Convert to Gemma chat template
python scripts/format_for_gemma.py --input data/v2_raw/ --output data/v2_gemma/
# 90/10 train/val split with stratification by contract_type and task
```

---

## Training Configuration (v2)

```python
# Continue from v1 adapter to preserve learning
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="jyoti0512shuklaorg/gemma4-legal-v1",  # NOT base — your v1 adapter
    max_seq_length=4096,
    load_in_4bit=True,
)

# LoRA already exists — no need to add new adapter
# But if continuing, may want to use slightly lower rank
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # Same as v1
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,
    lora_dropout=0.05,
)
```

```python
# Mix old + new data to prevent catastrophic forgetting
old_data = load_dataset("jyoti0512shukla/legal-finetune-v1", split="train")  # 1,827
new_data = load_dataset("jyoti0512shukla/legal-finetune-v2", split="train")  # 4,750
combined = concatenate_datasets([old_data, new_data])  # 6,577
```

```python
trainer = SFTTrainer(
    train_dataset=combined,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,           # FEWER than v1 (3) to prevent overfitting
        learning_rate=5e-5,           # LOWER than v1 (1e-4) — gentler updates
        bf16=True,
        warmup_steps=50,
        weight_decay=0.01,
        save_steps=200,
        eval_steps=200,
        evaluation_strategy="steps",  # Monitor val loss for catastrophic forgetting
    ),
)
```

---

## Validation Strategy — Catch Regressions Before Shipping

Build a held-out test set BEFORE training that exercises every failure mode:

```python
TEST_PROMPTS = [
    # SaaS IP test (v1 fails this)
    {
        "id": "saas_ip_correct",
        "task": "draft",
        "input": "Draft IP clause for SaaS subscription, Provider retains platform IP",
        "must_contain": ["Provider retains", "limited license", "non-exclusive"],
        "must_not_contain": ["vest in Customer", "deliverables", "Statement of Work"]
    },
    # No India hallucination
    {
        "id": "ca_no_india",
        "task": "draft",
        "input": "Draft payment clause for California vendor agreement",
        "must_not_contain": ["MSME", "Indian Contract Act", "RBI", "rupees", "INR"]
    },
    # No ERISA in assignment
    {
        "id": "assignment_no_erisa",
        "task": "draft",
        "input": "Draft assignment clause for commercial contract",
        "must_not_contain": ["ERISA", "412(p)", "pension"]
    },
    # SaaS must have SLA terms
    {
        "id": "saas_has_sla",
        "task": "draft",
        "input": "Draft SaaS service level clause",
        "must_contain": ["uptime", "%", "support", "response time"]
    },
    # Risk JSON format
    {
        "id": "risk_json_format",
        "task": "risk",
        "input": "[contract with clear liability cap]",
        "must_match_regex": r"^\w+\|\w+\|.*\|.*$"
    },
    # ... 30-50 such tests covering every failure mode
]
```

Run before training (baseline) and after training (v2). Don't ship if regression > 5%.

---

## Effort and Cost Summary

| Phase | Time | Cost |
|---|---|---|
| Source 250 contracts | 1-2 days | $0 (EDGAR free) |
| Clean and segment | 1 day | $0 |
| Generate examples with Claude | 2-3 days | $30-50 (Claude API) |
| Human review (10%) | 1 day | $0 |
| Format + split | 0.5 day | $0 |
| Build validation test set | 1 day | $0 |
| RunPod A100 training (4 hrs) | 4 hours | $10 |
| Evaluation | 1 day | $0 |
| **TOTAL** | **~7-9 days** | **~$40-60** |

---

## Expected Improvements (v2 vs v1)

| Metric | v1 (current) | v2 (target) |
|---|---|---|
| Correct IP clause for SaaS | ❌ Fails | ✅ Passes |
| No MSME in California contracts | ❌ Hallucinates | ✅ Clean |
| No ERISA in assignment clauses | ❌ Hallucinates | ✅ Clean |
| SaaS has SLA + support terms | ❌ Missing | ✅ Present |
| Definitions deduplication | ❌ Duplicates | ✅ Unique |
| Risk assessment accuracy | ⚠️ Hallucinates missing clauses | ✅ Reads document |
| Long clause coherence | ❌ Word salad after 200 tokens | ⚠️ Improved (model limit remains) |

---

## Key Principles for "Hugely Better" Results

1. **Negative examples are gold.** 500 negative examples will fix more failures than 5,000 positive examples. The model learns "don't do X" much faster than "always do Y".

2. **Clause-specific examples > full contract examples.** A model that's seen 2,000 IP clauses across 10 contract types learns IP allocation patterns deeply. A model that's seen 250 full contracts learns surface structure.

3. **Diversity within each type.** 25 enterprise SaaS contracts that all look similar teach less than 25 SaaS contracts spanning startup → enterprise.

4. **Contract type metadata in system prompts.** Make the contract type explicit at training time (`"You are drafting a SaaS Subscription Agreement"`) and at inference time. Don't make the model guess.

5. **Mix old data during training.** 50/50 old/new in each batch prevents forgetting risk/checklist/extraction skills.

6. **Validate before shipping.** A held-out test set catches regressions you can't see in training loss.

---

## Next Steps

1. Confirm the 10 contract types with you
2. I'll write `scripts/02_fetch_v2_corpus.py` to pull from EDGAR
3. I'll write `scripts/03_generate_v2_examples.py` (Claude API)
4. I'll write `scripts/04_validate_v2.py` (test set runner)
5. You source any contracts EDGAR can't provide
6. We run the generation pipeline
7. Push v2 dataset to git
8. Train on RunPod
9. Validate
10. Deploy if validation passes
