# Contract generation rules — re-read before every contract

## Diversity dimensions (deliberately vary)
- **Industry**: each new contract should add a new vertical or a 2nd contract in a vertical that already exists with a different angle. No 3 contracts in the same vertical with the same angle.
- **Deal size**: balance smb / mid_market / enterprise across the corpus.
- **Negotiation bias**: balance provider_friendly / balanced / customer_friendly. Provider-friendly is currently underrepresented.
- **Drafting style**: each new contract is assigned to a specific style and must commit to it (biglaw_formal, plain_english_common_paper, click_through_tos, modular_order_form, customer_paper_negotiated, modern_tech_minimalist).
- **Customer type**: vary across generic corp / LLC / state agency / regulated bank / law firm / professional corporation / broker dealer / non-profit / school district / etc.
- **Edge case**: every contract bakes in at least one distinguishing feature (source code escrow, MFN, audit rights, freemium-to-paid, multi-year escalator, sub-processor list, BYO-key, sovereign immunity, regulator pass-through, validation requirements, hardware hybrid, etc.). No two contracts in the same bucket should share the same edge case.

## Quality rules (always enforce)
- **Placeholder-first generation**: write the tokenized form directly. No raw values for entity fields. Tags: `{{PROVIDER_NAME}}`, `{{PROVIDER_STATE}}`, `{{PROVIDER_ENTITY_TYPE}}`, `{{PROVIDER_ADDRESS}}`, `{{CUSTOMER_NAME}}`, `{{CUSTOMER_STATE}}`, `{{CUSTOMER_ENTITY_TYPE}}`, `{{CUSTOMER_ADDRESS}}`, `{{EFFECTIVE_DATE}}`, `{{GOVERNING_LAW_STATE}}`, `{{VENUE_COUNTY}}`, `{{VENUE_CITY}}`, `{{ANNUAL_FEE_AMOUNT}}`, `{{TERM_YEARS}}`, `{{LIABILITY_CAP_MONTHS}}`, `{{NOTICE_DAYS}}`, `{{IMPLEMENTATION_FEE}}`.
- **No real-company addresses or names** in template body. Default values in entity_maps may use plausible-sounding names but must not match real entities — when in doubt, prefix with the contract's invented brand (`1 ForgeRunner Plaza`).
- **Regulatory frameworks stay literal**: HIPAA, FERPA, COPPA, OSFI, 21 CFR Part 11, GLBA, PCI-DSS, SOX, etc. are part of the template's character. Don't try to make them placeholders. The metadata pins which jurisdiction the template is for; the variation generator locks the customer state to match.
- **No PharmaTrack-class bugs**: a sentence describing the customer must use `{{CUSTOMER_*}}` slots, never `{{PROVIDER_*}}`. By construction, every entity reference is bound by position.
- **ALL CAPS legal disclaimers**: use the defined terms `CUSTOMER` and `PROVIDER`, not `{{CUSTOMER_NAME}}` / `{{PROVIDER_NAME}}`. Mixed case party names inside ALL CAPS paragraphs read badly.
- **Date stitching**: do not write `the {{EFFECTIVE_DATE}}` — drop the article. Same for `the {{TERM_YEARS}} years` (which works) vs `the {{TERM_YEARS}}-year period` (which works).
- **Article structure**: keep `ARTICLE N — TITLE` headers (em dash) so the clause extractor finds them. Standard layout for a SaaS contract:
  - ARTICLE 1 — DEFINITIONS
  - ARTICLE 2 — SUBSCRIPTION AND LICENSE (or "SERVICES AND SUBSCRIPTION")
  - ARTICLE 3 — IMPLEMENTATION
  - ARTICLE 4 — FEES AND PAYMENT
  - ARTICLE 5 — TERM AND RENEWAL
  - ARTICLE 6 — SERVICE LEVEL AGREEMENT AND SUPPORT
  - ARTICLE 7 — CUSTOMER DATA AND PRIVACY (extend with regulatory framework name where relevant: HIPAA, FERPA, PCI, etc.)
  - ARTICLE 8 — INTELLECTUAL PROPERTY RIGHTS
  - ARTICLE 9 — CONFIDENTIALITY
  - ARTICLE 10 — REPRESENTATIONS AND WARRANTIES
  - ARTICLE 11 — INDEMNIFICATION
  - ARTICLE 12 — LIMITATION OF LIABILITY
  - ARTICLE 13 — TERMINATION
  - ARTICLE 14 — GOVERNING LAW AND DISPUTE RESOLUTION
  - ARTICLE 15 — GENERAL PROVISIONS
- **Length target**: 4000–6500 words for biglaw_formal / customer_paper_negotiated / modular_order_form. 3000–4500 words for plain_english / click_through / modern_tech_minimalist.

## Pre-generation checklist (run mentally before writing each contract)
1. What industry / vertical? Does it duplicate an existing contract? If yes, what's the differentiating angle?
2. What deal size? Does the corpus need more of this size?
3. What negotiation bias? Does the corpus need more of this bias?
4. What drafting style? Am I committing to it?
5. What customer type? Is it a special type that needs preservation?
6. What edge case is unique to this contract? Have I used it elsewhere?
7. What regulatory framework is hardcoded? Does it match the customer type and jurisdiction?
8. What's the entity_map default value set? Do the company names not collide with real entities?
