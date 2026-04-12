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
5. What customer type? Is it a special type that needs preservation? **If so, put it in the CUSTOMER slot, not the PROVIDER slot** (the variation generator only preserves CUSTOMER fields via `SPECIAL_CUSTOMER_TYPES`). If the special party is structurally on the Provider side (e.g. a Bank that is itself the Discloser), use `party_roles` to relabel and accept that the variation generator will rename the Provider — or extend the generator to support a `provider_special_type` flag.
6. What edge case is unique to this contract? Have I used it elsewhere?
7. What regulatory framework is hardcoded? Does it match the customer type and jurisdiction?
8. What's the entity_map default value set? Do the company names not collide with real entities?
9. **Hardcoded party nicknames**: when defining parties in the preamble, use generic role labels like `("Party A")`, `("Company")`, `("Vendor")`, `("Employee")` — **never** use a literal nickname like `("StrataCore")` or `("Halcyon")` because the variation generator will rename `{{PROVIDER_NAME}}` and the body will then reference a nickname that no longer matches the renamed party.

## Per-category infrastructure checklist (run ONCE before writing the first contract of a new category)

These are one-time setup steps for each new contract type. The NDA bucket uncovered every item below the hard way; do them up front for the next category.

1. **Add the contract type to `AGREEMENT_LABEL_BY_TYPE`** in `v2_generate_drafting_examples.py` so the brief opener has a sensible default label.
2. **Add per-type clause overrides to `CLAUSE_DISPLAY_NAMES_BY_TYPE`** if any clause needs a different display name for this contract type (e.g. MSA's `intellectual_property` is "Work Product and IP Ownership").
3. **Add NEW clause types to `CLAUSE_TITLE_KEYWORDS`** for any article-naming convention this category introduces. Use specific multi-word keywords — `"PURPOSE"` is fine but `"NO LICENSE OR OWNERSHIP TRANSFER"` is better than just `"LICENSE"` because of substring collisions. **Verify the longest possible keyword wins** (the extractor uses `len(kw) > existing._kw_len`) — if two clause types both contain a shared word, the more specific keyword must be longer.
4. **Add NEW clause types to `CLAUSE_STANDARD_PROVISIONS`** with provisions that describe what the *body* of that clause is supposed to demonstrate — **not** what is defined or addressed in a different article of the same contract.
5. **Add per-type overrides to `CLAUSE_STANDARD_PROVISIONS_BY_TYPE`** wherever the standard provisions for a clause type differ from the generic list (e.g. MSA `intellectual_property` is work-product assignment, not SaaS license-grant).
6. **Add per-clause-type entries to `select_clause_types()`** so this contract type's "universal" clauses and rotating clauses are picked correctly. NDAs use `(confidentiality, term_survival, equitable_remedies)` as universals; SaaS/MSA use `(intellectual_property, limitation_of_liability, termination)`. Each new type should have its own set.
7. **Update `_PROVISION_KEYWORDS`** for every new provision you add. Use **specific phrases** as keywords, not generic single words. Avoid keywords that match common contract vocabulary (`confidential`, `purpose`, `retain`) because they create Defense 2 false positives (see Lesson 2 below).
8. **Update `relevant_kw` map in `build_brief()`** to include the new clause types. Use keywords that describe the clause's *substance*, not generic vocabulary that appears in many feature descriptions.
9. **Add per-type label handling to `build_brief()`** if the new contract type has different commercial terms (NDAs skip the fee block; DPAs probably will too; SOWs use a "Statement of Work fee" label).
10. **Standardize placeholder semantics across all templates of a category**: pick one meaning for `{{TERM_YEARS}}` (initial term OR survival period, not both) and stick with it for every template. If a particular template needs the other meaning, use a metadata flag like `suppress_initial_term: true` (NDA pattern) and special-case the brief.
11. **Run the generator on the first contract you write and inspect the extracted clause map** — `extract_clauses(text)` should return the clause types you expected. If not, your article titles don't match the keywords. Fix before scaling.

## Per-template authoring checklist (run for each contract template)

1. **Article titles use `ARTICLE N — TITLE` with em dash** (not hyphen) so the regex finds them.
2. **Article titles are unambiguous**: don't title an article `EXCLUSIONS FROM CONFIDENTIALITY` if you also have a `CONFIDENTIAL INFORMATION` article — the substring `CONFIDENTIALITY` will match both and one will silently overwrite the other in the extractor. Either rename the exclusions article to plain `EXCLUSIONS` or make sure the more specific keyword is longer.
3. **No hardcoded party nicknames** in the body. Use the placeholders or generic role labels. Grep for any literal company name before saving.
4. **Placeholder semantics match the category convention** (see infrastructure checklist item 10).
5. **Section numbering matches the article body** — sections numbered `8.1`, `8.2`, etc. inside `ARTICLE 8 — TITLE` (not `1.1`, `1.2`).
6. **Defined-term parentheticals** use generic roles or contract-type-appropriate labels: `(the "Provider")`, `(the "Customer")`, `(the "Disclosing Party")`, `(the "Receiving Party")`, `(the "Party A")`, `(the "Employee")`, etc.

## Per-batch validation checklist (run after writing 4-5 contracts of a new category)

This catches the recurring issues from the NDA bucket before they propagate to 25 contracts.

1. **Generator clean run**: `python3 scripts/v2_generate_drafting_examples.py --type <TYPE>` should produce 4 examples per contract with no warnings about missing clauses or leftover placeholders.
2. **Spot-check clause extraction**: pick one contract and run `extract_clauses(template_text)` — the keys should match the clauses you intended (e.g. `confidentiality` should map to your Confidentiality article, not your Exclusions article).
3. **Coverage round-trip audit**: run the cross-bucket coverage check from the NDA workflow. Drop rate should be 10–25%, mismatches should be 0. Higher drop rate means brief lists have provisions that bodies don't demonstrate (Defense 2 working). Lower drop rate near 0% with PARTIAL examples likely means false-positive keyword matches (see Lesson 2).
4. **Brief inspection**: print 2-3 briefs and read them as a human would. Specifically check:
   - Party block reads sensibly (special customer type preserved? party_roles correct?)
   - Commercial / term block uses the right label for this contract type
   - "Specific requirements:" line — if present, is the requirement actually demonstrated in this clause's body? If it belongs in a different clause, fix the `relevant_kw` map.
   - Standard provisions list — do the listed provisions belong in *this* clause's body, or are they describing what's defined in a different article?
5. **LLM-as-judge spot audit**: pick 8 (brief, output) pairs spanning the styles you've written and read each one carefully against this rubric:
   - **Coverage**: every standard provision in the brief is genuinely demonstrated in the body (not just keyword-matched)
   - **Faithfulness**: parties, jurisdictions, numbers in the brief match the body
   - **Style**: the body actually reads like the declared drafting style
   - **Specific requirements**: the "Specific requirements:" line is addressed in the body
   - **No hallucination / no leakage**: the body doesn't introduce off-topic content; no unrelated special features bleed into the brief
6. **Fix any issues found before writing the rest of the bucket** — the cost of fixing 4 templates is much lower than fixing 25.

## Lessons learned from the NDA bucket (apply to all future categories)

These are the recurring failure patterns we discovered in the NDA-bucket LLM-as-judge audit. They are NOT obvious from the diversity / quality rules above, and they cost a meaningful amount of debugging when discovered late.

### Lesson 1: A clause's standard-provision list should describe what THAT clause's body demonstrates — not what is defined or addressed in a different article

Concrete examples:
- "definition of Confidential Information" does NOT belong in the `confidentiality` clause's provision list, because the definition lives in `Article 1 — DEFINITIONS`. The `confidentiality` clause body demonstrates the *obligations* (use, disclosure, care, notification), not the definition.
- "no transfer of IP or ownership to the Receiving Party" does NOT belong in `confidentiality`, because it lives in the `no_license` article.
- "defined Purpose stated narrowly" does NOT belong in the `purpose` clause's provision list, because the Purpose itself is defined in `Article 1 — DEFINITIONS`. The `purpose` clause body demonstrates the *use restrictions* (solely for Purpose, no competitive use, no reverse engineering).

**Rule**: when adding provisions to a new clause type, ask "would this provision be in the *body* of this article, or is it defined / addressed elsewhere?" If elsewhere, do not list it.

### Lesson 2: Defense 2 keyword matching has known false-positive modes — design keywords to avoid them

The filter uses `any(kw in body)` substring matching. Three failure modes to avoid:

- **Generic single words match too much**: keywords like `"confidential"`, `"purpose"`, `"retain"`, `"reasonable"` appear in nearly every NDA body and cause provisions to be kept even when the body doesn't actually demonstrate them. Use phrases (3+ words) or longer terms.
- **Substrings without word boundaries**: `"retain"` matches `"retained"`, `"retainer"`, `"retains"` — sometimes correctly, sometimes not. If precision matters, use a longer phrase like `"retain ownership"` or `"retains all rights"`.
- **Generic vocabulary in the `relevant_kw` map** (the special-features filter): keywords like `"confidential"` or `"trade secret"` match almost any feature description that mentions Confidential Information, and cause unrelated features to be selected for a clause's brief. Use clause-substantive keywords like `"data room"`, `"clean team"`, `"watermark"`, `"PIPEDA"`, `"whistleblower"`, `"Procurement Integrity Act"` — things that uniquely identify which clause owns the concept.

**Rule**: every keyword should be specific enough that it would NOT match a feature description belonging to a different clause. When in doubt, prefer longer phrases.

### Lesson 3: `special_features` are contract-level — map each one to ONE clause, not all of them

The original `build_brief()` had a fallback that picked a random `special_feature` for any clause whose `relevant_kw` map didn't match — this leaked features into clauses where they didn't belong (e.g. asking the No License clause to "address" the DTSA whistleblower notice that actually belongs in Confidentiality).

**Rule**: when authoring a contract's `special_features` list, mentally tag each feature with the ONE clause whose body should demonstrate it. Then verify the `relevant_kw` map for that clause includes a keyword that matches the feature's description. **If no clause fits, the feature is contract-level prose for the recitals and should not appear in any clause-level brief.** The new `build_brief()` correctly OMITs the special_block when no relevant feature matches — keep that behavior; do not reintroduce a random fallback.

### Lesson 4: Placeholder semantics must be CONSISTENT across all templates of a category

In the NDA bucket, 24 of 25 templates used `{{TERM_YEARS}}` for the agreement's initial term, but the Atlas employee NDA used it for the post-employment survival period (because the agreement runs "throughout employment" with no specific term). This created a labeling inconsistency the brief builder couldn't detect.

**Rule**: pick one meaning for each placeholder in a new contract category and stick to it across every template. If a template genuinely needs a different meaning, set a metadata flag (the NDA pattern is `suppress_initial_term: true`) and special-case the brief builder. Document the canonical meaning in the per-category structural rules section of this file.

### Lesson 5: Hardcoded party nicknames in the template body break the variation generator

If you write `{{PROVIDER_NAME}} ("StrataCore")` in the preamble, then later sections that say "StrataCore shall..." will be referring to a name the variation generator just renamed. The body becomes incoherent on every variation.

**Rule**: defined-term parentheticals in the preamble must use generic role labels (`("Party A")`, `("Company")`, `("Vendor")`, `("Employee")`, `("Acquirer")`, etc.), not literal nicknames. Use the role label everywhere in the body too. Grep your template for any literal company name before saving.

### Lesson 6: `SPECIAL_CUSTOMER_TYPES` only preserves the CUSTOMER side — orient the entity map accordingly

The Keystone Canadian Bank NDA put the Bank in the PROVIDER slot because it was the Discloser, but `make_constrained_variation` only preserves CUSTOMER fields for special customer types. The variation generator renamed the Bank to a generic "Apex Workflow Platform, Inc." every run, destroying the regulatory framing.

**Rule**: if a party needs identity preservation (a regulated bank, a state agency, a Big 4 audit firm, etc.), put it in the CUSTOMER slot of the entity_map even if it is conceptually the Discloser / Provider in the contract. Use `party_roles` to relabel: `{"provider": "Vendor", "customer": "Bank"}`. The flow of disclosure / receipt is determined by the contract body, not by which slot the party occupies in the entity_map.

(Alternative: extend `SPECIAL_CUSTOMER_TYPES` to support a `provider_special_type` flag and preserve PROVIDER fields. This is the cleaner long-term fix — file as future work when needed.)

### Lesson 7: Article titles must be substring-disjoint within their clause-type group

The clause extractor uses substring keyword matching: `if kw in section_title_upper`. If two clause types both have keywords that appear in the same article title, the longer keyword wins via the `_kw_len` tiebreaker — but ONLY if both are registered. The Phase 1 NDAs had `confidentiality` matching `EXCLUSIONS FROM CONFIDENTIALITY` (because `CONFIDENTIALITY` is a substring of that title) and silently overwriting the real Confidential Information article.

**Rule**: when picking article titles for a new contract type, sanity-check that no title contains another clause type's keyword as a substring — and if it does, ensure the more specific keyword is longer. After writing the first template, verify with `extract_clauses(template_text)` that each clause type maps to the article you intended.

### Lesson 8: Run an LLM-as-judge audit on a sample after each new category — keyword matching is necessary but not sufficient

Defense 2 catches the gross failure modes (brief asks for X, body has nothing about X) but misses the subtler ones (brief asks for X, body has something that keyword-matches X but isn't really X). Reading 8 (brief, output) pairs as a human (or asking Claude to do it as a structured audit) catches issues that no automated check will.

**Rule**: after writing 4-5 contracts of a new category, sample 8 (brief, output) pairs spanning all the drafting styles you've written and apply the rubric in the per-batch validation checklist above. Fix any issues before scaling to the full bucket.

## Coverage discipline — defense in depth

The v2-adapter eval exposed a recurring failure: the brief asks for N
standard provisions, but the clause body only demonstrates M < N of them.
This teaches the model that items in a brief are optional, not required,
and the inference-time output drops provisions.

Three layers prevent this:

- **Defense 1 — pre-write clause checklist** (this document). Before writing
  any clause, consult the clause-specific checklist below and make sure the
  body covers every listed provision. Skipping a provision is fine — just
  skip it consistently (it won't be in the brief and won't be expected in
  the output).
- **Defense 2 — generator-level coverage filter** (in
  `scripts/v2_generate_drafting_examples.py`, `filter_provisions_by_coverage`).
  When `build_brief()` runs, it scans the clause body for each provision's
  keywords and drops any whose keywords don't appear. Even if a human
  author skips a provision, the brief is automatically trimmed to match.
- **Defense 3 — LLM-as-judge validation** (optional, only if residual drift
  shows up). Run a post-hoc audit that asks Claude to score each generated
  example for brief↔output alignment.

## NDA clause checklists (Defense 1)

### `confidentiality` — the core obligation clause
- [ ] Definition of Confidential Information (marked, identified, or what a reasonable person would treat as confidential)
- [ ] Use restriction — Confidential Information used SOLELY for the defined Purpose
- [ ] Disclosure restriction — limited to personnel with need to know who are bound by equivalent confidentiality
- [ ] Standard of care — at least the care the Receiving Party uses for its own, but no less than reasonable care
- [ ] No transfer of ownership or IP to the Receiving Party
- [ ] Obligation to notify the Disclosing Party of any unauthorized disclosure

### `purpose` — the narrow scope definition clause
- [ ] Defined Purpose stated narrowly (e.g. "evaluating a potential acquisition of Company")
- [ ] Use restriction limited to the Purpose
- [ ] No use for competitive advantage or unrelated business
- [ ] No reverse engineering of any tangible Confidential Information

### `exclusions` — what is NOT confidential
- [ ] Information already public without breach of this Agreement
- [ ] Information already known to the Receiving Party prior to disclosure
- [ ] Information independently developed without use of Confidential Information
- [ ] Information rightfully received from a third party without duty of confidence

### `compelled_disclosure` — what happens when a court orders disclosure
- [ ] Prompt written notice to the Disclosing Party before disclosure (where legally permitted)
- [ ] Cooperation with the Disclosing Party's efforts to obtain a protective order
- [ ] Disclosure limited to the portion legally required
- [ ] Continued confidentiality over non-disclosed portions

### `return_or_destruction` — what happens at end of term
- [ ] Return or destruction of all Confidential Information on the Disclosing Party's request
- [ ] Written certification of destruction on request
- [ ] Exception for archival backups and bona fide records retention
- [ ] Continued confidentiality obligations over any retained copies

### `term_survival` — how long the obligations last
- [ ] Term of the Agreement (end date or ongoing relationship)
- [ ] Survival of confidentiality obligations for a defined period (typically 3-5 years) after the term
- [ ] Trade secret carve-out — trade secret obligations survive as long as the information remains a trade secret
- [ ] Termination for material breach

### `no_license` — the "we are not transferring IP" clause
- [ ] No transfer of any ownership interest
- [ ] No license granted by implication, estoppel, or otherwise
- [ ] All rights reserved by the Disclosing Party
- [ ] Disclosing Party retains all patent, copyright, trademark, and trade secret rights

### `equitable_remedies` — the injunction clause
- [ ] Acknowledgment that breach may cause irreparable harm
- [ ] Right to seek injunctive relief without posting bond
- [ ] No obligation to prove actual damages
- [ ] Cumulative remedies in addition to any at law or in equity

## NDA-specific structural rules
- **Parties**: "Disclosing Party" and "Receiving Party" for one-way NDAs. For mutual NDAs, use "Party" / "Other Party" and swap language so either side can be the Discloser.
- **Purpose**: EVERY NDA must have a narrow, specifically-drafted Purpose definition — not "general business discussions". Make it specific: "evaluating a potential acquisition of Target by Acquirer", "evaluating the Vendor's platform for potential subscription", etc.
- **Article structure for a standalone NDA** (typically 8-12 articles, much shorter than SaaS/MSA):
  - ARTICLE 1 — DEFINITIONS
  - ARTICLE 2 — PURPOSE
  - ARTICLE 3 — CONFIDENTIAL INFORMATION
  - ARTICLE 4 — EXCLUSIONS FROM CONFIDENTIALITY
  - ARTICLE 5 — OBLIGATIONS OF THE RECEIVING PARTY
  - ARTICLE 6 — COMPELLED DISCLOSURE
  - ARTICLE 7 — RETURN OR DESTRUCTION
  - ARTICLE 8 — NO LICENSE OR OWNERSHIP TRANSFER
  - ARTICLE 9 — TERM AND SURVIVAL
  - ARTICLE 10 — EQUITABLE REMEDIES
  - ARTICLE 11 — GENERAL PROVISIONS
- **Length target**: 1800–3500 words for biglaw_formal / customer_paper_negotiated NDAs. 1000–2000 words for plain_english / click_through / modern_tech_minimalist NDAs. NDAs are short by design.
- **Placeholder note**: most NDAs don't have `ANNUAL_FEE_AMOUNT` or `LIABILITY_CAP_MONTHS`. Use `EFFECTIVE_DATE`, `TERM_YEARS` (for the survival period), `GOVERNING_LAW_STATE`, `VENUE_COUNTY`, plus the party slots.

## EMPLOYMENT clause checklists (Defense 1)

### `position_duties` — title, scope of duties, location
- [ ] Executive's title, reporting line, and scope of duties
- [ ] Full-time and exclusive services commitment
- [ ] Limited carve-out for outside boards / charitable / passive investments
- [ ] Principal place of employment with relocation triggers if any

### `compensation` — base / bonus / equity / benefits
- [ ] Annual base salary with payroll cadence
- [ ] Annual target bonus with discretion / performance metrics
- [ ] Equity participation in the Company's incentive plan
- [ ] Standard employee welfare benefits (medical, dental, retirement, PTO)
- [ ] Expense reimbursement under standard Company policies

### `termination` — termination triggers (employment-flavored)
- [ ] Termination upon Executive's death or disability
- [ ] Termination by the Company for Cause (with definition cross-reference)
- [ ] Termination by the Company without Cause
- [ ] Termination by Executive for Good Reason (with definition cross-reference)
- [ ] Termination by Executive without Good Reason
- [ ] Notice and effective-date mechanics for each trigger

### `severance` — what employee gets on termination
- [ ] Termination triggers (Cause / without Cause / Good Reason / death / disability)
- [ ] Severance multiple of base salary and / or bonus
- [ ] Continuation of group health benefits (COBRA-equivalent contribution)
- [ ] General release of claims as condition of severance
- [ ] No-mitigation clause and offset rules

### `restrictive_covenants` — non-compete / non-solicit / non-disparagement
- [ ] Non-competition restriction during employment and a defined post-employment period (subject to state-law enforceability — Cal. B&P § 16600 voids these in California)
- [ ] Non-solicitation of Company employees and customers
- [ ] Confidentiality of proprietary information surviving termination
- [ ] Non-disparagement covenant
- [ ] Tolling of restricted period for any breach by Executive

### `change_in_control` — CIC double-trigger and acceleration
- [ ] Definition of Change in Control event
- [ ] Double-trigger: termination without Cause or for Good Reason within a defined window after CIC
- [ ] Enhanced severance multiple compared to ordinary termination
- [ ] Accelerated vesting of outstanding equity awards
- [ ] Section 280G best-net cutback or gross-up treatment

### `confidentiality` — employment-specific override
- [ ] Confidential information of the Company defined to include trade secrets, customer data, financials, and personnel information
- [ ] Executive's duty to hold confidential information in strict confidence during and after employment
- [ ] No use of confidential information for any purpose other than performance of Executive's duties
- [ ] Defend Trade Secrets Act whistleblower immunity notice in the required statutory form (18 U.S.C. § 1833(b))
- [ ] Return of all confidential information and Company property on termination

### `intellectual_property` — invention assignment (employment-specific override)
- [ ] Executive's assignment of all Work Product to the Company
- [ ] Works made for hire to the maximum extent permitted by law
- [ ] Executive's pre-existing inventions excluded from assignment (Prior Inventions schedule)
- [ ] Moral rights waiver to the extent permitted
- [ ] State-law carve-outs for inventions on Executive's own time using no Company resources (e.g. Cal. Lab. Code § 2870)

### `dispute_resolution` — governing law / arbitration / equitable carve-out
- [ ] Governing law of a designated state
- [ ] Binding arbitration or exclusive court venue for disputes
- [ ] Carve-out for equitable relief in support of restrictive covenants
- [ ] Waiver of jury trial
- [ ] Fee-shifting or each-party-bears-its-own-fees rule

## EMPLOYMENT-specific structural rules

- **Parties**: PROVIDER slot = Employer (Company); CUSTOMER slot = Employee (always preserved as a special customer type so the employee identity isn't replaced by a random corporation). Use `party_roles` to relabel for the brief — typically `{provider: "Company", customer: "Executive"}` or `{provider: "Company", customer: "Employee"}`. For two-party employer structures (e.g. holding company + regulated subsidiary, Cathay-style), put the operating entity in the PROVIDER slot and reference the parent in the body via a separate defined term.
- **Compensation placeholder convention** (per Lesson 4 — placeholder semantics must be consistent across the category):
  - `EMPLOYEE_BASE_SALARY` = annual base salary as a fully-formed money string (e.g. `"Four Hundred Twenty-Five Thousand United States Dollars (US$425,000)"`)
  - `EMPLOYEE_TARGET_BONUS_PCT` = numeric percentage of base, no `%` sign (e.g. `"40"` for 40%)
  - `EMPLOYEE_INITIAL_EQUITY` = initial equity grant value at signing as a money string (e.g. `"One Million Two Hundred Thousand United States Dollars (US$1,200,000)"`)
  - `EMPLOYEE_TITLE` = Executive's title as plain text (e.g. `"Executive Vice President, General Counsel and Corporate Secretary"`)
  - `EMPLOYEE_REPORTS_TO` = title of person Executive reports to (e.g. `"Chief Executive Officer"`)
- **TERM_YEARS semantics for EMPLOYMENT**: `TERM_YEARS` represents the agreement's initial fixed term where applicable (e.g. Cathay's 3-year fixed term + auto-renew). For at-will agreements with no specific term length (most US executive agreements), set `"suppress_initial_term": true` in the metadata so the brief doesn't mislabel TERM_YEARS as an initial term. For Canadian indefinite-term agreements, set `suppress_initial_term: true` and let the body describe the indefinite-term mechanic explicitly.
- **No `ANNUAL_FEE_AMOUNT`, `LIABILITY_CAP_MONTHS`, `IMPLEMENTATION_FEE`** in EMPLOYMENT entity_maps — those are SaaS / MSA concepts and have no analogue in employment agreements.
- **Standard article structure for a biglaw_formal employment agreement** (10–13 articles):
  - ARTICLE 1 — DEFINITIONS
  - ARTICLE 2 — POSITION AND DUTIES
  - ARTICLE 3 — TERM AND COMMENCEMENT
  - ARTICLE 4 — COMPENSATION AND BENEFITS
  - ARTICLE 5 — TERMINATION
  - ARTICLE 6 — SEVERANCE AND TERMINATION BENEFITS
  - ARTICLE 7 — RESTRICTIVE COVENANTS
  - ARTICLE 8 — CONFIDENTIAL INFORMATION AND INVENTION ASSIGNMENT
  - ARTICLE 9 — CHANGE IN CONTROL
  - ARTICLE 10 — REPRESENTATIONS AND WARRANTIES
  - ARTICLE 11 — DISPUTE RESOLUTION AND GOVERNING LAW
  - ARTICLE 12 — GENERAL PROVISIONS
  - (optional) ARTICLE 13 — BANKING / INDUSTRY-SPECIFIC PROVISIONS (for regulated employers)
- **Length targets**:
  - biglaw_formal / customer_paper_negotiated executive employment: 5,000–9,000 words
  - modular_order_form employment-with-offer-addendum: 4,000–6,000 words for the master + 800–1,500 for each offer addendum
  - plain_english_common_paper offer letter: 800–2,000 words (offer letters are deliberately short)
  - click_through_tos onboarding flow (Rippling/Gusto-style): 600–1,500 words
  - modern_tech_minimalist startup hire (Stripe/Notion-style): 1,000–2,500 words
- **Jurisdiction-specific must-haves**:
  - **California**: Cal. B&P § 16600 acknowledgment (no post-employment non-compete enforceable; non-solicit must be customer-specific not employee-broad), Cal. Lab. Code § 2870 invention exemption notice, accrued PTO must be paid out as wages on termination (no use-it-or-lose-it), wage statement compliance reference
  - **Massachusetts**: 2018 MA Noncompetition Agreement Act compliance — non-compete must be in writing 10 days before employment starts, garden leave OR mutually-agreed consideration, max 12 months post-employment, geographic and substantive narrowness
  - **New York**: standard at-will, no specific non-compete bar (yet), but NY follows reasonableness review
  - **Texas**: at-will, non-competes generally enforceable if supported by consideration, binding arbitration is common
  - **Canada (BC / Ontario / Quebec)**: indefinite term (no at-will), statutory severance under provincial Employment Standards Act, common-law reasonable notice, "Actively Employed" defined-term mechanic for equity vesting, possible Full and Final Release schedule
- **Defend Trade Secrets Act whistleblower notice (18 U.S.C. § 1833(b))**: REQUIRED statutory language for any employment agreement that contains confidentiality / trade secret provisions and is governed by US law. Include verbatim in the Confidentiality clause. Without it, the employer loses the right to recover exemplary damages and attorneys' fees in any DTSA misappropriation suit.
- **DEI / California-mandatory provisions**: For any California employment agreement, include at least one provision that complies with California's wage statement requirements and acknowledges the at-will nature explicitly. Avoid blanket "Employee waives X" language that California courts routinely strike.

## SOW clause checklists (Defense 1)

An SOW (Statement of Work) is a project-specific document that operates UNDER a parent Master Services Agreement. It does NOT restate framework terms (limitation of liability, indemnification, governing law, IP ownership defaults) — those live in the parent MSA. The SOW's body is purely project-specific.

### `sow_scope` — what work is being done
- [ ] Specific project description with concrete in-scope activities (NOT generic "consulting services")
- [ ] Explicit out-of-scope list — what the project will NOT cover
- [ ] Project objectives or success criteria (what done looks like)
- [ ] Reference to the parent MSA that this SOW operates under

### `deliverables_acceptance` — concrete artifacts and how Customer accepts them
- [ ] Numbered or tabular list of deliverables with descriptions
- [ ] Objective acceptance criteria for each deliverable (what "complete" means)
- [ ] Customer review and acceptance period (typically 5-10 business days)
- [ ] Provider cure right for non-conforming deliverables on rejection notice
- [ ] Deemed acceptance after the review window expires without rejection notice

### `milestones_schedule` — when work happens
- [ ] Project start date (or kickoff date)
- [ ] Target completion date
- [ ] Named milestones with target dates
- [ ] Inter-milestone dependencies where applicable
- [ ] Status reporting cadence and escalation procedure for slippage

### `fees_payment_sow` — how Provider gets paid
- [ ] Pricing structure: T&M, fixed-fee, milestone-based, or hybrid
- [ ] For T&M: rate card or blended rate with currency and any not-to-exceed cap
- [ ] For fixed-fee: total fee amount with currency
- [ ] For milestone-based: payment trigger per milestone with amount
- [ ] Expense reimbursement policy with pre-approval threshold
- [ ] Invoicing cadence (typically monthly) and payment terms (typically net 30 / net 45)
- [ ] Late payment interest or service-suspension right

### `change_control` — how scope changes are handled
- [ ] Definition of a Change Request with required content (description, impact, proposed dates)
- [ ] Provider's obligation to assess impact on scope, schedule, and fees within a defined window
- [ ] Customer's written approval required before any change is implemented (signed Change Order)
- [ ] No-oral-changes language — scope changes only by signed Change Order
- [ ] Effect of disputed Change Requests on continuing work (typically Provider continues per current scope)

### `client_responsibilities` — what Customer owes the Provider to enable the work
- [ ] Named Customer personnel and decision-makers with authority
- [ ] Timely Customer access to systems, data, facilities, and credentials
- [ ] Customer responsibility for accuracy and completeness of provided materials
- [ ] Customer review and approval turnaround commitment (e.g., 5 business days)
- [ ] Relief or extension if Customer delays prevent Provider from meeting milestones

### `assumptions_dependencies` — what the project pricing assumes
- [ ] Numbered list of assumptions on which pricing and schedule are based
- [ ] Third-party dependencies (vendors, software licenses, regulatory approvals)
- [ ] Change control trigger if any assumption proves incorrect
- [ ] Customer obligation to notify Provider of any changes to assumptions

### `personnel` — SOW-flavored override
- [ ] Named key personnel assigned to this project (with titles)
- [ ] Minimum allocation percentage of each key person to the project
- [ ] No replacement of key personnel without Customer prior written consent
- [ ] Background checks and Customer security training where applicable
- [ ] Non-solicitation of project personnel during the engagement (and a tail period)

## SOW-specific structural rules

- **Parties**: PROVIDER slot = the services firm / consultancy / vendor; CUSTOMER slot = the client. Use `party_roles` to relabel: typically `{"provider": "Provider", "customer": "Client"}` or `{"provider": "Consultant", "customer": "Client"}`. For MSAs that already define the parties as `Provider` and `Customer`, the SOW should use the same labels — the SOW is incorporated INTO the MSA and inherits its defined terms.
- **SOW operates under a parent MSA**: every SOW MUST contain an "Incorporation by Reference" clause stating that this SOW is incorporated into the parent MSA dated `{{MSA_EFFECTIVE_DATE}}`, and that conflicts between the SOW and MSA are resolved in favor of the MSA except for project-specific commercial terms (which override the MSA). Use the placeholder `{{MSA_EFFECTIVE_DATE}}` in the preamble or in the Incorporation article.
- **NO framework clauses**: SOWs do NOT contain Limitation of Liability, Indemnification, Confidentiality, IP Ownership, Governing Law, Dispute Resolution, or Insurance articles. Those live in the parent MSA. If the LLM-as-judge audit finds an SOW with any of those, it's wrong — fix the template to reference the MSA instead.
- **Standard article structure for a biglaw_formal SOW** (8–11 articles, project-focused):
  - ARTICLE 1 — PROJECT DESCRIPTION AND SCOPE
  - ARTICLE 2 — DELIVERABLES AND ACCEPTANCE
  - ARTICLE 3 — PROJECT SCHEDULE AND MILESTONES
  - ARTICLE 4 — KEY PERSONNEL
  - ARTICLE 5 — CLIENT RESPONSIBILITIES
  - ARTICLE 6 — ASSUMPTIONS AND DEPENDENCIES
  - ARTICLE 7 — FEES AND PAYMENT
  - ARTICLE 8 — CHANGE CONTROL
  - ARTICLE 9 — INCORPORATION OF MASTER SERVICES AGREEMENT
  - (optional) ARTICLE 10 — STATUS REPORTING AND GOVERNANCE
  - (optional) ARTICLE 11 — PROJECT-SPECIFIC ACCEPTANCE TESTING (if the deliverables are software/products that need test plans)
- **Length targets**:
  - biglaw_formal / customer_paper_negotiated SOWs: 2,000–4,500 words
  - modular_order_form SOWs (where the SOW is itself a form with fillable sections): 1,500–3,000 words
  - plain_english_common_paper SOWs: 1,200–2,500 words
  - modern_tech_minimalist SOWs (Stripe/Notion-style): 1,000–2,000 words
  - click_through_tos SOWs are not common — skip this style for SOW or use it for self-serve professional services portals (1,000–2,000 words)
- **Pricing structure variation across the bucket** — distribute across: time-and-materials with rate card (8 contracts), fixed-fee (8 contracts), milestone-based (5 contracts), hybrid T&M-with-fixed-fee-cap (4 contracts). This teaches the model the full vocabulary of services pricing.
- **Placeholder convention for SOW**:
  - `EFFECTIVE_DATE` = the SOW effective date (when work starts)
  - `MSA_EFFECTIVE_DATE` = the parent MSA's effective date (NEW placeholder for SOW)
  - `PROJECT_NAME` = a short name for the project (NEW placeholder)
  - `SOW_FEE_AMOUNT` = total project fee for fixed-fee SOWs as a money string (NEW; do NOT reuse `ANNUAL_FEE_AMOUNT` because SOWs are project totals, not annual)
  - `HOURLY_RATE` = blended rate for T&M SOWs as a money string (NEW)
  - `PROJECT_START_DATE` = the kickoff date (NEW)
  - `PROJECT_END_DATE` = the target completion date (NEW)
  - `CHANGE_ORDER_THRESHOLD` = the dollar threshold above which a Change Order is required for impact assessment (NEW; e.g., "Five Thousand United States Dollars (US$5,000)")
  - `EXPENSE_PREAPPROVAL_THRESHOLD` = the per-expense pre-approval threshold (NEW; e.g., "Five Hundred United States Dollars (US$500)")
  - `NOTICE_DAYS` = used for status reporting and escalation cadence
  - `GOVERNING_LAW_STATE` and `VENUE_COUNTY` are typically NOT in an SOW because they're in the parent MSA — but include them in the entity_map so the variation generator doesn't error, and reference them only in the parent-MSA citation
- **NO `LIABILITY_CAP_MONTHS`, `IMPLEMENTATION_FEE`, `ANNUAL_FEE_AMOUNT`** in SOW templates — wrong frame of reference.
- **Industry / vertical balance**: SOWs are most common in IT services, management consulting, design/UX, marketing/agency, engineering/architecture, custom software development, data analytics. Distribute the 25 contracts across at least 8 of these verticals.
- **Worker classification reminder**: SOWs engaging individual humans (rare — usually Independent Contractor agreements) need a "no employment relationship" acknowledgment. Most SOWs are firm-to-firm and don't need this.

## DPA clause checklists (Defense 1)

A DPA (Data Processing Addendum) is a privacy-law-driven addendum that operates UNDER a parent SaaS Agreement, MSA, or BAA. It is shaped by GDPR Article 28 and analogous laws (UK GDPR, CCPA/CPRA service-provider terms, India DPDP Act 2023, LGPD, PIPEDA). The DPA's body is purely about the personal data processing relationship — it does NOT restate framework terms (limitation of liability, indemnification, governing law, IP ownership defaults), which live in the parent agreement.

### `controller_processor_roles` — who plays which role
- [ ] Designation of each Party as Controller, Processor, or Sub-processor for each processing activity
- [ ] Controller's documented instructions as the basis for the Processor's processing
- [ ] Processor's obligation to process only on documented Controller instructions
- [ ] Processor's notification obligation if any instruction would violate applicable data protection law

### `processing_scope` — what the processing is about (Article 28(3) GDPR)
- [ ] Subject matter and duration of the processing
- [ ] Nature and purpose of the processing
- [ ] Types / categories of Personal Data processed
- [ ] Categories of Data Subjects
- [ ] Controller's representation that there is a lawful basis for the processing

### `security_measures` — TOMs (Annex II)
- [ ] Implementation of appropriate technical and organizational measures
- [ ] Encryption in transit and at rest where appropriate
- [ ] Confidentiality, integrity, availability, and resilience of processing systems
- [ ] Ability to restore access after a physical or technical incident (backup / recovery)
- [ ] Regular testing and evaluation of the effectiveness of the TOMs
- [ ] Personnel confidentiality and need-to-know access controls
- (Annex II should detail specific TOMs: encryption standards, access controls, network segregation, security incident management, audit logging, vulnerability management, secure SDLC, etc.)

### `subprocessors` — Article 28(2) and (4) GDPR
- [ ] General or specific authorization for sub-processors
- [ ] Maintained list of sub-processors with notification of changes
- [ ] Controller's right to object to new sub-processors within a defined window
- [ ] Flow-down of equivalent data protection obligations to sub-processors
- [ ] Processor's primary liability for sub-processor acts and omissions

### `breach_notification` — Article 33 GDPR
- [ ] Notification without undue delay after becoming aware of a Personal Data Breach
- [ ] Specific notification timeline (typically 24, 48, or 72 hours)
- [ ] Minimum content of the notification (nature, categories, approximate numbers, contact point, likely consequences, measures taken)
- [ ] Cooperation with Controller's investigation and remediation
- [ ] Record-keeping of all Personal Data Breaches

### `international_transfers` — Chapter V GDPR / Schrems II
- [ ] Permitted transfer mechanisms (adequacy decisions, SCCs, UK IDTA, supplementary measures)
- [ ] Incorporation of EU SCCs (Commission Decision 2021/914) where applicable, with module selection (Module Two: Controller to Processor; Module Three: Processor to Sub-processor)
- [ ] Incorporation of UK International Data Transfer Addendum where applicable
- [ ] Transfer Impact Assessment (TIA) obligation for restricted transfers
- [ ] Controller's right to require additional safeguards or to suspend transfers

### `data_subject_rights` — Articles 12-22 GDPR
- [ ] Processor's obligation to assist Controller in responding to Data Subject requests
- [ ] Covered rights (access, rectification, erasure, restriction, portability, objection)
- [ ] Forward Data Subject requests received directly to Controller without delay
- [ ] Limitation that Processor will not respond directly to Data Subjects without Controller authorization

### `dpia_cooperation` — Articles 35-36 GDPR
- [ ] Assist Controller with Data Protection Impact Assessments (DPIAs) under Article 35
- [ ] Assist with prior consultations with supervisory authorities under Article 36
- [ ] Scope of the assistance (information, documentation, attendance at meetings)
- [ ] Controller's responsibility for the DPIA itself; Processor's role is supportive

### `deletion_return` — Article 28(3)(g) GDPR
- [ ] At Controller's choice, delete or return all Personal Data at the end of the services
- [ ] Deletion of all existing copies unless retention is required by applicable law
- [ ] Certification of deletion on Controller's request
- [ ] Exception for backup or archival copies retained pursuant to a documented retention schedule
- [ ] Continued application of confidentiality and security obligations to any retained copies

### `audit_rights` — Article 28(3)(h) GDPR
- [ ] Controller's right to audit the Processor's compliance with the DPA
- [ ] Audit notice period and audit frequency limitations (typically 30-60 days notice, no more than once per 12 months absent cause)
- [ ] Processor's right to provide third-party audit reports (SOC 2 Type II, ISO 27001) in lieu of on-site audits
- [ ] Scope limitations to protect other customers' confidential information and Processor trade secrets
- [ ] Cost allocation for audits beyond standard third-party reports

## DPA-specific structural rules

- **Parties**: PROVIDER slot = the Processor (typically the SaaS vendor / service provider); CUSTOMER slot = the Controller (the customer of the parent agreement). Use `party_roles` to relabel: typically `{"provider": "Processor", "customer": "Controller"}`. For joint controller arrangements, use `{"provider": "Joint Controller A", "customer": "Joint Controller B"}`.
- **DPA operates under a parent agreement**: every DPA MUST contain an "Incorporation by Reference" or equivalent provision stating that this DPA is incorporated into the parent SaaS Agreement, MSA, or BAA dated `{{PARENT_AGREEMENT_DATE}}`. Use the placeholder `{{PARENT_AGREEMENT_NAME}}` to identify the parent (e.g., "the SaaS Subscription Agreement", "the Master Services Agreement", "the Business Associate Agreement"). Conflicts between the DPA and the parent are typically resolved in favor of the DPA on personal data matters and in favor of the parent on commercial matters.
- **NO framework clauses**: DPAs do NOT contain Limitation of Liability, Indemnification, IP Ownership, Governing Law, Dispute Resolution, or Insurance articles. Those live in the parent agreement. The DPA only references them. (Some heavily-negotiated enterprise DPAs do include privacy-specific liability carve-outs that override the parent's liability cap — flag those as a special_feature when present.)
- **Standard article structure for a biglaw_formal DPA** (10–14 articles plus three annexes):
  - ARTICLE 1 — DEFINITIONS
  - ARTICLE 2 — RELATIONSHIP OF THE PARTIES (Controller/Processor roles)
  - ARTICLE 3 — SCOPE OF PROCESSING (subject matter, nature, purpose, duration, data categories, data subject categories — references Annex I)
  - ARTICLE 4 — PROCESSOR OBLIGATIONS (general processor obligations, instruction limitation, personnel confidentiality)
  - ARTICLE 5 — TECHNICAL AND ORGANIZATIONAL MEASURES (references Annex II)
  - ARTICLE 6 — SUB-PROCESSORS (general/specific authorization, list, change notification, objection rights, flow-down — references Annex III)
  - ARTICLE 7 — DATA SUBJECT RIGHTS (assistance with DSRs)
  - ARTICLE 8 — PERSONAL DATA BREACH NOTIFICATION
  - ARTICLE 9 — DATA PROTECTION IMPACT ASSESSMENTS (DPIA cooperation, prior consultation)
  - ARTICLE 10 — INTERNATIONAL TRANSFERS (SCCs, IDTA, supplementary measures, TIA)
  - ARTICLE 11 — AUDIT RIGHTS (controller audit, third-party reports)
  - ARTICLE 12 — DELETION OR RETURN OF PERSONAL DATA
  - ARTICLE 13 — TERM AND TERMINATION (typically tied to the parent agreement)
  - ARTICLE 14 — INCORPORATION OF THE PARENT AGREEMENT AND ORDER OF PRECEDENCE
  - ANNEX I — Subject Matter and Details of Processing (Article 28(3)(a) GDPR)
  - ANNEX II — Technical and Organizational Measures
  - ANNEX III — Approved Sub-processors
- **Length targets**:
  - biglaw_formal / customer_paper_negotiated DPAs: 3,500–6,500 words (annexes are substantial)
  - modular_order_form DPAs (DPA-as-addendum referenced from a Master): 2,500–4,500 words
  - plain_english_common_paper DPAs: 2,000–3,500 words
  - modern_tech_minimalist DPAs (Stripe / Linear / Notion-style): 1,500–2,800 words
  - click_through_tos DPAs (self-serve product DPA accepted in-product): 1,500–2,500 words
- **Placeholder convention for DPA**:
  - `EFFECTIVE_DATE` = the DPA effective date
  - `PARENT_AGREEMENT_DATE` = the parent agreement's effective date (NEW)
  - `PARENT_AGREEMENT_NAME` = friendly name for the parent (NEW; e.g., "the SaaS Subscription Agreement", "the Master Services Agreement", "the Business Associate Agreement")
  - `DATA_RESIDENCY_REGION` = where Personal Data is stored / processed (NEW; e.g., "European Union (Frankfurt and Dublin)", "United States (us-east-1, us-west-2)", "United Kingdom (London)")
  - `BREACH_NOTIFICATION_HOURS` = numeric hours for breach notification (NEW; typically "24", "48", or "72")
  - `AUDIT_NOTICE_DAYS` = notice required for audits (NEW; typically "thirty (30)" or "sixty (60)")
  - `AUDIT_FREQUENCY` = audit frequency limitation (NEW; e.g., "once per twelve (12) month period")
  - `PROCESSING_PURPOSE` = the high-level purpose for processing (NEW; e.g., "providing the SaaS Service to the Controller and its end users")
  - `DATA_CATEGORIES` = types of personal data processed (NEW; e.g., "names, email addresses, IP addresses, usage telemetry")
  - `DATA_SUBJECT_CATEGORIES` = who the data subjects are (NEW; e.g., "Controller's employees, contractors, and end users")
  - `GOVERNING_LAW_STATE` and `VENUE_COUNTY` are typically NOT in a DPA (parent controls), but include in entity_map so the variation generator doesn't error
- **NO `LIABILITY_CAP_MONTHS`, `IMPLEMENTATION_FEE`, `ANNUAL_FEE_AMOUNT`, `TERM_YEARS`** in DPA templates — wrong frame of reference. The DPA term typically tracks the parent agreement's term automatically.
- **Regulatory framework variation across the bucket** — distribute across:
  - **EU GDPR** (with SCCs 2021/914 incorporated): 8 contracts
  - **UK GDPR** (with UK IDTA / UK Addendum): 5 contracts
  - **California CCPA / CPRA** (service provider terms): 5 contracts
  - **India DPDP Act 2023**: 3 contracts
  - **Brazil LGPD**: 2 contracts
  - **Canada PIPEDA + provincial**: 2 contracts
  
  Many DPAs cover multiple frameworks (EU GDPR + UK GDPR + CCPA is the most common combination); count the primary framework for distribution purposes.
- **Industry verticals**: privacy-sensitive industries that produce many DPAs in the wild — B2B SaaS general, healthcare SaaS (BAA + DPA combo), HR tech, EdTech (FERPA + GDPR), AdTech / MarTech, fintech, customer support tooling, analytics platforms, communications platforms, AI/ML / LLM platforms, e-commerce platforms, backup/storage. Distribute the 25 contracts across at least 10 of these verticals.
- **Common combinations to author**:
  - SaaS vendor → enterprise customer (most common)
  - Healthcare SaaS vendor → hospital system (DPA + BAA)
  - EdTech vendor → school district (DPA + FERPA addendum)
  - Communications platform → enterprise (DPA + carrier-level supplementary terms)
  - AdTech / MarTech → publisher (data processor with downstream sharing)
  - AI/ML platform → enterprise (DPA + model training data carve-outs)
  - Joint controller arrangements (Article 26 GDPR) — rarer but distinctive
- **DPA-as-amendment vs DPA-as-standalone**: most DPAs are addenda incorporated into a parent agreement. A few enterprise DPAs are stand-alone documents with the parent agreement referenced by exhibit. Both patterns are valid; biglaw_formal templates can use either.

## INDEPENDENT_CONTRACTOR (IC) clause checklists (Defense 1)

An Independent Contractor Agreement is a standalone contract between a Company and an individual (or single-person LLC) engaged on a 1099 basis. Unlike SOW or DPA, it does NOT operate under a parent agreement — it is the entire agreement. The defining feature is the worker-classification anchor: explicit 1099 / W-9 / no benefits / no withholding language that establishes the IC is NOT an employee.

### `ic_engagement` — what work is being done
- [ ] Specific scope of services (NOT generic "consulting services")
- [ ] Term: fixed-term (start date + end date), project-based (until completion), or open-ended (until terminated)
- [ ] Contractor's commitment to perform in a professional and workmanlike manner
- [ ] Personal services / no subcontracting or delegation without Company consent (this is what makes it "personal services" under tax classification)

### `ic_compensation` — how the Contractor gets paid
- [ ] Fee structure: hourly rate, fixed fee, milestone payments, monthly retainer, or hybrid (e.g., flat monthly minimum + hourly above the cap, like the Pro-Dex pattern)
- [ ] Invoicing cadence (monthly, biweekly, on milestone) and payment terms (net 15, net 30, net 45)
- [ ] Expense reimbursement policy with pre-approval threshold (or no expense reimbursement at all)
- [ ] Explicit "no benefits, no withholding — Contractor is responsible for own taxes" language

### `ic_classification` — the worker-classification anchor (REQUIRED IN EVERY IC AGREEMENT)
- [ ] Explicit independent contractor relationship — no employer-employee, partnership, joint venture, or agency relationship
- [ ] Contractor responsible for self-employment tax, Social Security, Medicare, and income tax payments
- [ ] No Company withholding from Contractor's compensation
- [ ] Contractor not entitled to employee benefits (medical, retirement, vacation, workers comp, unemployment insurance)
- [ ] Contractor's obligation to provide IRS Form W-9; Company will report on Form 1099-NEC

### `ic_intellectual_property` — work product ownership
- [ ] Contractor's assignment of all Work Product to the Company (or, in rare cases, Contractor retains with license to Company)
- [ ] "Works made for hire to the maximum extent permitted by U.S. copyright law" language WITH explicit assignment as backup (CRITICAL: under U.S. copyright law, "work for hire" only applies to specific categories; assignment is required for everything else)
- [ ] Pre-existing IP carve-out: Contractor retains pre-existing IP, with limited license to Company as needed for the deliverables (modern best practice — most older templates omit this)
- [ ] Moral rights waiver to the extent permitted by applicable law
- [ ] Contractor's representations of original authorship and non-infringement of third-party IP

### `ic_termination` — how the engagement ends
- [ ] Termination for convenience by either party on stated notice (typical: 10-30 days; transition contracts use 3 days; some click-through ICs allow immediate termination)
- [ ] Termination by the Company for cause (material breach, misconduct, regulatory disqualification, conviction)
- [ ] Contractor's deliverables and final invoice obligations on termination
- [ ] Survival of confidentiality, IP assignment, restrictive covenants, and indemnification

### `ic_restrictive_covenants` — limits on Contractor's other engagements
- [ ] Contractor free to perform services for other clients (this is part of what makes them an IC, not an employee)
- [ ] During-engagement non-compete: Contractor cannot work for direct competitors during the engagement
- [ ] Non-solicitation of Company employees and contractors during and after the engagement (typical: 6-12 months post-engagement; matched to engagement duration is also common)
- [ ] No use of Company Confidential Information in providing services to third parties
- [ ] Contractor's obligation to disclose conflicts of interest with Company customers, vendors, or competitors
- (Note: post-engagement non-competes are LARGELY UNENFORCEABLE for ICs in most US states. California voids them entirely under B&P § 16600. Most modern IC agreements omit them.)

### `confidentiality` — IC-flavored confidentiality (override)
- [ ] Definition of Company Confidential Information
- [ ] Contractor's obligation to hold Confidential Information in strict confidence during and after the engagement
- [ ] No use of Confidential Information for any purpose other than performing the services
- [ ] Return or destruction of all Company materials on termination
- [ ] Injunctive relief for breach (no proof of actual damages required, no bond)

### `indemnification` — IC-flavored indemnification (override)
- [ ] Contractor indemnifies Company for any claim that Contractor is or should be classified as an employee (the misclassification indemnity — this is the IC-distinctive provision)
- [ ] Contractor indemnifies Company for any tax liability resulting from Contractor's failure to pay self-employment / Social Security / income taxes
- [ ] Contractor indemnifies Company for breach of IP non-infringement representations
- [ ] Contractor maintains professional liability / E&O insurance where engagement risk warrants

## INDEPENDENT_CONTRACTOR-specific structural rules

- **Parties**: PROVIDER slot = the Company (the entity hiring the Contractor); CUSTOMER slot = the Contractor (typically an individual, sometimes a single-person LLC). Use `party_roles` to relabel: typically `{"provider": "Company", "customer": "Contractor"}` or `{"provider": "Company", "customer": "Consultant"}`. The Contractor identity is preserved as a special_customer_type to prevent the variation generator from replacing the individual with a generic SaaS company.
- **NO parent agreement reference**: IC agreements are standalone, unlike SOW and DPA. They contain their own confidentiality, IP, and dispute resolution articles.
- **Standard article structure for a biglaw_formal IC** (8–12 articles):
  - ARTICLE 1 — DEFINITIONS (optional; many IC agreements skip a definitions article and define terms inline)
  - ARTICLE 2 — ENGAGEMENT
  - ARTICLE 3 — TERM AND TERMINATION
  - ARTICLE 4 — FEES AND EXPENSES
  - ARTICLE 5 — INDEPENDENT CONTRACTOR RELATIONSHIP (the worker-classification anchor)
  - ARTICLE 6 — WORK PRODUCT AND INTELLECTUAL PROPERTY
  - ARTICLE 7 — CONFIDENTIALITY AND NON-DISCLOSURE
  - ARTICLE 8 — RESTRICTIVE COVENANTS AND CONFLICTS OF INTEREST
  - ARTICLE 9 — INDEMNIFICATION
  - ARTICLE 10 — INSURANCE (optional; only if engagement risk warrants)
  - ARTICLE 11 — DISPUTE RESOLUTION AND GOVERNING LAW
  - ARTICLE 12 — MISCELLANEOUS / GENERAL PROVISIONS
- **Length targets**:
  - biglaw_formal / customer_paper_negotiated IC: 2,500–5,000 words
  - plain_english_common_paper IC: 1,500–2,800 words
  - modern_tech_minimalist IC: 1,200–2,200 words
  - modular_order_form IC (master IC + per-engagement order form): 2,000–3,500 words for the master + 500–1,000 for each order form
  - click_through_tos IC (platform IC, Uber/Upwork-style): 1,500–2,500 words
- **Placeholder convention for IC**:
  - `EFFECTIVE_DATE` = the IC agreement's effective date
  - `IC_FEE_STRUCTURE` = friendly description of the fee structure (NEW; e.g., `"hourly with monthly cap"`, `"fixed fee"`, `"monthly retainer"`, `"milestone-based"`)
  - `IC_HOURLY_RATE` = hourly rate as a money string (NEW; e.g., `"Two Hundred United States Dollars (US$200)"`)
  - `IC_FIXED_FEE` = total fixed fee as a money string (NEW)
  - `IC_MONTHLY_RETAINER` = monthly retainer as a money string (NEW)
  - `IC_HOURS_PER_MONTH_CAP` = hours cap for hybrid retainer + hourly (NEW; e.g., `"forty (40)"`)
  - `IC_NOTICE_DAYS_CONVENIENCE` = notice for termination for convenience (NEW; e.g., `"three (3)"`, `"ten (10)"`, `"thirty (30)"`)
  - `IC_REPORT_TO` = title of Company contact the Contractor reports to (NEW; e.g., `"the Chief Operating Officer"`)
  - `IC_TERM_MONTHS` = engagement duration in months (NEW; e.g., `"six (6)"`)
  - `PROJECT_NAME` = optional engagement name (reused from SOW)
  - `PROJECT_START_DATE`, `PROJECT_END_DATE` = optional engagement start/end (reused from SOW)
  - `EFFECTIVE_DATE`, `GOVERNING_LAW_STATE`, `VENUE_COUNTY`, `VENUE_CITY` = standard
- **NO `LIABILITY_CAP_MONTHS`, `IMPLEMENTATION_FEE`, `ANNUAL_FEE_AMOUNT`, `TERM_YEARS`** in IC templates — wrong frame of reference.
- **Worker classification reminder**: IC agreements should bake in language that supports IC status under the various tests:
  - Common-law test: control over how work is done (Contractor controls), opportunity for profit/loss, investment in equipment, permanency of relationship, type of relationship in employment context
  - California ABC test (Cal AB-5 / Dynamex): (A) free from control, (B) outside the usual course of the hiring entity's business, (C) customarily engaged in independently established trade
  - Some states use IRS 20-factor test
  - The agreement should state that the Contractor controls the means and methods of performing the services; uses own equipment; provides services to other clients; and is not subject to Company's day-to-day supervision
- **Industry verticals**: distribute the 25 contracts across at least 12 IC engagement types:
  - Outgoing CEO transition consultant (modeled on Pro-Dex)
  - Strategy / management consulting
  - Interim CFO / fractional CFO
  - Interim CMO / fractional CMO
  - Expert witness (medical, technical, financial)
  - Freelance designer / developer / writer / videographer
  - Marketing consultant / agency engagement
  - Virtual assistant / project manager
  - Side-hustle moonlighter
  - Board advisor / advisory board
  - Academic researcher
  - Defense / federal contractor (FAR/DFARS)
  - Healthcare consultant (BAA references)
  - Pharma consultant
  - Bank vendor consultant (FFIEC, OCC Bulletin 2013-29)
  - State government contractor (sovereign immunity, MWBE, FOIL)
  - Litigation expert / damages expert
  - International consultant (W-8BEN, treaty)
  - Platform IC (Uber/Upwork-style click-through)
- **Common edge cases worth baking in**:
  - **Cal AB-5 / ABC test compliance** — explicitly satisfy the B prong (outside usual course of business) and C prong (customarily engaged in independently established trade)
  - **NY Freelance Isn't Free Act** — written contract for >$800, 30-day payment timing, anti-retaliation language
  - **International contractor** — Form W-8BEN, IRC § 1442 / 1446 withholding rules, FATCA, treaty positions
  - **Tax indemnification** — Contractor indemnifies Company against any reclassification claim
  - **Pre-existing IP carve-out vs license-back** — modern best practice (carve-out) vs older templates (license-back to Company)
  - **Hybrid compensation** — flat monthly minimum + hourly above cap (Pro-Dex pattern)
  - **No post-engagement non-compete** — most US states make these unenforceable for ICs; the absence is a deliberate choice
  - **Customer-imposed conditions flow-down** — Company's customers may impose conditions (confidentiality, security clearance) that flow through to Contractor
  - **Personal services / no subcontracting** — required for tax classification; the Contractor must perform the services personally

---

## LICENSE clause checklists (Defense 1)

License agreements grant rights in IP (patent, trademark, copyright, software, trade secret, know-how) from a Licensor to a Licensee. The clause body must demonstrate the IP-license-specific structure that distinguishes a license from a service contract.

### `license_grant`
- [ ] Identification of the Licensed IP (patent, trademark, copyright, software, trade secret, or know-how)
- [ ] Exclusivity (exclusive, sole, or non-exclusive) — and the exact difference (sole = Licensor cannot grant to others but can use itself; exclusive = no one else, including Licensor)
- [ ] Field of use limitations (e.g. "veterinary applications only" or "automotive only, excluding aerospace")
- [ ] Territory in which the Licensee may exercise the rights (worldwide, US, EU, country-specific)
- [ ] Term of the license (perpetual, fixed, term of patent, or coterminous with another agreement)
- [ ] Sublicensing rights (none / one-tier / multi-tier / with Licensor consent / freely)
- [ ] Specific verbs of grant ("make, have made, use, have used, sell, offer to sell, import, perform, display, reproduce, distribute") — pick the verbs that match the IP type

### `license_royalties_payment`
- [ ] Upfront license fee or signing payment
- [ ] Running royalty rate (percentage of Net Sales or per-unit)
- [ ] Definition of Net Sales (with deductions for returns, taxes, freight, discounts)
- [ ] Minimum annual royalty or guaranteed payment (and what happens if Licensee misses it)
- [ ] Milestone payments tied to development, regulatory, or commercial events (especially for patent licenses)
- [ ] Royalty reporting cadence (typically quarterly) and royalty report contents
- [ ] Currency, payment timing, and late payment interest
- [ ] Royalty stacking provisions (where multiple licenses apply to the same product) — optional but common in pharma/telecom

### `license_restrictions`
- [ ] No reverse engineering, decompilation, or disassembly (for software licenses)
- [ ] No transfer, sublicense, or assignment outside the granted scope
- [ ] No use outside the licensed field of use or territory
- [ ] No creation of derivative works without Licensor consent (where applicable — depends on copyright vs patent)
- [ ] No removal of proprietary notices or markings
- [ ] Licensor's reservation of all rights not expressly granted (no implied licenses, no estoppel)

### `license_term_termination`
- [ ] Initial term and renewal options
- [ ] Termination for material breach with cure period (typically 30-60 days)
- [ ] Termination for insolvency or bankruptcy (subject to 11 U.S.C. § 365(n) safe harbor for IP licenses)
- [ ] Licensor's termination right for failure to meet milestones or minimum royalties
- [ ] Post-termination sell-off or wind-down period for inventory (typically 6-12 months)
- [ ] Survival of payment obligations, confidentiality, indemnification, and accrued liabilities

### `license_warranties_indemnification`
- [ ] Licensor's warranty of title and right to grant the license
- [ ] Licensor's warranty (or disclaimer) of non-infringement of third-party IP — this is the key negotiation point
- [ ] Licensor's indemnification of Licensee against third-party infringement claims arising out of the Licensed IP
- [ ] Indemnification procedure (notice, control of defense, cooperation)
- [ ] Licensee's indemnification of Licensor for use outside the licensed scope
- [ ] Limitation of liability and exclusion of consequential damages
- [ ] Optional: license-fee cap on infringement indemnity

### `license_audit_rights`
- [ ] Licensee's obligation to maintain complete books and records of Net Sales and royalty calculations (typically for 3-5 years)
- [ ] Licensor's right to audit Licensee's books on reasonable notice (typically 30 days)
- [ ] Audit frequency limitation (typically once per calendar year)
- [ ] Use of an independent certified public accountant agreed by both parties
- [ ] Licensee's payment of audit costs if the underpayment exceeds a stated threshold (typically 5%)
- [ ] Licensee's payment of any underreported royalties plus interest at a stated rate

### `license_improvements`
- [ ] Definition of Improvements to the Licensed IP
- [ ] Ownership of Improvements made by Licensor (typically retained by Licensor and automatically licensed to Licensee on the same terms)
- [ ] Ownership of Improvements made by Licensee (varies — sole Licensee ownership, joint ownership, or grant-back to Licensor)
- [ ] Grant-back license from Licensee to Licensor for Licensee's Improvements (scope, exclusivity, royalty)
- [ ] Licensee's obligation to disclose Improvements to Licensor

### `license_quality_control` (especially for trademark licenses)
- [ ] Licensor's quality standards for Licensee's use of the Licensed IP
- [ ] Licensee's obligation to submit samples to Licensor for approval prior to use
- [ ] Licensor's right to inspect Licensee's facilities and operations
- [ ] Licensor's right to require corrective action for non-conforming use
- [ ] Naked-license avoidance language (Licensor exercises control to maintain trademark validity — required to prevent the trademark from being deemed abandoned)

### `confidentiality` (LICENSE override)
- [ ] Definition of Confidential Information including the Licensed IP itself, royalty reports, and financial information exchanged
- [ ] Mutual obligation (or one-way, depending on the deal) to hold Confidential Information in strict confidence
- [ ] No use of Confidential Information for any purpose other than exercising rights or fulfilling obligations under the License
- [ ] Return or destruction of Confidential Information on termination, except as needed to exercise surviving rights
- [ ] Trade secret carve-out — trade secret protection survives as long as the information remains a trade secret

---

## LICENSE-specific structural rules

- **Parties**: `Licensor` (the IP owner) and `Licensee` (the recipient of the rights). Variants include `Patent Owner`, `Trademark Owner`, `Copyright Owner`, `Sponsor` (in pharma deals), `Foundation` (in standard-essential-patent / FRAND deals).
- **Standalone agreement**: License agreements are standalone — they do NOT reference a parent MSA or framework agreement (unlike SOW or DPA).
- **Recitals matter**: For biglaw_formal style, include WHEREAS recitals that identify the IP and the licensing context (especially for pharma deals — recite ownership of the patent, regulatory approval status, and Licensor's right to license).
- **Standard 8-12 article structure**:
  1. Definitions (Net Sales, Field, Territory, Improvements, Licensed IP)
  2. Grant of License
  3. Restrictions on Use / Reservation of Rights
  4. Royalties and Payment Terms
  5. Records and Royalty Audit
  6. Improvements and Grant-Back (or omitted if irrelevant)
  7. Quality Control (trademark licenses only — required to avoid naked license)
  8. Confidentiality
  9. Warranties and Infringement Indemnification
  10. Limitation of Liability
  11. Term and Termination of License
  12. Miscellaneous (assignment, governing law, notices, entire agreement)
- **Length targets**: biglaw_formal 7,000-12,000 chars (longer for pharma with milestone schedules and royalty stacking); plain_english/minimalist 4,000-7,000 chars; customer_paper 8,000-13,000 chars; modular_order_form 5,000-9,000 chars (the meat is in the schedules).
- **No "Services" article**: License agreements grant rights in pre-existing IP — they don't describe ongoing services. (Hybrid license + services deals exist but are typically structured as a service contract with a license carve-out, not a license with services.)

### LICENSE placeholder convention
- `LICENSE_GRANT_TYPE` — `exclusive`, `non-exclusive`, or `sole`
- `LICENSE_FIELD_OF_USE` — e.g. "human therapeutic and diagnostic applications" or "veterinary use only"
- `LICENSE_TERRITORY` — e.g. "worldwide", "United States and Canada", "European Union"
- `LICENSE_TERM` — e.g. "perpetual", "the term of the Licensed Patents", "ten (10) years"
- `LICENSE_UPFRONT_FEE` — e.g. "Five Million United States Dollars (US$5,000,000)"
- `LICENSE_ROYALTY_RATE` — e.g. "five percent (5%) of Net Sales", "US$2.50 per Licensed Product"
- `LICENSE_MIN_ANNUAL_ROYALTY` — e.g. "Two Hundred Fifty Thousand United States Dollars (US$250,000)"
- `LICENSE_AUDIT_FREQUENCY` — e.g. "once per calendar year"
- `LICENSE_SUBLICENSE_RIGHTS` — e.g. "no sublicensing", "sublicensing permitted with Licensor consent", "freely sublicensable"
- `LICENSED_IP_DESCRIPTION` — e.g. "U.S. Patent No. 9,876,543", "the BEACON® word mark", "the source code for the Acme Inventory Management Platform v3.2"

### 25-contract roster guidance
- **Industry verticals to cover**:
  - Pharma / biotech (compound license, royalty stacking, milestones)
  - Fashion / consumer brands (trademark license, quality control)
  - Software (perpetual on-prem, source code, OEM, SDK, dual GPL/commercial)
  - Music (master license, synch license, beat license)
  - Stock content (photo, footage, font)
  - Telecom (FRAND / standard-essential patent license)
  - University tech transfer (Bayh-Dole patent license)
  - Foundation models / AI (model weights enterprise license)
  - Genomic data (research license)
  - Celebrity name & likeness
  - Franchise / brand collab
- **Common edge cases worth baking in**:
  - **Bayh-Dole compliance** — march-in rights, US manufacturing preference, government use license (for university tech transfer of federally funded inventions)
  - **FRAND obligations** — fair, reasonable, non-discriminatory licensing for standard-essential patents
  - **Royalty stacking** — express provisions for combining royalty obligations across multiple licenses
  - **Bankruptcy safe harbor** — 11 U.S.C. § 365(n) for IP licenses
  - **Naked license prevention** — quality control language for trademark licenses (without it, the trademark may be deemed abandoned)
  - **Most Favored Nation (MFN)** — Licensee gets the benefit of any better terms Licensor offers a third party
  - **Improvement ownership** — grant-back vs sole-ownership vs joint-ownership
  - **Sell-off period** — Licensee may continue to sell inventory for 6-12 months after termination
  - **Field-of-use carve-out** — distinguishes the licensed application from reserved fields (e.g. "human use only, veterinary reserved")
  - **Cross-license** — bilateral rights exchange between two patent portfolios with no cash royalty
