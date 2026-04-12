#!/usr/bin/env python3
"""Run the v2-fine-tuned Gemma 4 on targeted drafting prompts.

Loads the base Gemma 4 26B-A4B in 4-bit + the v2 LoRA adapter, then runs
a set of targeted drafting prompts that exercise capabilities the v2
training data explicitly taught (HIPAA BAA, OSFI, FAR/DFARS, PCAOB,
plain-English style, click-through ToS, MSA structure). Outputs each
response with timing info so we can eyeball the quality.

Run on the RunPod pod:
    python3 scripts/v2_run_eval_prompts.py
"""

import json
import os
import sys
import time

# Force unbuffered output so prints hit disk immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


PROMPTS = [
    {
        "id": "P1_saas_hipaa_telemedicine",
        "label": "SaaS / HIPAA telemedicine for an academic medical center",
        "what_v2_should_do": "Produce a HIPAA Business Associate clause with BAA reference, FDA SaMD language if clinical, data breach 72-hour notification, and security safeguards section. TeleVita-style.",
        "prompt": (
            "Draft the Customer Data and HIPAA Compliance clause (Article 7) for a "
            "Software-as-a-Service Subscription Agreement between Helios Platform Inc. "
            "(a Delaware corporation, the \"Provider\") and Northridge University Medical "
            "Center (a Pennsylvania non-profit academic medical center, the \"Customer\"). "
            "Effective Date: February 11, 2025. Industry: telemedicine / remote patient "
            "monitoring. Deal size: enterprise. Negotiation posture: customer-friendly. "
            "Regulatory frameworks to address: HIPAA, HITECH, FDA Software as a Medical "
            "Device (SaMD), 21 CFR Part 820, NIH Data Management and Sharing Policy. "
            "Drafting style: BigLaw formal. Cover the standard provisions: Business "
            "Associate status with BAA reference, permitted uses limited to the Services, "
            "no AI training on PHI, HIPAA Security Rule safeguards with encryption, "
            "breach notification within 72 hours, data residency in the contiguous U.S., "
            "data return and deletion on termination, and regulator cooperation."
        ),
    },
    {
        "id": "P2_saas_canadian_bank_osfi",
        "label": "SaaS / Canadian Schedule I bank with OSFI",
        "what_v2_should_do": "Use Canadian-specific language: OSFI Guidelines B-10/B-13, PIPEDA, Schedule I bank, exit plan, concentration risk. NorthernEdge-style.",
        "prompt": (
            "Draft the Termination clause (Article 13) for a Software-as-a-Service "
            "Subscription Agreement between Brightline Software, Inc. (a Canadian "
            "corporation, the \"Provider\") and Maple Capital Bank (a Canadian Schedule I "
            "bank, the \"Customer\"). Effective Date: February 18, 2025. Industry: Canadian "
            "banking / cloud infrastructure. Deal size: enterprise. Negotiation posture: "
            "customer-friendly. Regulatory frameworks to address: OSFI Guideline B-10 "
            "(Third-Party Risk), OSFI Guideline B-13 (Technology & Cyber Risk), PIPEDA, "
            "Bank Act (Canada). Governing law: Ontario. Drafting style: BigLaw formal. "
            "Cover the standard provisions: termination for material breach with cure "
            "period, termination for insolvency under Canadian Bankruptcy and Insolvency "
            "Act, exit and stressed-exit plan per OSFI B-10, transition assistance, "
            "regulator-directed termination right."
        ),
    },
    {
        "id": "P3_msa_big4_audit_pcaob",
        "label": "MSA / Big 4 financial statement audit (PCAOB) — v1 never saw an MSA",
        "what_v2_should_do": "Use PCAOB partner rotation, auditor independence rules, Working Papers stay with auditor, Form 8-K reporting, Audit Committee pre-approval. CapstoneAudit-style.",
        "prompt": (
            "Draft the Key Personnel and Partner Rotation clause (Article 6) for a "
            "Financial Statement Audit Master Engagement Agreement between Capstone Audit "
            "& Advisory LLP (a Delaware limited liability partnership, the \"Auditor\") "
            "and Aurora Energy Holdings, Inc. (a Delaware publicly traded corporation, "
            "the \"Company\"). Effective Date: October 14, 2024. Industry: Big 4-style "
            "financial statement audit (PCAOB-registered). Deal size: enterprise. "
            "Negotiation posture: customer-friendly. Regulatory frameworks to address: "
            "PCAOB Standards, PCAOB Rule 3526 / 3527 partner rotation, AICPA Code of "
            "Professional Conduct, SEC auditor independence rules, Sarbanes-Oxley Act. "
            "Drafting style: BigLaw formal. Cover the standard provisions: engagement "
            "partner identification on the engagement letter, concurring partner for "
            "engagement quality review under PCAOB AS 1220, 5-year lead partner rotation "
            "per PCAOB Rule 3526/3527, replacement procedure with Audit Committee notice, "
            "and auditor independence cooling-off restrictions on Customer hiring."
        ),
    },
    {
        "id": "P4_saas_plain_english",
        "label": "SaaS / Plain English Common Paper style",
        "what_v2_should_do": "Use \"we\"/\"you\" voice, short sentences, no WHEREAS, no Latin. BrightStep/PulseLoop-style.",
        "prompt": (
            "Draft the Limitation of Liability clause (Article 12) for a Software-as-a-"
            "Service Subscription Agreement between PulseLoop, Inc. (a Delaware corporation, "
            "the \"Provider\") and Saltwater Apparel Co. (a North Carolina limited liability "
            "company, the \"Customer\"). Effective Date: April 8, 2024. Industry: marketing "
            "technology / email and SMS automation for D2C brands. Deal size: SMB. "
            "Negotiation posture: balanced. Regulatory frameworks to address: CCPA, CPRA, "
            "CAN-SPAM, TCPA. Drafting style: Plain English / Common Paper — short "
            "sentences, \"we\" and \"you\", minimal Latin, defined terms used inline, no "
            "WHEREAS recitals. Cover the standard provisions: aggregate cap tied to fees "
            "paid in the prior 12 months, exclusion of indirect / consequential damages, "
            "TCPA class action exposure carved out from the cap, exceptions for "
            "confidentiality / indemnity / data breaches, basis-of-the-bargain language."
        ),
    },
    {
        "id": "P5_saas_clickthrough",
        "label": "SaaS / Click-through ToS for a CI/CD platform",
        "what_v2_should_do": "Use ALL CAPS \"TO THE MAXIMUM EXTENT PERMITTED BY LAW\", $100 minimum floor, class action waiver, no signature blocks. ForgeRunner-style.",
        "prompt": (
            "Draft the Limitation of Liability clause (Article 12) for a click-through "
            "Terms of Service for ForgeRunner, Inc. (a Delaware corporation, the "
            "\"Provider\") and Pacific Loop Studios, LLC (a California limited liability "
            "company, the \"Customer\"). Effective Date: November 12, 2024. Industry: "
            "developer tools / cloud CI/CD platform. Deal size: SMB. Negotiation posture: "
            "provider-friendly. Drafting style: click-through Terms of Service — terse "
            "numbered lists, no preamble, accept-by-clicking style. Cover the standard "
            "provisions: cap at the greater of prior 12-month fees or $100, class action "
            "waiver, ALL CAPS disclaimers, exceptions for the narrow indemnity and "
            "payment obligations, Free Tier users capped at $100 hard, and the basis-of-"
            "the-bargain acknowledgment."
        ),
    },
    {
        "id": "P6_msa_federal_it_far",
        "label": "MSA / Federal IT contractor with FAR/DFARS — v1 never saw this",
        "what_v2_should_do": "Use FAR/DFARS clause references, NIST SP 800-171, DFARS 252.204-7012, ITAR/EAR, citizenship requirements for CUI. PatriotIT-style.",
        "prompt": (
            "Draft the Information Security, CUI, and Export Controls clause (Article 7) "
            "for a Federal Information Technology Services Master Agreement between "
            "Patriot IT Solutions, Inc. (a Virginia corporation, the \"Contractor\") and "
            "the United States Department of Veterans Affairs (a federal executive "
            "agency, the \"Government\"). Effective Date: January 8, 2025. Industry: "
            "federal government IT services. Deal size: enterprise. Negotiation posture: "
            "customer-friendly. Regulatory frameworks to address: FAR, DFARS, DFARS "
            "252.204-7012, NIST SP 800-171, CMMC, ITAR (22 C.F.R. Parts 120-130), EAR "
            "(15 C.F.R. Parts 730-774), NISPOM. Drafting style: BigLaw formal. Cover the "
            "standard provisions: NIST SP 800-171 compliance for CUI handling, CMMC "
            "certification maintenance, 72-hour cyber incident reporting to DoD under "
            "DFARS 252.204-7012, ITAR/EAR compliance for controlled technical data, U.S. "
            "citizen-only access to CUI/CDI/classified information, no cloud storage "
            "outside the United States, and FOCI compliance under NISPOM."
        ),
    },
    {
        "id": "P7_saas_full_contract_modular",
        "label": "SaaS / Full modular MSA + Order Form contract",
        "what_v2_should_do": "Produce a full multi-article contract (at least articles 1-7) with modular MSA + Order Form references throughout. BrandWaffle/MedPath-style.",
        "prompt": (
            "Draft a Master Subscription Agreement between MedPath Health Systems, Inc. "
            "(a Delaware corporation, the \"Provider\") and Lakeshore Regional Health "
            "System (a Michigan non-profit hospital system, the \"Customer\"). Effective "
            "Date: October 21, 2024. Industry: healthcare / electronic health record "
            "platform (hospital system enterprise). Deal size: enterprise. Negotiation "
            "posture: customer-friendly. Regulatory frameworks to address: HIPAA, HITECH, "
            "HITRUST CSF, state breach notification laws. Drafting style: Modular MSA + "
            "Order Form — bare master agreement with all commercials pushed into a "
            "separate Order Form. Produce at least Articles 1 (Definitions), 2 "
            "(Subscription and License), 3 (Implementation and Professional Services), "
            "4 (Subscription Fees and Payment), 5 (Term and Renewal), 6 (Service Level "
            "Agreement and Support), and 7 (Customer Data, Privacy, and HIPAA), with the "
            "bare-master + Order Form pattern visible throughout."
        ),
    },
]


def main():
    print("=" * 70)
    print("v2 adapter eval — targeted drafting prompts")
    print("=" * 70)

    # ── 1. Load base + v2 adapter ───────────────────────────────────────
    print("Loading base Gemma 4 26B-A4B (4-bit) + v2 adapter...")
    t0 = time.time()
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = "/workspace/gemma4-legal-v2-only/final",
        max_seq_length = 8192,
        dtype          = None,
        load_in_4bit   = True,
    )
    FastLanguageModel.for_inference(model)
    print(f"✓ Loaded in {time.time()-t0:.1f}s")

    # ── 2. Run each prompt — save to disk IMMEDIATELY after each ──────
    os.makedirs("/workspace/v2_eval_results", exist_ok=True)
    results = []
    for i, p in enumerate(PROMPTS, start=1):
        print()
        print("━" * 70)
        print(f"[{i}/{len(PROMPTS)}] {p['id']}")
        print(f"    {p['label']}")
        print(f"    expected: {p['what_v2_should_do']}")
        print("━" * 70)
        print(f"PROMPT: {p['prompt'][:300]}...")

        # Gemma 4 multimodal processor expects content as a list of typed dicts
        messages = [{"role": "user", "content": [{"type": "text", "text": p["prompt"]}]}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        t0 = time.time()
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=2000,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        elapsed = time.time() - t0

        # Decode only the generated portion (skip the input)
        gen_tokens = outputs[0][inputs.shape[1]:]
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        print(f"\nRESPONSE ({len(response)} chars, {len(gen_tokens)} tokens, {elapsed:.1f}s, "
              f"{len(gen_tokens)/elapsed:.1f} tok/s):", flush=True)
        print(response, flush=True)

        result = {
            "id": p["id"],
            "label": p["label"],
            "prompt": p["prompt"],
            "response": response,
            "tokens": int(len(gen_tokens)),
            "seconds": round(elapsed, 1),
        }
        results.append(result)

        # Save per-prompt file IMMEDIATELY so the user can see each draft as it completes
        per_prompt_path = f"/workspace/v2_eval_results/{p['id']}.txt"
        with open(per_prompt_path, "w") as f:
            f.write(f"=== {p['label']} ===\n\n")
            f.write(f"PROMPT:\n{p['prompt']}\n\n")
            f.write(f"RESPONSE ({result['tokens']} tokens, {result['seconds']}s):\n{response}\n")
        print(f"✓ Saved draft to {per_prompt_path}", flush=True)

        # Also update the partial master JSON
        with open("/workspace/v2_eval_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 3. Final summary (per-prompt files already saved above) ─────
    out_path = "/workspace/v2_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print()
    print("=" * 70)
    print(f"Saved full results to {out_path}")
    print(f"Per-prompt drafts: /workspace/v2_eval_results/")
    print(f"Total prompts: {len(results)}")
    print(f"Total time: {sum(r['seconds'] for r in results):.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
