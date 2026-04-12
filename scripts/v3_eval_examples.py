#!/usr/bin/env python3
"""Run 5 law-firm-realistic examples per task against the trained v3 adapter.

Tasks tested (7):
  1. drafting_clause — Draft a specific clause
  2. drafting_full_contract — Draft a complete contract
  3. extraction — Extract key terms from a contract
  4. qa — Answer questions about a contract
  5. checklist_qa — Audit a contract for required provisions
  6. risk_flagging — Flag risky or unfair terms
  7. summary — Summarize a clause in plain English

Saves results to /workspace/v3_eval_results.json (on pod)
which is then SCP'd to local machine.

Usage:
    python scripts/v3_eval_examples.py --hf-token $HF_TOKEN --adapter-repo jyoti0512shuklaorg/gemma4-legal-v3
"""

import argparse
import json
import time
from pathlib import Path


EVAL_PROMPTS = {
    "drafting_clause": [
        {
            "name": "SaaS Limitation of Liability",
            "prompt": "Draft the Limitation of Liability clause for a SaaS Subscription Agreement between CloudMetrics Inc. (a Delaware corporation, the Provider) and National Health Systems, Inc. (a Tennessee corporation, the Customer). Key commercial terms: annual subscription fee of $480,000, initial term of 3 years, liability cap equal to 12 months of fees. Industry: healthcare SaaS. Regulatory frameworks: HIPAA, HITECH. Negotiation posture: balanced. Drafting style: BigLaw formal. Cover the standard provisions: aggregate cap tied to fees paid in the prior 12 months, exclusion of indirect consequential special and punitive damages, exceptions for confidentiality indemnity and data breaches, basis-of-the-bargain language."
        },
        {
            "name": "NDA Confidentiality (M&A)",
            "prompt": "Draft the Confidentiality Obligations clause for a Mutual Non-Disclosure Agreement between Goldman Sachs & Co. LLC (a New York limited liability company, the Disclosing Party) and Riverton Manufacturing Corp. (a Michigan corporation, the Receiving Party). This NDA is for a potential M&A transaction — the Receiving Party is evaluating an acquisition of a division of the Disclosing Party. Drafting style: BigLaw formal. Cover: use restriction (solely for evaluating the Transaction), disclosure restriction (limited to Representatives with need to know), standard of care (at least the care used for own confidential information, no less than reasonable care), obligation to notify of unauthorized disclosure."
        },
        {
            "name": "Employment Non-Compete",
            "prompt": "Draft the Restrictive Covenants clause for an Employment Agreement between Apex Financial Technologies, Inc. (a Delaware corporation, the Company) and Sarah Chen (the Executive, VP of Product). The Executive is based in Massachusetts. Key terms: 12-month post-termination non-compete, 18-month non-solicitation of employees and customers, garden leave provision during non-compete period with full base salary continuation. Must comply with the Massachusetts Noncompetition Agreement Act (MGL c.149 §24L). Drafting style: BigLaw formal."
        },
        {
            "name": "License Grant (Patent)",
            "prompt": "Draft the Grant of License clause for a Patent License Agreement between MIT (a Massachusetts non-profit higher education corporation, the Licensor) and NeuralWave Therapeutics, Inc. (a Delaware corporation, the Licensee). Grant: exclusive license. Field of use: development and commercialization of neurostimulation devices for treatment of treatment-resistant depression. Territory: worldwide. Sublicensing: permitted with Licensor consent. Must include reservation of rights for the U.S. government (Bayh-Dole reserved license under 35 U.S.C. § 202(c)(4)) and reservation for university research purposes. Drafting style: BigLaw formal."
        },
        {
            "name": "IC Worker Classification",
            "prompt": "Draft the Independent Contractor Relationship clause for an Independent Contractor Agreement between DataForge Analytics LLC (a Delaware LLC, the Company) and Michael Torres (an individual, the Contractor). The Contractor is a data scientist providing analytics consulting. Cover: explicit independent contractor status (no employer-employee relationship), Contractor responsible for all self-employment taxes and FICA, no Company withholding, no employee benefits, Form W-9 and 1099-NEC reporting. Include tax indemnification by Contractor against any reclassification claim. Drafting style: plain English."
        }
    ],
    "drafting_full_contract": [
        {
            "name": "SaaS Agreement (Fintech)",
            "prompt": "Draft a complete SaaS Subscription Agreement between PayStream Technologies, Inc. (a Delaware corporation, the Provider) and Meridian Federal Credit Union (a federally chartered credit union, the Customer). Industry: fintech / financial services. Annual subscription fee: $360,000. Initial term: 3 years with auto-renewal. Liability cap: 12 months of fees. Governing law: Delaware, venue in Wilmington. Regulatory frameworks: GLBA Safeguards Rule, FFIEC guidance, NCUA regulations. Negotiation posture: customer-friendly. Drafting style: BigLaw formal. Produce the complete agreement with all standard articles."
        },
        {
            "name": "NDA (Bilateral Tech Due Diligence)",
            "prompt": "Draft a complete Mutual Non-Disclosure Agreement between Horizon Robotics, Inc. (a California corporation) and Atlas Semiconductor Corp. (a Delaware corporation). This is for bilateral technology due diligence — both companies are exploring a potential strategic partnership involving shared IP. Initial term: 2 years, survival: 5 years. Governing law: California. Drafting style: modern tech minimalist. Produce the complete agreement."
        },
        {
            "name": "Independent Contractor (Fractional CFO)",
            "prompt": "Draft a complete Independent Contractor Agreement between GreenLeaf Brands, Inc. (a Delaware corporation, the Company) and James Whitfield (an individual, the Fractional CFO). Engagement: fractional CFO services, 20 hours/week. Monthly retainer: $15,000. Term: 12 months. Termination for convenience: 30 days notice. The Fractional CFO serves multiple clients simultaneously. Governing law: New York. Drafting style: plain English. Produce the complete agreement."
        },
        {
            "name": "MSA (IT Consulting)",
            "prompt": "Draft a complete Master Services Agreement between Deloitte Digital LLC (a Delaware LLC, the Provider) and Amtrak (National Railroad Passenger Corporation, a District of Columbia corporation, the Customer). Industry: transportation / government-adjacent. Annual fees across all SOWs: $2.4M. Liability cap: 24 months of fees. Governing law: New York. Drafting style: customer paper / heavily negotiated. Produce the complete agreement."
        },
        {
            "name": "License Agreement (Software OEM)",
            "prompt": "Draft a complete Source Code License Agreement between DataGrid Systems, Inc. (a Delaware corporation, the Licensor) and Siemens Digital Industries Software LLC (a Delaware LLC, the Licensee). The Licensee is embedding DataGrid's real-time data processing engine into its industrial IoT platform. Grant: non-exclusive, perpetual, worldwide. License fee: $2.5M one-time. No per-end-customer royalty. Source code escrow required. Governing law: California. Drafting style: BigLaw formal. Produce the complete agreement."
        }
    ],
    "extraction": [
        {
            "name": "Extract from SaaS Agreement",
            "prompt": "Extract the key terms from this contract:\n\nSOFTWARE AS A SERVICE SUBSCRIPTION AGREEMENT\n\nThis Agreement is entered into as of March 15, 2025 by and between CloudVault Technologies, Inc., a Delaware corporation located at 500 Howard Street, San Francisco, CA 94105 (\"Provider\"), and First National Bank of Chicago, a national banking association located at 1 South Dearborn Street, Chicago, IL 60603 (\"Customer\").\n\nThe Provider grants the Customer a non-exclusive subscription to access the CloudVault Enterprise Platform for a term of three (3) years commencing on the Effective Date, automatically renewing for successive one-year terms. The annual subscription fee is Four Hundred Twenty Thousand United States Dollars ($420,000). The aggregate liability of either party shall not exceed twelve (12) months of fees. This Agreement shall be governed by the laws of the State of Illinois. Disputes shall be resolved by arbitration in Chicago, Illinois under the AAA Commercial Arbitration Rules. Either party may terminate for material breach with thirty (30) days written notice and opportunity to cure."
        },
        {
            "name": "Extract from Employment Agreement",
            "prompt": "Extract the key terms from this contract:\n\nEXECUTIVE EMPLOYMENT AGREEMENT\n\nThis Agreement is made as of January 8, 2025 between Vertex Pharmaceuticals Incorporated, a Massachusetts corporation (\"Company\"), and Dr. Rebecca Lin, M.D., Ph.D. (\"Executive\"). The Executive shall serve as Chief Medical Officer reporting to the CEO. Annual base salary: Six Hundred Fifty Thousand Dollars ($650,000). Target annual bonus: 60% of base salary. Initial equity grant: 150,000 RSUs vesting over 4 years. Term: at-will employment with 12 months enhanced severance on termination without Cause or resignation for Good Reason. Non-compete: 12 months post-termination, Massachusetts only. Governing law: Commonwealth of Massachusetts."
        },
        {
            "name": "Extract from License Agreement",
            "prompt": "Extract the key terms from this contract:\n\nPATENT LICENSE AGREEMENT\n\nThis Agreement is effective April 1, 2025 between BioGenesis Research Institute (\"Licensor\") and Meridian Therapeutics, Inc. (\"Licensee\"). Licensor grants Licensee an exclusive, worldwide license under U.S. Patent No. 11,456,789 covering monoclonal antibody platform technology for human oncology applications. Upfront payment: $10,000,000. Running royalty: 6% of Net Sales. Minimum annual royalty: $2,000,000 beginning in the first year after First Commercial Sale. Sublicensing permitted with Licensor consent. Term: life of the last-to-expire Licensed Patent. Governing law: State of Delaware."
        },
        {
            "name": "Extract from NDA",
            "prompt": "Extract the key terms from this contract:\n\nMUTUAL NON-DISCLOSURE AGREEMENT\n\nThis NDA is entered into as of February 20, 2025 between Amazon Web Services, Inc. (\"Party A\") and Snowflake Inc. (\"Party B\"). Purpose: evaluation of a potential data integration partnership. Term: 2 years from the Effective Date. Survival of confidentiality obligations: 5 years after termination. Governing law: State of Washington. Venue: King County, Washington. Either party may terminate on 30 days written notice."
        },
        {
            "name": "Extract from Independent Contractor",
            "prompt": "Extract the key terms from this contract:\n\nINDEPENDENT CONTRACTOR AGREEMENT\n\nThis Agreement is dated March 3, 2025 between Beacon Consulting Group, Inc., a New York corporation (\"Company\") and David Park (\"Contractor\"). The Contractor shall provide management consulting services at an hourly rate of $275 per hour, not to exceed $150,000 in total fees. Term: 9 months. Termination for convenience: 15 days notice. The Contractor is an independent contractor, not an employee. The Contractor shall provide a Form W-9. Governing law: State of New York."
        }
    ],
    "qa": [
        {
            "name": "Liability cap question",
            "prompt": "What does this contract say about limitation of liability?\n\nARTICLE 8 — LIMITATION OF LIABILITY\n\n8.1 Aggregate Cap. THE AGGREGATE LIABILITY OF EITHER PARTY ARISING OUT OF OR RELATED TO THIS AGREEMENT, WHETHER IN CONTRACT, TORT, OR OTHERWISE, SHALL NOT EXCEED THE TOTAL FEES PAID OR PAYABLE BY CUSTOMER TO PROVIDER IN THE TWELVE (12) MONTH PERIOD IMMEDIATELY PRECEDING THE EVENT GIVING RISE TO THE CLAIM.\n\n8.2 Exclusion of Damages. NEITHER PARTY SHALL BE LIABLE TO THE OTHER FOR ANY INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL, EXEMPLARY, OR PUNITIVE DAMAGES, INCLUDING LOST PROFITS, LOST REVENUES, OR LOST DATA, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.\n\n8.3 Exceptions. The limitations in Sections 8.1 and 8.2 shall not apply to: (a) either Party's indemnification obligations under Article 9; (b) breaches of confidentiality under Article 6; (c) Provider's infringement of Customer's intellectual property rights; (d) either Party's willful misconduct or fraud; or (e) Customer's obligation to pay fees due under this Agreement."
        },
        {
            "name": "Termination rights question",
            "prompt": "What are the termination rights in this agreement and what happens after termination?\n\nARTICLE 11 — TERM AND TERMINATION\n\n11.1 Term. The initial term is three (3) years from the Effective Date, automatically renewing for successive one-year terms unless either Party provides written notice of non-renewal at least ninety (90) days before the end of the then-current term.\n\n11.2 Termination for Cause. Either Party may terminate this Agreement upon thirty (30) days' written notice if the other Party materially breaches and fails to cure within the notice period.\n\n11.3 Termination for Insolvency. Either Party may terminate immediately if the other Party becomes insolvent, files for bankruptcy, or makes a general assignment for the benefit of creditors.\n\n11.4 Effects of Termination. Upon termination: (a) Customer's access to the Platform shall cease; (b) Provider shall make Customer Data available for export for sixty (60) days; (c) Provider shall delete Customer Data within ninety (90) days after the export period; (d) all fees accrued through the termination date remain payable; (e) Sections 6, 7, 8, 9, and 13 shall survive."
        },
        {
            "name": "Data protection question",
            "prompt": "What are the data protection obligations in this DPA?\n\nARTICLE 4 — PROCESSOR OBLIGATIONS\n\n4.1 The Processor shall process Personal Data only on documented instructions from the Controller, including with regard to transfers of Personal Data to a third country, unless required to do so by applicable law.\n\n4.2 The Processor shall ensure that persons authorised to process the Personal Data have committed themselves to confidentiality.\n\n4.3 The Processor shall implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk, including: (a) the pseudonymisation and encryption of Personal Data; (b) the ability to ensure ongoing confidentiality, integrity, availability, and resilience; (c) the ability to restore availability and access in a timely manner in the event of an incident; (d) a process for regularly testing effectiveness of security measures.\n\n4.4 The Processor shall notify the Controller without undue delay after becoming aware of a Personal Data breach, and in any event within forty-eight (48) hours."
        },
        {
            "name": "IP ownership question",
            "prompt": "Who owns the intellectual property created under this agreement?\n\nARTICLE 7 — INTELLECTUAL PROPERTY\n\n7.1 Provider IP. Provider retains all right, title, and interest in the Platform, the Provider Technology, and all improvements, enhancements, and modifications thereto, including any developed during the performance of Services.\n\n7.2 Customer Data. Customer retains all right, title, and interest in Customer Data. Provider acquires no rights in Customer Data except the limited license to process it for purposes of providing the Services.\n\n7.3 Work Product. All deliverables, reports, analyses, and other work product created by Provider specifically for Customer under a Statement of Work shall be owned by Customer upon full payment, subject to Provider's retained rights in Provider's pre-existing IP and tools.\n\n7.4 Feedback. If Customer provides suggestions or feedback regarding the Platform, Customer grants Provider a non-exclusive, perpetual, royalty-free license to use such feedback."
        },
        {
            "name": "Force majeure question",
            "prompt": "What are the force majeure provisions?\n\nARTICLE 14 — FORCE MAJEURE\n\n14.1 Neither Party shall be liable for any delay or failure to perform its obligations (other than payment obligations) to the extent caused by a Force Majeure Event, provided that the affected Party: (a) promptly notifies the other Party in writing of the Force Majeure Event and its expected duration; (b) uses commercially reasonable efforts to mitigate the impact; and (c) resumes performance as soon as practicable after the Force Majeure Event ceases.\n\n14.2 \"Force Majeure Event\" means any event beyond the reasonable control of the affected Party, including: acts of God, fire, flood, earthquake, pandemic, epidemic, war, terrorism, civil unrest, government action, embargo, sanctions, labor disputes (excluding those involving the affected Party's own employees), failure of third-party telecommunications or power supply, and cyberattacks (provided the affected Party maintained commercially reasonable cybersecurity measures).\n\n14.3 If a Force Majeure Event continues for more than ninety (90) consecutive days, either Party may terminate this Agreement on thirty (30) days' written notice without liability."
        }
    ],
    "checklist_qa": [
        {
            "name": "SaaS agreement checklist",
            "prompt": "Audit this SaaS agreement for required provisions. For each standard clause, indicate if it is PRESENT, WEAK, or MISSING, and explain why.\n\nThis SaaS Agreement between TechCorp and BigBank includes: (1) Service description and SLA with 99.9% uptime commitment and service credits. (2) Fees of $500K/year, payable annually in advance. (3) Limitation of liability capped at 12 months of fees, excluding indirect damages. (4) Mutual confidentiality with 5-year survival. (5) Customer data ownership with Provider processing license. (6) Termination for cause with 30-day cure. (7) Governing law: New York.\n\nNotably absent from the agreement: no indemnification clause, no force majeure clause, no data protection / privacy clause, no dispute resolution mechanism specified, and no assignment / change of control provision."
        },
        {
            "name": "Employment agreement checklist",
            "prompt": "Audit this employment agreement for required provisions. For each standard clause, indicate if it is PRESENT, WEAK, or MISSING.\n\nThis Executive Employment Agreement covers: (1) Position: CEO, reporting to the Board. (2) Base salary: $800K. (3) Target bonus: 100% of base. (4) Equity: 500K options, 4-year vest. (5) Termination by Company for Cause defined. (6) Non-solicitation: 12 months. (7) Confidentiality: perpetual.\n\nMissing: no severance provisions, no change-in-control protections, no non-compete (despite California location), no IP assignment clause, no disability/death provisions, no governing law specified."
        },
        {
            "name": "NDA checklist",
            "prompt": "Audit this NDA for required provisions. For each standard clause, indicate if it is PRESENT, WEAK, or MISSING.\n\nThis Mutual NDA covers: (1) Definition of Confidential Information (broad — all non-public information). (2) Use restriction: solely for evaluating a potential business relationship. (3) Standard of care: reasonable care. (4) Term: 3 years. (5) Governing law: Delaware.\n\nMissing: no exclusions from confidential information (publicly known, independently developed, etc.), no compelled disclosure procedure, no return/destruction obligation, no equitable remedies provision, no no-license/IP-reservation clause."
        },
        {
            "name": "License agreement checklist",
            "prompt": "Audit this patent license for required provisions. For each standard clause, indicate if it is PRESENT, WEAK, or MISSING.\n\nThis Patent License covers: (1) Grant: exclusive license to make, use, sell in the Field. (2) Royalty: 5% of Net Sales. (3) Term: life of patent. (4) Licensor's warranty of title. (5) Governing law: Massachusetts.\n\nMissing: no field of use limitation specified, no territory defined (worldwide implied?), no minimum annual royalty, no audit rights for royalty verification, no sublicensing terms, no infringement indemnification, no improvements/grant-back provision, no bankruptcy safe harbor (Section 365(n))."
        },
        {
            "name": "DPA checklist",
            "prompt": "Audit this Data Processing Addendum for GDPR Article 28 compliance. For each required provision, indicate if it is PRESENT, WEAK, or MISSING.\n\nThis DPA covers: (1) Processor acts only on Controller instructions. (2) Confidentiality of personnel. (3) Security measures (ISO 27001 certified). (4) Sub-processor management with general authorization and 30-day objection window. (5) Assist with data subject rights.\n\nMissing: no breach notification timeline specified, no deletion/return of data on termination, no audit rights for Controller, no DPIA cooperation provision, no international transfer mechanism (no SCCs), no data processing scope/categories defined in an annex."
        }
    ],
    "risk_flagging": [
        {
            "name": "One-sided liability clause",
            "prompt": "Flag any risks in this clause:\n\nLIMITATION OF LIABILITY. IN NO EVENT SHALL PROVIDER'S TOTAL LIABILITY UNDER THIS AGREEMENT EXCEED ONE HUNDRED DOLLARS ($100.00). PROVIDER SHALL NOT BE LIABLE FOR ANY DAMAGES WHATSOEVER, INCLUDING DIRECT, INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL, EXEMPLARY, OR PUNITIVE DAMAGES. CUSTOMER ACKNOWLEDGES THAT THE FEES REFLECT THIS ALLOCATION OF RISK AND THAT PROVIDER WOULD NOT ENTER INTO THIS AGREEMENT WITHOUT THESE LIMITATIONS."
        },
        {
            "name": "Broad IP assignment",
            "prompt": "Flag any risks in this clause:\n\nINTELLECTUAL PROPERTY. All ideas, inventions, discoveries, improvements, works of authorship, and developments, whether or not patentable, made, conceived, or reduced to practice by Contractor at any time during the term of this Agreement, whether or not during working hours and whether or not related to the Services, shall be the sole and exclusive property of the Company. Contractor hereby irrevocably assigns to the Company all right, title, and interest in and to all such intellectual property."
        },
        {
            "name": "Unlimited indemnification",
            "prompt": "Flag any risks in this clause:\n\nINDEMNIFICATION. Customer shall indemnify, defend, and hold harmless Provider and its affiliates, officers, directors, employees, agents, successors, and assigns from and against any and all claims, losses, damages, liabilities, costs, and expenses (including reasonable attorneys' fees) arising out of or related to: (a) Customer's use of the Platform; (b) Customer's breach of this Agreement; (c) any third-party claim related to Customer's business operations; (d) any regulatory investigation or enforcement action involving Customer; or (e) any claim by Customer's employees, contractors, or agents. This indemnification obligation shall survive termination of this Agreement and shall not be subject to any limitation of liability."
        },
        {
            "name": "Auto-renewal with penalty",
            "prompt": "Flag any risks in this clause:\n\nTERM AND RENEWAL. This Agreement shall have an initial term of five (5) years. The Agreement shall automatically renew for successive three (3) year terms unless Customer provides written notice of non-renewal at least one hundred eighty (180) days prior to the end of the then-current term. If Customer terminates during any renewal term, Customer shall pay an early termination fee equal to seventy-five percent (75%) of the remaining fees for the balance of the then-current term."
        },
        {
            "name": "Unilateral amendment right",
            "prompt": "Flag any risks in this clause:\n\nAMENDMENTS. Provider may modify the terms of this Agreement, the Service Level Agreement, the Acceptable Use Policy, and the Privacy Policy at any time by posting the modified terms on the Provider's website. Customer's continued use of the Platform after the effective date of any modification constitutes Customer's acceptance of the modified terms. If Customer does not agree to the modified terms, Customer's sole remedy is to terminate this Agreement, subject to the early termination fee in Section 11.4."
        }
    ],
    "summary": [
        {
            "name": "Summarize indemnification clause",
            "prompt": "Summarize this clause in plain English that a non-lawyer business executive would understand:\n\nARTICLE 9 — INDEMNIFICATION\n\n9.1 Provider Indemnification. Provider shall defend, indemnify, and hold harmless Customer and its Affiliates, officers, directors, employees, and agents from and against any third-party claim alleging that the Platform, as provided by Provider, infringes any United States patent, copyright, or trade secret. If the Platform becomes subject to a claim, Provider may, at its option: (a) obtain for Customer the right to continue using the Platform; (b) modify the Platform to be non-infringing; or (c) terminate Customer's subscription and refund prepaid fees on a pro-rata basis.\n\n9.2 Customer Indemnification. Customer shall defend Provider against any claim arising from: (a) Customer Data; (b) Customer's use of the Platform in violation of this Agreement; or (c) Customer's violation of applicable law.\n\n9.3 Procedure. The indemnified party shall promptly notify the indemnifying party, tender control of the defense, and cooperate at the indemnifying party's expense."
        },
        {
            "name": "Summarize data protection clause",
            "prompt": "Summarize this clause in plain English:\n\nARTICLE 5 — DATA PROTECTION\n\n5.1 Customer Data Ownership. As between the parties, Customer retains all right, title, and interest in Customer Data. Provider shall not access, use, or process Customer Data except as necessary to provide the Services and as permitted under this Agreement.\n\n5.2 Security. Provider shall maintain administrative, physical, and technical safeguards designed to protect Customer Data, including: (a) encryption of Customer Data at rest (AES-256) and in transit (TLS 1.3); (b) role-based access controls; (c) annual SOC 2 Type II audits; (d) vulnerability scanning and penetration testing at least annually.\n\n5.3 Breach Notification. Provider shall notify Customer of any Security Incident involving unauthorized access to Customer Data within seventy-two (72) hours of becoming aware of the incident.\n\n5.4 Data Return. Upon termination, Provider shall make Customer Data available for export for sixty (60) days, after which Provider shall securely delete all Customer Data within thirty (30) days and certify deletion in writing."
        },
        {
            "name": "Summarize non-compete clause",
            "prompt": "Summarize this clause in plain English:\n\nARTICLE 6 — RESTRICTIVE COVENANTS\n\n6.1 Non-Competition. During the Employment Period and for a period of twelve (12) months following termination of employment for any reason (the \"Restricted Period\"), Executive shall not, directly or indirectly, engage in, own, manage, operate, control, finance, or participate in the ownership, management, operation, or control of any business that competes with the Company's business within the United States. For purposes of this Section, a competing business is any enterprise that derives more than ten percent (10%) of its annual revenue from products or services that are substantially similar to those offered by the Company.\n\n6.2 Non-Solicitation. During the Restricted Period, Executive shall not solicit or recruit any employee, contractor, or consultant of the Company, or induce any such person to leave the Company's service.\n\n6.3 Garden Leave. During the Restricted Period, the Company shall continue to pay Executive's base salary as garden leave compensation, subject to Executive's compliance with this Article 6.\n\n6.4 Blue Pencil. If a court determines that any restriction in this Article 6 is unenforceable, the court shall reform the restriction to the minimum extent necessary to make it enforceable."
        },
        {
            "name": "Summarize force majeure clause",
            "prompt": "Summarize this clause in plain English:\n\nARTICLE 14 — FORCE MAJEURE\n\n14.1 Neither Party shall be liable for any delay or failure to perform its obligations (other than payment obligations) to the extent caused by a Force Majeure Event, provided that the affected Party: (a) promptly notifies the other Party; (b) uses commercially reasonable efforts to mitigate; and (c) resumes performance as soon as practicable.\n\n14.2 \"Force Majeure Event\" means fire, flood, earthquake, pandemic, war, terrorism, government action, embargo, labor disputes, failure of telecommunications or power, and cyberattacks (provided commercially reasonable cybersecurity was maintained).\n\n14.3 If a Force Majeure Event continues for more than ninety (90) consecutive days, either Party may terminate without liability."
        },
        {
            "name": "Summarize royalty clause",
            "prompt": "Summarize this clause in plain English:\n\nARTICLE 4 — ROYALTIES AND PAYMENT\n\n4.1 Upfront Payment. Licensee shall pay Licensor Twenty-Five Million United States Dollars (US$25,000,000) within thirty days of the Effective Date.\n\n4.2 Milestone Payments. Licensee shall pay: $5M on IND filing, $10M on Phase II start, $20M on Phase III start, $30M on NDA filing, $50M on FDA approval, $25M on EMA approval.\n\n4.3 Running Royalty. Eight percent (8%) of Net Sales, on a country-by-country and product-by-product basis.\n\n4.4 Royalty Stacking. If Licensee pays royalties to third parties for the same product, Licensee may credit 50% of such third-party royalties against the royalty in 4.3, provided the royalty shall not fall below 4% of Net Sales (the floor).\n\n4.5 Minimum Annual Royalty. Five Million Dollars per year after First Commercial Sale, creditable against running royalties."
        }
    ]
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token", required=True)
    parser.add_argument("--adapter-repo", default="jyoti0512shuklaorg/gemma4-legal-v3")
    parser.add_argument("--base-model", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--output", default="/workspace/v3_eval_results.json")
    args = parser.parse_args()

    print("=" * 70)
    print("v3 Eval — 5 examples per task, 7 tasks = 35 total")
    print("=" * 70)

    from huggingface_hub import login
    login(token=args.hf_token)

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    print("Loading base model + adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_repo,
        max_seq_length=8192,
        dtype=None,
        load_in_4bit=True,
        token=args.hf_token,
    )
    FastLanguageModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma")
    print("✓ Model loaded with adapter")

    results = {}
    total = sum(len(v) for v in EVAL_PROMPTS.values())
    done = 0

    for task, examples in EVAL_PROMPTS.items():
        print(f"\n{'='*50}")
        print(f"Task: {task} ({len(examples)} examples)")
        print(f"{'='*50}")
        task_results = []

        for ex in examples:
            done += 1
            print(f"\n[{done}/{total}] {ex['name']}...")
            start = time.time()

            messages = [{"role": "user", "content": ex["prompt"]}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer.tokenizer(text, return_tensors="pt").to(model.device)

            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
            )
            response = tokenizer.tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            elapsed = time.time() - start
            print(f"  Generated {len(response)} chars in {elapsed:.1f}s")
            print(f"  Preview: {response[:200]}...")

            task_results.append({
                "name": ex["name"],
                "prompt": ex["prompt"],
                "response": response,
                "chars": len(response),
                "time_seconds": round(elapsed, 1),
            })

        results[task] = task_results

    # Save results
    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n✓ Results saved to {output_path}")
    print(f"  Total examples: {total}")
    print(f"  Tasks: {list(results.keys())}")


if __name__ == "__main__":
    main()
