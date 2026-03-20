# legal-finetune

QLoRA fine-tuning pipeline for private legal LLM deployment.

Fine-tunes **Saul-7B** (or any Mistral-based legal model) on real contract data from EDGAR, CUAD, and Claude-distilled examples. The resulting model runs entirely on-prem — no API calls, no data leaves your infrastructure.

## Pipeline

```
Step 1: Fetch EDGAR EX-10.x contracts          → data/raw/edgar/
Step 2: Segment contracts into clauses          → data/processed/clauses/
Step 3: Load CUAD annotated dataset             → data/processed/cuad/
Step 4: Distill with Claude (gold examples)     → data/processed/distilled/
Step 5: Build unified training dataset          → data/training/
Step 6: QLoRA fine-tune on Colab/A100           → outputs/
Step 7: Evaluate against held-out test set      → outputs/eval/
Step 8: Export merged model for vLLM            → outputs/merged/
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. Fetch 500 EDGAR contracts
python scripts/01_fetch_edgar.py --count 50 --contract-types saas,msa,nda,employment

# 2. Segment into clauses
python scripts/02_segment_clauses.py

# 3. Load CUAD
python scripts/03_load_cuad.py

# 4. Distill with Claude (requires ANTHROPIC_API_KEY)
python scripts/04_distill_claude.py --examples-per-type 100

# 5. Build training dataset
python scripts/05_build_dataset.py --val-split 0.1

# 6. Train (run on Colab or A100 machine)
python scripts/06_train.py --config configs/saul_7b.yaml

# 7. Evaluate
python scripts/07_eval.py --config configs/saul_7b.yaml

# 8. Export for vLLM
python scripts/08_export.py --config configs/saul_7b.yaml
```

## Project Structure

```
legal-finetune/
├── configs/
│   ├── base.yaml               # shared training defaults
│   └── saul_7b.yaml            # Saul-specific overrides
├── src/
│   ├── data/
│   │   ├── edgar.py            # EDGAR EFTS API client
│   │   ├── cuad.py             # CUAD dataset loader
│   │   ├── segmenter.py        # Contract → clause segmentation
│   │   └── schema.py           # Data models (ClauseType, ContractType)
│   ├── distill/
│   │   ├── claude_teacher.py   # Claude API for generating gold examples
│   │   ├── quality_filter.py   # Grade + filter training data
│   │   └── augmentor.py        # Instruction variation generation
│   ├── train/
│   │   ├── dataset.py          # HuggingFace Dataset builder
│   │   ├── qlora.py            # QLoRA training loop
│   │   └── eval.py             # Evaluation metrics
│   └── export/
│       └── merge.py            # Merge LoRA adapters for deployment
├── scripts/                    # Numbered pipeline scripts
├── notebooks/
│   └── colab_train.ipynb       # One-click Colab training notebook
├── data/                       # Local data (gitignored)
└── tests/
```
