#!/bin/bash
# ============================================================================
# Run the full data pipeline (steps 1-5) on your laptop.
# After this completes, upload data/training/ to Google Drive and open
# notebooks/colab_train.ipynb on Colab.
#
# Usage:
#   ./scripts/run_data_pipeline.sh
#   ./scripts/run_data_pipeline.sh --with-claude    # includes Claude distillation
# ============================================================================

set -e

echo "================================================"
echo "  Legal Fine-Tuning Data Pipeline"
echo "================================================"
echo ""

# Check Python
python3 --version || { echo "Python 3 required"; exit 1; }

# Install deps if needed
echo "[1/6] Checking dependencies..."
pip install -q requests beautifulsoup4 lxml tqdm pyyaml datasets jsonlines anthropic 2>/dev/null

# Step 1: Fetch EDGAR
echo ""
echo "[2/6] Fetching EDGAR contracts (50 per type)..."
python3 -m scripts.01_fetch_edgar --count 50 --contract-types msa,saas,nda,employment,software_license

# Step 2: Segment clauses
echo ""
echo "[3/6] Segmenting contracts into clauses..."
python3 -m scripts.02_segment_clauses

# Step 3: Load CUAD
echo ""
echo "[4/6] Loading CUAD dataset..."
python3 -m scripts.03_load_cuad

# Step 4: Claude distillation (optional)
if [[ "$1" == "--with-claude" ]]; then
    if [[ -z "$ANTHROPIC_API_KEY" ]]; then
        echo ""
        echo "ERROR: ANTHROPIC_API_KEY not set. Run:"
        echo "  export ANTHROPIC_API_KEY=sk-ant-..."
        exit 1
    fi
    echo ""
    echo "[5/6] Distilling with Claude (100 examples per type, ~\$40)..."
    python3 -m scripts.04_distill_claude --examples-per-type 100
else
    echo ""
    echo "[5/6] Skipping Claude distillation (run with --with-claude to enable)"
fi

# Step 5: Build dataset
echo ""
echo "[6/6] Building unified training dataset..."
python3 -m scripts.05_build_dataset --val-split 0.1

echo ""
echo "================================================"
echo "  Data pipeline complete!"
echo "================================================"
echo ""
echo "Training data is at: data/training/"
echo ""
echo "Next steps:"
echo "  1. Upload data/training/ to Google Drive:"
echo "     → My Drive/legal-finetune/data/training/"
echo ""
echo "  2. Open notebooks/colab_train.ipynb in Colab"
echo "     → Runtime → Change runtime type → A100"
echo "     → Run all cells"
echo ""
echo "  3. After training, download the merged model:"
echo "     → My Drive/legal-finetune/outputs/merged/"
echo "     → Upload to your GCP VM"
echo "     → vllm serve ./merged --dtype half --chat-template /tmp/mistral.jinja"
echo ""
