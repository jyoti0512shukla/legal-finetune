#!/bin/bash
# End-to-end v3 training on RunPod — runs unattended, stops pod when done.
#
# Usage (from inside the pod):
#   cd /workspace/legal-finetune && bash scripts/run_v3_training.sh
#
# What it does:
#   1. Pulls latest code from git
#   2. Installs/upgrades dependencies
#   3. Runs v3 training (~4 hours on A100 80GB)
#   4. Pushes adapter to HuggingFace
#   5. Stops the pod via RunPod API
#
# Prerequisites:
#   - HF_TOKEN set in environment (or in ~/.huggingface/token)
#   - RUNPOD_API_KEY set in environment
#   - RUNPOD_POD_ID set in environment

set -euo pipefail

LOG="/workspace/v3_training.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "v3 Training — $(date)"
echo "============================================================"

# ── 0. Check prerequisites ─────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    # Try to read from huggingface cache
    if [ -f ~/.huggingface/token ]; then
        export HF_TOKEN=$(cat ~/.huggingface/token)
        echo "✓ HF_TOKEN loaded from ~/.huggingface/token"
    elif [ -f ~/.cache/huggingface/token ]; then
        export HF_TOKEN=$(cat ~/.cache/huggingface/token)
        echo "✓ HF_TOKEN loaded from ~/.cache/huggingface/token"
    else
        echo "ERROR: HF_TOKEN not set and no cached token found"
        exit 1
    fi
else
    echo "✓ HF_TOKEN is set"
fi

if [ -z "${RUNPOD_POD_ID:-}" ]; then
    # Try to detect pod ID from hostname or environment
    if [ -f /etc/hostname ]; then
        # RunPod sets RUNPOD_POD_ID in the environment typically
        echo "WARNING: RUNPOD_POD_ID not set — will skip auto-shutdown"
    fi
else
    echo "✓ RUNPOD_POD_ID=${RUNPOD_POD_ID}"
fi

if [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "WARNING: RUNPOD_API_KEY not set — will skip auto-shutdown"
fi

# ── 1. Pull latest code ────────────────────────────────────────────
cd /workspace/legal-finetune
echo ""
echo "Pulling latest code..."
git pull --ff-only || {
    echo "WARNING: git pull failed (maybe local changes) — continuing with current code"
}
echo "✓ Code is up to date"

# ── 2. Install/upgrade dependencies ────────────────────────────────
echo ""
echo "Checking dependencies..."
pip install -q --upgrade unsloth trl transformers datasets huggingface_hub 2>&1 | tail -5
echo "✓ Dependencies ready"

# ── 2b. Regenerate training data (not in git — too large) ──────────
echo ""
echo "Regenerating training data from templates..."
cd /workspace/legal-finetune

echo "  Step 1/4: Drafting examples (per-type)..."
for TYPE in SAAS MSA NDA EMPLOYMENT SOW DPA INDEPENDENT_CONTRACTOR LICENSE; do
    python scripts/v2_generate_drafting_examples.py --type "$TYPE" 2>&1 | grep "Total examples"
done

echo "  Step 2/4: v2 training dataset (drafting)..."
python scripts/v2_build_training_dataset.py \
    --types SAAS MSA NDA EMPLOYMENT SOW DPA INDEPENDENT_CONTRACTOR LICENSE \
    2>&1 | tail -3

echo "  Step 3/4: Multi-task dataset (extraction, Q&A, checklist, risk, summary)..."
python scripts/v2_build_multitask_dataset.py \
    --include-cuad --include-legal-summarization --include-unfair-tos --include-acord \
    2>&1 | tail -3

echo "  Step 4/4: Combined dataset..."
python scripts/v2_combine_datasets.py 2>&1 | tail -3

# Verify the files exist
if [ ! -f data/v2/training/v2_combined_train.jsonl ]; then
    echo "ERROR: Training data generation failed — v2_combined_train.jsonl not found"
    exit 1
fi
TRAIN_COUNT=$(wc -l < data/v2/training/v2_combined_train.jsonl)
VAL_COUNT=$(wc -l < data/v2/training/v2_combined_val.jsonl)
echo "✓ Training data ready: ${TRAIN_COUNT} train + ${VAL_COUNT} val examples"

# ── 3. Run training ────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Starting v3 training — $(date)"
echo "============================================================"

python scripts/v3_train_gemma4.py \
    --hf-token "$HF_TOKEN" \
    --output-dir /workspace/gemma4-legal-v3 \
    --hf-repo jyoti0512shuklaorg/gemma4-legal-v3 \
    --epochs 2 \
    --max-seq-length 8192 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 2e-4 \
    --warmup-steps 100 \
    --save-steps 500 \
    --eval-steps 500

TRAIN_EXIT=$?

echo ""
echo "============================================================"
echo "Training finished — exit code ${TRAIN_EXIT} — $(date)"
echo "============================================================"

# ── 4. Stop the pod ────────────────────────────────────────────────
if [ "$TRAIN_EXIT" -eq 0 ] && [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
    echo ""
    echo "Training succeeded. Stopping pod ${RUNPOD_POD_ID}..."

    # Use RunPod GraphQL API to stop the pod
    curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"mutation { podStop(input: { podId: \\\"${RUNPOD_POD_ID}\\\" }) { id desiredStatus } }\"}" \
        && echo "" && echo "✓ Pod stop requested" \
        || echo "WARNING: Failed to stop pod — stop it manually"
else
    if [ "$TRAIN_EXIT" -ne 0 ]; then
        echo "Training FAILED (exit code ${TRAIN_EXIT}). Pod NOT stopped — investigate the error."
    else
        echo "No RUNPOD_POD_ID or RUNPOD_API_KEY set. Pod NOT stopped — stop it manually."
    fi
fi

echo ""
echo "Log saved to: $LOG"
echo "Done — $(date)"
