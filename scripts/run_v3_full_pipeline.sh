#!/bin/bash
# Full pipeline: Train → Eval → Upload adapter → Kill pod
# Runs unattended on the pod via nohup.
set -euo pipefail

# These MUST be set as pod environment variables (via --env flag on pod create)
# or exported before running this script. Never hardcode secrets in scripts.
: "${HF_TOKEN:?HF_TOKEN not set}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set}"
: "${RUNPOD_POD_ID:?RUNPOD_POD_ID not set}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf_cache}"

LOG="/workspace/v3_full_pipeline.log"
cd /workspace/legal-finetune

echo "============================================================" | tee -a $LOG
echo "FULL PIPELINE START — $(date)" | tee -a $LOG
echo "============================================================" | tee -a $LOG

# ── Phase 1: TRAINING ──────────────────────────────────────────────
echo "" | tee -a $LOG
echo ">>> PHASE 1: TRAINING" | tee -a $LOG
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" | tee -a $LOG

# Verify Flash Attention 2
python -c "import flash_attn; print(f'FA2 version: {flash_attn.__version__}')" 2>&1 | tee -a $LOG || {
    echo "WARNING: Flash Attention 2 not available — training will be slower" | tee -a $LOG
}

python scripts/v3_train_gemma4.py \
    --hf-token "$HF_TOKEN" \
    --output-dir /workspace/gemma4-legal-v3 \
    --hf-repo jyoti0512shuklaorg/gemma4-legal-v3 \
    --train-file data/v2/training/v2_combined_train.jsonl \
    --val-file data/v2/training/v2_combined_val.jsonl \
    --epochs 1 \
    --max-seq-length 8192 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 2e-4 \
    --warmup-steps 50 \
    --save-steps 500 \
    --eval-steps 500 \
    --logging-steps 25 \
    2>&1 | tee -a $LOG

TRAIN_EXIT=${PIPESTATUS[0]}
echo "" | tee -a $LOG
echo "TRAINING EXIT CODE: $TRAIN_EXIT — $(date)" | tee -a $LOG

if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "TRAINING FAILED — skipping eval. Pod NOT killed." | tee -a $LOG
    exit 1
fi

# ── Phase 2: EVAL ─────────────────────────────────────────────────
echo "" | tee -a $LOG
echo ">>> PHASE 2: EVAL (35 examples across 7 tasks)" | tee -a $LOG

python scripts/v3_eval_examples.py \
    --hf-token "$HF_TOKEN" \
    --adapter-repo jyoti0512shuklaorg/gemma4-legal-v3 \
    --output /workspace/v3_eval_results.json \
    2>&1 | tee -a $LOG

EVAL_EXIT=${PIPESTATUS[0]}
echo "" | tee -a $LOG
echo "EVAL EXIT CODE: $EVAL_EXIT — $(date)" | tee -a $LOG

# ── Phase 3: TERMINATE POD ─────────────────────────────────────────
echo "" | tee -a $LOG
echo ">>> PHASE 3: TERMINATE POD" | tee -a $LOG

curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"mutation { podTerminate(input: { podId: \\\"${RUNPOD_POD_ID}\\\" }) }\"}" \
    2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo "FULL PIPELINE DONE — $(date)" | tee -a $LOG
echo "============================================================" | tee -a $LOG
