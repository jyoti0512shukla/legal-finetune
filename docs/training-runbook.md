# Training Runbook — Lessons from v3 Training (April 2026)

Every mistake we made, what caused it, and how to avoid it next time. Written from painful experience over 3 days and ~$35 of wasted GPU spend.

---

## Mistake #1: Wrong Docker Image → No Flash Attention 2

**What happened:** Used `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`. After installing unsloth, it upgraded torch to 2.10.0. Flash Attention 2 has no prebuilt wheel for torch 2.10.0, so it silently fell back to Xformers. Training ran at 22s/step instead of ~7-8s/step.

**Root cause:** Unsloth's latest version (2026.4.4) requires torch 2.10+, but the flash-attn pip package only has wheels for torch ≤2.5.x. Building from source takes 30+ minutes. The `unslothai/unsloth:latest` Docker image has everything pre-built but takes 20+ minutes to pull on RunPod (and sometimes fails).

**What we tried:**
- `pip install flash-attn --no-build-isolation` → installed but crashed on import (`undefined symbol`)
- Pin torch 2.5.1 + flash-attn → FA2 loaded, but torchao (required by unsloth) crashed (`torch.int1` not in 2.5)
- Install matching torchvision → still crashed on torchao
- `unslothai/unsloth:latest` Docker image → never finished pulling after 20+ minutes

**Current status:** No fix. FA2 is incompatible with the current unsloth+torch combo. Training runs at ~13-22s/step on Xformers (depends on GPU).

**For next time:**
- Check if `unslothai/unsloth:latest` works on RunPod before renting a GPU pod. Test on a CPU pod first ($0.05/hr).
- Check the unsloth GitHub issues for torch/FA2 compatibility before choosing versions.
- If FA2 still can't work: budget for the slower speed. Don't waste GPU hours debugging.
- The speed without FA2: **A100 ≈ 22s/step, H100 ≈ 13s/step** at seq_len=8192.

---

## Mistake #2: 50GB Container Disk → "No space left on device"

**What happened:** Created the first pod with `--container-disk-in-gb 50`. The Gemma 4 26B model weights (~17GB quantized) + pip packages + HuggingFace cache filled the container disk during model download. Training crashed with `RuntimeError: No space left on device`.

**Fix:** Two things, do BOTH:
1. `--container-disk-in-gb 100` (enough for model + packages)
2. `--env '{"HF_HOME":"/workspace/hf_cache"}'` (redirects HF download cache to the larger workspace volume)

---

## Mistake #3: HF_TOKEN Not Available in nohup Background Process

**What happened:** Set `HF_TOKEN` as an env var in the SSH session, then launched training with `nohup ... &`. The background process didn't inherit the session env vars. Training script crashed with `HF_TOKEN: unbound variable`.

**Fix:** Hardcode credentials directly in the wrapper script. Never rely on SSH session env vars for nohup processes.

```bash
# WRONG — won't work in nohup
export HF_TOKEN="..."
nohup bash train.sh &

# RIGHT — hardcode in the script itself
cat > /workspace/run.sh << 'EOF'
export HF_TOKEN="..."
python train.py --hf-token "$HF_TOKEN"
EOF
nohup bash /workspace/run.sh &
```

---

## Mistake #4: Killed Running Training to Try Speed Optimization

**What happened:** Training was at step 1000/2376 (42% done, ~8 hours remaining on A100). Killed it to try FA2 on a new pod. FA2 didn't work (see Mistake #1). Lost all progress. Had to start over from step 0.

**Cost of the mistake:** ~$15 wasted (5 hours of A100 training thrown away + 2 hours debugging FA2).

**Rule:** **NEVER kill a running training to optimize speed.** The cost of restarting almost always exceeds the savings from optimization. Only restart if the job is BROKEN (crash, OOM, wrong data), not slow.

**Exception:** If checkpoints are being pushed to HuggingFace (which we now do), killing is safe because you can resume.

---

## Mistake #5: Recommended H100 When Budget Was Tight

**What happened:** User had $26 balance. Recommended H100 at $2.99/hr. Training needed ~9.2 hours = $27.50. Balance ran out at ~80% completion. Pod terminated. Adapter never uploaded. $25 wasted.

**Rule:** Always calculate: `estimated_hours × $/hr × 1.15 safety margin`. If the result exceeds the user's balance, recommend the cheaper GPU or a smaller dataset.

| Budget | A100 ($1.49/hr) covers | H100 ($2.99/hr) covers |
|--------|----------------------|----------------------|
| $15 | 10 hours | 5 hours |
| $20 | 13.4 hours | 6.7 hours |
| $25 | 16.8 hours | 8.4 hours |
| $30 | 20.1 hours | 10 hours |

For 19K examples: A100 needs ~14.5h ($21.60), H100 needs ~8.5h ($25.40).

---

## Mistake #6: No Checkpoint-to-HuggingFace → Lost All Progress When Pod Died

**What happened:** Training ran for 8+ hours. Pod was terminated (balance ran out). Checkpoints were saved to local disk only. When the pod was destroyed, all checkpoints were lost. No way to resume.

**Fix:** Push checkpoints to HuggingFace every N steps. Each checkpoint is a separate HF branch (`checkpoint-100`, `checkpoint-200`, etc.). On restart, the script checks HF for existing checkpoints and auto-resumes.

Now implemented in `v3_train_gemma4.py`:
- `PushCheckpointToHub` callback fires on every save (default: every 100 steps)
- On startup: scans HF branches for `checkpoint-*`, downloads latest, resumes

**Cost of pushing:** ~1-2 minutes per checkpoint (adapter is ~200MB). At every 100 steps, that's ~2% overhead. Worth it.

---

## Mistake #7: Gemma 4 Multimodal Tokenizer Breaks Token Counting

**What happened:** The training script had `tokenizer(text, truncation=False)["input_ids"]` for counting token lengths. Gemma 4's tokenizer is a `Gemma4Processor` (multimodal), not a plain text tokenizer. This call crashes with `TypeError: 'NoneType' object is not subscriptable`.

**Fix:** Use `tokenizer.tokenizer.encode(text)` instead. The `.tokenizer` attribute accesses the underlying text-only tokenizer.

Now fixed in the committed script.

---

## Mistake #8: `save_steps` Not a Multiple of `eval_steps` → Crash on Startup

**What happened:** Set `--save-steps 100` and `--eval-steps 500`. With `load_best_model_at_end=True`, HuggingFace Trainer requires save_steps to be a multiple of eval_steps. Training crashed before the first step.

**Fix:** Set `load_best_model_at_end=False` (not critical for QLoRA), or make eval_steps a multiple of save_steps.

Now fixed in the committed script.

---

## Mistake #9: SSH Port Not Exposed on Pod

**What happened:** Created the first pod without `--ports "22/tcp"`. Only an HTTP port was visible. SSH was not accessible.

**Fix:** Always include `--ports "22/tcp"` in the pod create command.

---

## Mistake #10: tmux Not Available on RunPod Image

**What happened:** Tried `tmux new -s train ...` to keep training alive after SSH disconnect. Got `tmux: command not found`.

**Fix:** Use `nohup ... &` instead. Or `apt-get install -y tmux` first if you want tmux.

---

## Mistake #11: Tried Unsloth Docker Image → Never Finished Pulling

**What happened:** Created a pod with `unslothai/unsloth:latest`. The image is ~20GB. After 20+ minutes, the runtime was still `null` (image hadn't finished pulling). Killed the pod and switched to the proven PyTorch image.

**Lesson:** Large custom images may take forever to pull depending on the RunPod datacenter. The base `runpod/pytorch` image is cached on most machines and boots in ~2 minutes.

**For next time:** If you need a custom image, test it on a CPU pod first to verify it pulls quickly in the target datacenter.

---

## Mistake #12: GitHub Push Protection Blocked Secrets in Script

**What happened:** Hardcoded HF token and RunPod API key in `run_v3_full_pipeline.sh`. GitHub's push protection detected the secrets and rejected the push.

**Fix:** Use environment variable references (`${HF_TOKEN}`) in committed scripts. Only hardcode secrets in runtime scripts created on the pod (which are never committed).

---

## Mistake #13: Large Training Files (>100MB) Can't Be Pushed to GitHub

**What happened:** Training JSONL files (v2_combined_train.jsonl = 244MB) exceeded GitHub's 100MB file size limit. Push rejected.

**Fix:** Added them to `.gitignore`. SCP the files directly to the pod. The generation scripts (which ARE in git) can regenerate them on the pod if needed.

---

## The Correct Pod Setup (Proven Working)

```bash
runpodctl pod create \
  --name "gemma4-legal-train" \
  --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --gpu-id "NVIDIA A100-SXM4-80GB" \
  --gpu-count 1 \
  --volume-in-gb 100 \
  --container-disk-in-gb 100 \
  --volume-mount-path /workspace \
  --ports "22/tcp" \
  --env '{"HF_TOKEN":"<token>","RUNPOD_API_KEY":"<key>","HF_HOME":"/workspace/hf_cache","TRANSFORMERS_CACHE":"/workspace/hf_cache"}'
```

## The Correct Launch Sequence

```bash
# 1. Wait for SSH
runpodctl ssh info <pod_id>

# 2. SSH in
ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no root@<ip> -p <port>

# 3. Install deps
pip install -q --upgrade unsloth trl transformers datasets huggingface_hub

# 4. Clone repo
cd /workspace && git clone <repo> && cd legal-finetune && git checkout v3

# 5. SCP training data (from local machine, separate terminal)
scp -P <port> -i ~/.runpod/ssh/RunPod-Key-Go \
  data/v2/training/v2_combined_*.jsonl root@<ip>:/workspace/legal-finetune/data/v2/training/

# 6. Create wrapper script WITH hardcoded credentials
cat > /workspace/run.sh << 'EOF'
#!/bin/bash
export HF_TOKEN="<token>"
export RUNPOD_API_KEY="<key>"
export RUNPOD_POD_ID="<pod_id>"
export HF_HOME="/workspace/hf_cache"
cd /workspace/legal-finetune
python scripts/v3_train_gemma4.py \
    --hf-token "$HF_TOKEN" --output-dir /workspace/gemma4-legal-v3 \
    --hf-repo jyoti0512shuklaorg/gemma4-legal-v3 \
    --train-file data/v2/training/v2_combined_train.jsonl \
    --val-file data/v2/training/v2_combined_val.jsonl \
    --epochs 1 --max-seq-length 8192 --batch-size 1 --grad-accum 8 \
    --lr 2e-4 --warmup-steps 50 --save-steps 100 --eval-steps 500 --logging-steps 25
echo "=== TRAIN DONE $(date) ==="
# Auto-kill pod
curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"mutation { podTerminate(input: { podId: \\\"${RUNPOD_POD_ID}\\\" }) }\"}"
EOF

# 7. Launch
nohup bash /workspace/run.sh > /workspace/run.log 2>&1 &

# 8. Verify first 10 steps are running, then disconnect
sleep 300 && grep -oP '\d+/2376' /workspace/run.log | tail -1
```

## Speed Reference

| GPU | s/step (no FA2) | 19K examples (1 epoch) | Cost |
|-----|----------------|----------------------|------|
| A100 SXM 80GB ($1.49/hr) | ~22s | ~14.5 hours | ~$22 |
| H100 SXM 80GB ($2.99/hr) | ~13s | ~8.5 hours | ~$25 |

## Budget Calculator

```
Steps = num_examples / effective_batch_size
Hours = steps × seconds_per_step / 3600
Cost  = hours × $/hr
Budget needed = cost × 1.15 (safety margin)
```

Example: 19,002 examples, batch 8, H100 at 13s/step:
```
Steps = 19002 / 8 = 2376
Hours = 2376 × 13 / 3600 = 8.58
Cost  = 8.58 × $2.99 = $25.65
Budget = $25.65 × 1.15 = $29.50
```

## Checkpoint Safety Net

Every 100 steps, the adapter is pushed to HuggingFace as a branch (`checkpoint-100`, `checkpoint-200`, etc.). If the pod dies:

1. Add funds to RunPod
2. Create a new pod (same command as above)
3. Run the same training script — it auto-detects the latest checkpoint on HF and resumes

Maximum progress loss: 100 steps (~22 minutes on A100, ~13 minutes on H100).

## Monitoring Commands

```bash
# Step count
ssh ... "grep -oP '\d+/2376' /workspace/run.log | tail -1"

# Per-step speed
ssh ... "grep -oP '\d+/2376.*?s/it' /workspace/run.log | tail -3"

# GPU stats
ssh ... "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader"

# Check if still running
ssh ... "pgrep -f v3_train_gemma4 && echo RUNNING || echo STOPPED"

# Check checkpoint branches on HF (from local, no SSH)
python3 -c "from huggingface_hub import list_repo_refs; refs=list_repo_refs('jyoti0512shuklaorg/gemma4-legal-v3',token='<token>'); print([b.name for b in refs.branches])"
```
