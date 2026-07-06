#!/bin/bash
# Billing-grade benchmark -> billing/rate_table.json (one record per model/tier/TP).
#
# Defaults below target the A100 REFERENCE tier first (plan step 4: validate on
# A100, then the rest). Override the GPU tier / count / TP / model on the sbatch
# command line; the SBATCH directives in this file are just defaults.
#
# Examples
# --------
#   # A100 reference, Qwen2.5-72B TP=4 (uses the defaults in this file):
#   sbatch benchmark/bench_billing.sbatch
#
#   # H200, Qwen2.5-72B TP=2:
#   sbatch --constraint=H200 --gres=gpu:2 \
#          --export=ALL,MODEL_KEY=qwen2.5_72B,MODEL_PATH=/project/rcc/mehta5/vllm/models/Qwen2.5-72B-Instruct,TP=2 \
#          benchmark/bench_billing.sbatch
#
#   # Qwen3-4B single-GPU anchor on A100:
#   sbatch --gres=gpu:1 \
#          --export=ALL,MODEL_KEY=qwen3_4b,MODEL_PATH=/project/rcc/mehta5/vllm/models/Qwen3-4B,TP=1 \
#          benchmark/bench_billing.sbatch
#
# IMPORTANT: --gres GPU count must equal TP (and match --constraint to the tier).
#
#SBATCH --job-name=bench_billing
#SBATCH --account=rcc-staff
#SBATCH --partition=test
# Feature names are CASE-SENSITIVE on this cluster: lowercase `a100` = beagle3
# (40GB, plentiful); uppercase `A100` = midway3 (512g, more VRAM). Also a40
# (lowercase), H100/H200/L40S (uppercase). Override --constraint per tier.
#SBATCH --constraint=a100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/project/rcc/mehta5/vllm/benchmark/logs/bench_billing-%j.out
#SBATCH --error=/project/rcc/mehta5/vllm/benchmark/logs/bench_billing-%j.err

set -euo pipefail

REPO=/project/rcc/mehta5/vllm
ENV_PATH=/project/rcc/mehta5/conda-envs/vllm-probe
HF_CACHE=/project/rcc/mehta5/hf_cache
INDUCTOR_CACHE=/project/rcc/mehta5/torchinductor_cache

# Parameters (override via --export=ALL,MODEL_KEY=...,MODEL_PATH=...,TP=...).
MODEL_KEY=${MODEL_KEY:-qwen2.5_72B}
MODEL_PATH=${MODEL_PATH:-/project/rcc/mehta5/vllm/models/Qwen2.5-72B-Instruct}
TP=${TP:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}     # set to match the PROD serve config for this model
EXTRA_BENCH_ARGS=${EXTRA_BENCH_ARGS:-}   # e.g. "--enforce-eager"

mkdir -p "$REPO/benchmark/logs" "$REPO/benchmark/results" "$HF_CACHE" "$INDUCTOR_CACHE"

# Per-job TMPDIR hygiene: drop any stale TMPDIR/SLURM scratch path inherited from
# a previous (now-cancelled) job, or torch.compile's autotune cache will try to
# write into a /scratch path that no longer exists and kill the engine on load.
unset TMPDIR SLURM_TMPDIR
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

export HF_HOME=${HF_CACHE}
export HUGGINGFACE_HUB_CACHE=${HF_CACHE}
export TORCHINDUCTOR_CACHE_DIR=${INDUCTOR_CACHE}

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate "${ENV_PATH}"

echo "=== bench_billing: ${MODEL_KEY} TP=${TP} on $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true

cd "$REPO"
python benchmark/bench_billing.py \
  --model-key "${MODEL_KEY}" \
  --model-path "${MODEL_PATH}" \
  --tp "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  ${EXTRA_BENCH_ARGS}

echo "=== bench_billing done: ${MODEL_KEY} TP=${TP} ==="
