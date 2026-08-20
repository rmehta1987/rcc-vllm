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

# HARDWARE RESTRICTION (CLAUDE.md): no H200 without explicit user permission.
case "${SLURM_JOB_NODELIST:-}" in
  *midway3-060*) echo "FENCE VIOLATION: H200 node ${SLURM_JOB_NODELIST} -- aborting"; exit 3 ;;
esac

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

# Serving env follows the MODEL, mirroring ai-session/launch_ai_session.sh. The default
# vllm-probe env is vLLM 0.10.2 and cannot load Qwen3_5* architectures; those models serve
# (and must therefore be RATE-MEASURED) on vllm-serve-cu129 (vLLM 0.26.0). A rate row is
# only valid for the env and flags it was measured under, so this MUST match the launcher.
# FORCE_CU129=1 measures a model that normally serves on 0.10.2 against the 0.26.0 env
# instead. Used for the single-env consolidation: every served model needs a rate row
# measured on the env it will actually run under, and metering.py invalidates any row
# whose recorded vllm_version differs from the running engine.
CU129_ENV=${FORCE_CU129:-0}
case "${MODEL_KEY}" in
  qwen3.5*|qwen3.6*|qwen3.8*) CU129_ENV=1
    ENV_PATH=/project/rcc/mehta5/conda-envs/vllm-serve-cu129
    export HF_HOME=/project/rcc/mehta5/hf-cache
    echo "[bench] ${MODEL_KEY} requires vLLM 0.26.0; ENV_PATH=${ENV_PATH}" >&2 ;;
esac

if [ "$CU129_ENV" = "1" ]; then
  ENV_PATH=/project/rcc/mehta5/conda-envs/vllm-serve-cu129
  export HF_HOME=/project/rcc/mehta5/hf-cache
  # The three env fixes the cu129 env REQUIRES on a GPU compute node (see
  # tools/serve_cu129.sbatch): real CUDA toolkit + host compiler for FlashInfer's JIT,
  # the env's newer libstdc++ ahead of el8's, and DeepGEMM off (its JIT FP8 GEMM needs a
  # driver-API symbol driver 535 lacks). Without these the engine dies on load.
  module load python/miniforge-25.3.0 cuda/12.8 gcc/12.2.0
  eval "$(mamba shell hook --shell bash)"
  mamba activate "${ENV_PATH}"
  : "${CUDA_HOME:=$(dirname "$(dirname "$(command -v nvcc)")")}"
  export CUDA_HOME
  export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"
  export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
  # Keep multi-GB compile/JIT caches OFF the quota-limited home fileset.
  export VLLM_CACHE_ROOT=/project/rcc/mehta5/.serve-cache/vllm
  export XDG_CACHE_HOME="$TMPDIR/xdg" TRITON_CACHE_DIR="$TMPDIR/triton"
  export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_ext" TORCH_HOME="$TMPDIR/torch_home"
  export FLASHINFER_WORKSPACE_BASE="$TMPDIR/flashinfer"
  mkdir -p "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
           "$TORCH_EXTENSIONS_DIR" "$TORCH_HOME" "$FLASHINFER_WORKSPACE_BASE"
else
  module load python/miniforge-25.3.0
  eval "$(mamba shell hook --shell bash)"
  mamba activate "${ENV_PATH}"
fi

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
