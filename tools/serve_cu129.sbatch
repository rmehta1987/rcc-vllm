#!/bin/bash
#SBATCH --job-name=mrefresh-nest-serve
#SBATCH --partition=test
#SBATCH --account=rcc-staff
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=/project/rcc/mehta5/vllm/tools/.mrefresh-serve-%j.log

# Canonical SERVE template for the model-refresh loop, on the PROVEN cu129 serve env.
# (jobs 52984938 Qwen3-4B TP=1, 52985604 Qwen3.5-122B-A10B-FP8 TP=2; cluster_provenance.md).
#
# The loop MUST use this instead of ai-session/launch_ai_session.sh (which hardcodes the wrong
# env and a floor-billed <key>:<port> job name). Submit it, overriding the per-candidate SBATCH
# knobs on the CLI and passing the rest via --export, e.g.:
#
#   sbatch --job-name=mrefresh-nest-stage1 --constraint=H200 --gres=gpu:2 --time=00:30:00 \
#     --export=ALL,MODEL_DIR=/project/rcc/mehta5/vllm/models/Qwen3.5-122B-A10B-FP8,\
# SERVED_NAME=bench-qwen35-122b,TP=2,PORT=8412 \
#     tools/serve_cu129.sbatch
#
# Tier A (A100, BF16):  --constraint="A100|a100" --gres=gpu:2   TP=2   (case-sensitive feature!)
# Tier B (H200, FP8):   --constraint=H200        --gres=gpu:2   TP=2
#
# FENCES (self-asserted below so a mistake fails fast, never floor-bills):
#   - job name MUST start 'mrefresh-nest'  (billing_sweep only bills <registry-key>:<port>;
#     this name is invisible to the sweep AND to production discovery, and the orchestrator
#     cancels the whole 'mrefresh-nest*' family by name on a terminal stop).
#   - served-model-name MUST start 'bench-' (NEVER a MODEL_REGISTRY key).

set -euo pipefail

ENV=/project/rcc/mehta5/conda-envs/vllm-serve-cu129
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the staged model directory}"
SERVED_NAME="${SERVED_NAME:?set SERVED_NAME (must start with bench-)}"
TP="${TP:-2}"
PORT="${PORT:?set PORT}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
EAGER="${EAGER:-1}"   # 1 = enforce-eager (proven-safe on this driver). Set 0 for CUDA graphs
                      # (higher Gate-2 throughput) ONLY after confirming it loads cleanly here.

# --- fence assertions (fail before any GPU work) ---------------------------- #
case "${SLURM_JOB_NAME:-}" in
  mrefresh-nest*) : ;;
  *) echo "FENCE VIOLATION: job name '${SLURM_JOB_NAME:-}' must start 'mrefresh-nest' (else it is floor-billed / seen by production)"; exit 3 ;;
esac
case "$SERVED_NAME" in
  bench-*) : ;;
  *) echo "FENCE VIOLATION: SERVED_NAME '$SERVED_NAME' must start 'bench-' (never a MODEL_REGISTRY key)"; exit 3 ;;
esac

# --- the three env fixes the cu129 serve env REQUIRES on a GPU compute node --- #
# (bare '$ENV/bin/vllm' or 'mamba activate' alone is NOT enough; see cluster_provenance.md)
# 1. real CUDA toolkit + host compiler for FlashInfer's first-use JIT (caches to ~/.cache/flashinfer)
module load python/miniforge-25.3.0 cuda/12.8 gcc/12.2.0
eval "$(mamba shell hook --shell bash)"
mamba activate "$ENV"
: "${CUDA_HOME:=$(dirname "$(dirname "$(command -v nvcc)")")}"
export CUDA_HOME
# 2. env's libstdc++ (CXXABI_1.3.15 for ICU/sqlite3) must beat el8 /lib64 and the gcc module's -> prepend LAST
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
# 3. DeepGEMM's JIT FP8 GEMM needs a driver-API symbol driver 535 lacks -> use prebuilt CUTLASS FP8 MoE
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0

# Per-job TMPDIR hygiene (a stale TMPDIR kills the engine on load).
unset TMPDIR SLURM_TMPDIR
export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export HF_HOME=/project/rcc/mehta5/hf-cache

# Qwen3 / Qwen3.5 ship "thinking" ON by default; split it from the answer so the loop's Gate-1
# accept-check and Gate-2 scoring see the code, not the raw <think> block. Non-Qwen3 models: no-op.
# NOTE (verified job 52987444): with the parser on, the ANSWER is in the response `content` and the CoT
# in `reasoning`; a tight max-tokens can leave content=null (still thinking, finish_reason=length). Give a
# generous token budget or set enable_thinking=false -- identically for baseline+candidate (session_start.md §3).
REASONING_FLAG=""
case "$(basename "$MODEL_DIR")" in
  Qwen3*|qwen3*) REASONING_FLAG="--reasoning-parser qwen3" ;;
esac
EAGER_FLAG=""; [ "$EAGER" = "1" ] && EAGER_FLAG="--enforce-eager"

echo "=== $(date '+%F %T') node $(hostname) serving $SERVED_NAME (TP=$TP eager=$EAGER) ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

exec vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --trust-remote-code \
  --enable-prefix-caching \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  ${REASONING_FLAG} \
  ${EAGER_FLAG}
