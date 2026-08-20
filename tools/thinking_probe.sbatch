#!/bin/bash
#SBATCH --job-name=mrefresh-nest-think
#SBATCH --partition=pedramh-gpu
#SBATCH --account=pi-pedramh
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH --output=/project/rcc/mehta5/vllm/tools/.think-probe-%j.log
# Thinking on/off probe. Serves once, then runs the same prompts with thinking disabled and
# enabled, so the delta is attributable. Params: MODEL_DIR, SERVED_NAME, MODE (gemma|qwen),
# TAG, TP, PORT, RPARSER, MAX_TOKENS.
#
# HARDWARE (CLAUDE.md): H200 off limits; runs on pedramh-gpu (midway3-0423, H100).
set -uo pipefail

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR}"
SERVED="${SERVED_NAME:?set SERVED_NAME}"
MODE="${MODE:-gemma}"
TAG="${TAG:-gemma4}"
TP="${TP:-2}"
PORT="${PORT:-8520}"
RPARSER="${RPARSER:-gemma4}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
ENV=/project/rcc/mehta5/conda-envs/vllm-serve-cu129
OUT=/project/rcc/mehta5/vllm/_scratch/thinking_${TAG}.json

case "${SLURM_JOB_NAME:-}" in mrefresh-nest*) : ;; *) echo "FENCE: job name"; exit 3 ;; esac
case "${SERVED}" in bench-*) : ;; *) echo "FENCE: served name must start bench-"; exit 3 ;; esac
case "${SLURM_JOB_NODELIST:-}" in *midway3-060*) echo "FENCE: H200"; exit 3 ;; esac

module load python/miniforge-25.3.0 cuda/12.8 gcc/12.2.0
eval "$(mamba shell hook --shell bash)"; mamba activate "$ENV"
: "${CUDA_HOME:=$(dirname "$(dirname "$(command -v nvcc)")")}"; export CUDA_HOME
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
unset TMPDIR SLURM_TMPDIR; export TMPDIR="/tmp/${USER}-vllmjob/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export HF_HOME=/project/rcc/mehta5/hf-cache
export VLLM_CACHE_ROOT=/project/rcc/mehta5/.serve-cache/vllm
export TRITON_CACHE_DIR=/project/rcc/mehta5/.serve-cache/triton
export XDG_CACHE_HOME="$TMPDIR/xdg" TORCH_EXTENSIONS_DIR="$TMPDIR/torch_ext"
export TORCH_HOME="$TMPDIR/torch_home" FLASHINFER_WORKSPACE_BASE="$TMPDIR/flashinfer"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-2400}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCH_HOME" "$FLASHINFER_WORKSPACE_BASE"

echo "=== $(date '+%F %T') node $(hostname) thinking-probe $SERVED (mode=$MODE TP=$TP) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
_VIS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$_VIS" -ne "$TP" ] && { echo "GPU VISIBILITY MISMATCH: TP=$TP but $_VIS visible"; exit 4; }

vllm serve "$MODEL_DIR" --served-model-name "$SERVED" --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size "$TP" --trust-remote-code --enable-prefix-caching \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.90 \
  --reasoning-parser "$RPARSER" --enforce-eager &
PID=$!
python /project/rcc/mehta5/vllm/tools/thinking_probe.py \
  "http://127.0.0.1:${PORT}/v1" "$SERVED" "$OUT" "$MODE" "$MAX_TOKENS"
rc=$?
kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true
echo "=== thinking probe rc=$rc -> $OUT ==="
