#!/bin/bash
#SBATCH --job-name=mrefresh-nest-mtp
#SBATCH --partition=pedramh-gpu
#SBATCH --account=pi-pedramh
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=/project/rcc/mehta5/vllm/tools/.mtp-smoke-%j.log
# A/B smoke: Qwen3.8-27B with vLLM MTP speculative decoding vs control, at reasoning_effort.
#
# HARDWARE RESTRICTION (AGENTS.md): H200 nodes are off limits without explicit user
# permission. This runs on pedramh-gpu (midway3-0423, H100 80GB) -- NOT H200. The
# --exclude below is a belt-and-braces guard; the partition contains no H200 anyway.
#
# Answers three things in one reservation:
#   1. Does vLLM 0.26.0 actually load the Qwen3_5MTP draft head from our staged checkpoint?
#      (config/speculative.py:505 keys on model_type qwen3_5 and reads mtp_num_hidden_layers,
#      which in OUR config lives under text_config -- unproven that it resolves. This settles it.)
#   2. Does reasoning_effort=xhigh serve, and what does it cost in tokens/time?
#   3. Does the model EMIT tool calls? (Qwen2.5-Coder never did; that broke opencode silently.)
set -uo pipefail

MODEL_DIR=/project/rcc/mehta5/vllm/models/Qwen3.8-27B
ENV=/project/rcc/mehta5/conda-envs/vllm-serve-cu129
OUT_DIR=/project/rcc/mehta5/vllm/_scratch
TP="${TP:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

# --- fences (mirror tools/serve_cu129.sbatch) ------------------------------- #
case "${SLURM_JOB_NAME:-}" in
  mrefresh-nest*) : ;;
  *) echo "FENCE VIOLATION: job name must start 'mrefresh-nest'"; exit 3 ;;
esac
case "${SLURM_JOB_NODELIST:-}" in
  *midway3-060*) echo "FENCE VIOLATION: landed on an H200 node (${SLURM_JOB_NODELIST}) -- aborting"; exit 3 ;;
esac
if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi H200; then
  echo "FENCE VIOLATION: nvidia-smi reports an H200 GPU -- aborting"; exit 3
fi

# --- proven cu129 serve-env setup (copied from tools/serve_cu129.sbatch) ----- #
module load python/miniforge-25.3.0 cuda/12.8 gcc/12.2.0
eval "$(mamba shell hook --shell bash)"
mamba activate "$ENV"
: "${CUDA_HOME:=$(dirname "$(dirname "$(command -v nvcc)")")}"
export CUDA_HOME
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
unset TMPDIR SLURM_TMPDIR
export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export HF_HOME=/project/rcc/mehta5/hf-cache
export VLLM_CACHE_ROOT=/project/rcc/mehta5/.serve-cache/vllm
export XDG_CACHE_HOME="$TMPDIR/xdg"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_ext"
export TORCH_HOME="$TMPDIR/torch_home"
export FLASHINFER_WORKSPACE_BASE="$TMPDIR/flashinfer"
mkdir -p "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TORCH_HOME" "$FLASHINFER_WORKSPACE_BASE"

echo "=== $(date '+%F %T') node $(hostname) MTP A/B on $(basename "$MODEL_DIR") TP=$TP ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

run_arm () {                     # $1 = arm label, $2 = port, $3.. = extra serve args
  local arm="$1"; shift
  local port="$1"; shift
  local out="$OUT_DIR/mtp_smoke_${arm}.json"
  echo "=== ARM ${arm}: serving on port ${port} (extra: $*) ==="
  vllm serve "$MODEL_DIR" \
    --served-model-name "bench-qwen38-27b-${arm}" \
    --host 127.0.0.1 --port "$port" \
    --tensor-parallel-size "$TP" \
    --trust-remote-code \
    --enable-prefix-caching \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --reasoning-parser qwen3 \
    --enforce-eager \
    "$@" &
  local pid=$!
  python /project/rcc/mehta5/vllm/tools/mtp_smoke.py \
    "http://127.0.0.1:${port}/v1" "bench-qwen38-27b-${arm}" "$out" "$arm"
  local rc=$?
  echo "ARM ${arm}: client rc=${rc}"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  sleep 20                       # let the GPUs release before the next arm loads
  return 0
}

# MTP arm FIRST -- it is the primary question; if it fails we still get the control.
run_arm mtp 8471 --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
run_arm control 8472

echo "=== MTP A/B DONE $(date '+%F %T') ==="
for f in "$OUT_DIR"/mtp_smoke_mtp.json "$OUT_DIR"/mtp_smoke_control.json; do
  [ -s "$f" ] && echo "--- $f" && python -c "
import json,sys
d=json.load(open('$f'))
print(' ready=',d.get('ready'),'load_s=',d.get('time_to_ready_s'))
a=d.get('probe_a_xhigh',{})
print(' xhigh: ok=',a.get('ok'),'finish=',a.get('finish_reason'),'compl_tok=',a.get('completion_tokens'),
      'reasoning_chars=',a.get('reasoning_chars'),'s=',a.get('elapsed_s'),'err=',str(a.get('error'))[:120])
print(' throughput agg tok/s=',d.get('throughput_agg_tok_per_s'))
c=d.get('probe_c_toolcall',{})
print(' toolcall: n=',c.get('n_tool_calls'),'has_content=',c.get('has_content'))
m=d.get('metrics_after',{})
print(' spec acceptance=',m.get('_derived_acceptance_rate'),'accepted=',m.get('_derived_accepted_total'),
      'draft=',m.get('_derived_draft_total'))
"
done
