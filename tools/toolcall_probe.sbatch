#!/bin/bash
#SBATCH --job-name=mrefresh-nest-toolcall
#SBATCH --partition=pedramh-gpu
#SBATCH --account=pi-pedramh
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:50:00
#SBATCH --output=/project/rcc/mehta5/vllm/tools/.toolcall-probe-%j.log
# Does Qwen3.8-27B emit tool calls, and which vLLM parser decodes them?
#
# HARDWARE RESTRICTION (CLAUDE.md): H200 nodes are off limits without explicit user
# permission. Runs on pedramh-gpu (midway3-0423, H100 80GB). Fences below abort if
# we ever land on an H200.
#
# Three arms, one per candidate parser registered in vLLM 0.26.0
# (vllm/tool_parsers/__init__.py): hermes, qwen3_coder, qwen3_xml.
# No MTP here -- speculative decoding is irrelevant to tool-call formatting and would
# only add a variable.
set -uo pipefail

MODEL_DIR=/project/rcc/mehta5/vllm/models/Qwen3.8-27B
ENV=/project/rcc/mehta5/conda-envs/vllm-serve-cu129
OUT_DIR=/project/rcc/mehta5/vllm/_scratch
TP="${TP:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

case "${SLURM_JOB_NAME:-}" in
  mrefresh-nest*) : ;;
  *) echo "FENCE VIOLATION: job name must start 'mrefresh-nest'"; exit 3 ;;
esac
case "${SLURM_JOB_NODELIST:-}" in
  *midway3-060*) echo "FENCE VIOLATION: H200 nodelist ${SLURM_JOB_NODELIST} -- aborting"; exit 3 ;;
esac
if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi H200; then
  echo "FENCE VIOLATION: nvidia-smi reports an H200 -- aborting"; exit 3
fi

module load python/miniforge-25.3.0 cuda/12.8 gcc/12.2.0
eval "$(mamba shell hook --shell bash)"
mamba activate "$ENV"
: "${CUDA_HOME:=$(dirname "$(dirname "$(command -v nvcc)")")}"
export CUDA_HOME
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
unset TMPDIR SLURM_TMPDIR
export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export HF_HOME=/project/rcc/mehta5/hf-cache
export VLLM_CACHE_ROOT=/project/rcc/mehta5/.serve-cache/vllm
export XDG_CACHE_HOME="$TMPDIR/xdg" TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor" TORCH_EXTENSIONS_DIR="$TMPDIR/torch_ext"
export TORCH_HOME="$TMPDIR/torch_home" FLASHINFER_WORKSPACE_BASE="$TMPDIR/flashinfer"
mkdir -p "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TORCH_HOME" "$FLASHINFER_WORKSPACE_BASE"

echo "=== $(date '+%F %T') node $(hostname) tool-call probe on $(basename "$MODEL_DIR") ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

probe_parser () {
  local parser="$1" port="$2"
  local out="$OUT_DIR/toolcall_${parser}.json"
  echo "=== PARSER ${parser} on port ${port} ==="
  vllm serve "$MODEL_DIR" \
    --served-model-name "bench-qwen38-27b-tc" \
    --host 127.0.0.1 --port "$port" \
    --tensor-parallel-size "$TP" \
    --trust-remote-code --enable-prefix-caching \
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 --enforce-eager \
    --enable-auto-tool-choice --tool-call-parser "$parser" &
  local pid=$!
  python /project/rcc/mehta5/vllm/tools/toolcall_probe.py \
    "http://127.0.0.1:${port}/v1" "bench-qwen38-27b-tc" "$out" "$parser"
  echo "PARSER ${parser}: client rc=$?"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  sleep 20
  return 0
}

probe_parser hermes      8481
probe_parser qwen3_coder 8482
probe_parser qwen3_xml   8483

echo "=== TOOL-CALL PROBE DONE $(date '+%F %T') ==="
for p in hermes qwen3_coder qwen3_xml; do
  f="$OUT_DIR/toolcall_${p}.json"
  [ -s "$f" ] && python -c "
import json
d=json.load(open('$f'))
print('--- parser', d['parser'], 'verdict', d.get('verdict'), 'ready_s', d.get('time_to_ready_s'))
for n,c in (d.get('cases') or {}).items():
    print('   ', n, 'n_calls=', c.get('n_tool_calls'), 'raw_tag=', c.get('raw_has_tool_call_tag'),
          'calls=', json.dumps(c.get('parsed_calls'))[:180], 'err=', str(c.get('error'))[:120])
"
done
