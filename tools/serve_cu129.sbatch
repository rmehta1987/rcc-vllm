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
# 16384, not 8192: the frozen Gate-1/2 DECODE (prereg.md §3) sets max_tokens=8192, so an 8192-token
# context window leaves NO room for the prompt -> vLLM 400 "maximum context length" (Tier-A job 53030556).
# 16384 lets the frozen completion budget plus a Gate-2-sized prompt fit; far below the model cap (262144).
# Scoring-neutral: max_tokens is unchanged, greedy output is identical; it only lets the request be accepted.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
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

# Keep serve-time compile/runtime caches OFF the quota-limited home fileset. A FULL ~ (Midway3-home
# 35/35G) crashed a Stage-1 load mid-profiling with "[Errno 122] Disk quota exceeded" (job 53061850).
# vLLM's own cache is multi-GB -> /project (rcc-staff group quota, ~875G free); transient JIT/compile
# caches -> node-local $TMPDIR (quota-free). HF_HOME is already off home (above).
export VLLM_CACHE_ROOT=/project/rcc/mehta5/.serve-cache/vllm
export XDG_CACHE_HOME="$TMPDIR/xdg"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_ext"
export TORCH_HOME="$TMPDIR/torch_home"
# FlashInfer derives ALL its JIT/cubin/cache dirs from FLASHINFER_WORKSPACE_BASE (env.py:52,
# default $HOME); it does NOT honor XDG. Without this its sm_90a JIT .so would still land on home
# -> the one remaining serve-time home write. Node-local, quota-free, per-job (recompiles cheaply).
export FLASHINFER_WORKSPACE_BASE="$TMPDIR/flashinfer"
mkdir -p "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TORCH_HOME" "$FLASHINFER_WORKSPACE_BASE"

# Qwen3 / Qwen3.5 ship "thinking" ON by default; split it from the answer so the loop's Gate-1
# accept-check and Gate-2 scoring see the code, not the raw <think> block. Non-Qwen3 models: no-op.
# NOTE (verified job 52987444): with the parser on, the ANSWER is in the response `content` and the CoT
# in `reasoning`; a tight max-tokens can leave content=null (still thinking, finish_reason=length). Give a
# generous token budget or set enable_thinking=false -- identically for baseline+candidate (session_start.md §3).
REASONING_FLAG=""
case "$(basename "$MODEL_DIR")" in
  Qwen3*|qwen3*) REASONING_FLAG="--reasoning-parser qwen3" ;;
esac
# DeepSeek-V4's fp8_ds_mla attention layout REQUIRES an fp8 kv-cache; with the default (auto) it
# aborts engine init: "DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, got auto" (job
# 53061901). A model-specific serve REQUIREMENT (like the reasoning-parser), not a decode knob --
# the frozen DECODE (sampling) is unchanged; only V4 is served in its mandated kv-cache dtype.
KV_CACHE_FLAG=""
case "$(basename "$MODEL_DIR")" in
  *V4-Flash*|*DeepSeek-V4*|*deepseek-v4*) KV_CACHE_FLAG="--kv-cache-dtype fp8" ;;
esac
EAGER_FLAG=""; [ "$EAGER" = "1" ] && EAGER_FLAG="--enforce-eager"

echo "=== $(date '+%F %T') node $(hostname) serving $SERVED_NAME (TP=$TP eager=$EAGER) ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

# Build the serve argv ONCE so the default (pure-serve) path and the SMOKE
# (co-located accept-check) path below cannot drift. Unquoted ${REASONING_FLAG}/
# ${EAGER_FLAG} word-split to zero-or-more argv elements exactly as the old
# backslash-continued form did.
SERVE_CMD=(vllm serve "$MODEL_DIR"
  --served-model-name "$SERVED_NAME"
  --host 0.0.0.0
  --port "$PORT"
  --tensor-parallel-size "$TP"
  --trust-remote-code
  --enable-prefix-caching
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  ${REASONING_FLAG}
  ${KV_CACHE_FLAG}
  ${EAGER_FLAG})

if [ "${SMOKE:-0}" != "1" ] && [ "${BENCH:-0}" != "1" ]; then
  # Default, PROVEN behavior (jobs 52984938 / 52985604): pure serve; the time-box
  # terminates it. Byte-identical argv to the pre-SMOKE template.
  exec "${SERVE_CMD[@]}"
fi

# --- BENCH=1: Stage-2 raw-code-gen over the frozen 60-problem LCB subset ---------- #
# Same co-located client/server pattern as SMOKE (serve on 127.0.0.1, generate, WRITE a
# completions file, EXIT before the box kill so the orchestrator's mrefresh-nest* wait-loop
# advances). GENERATION ONLY -- scoring is CPU work on caslake (session_start.md §4). The
# baseline (bench-coder32b) and every candidate run this SAME path -> uniform env/decode/harness.
if [ "${BENCH:-0}" = "1" ]; then
  SMOKE_PY="/project/rcc/mehta5/vllm/tools/stage1_smoke.py"   # reuse the fail-cheap preflight
  BENCH_PY="/project/rcc/mehta5/vllm/tools/stage2_bench.py"
  RESULT_PATH="${RESULT_PATH:-/project/rcc/mehta5/vllm/_scratch/stage2_gen_${SERVED_NAME}.json}"
  mkdir -p "$(dirname "$RESULT_PATH")"

  # Fail-cheap pre-flight BEFORE the multi-minute weight load, so a doomed reservation dies in
  # seconds (session_start.md §4). Bounded so an interruptible GPU hang still yields a verdict.
  if ! timeout -k 30 420 python "$SMOKE_PY" preflight "$MODEL_DIR"; then
    echo "STAGE2 PREFLIGHT_FAIL/timeout for $SERVED_NAME -- not loading weights"
    timeout 30 python "$BENCH_PY" writefail "$RESULT_PATH" "$SERVED_NAME" preflight_failed_or_timeout || true
    exit 0
  fi

  # Absolute client deadline = BENCH_TEARDOWN_S before the job's REAL box end (derived from Slurm's
  # own EndTime, robust to ANY --time; fall back to the frozen 02:00:00 box + $SECONDS otherwise).
  BENCH_TEARDOWN_S="${BENCH_TEARDOWN_S:-240}"
  BENCH_BOX_S="${BENCH_BOX_S:-7200}"   # fallback box only (02:00:00, prereg.md §7)
  _end_iso="$(timeout 15 scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -oP 'EndTime=\K\S+' || true)"
  _end_epoch="$(date -d "${_end_iso:-invalid}" +%s 2>/dev/null || true)"
  if [ -n "${_end_epoch:-}" ]; then
    export NEST_DEADLINE_EPOCH=$(( _end_epoch - BENCH_TEARDOWN_S ))
  else
    export NEST_DEADLINE_EPOCH=$(( ($(date +%s) - SECONDS) + BENCH_BOX_S - BENCH_TEARDOWN_S ))
  fi

  "${SERVE_CMD[@]}" &
  SERVE_PID=$!
  export NEST_SERVE_PID="$SERVE_PID"
  rc=0
  python "$BENCH_PY" client "http://127.0.0.1:$PORT" "$SERVED_NAME" "$RESULT_PATH" || rc=$?
  echo "STAGE2 bench rc=$rc for $SERVED_NAME"
  kill "$SERVE_PID" 2>/dev/null || true
  wait "$SERVE_PID" 2>/dev/null || true
  exit "$rc"
fi

# --- SMOKE=1: self-contained Stage-1 accept-check (session_start.md §2 Stage 1) --- #
# The orchestrator only starts the next driver session once NO mrefresh-nest* job is
# in the queue (run_model_refresh_loop.sbatch wait-loop), so a Stage-1 job must SERVE,
# self-test, WRITE A VERDICT, and EXIT (not block for the full box). Co-locating client
# and server on 127.0.0.1 avoids any cross-node HTTP. Fail-cheap pre-flight first, so a
# doomed reservation dies in seconds instead of after the multi-minute weight load.
SMOKE_PY="/project/rcc/mehta5/vllm/tools/stage1_smoke.py"
RESULT_PATH="${RESULT_PATH:-/project/rcc/mehta5/vllm/_scratch/stage1_${SERVED_NAME}.json}"
mkdir -p "$(dirname "$RESULT_PATH")"

# Bound the pre-flight so an interruptible GPU/driver hang still yields a verdict (a truly
# wedged D-state GPU cannot be killed from userspace -> it falls through to Slurm's --time
# kill and the loop's §9 infra-retry, the correct routing for a hardware wedge).
if ! timeout -k 30 420 python "$SMOKE_PY" preflight "$MODEL_DIR"; then
  echo "STAGE1 PREFLIGHT_FAIL/timeout for $SERVED_NAME -- not loading weights"
  timeout 30 python "$SMOKE_PY" writefail "$RESULT_PATH" "$SERVED_NAME" preflight_failed_or_timeout || true
  exit 0   # leave the queue cleanly; the JSON verdict records the NO-GO
fi

# Absolute client deadline = SMOKE_TEARDOWN_S before the job's REAL box end, so the client
# always writes a verdict + returns before Slurm's wall-clock kill. Derive the box end from
# Slurm's own EndTime (robust to ANY --time, no hand-kept constant to desync); fall back to
# the frozen 30-min box + $SECONDS (elapsed since job start) only if scontrol is unavailable.
SMOKE_TEARDOWN_S="${SMOKE_TEARDOWN_S:-180}"
SMOKE_BOX_S="${SMOKE_BOX_S:-1800}"   # fallback box only (00:30:00, prereg.md §7)
_end_iso="$(timeout 15 scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -oP 'EndTime=\K\S+' || true)"
_end_epoch="$(date -d "${_end_iso:-invalid}" +%s 2>/dev/null || true)"
if [ -n "${_end_epoch:-}" ]; then
  export SMOKE_DEADLINE_EPOCH=$(( _end_epoch - SMOKE_TEARDOWN_S ))
else
  export SMOKE_DEADLINE_EPOCH=$(( ($(date +%s) - SECONDS) + SMOKE_BOX_S - SMOKE_TEARDOWN_S ))
fi

"${SERVE_CMD[@]}" &
SERVE_PID=$!
export SMOKE_SERVE_PID="$SERVE_PID"   # lets the client fail fast if the serve crashes on load
rc=0
python "$SMOKE_PY" client "http://127.0.0.1:$PORT" "$SERVED_NAME" "$RESULT_PATH" || rc=$?
echo "STAGE1 accept-check rc=$rc for $SERVED_NAME"
# Best-effort graceful stop; the GUARANTEED backstop is Slurm's cgroup teardown at 'exit',
# which reclaims the whole TP worker tree regardless of this kill.
kill "$SERVE_PID" 2>/dev/null || true
wait "$SERVE_PID" 2>/dev/null || true
exit "$rc"
