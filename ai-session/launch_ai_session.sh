#!/bin/bash
# Launch ONE vLLM server as a Slurm job for an ai-session, then print a single
# JSON line (jobid/port/...) on stdout for ai_session.py to capture.
# Adapted from decrypto/slurm/launch_servers_midway.sh.
#
# Phase-1 runtime target: --partition=test --account=rcc-staff (the high-priority
# queue all sibling-project work used; exposes every GPU tier). Account /
# partition / constraint / gres / TP are PARAMETERS (env or flags) so a future
# per-PI deployment is a one-line change, not a code edit.
#
# All parameters have env defaults; flags override env. Required: MODEL_KEY + MODEL_PATH.
#   MODEL_KEY MODEL_PATH TP CONSTRAINT GRES ACCOUNT PARTITION TIME_LIMIT
#   CPUS MEM MAX_MODEL_LEN GPU_MEM_UTIL ENFORCE_EAGER PORT
#   ENABLE_LORA LORA_MODULES MAX_LORA_RANK    (fine-tuned adapter serving)
#
# Standalone example:
#   ./launch_ai_session.sh --model-key qwen2.5_72B \
#       --model-path /project/rcc/mehta5/vllm/models/Qwen2.5-72B-Instruct \
#       --tp 4 --constraint A100 --gres gpu:4

set -euo pipefail

REPO=/project/rcc/mehta5/vllm
ENV_PATH=/project/rcc/mehta5/conda-envs/vllm-probe
# HF + torch-inductor caches: the vLLM job WRITES these, so they must live where
# the running user can write. Default (empty here) is resolved to the per-user
# state dir below, once STATE_BASE is known. Override via env only if you point
# them at another writable location; the old /project/rcc/mehta5 default is NOT
# writable by users outside rcc-staff.
HF_CACHE=${HF_CACHE:-}
INDUCTOR_CACHE=${INDUCTOR_CACHE:-}

# -- defaults (override via env or flags) ----------------------------------- #
MODEL_KEY=${MODEL_KEY:-}
MODEL_PATH=${MODEL_PATH:-}
TP=${TP:-4}
CONSTRAINT=${CONSTRAINT:-A100}
GRES=${GRES:-}   # resolved after flag parsing so a --tp flag is honored
ACCOUNT=${ACCOUNT:-}       # no default: unique per user/PI, must be supplied
PARTITION=${PARTITION:-}   # no default: unique per user, must be supplied
TIME_LIMIT=${TIME_LIMIT:-02:00:00}
CPUS=${CPUS:-16}
MEM=${MEM:-128G}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
AGENT_CLIENT=${AGENT_CLIENT:-0}
ENABLE_LORA=${ENABLE_LORA:-0}
LORA_MODULES=${LORA_MODULES:-}      # space-separated name=/abs/path pairs (no spaces in paths)
MAX_LORA_RANK=${MAX_LORA_RANK:-16}  # must be >= the largest adapter r; CLI computes this
PORT=${PORT:-}

# -- flag parsing (overrides env) ------------------------------------------- #
while [ $# -gt 0 ]; do
  case "$1" in
    --model-key) MODEL_KEY="$2"; shift 2;;
    --model-path) MODEL_PATH="$2"; shift 2;;
    --tp) TP="$2"; shift 2;;
    --constraint) CONSTRAINT="$2"; shift 2;;
    --gres) GRES="$2"; shift 2;;
    --account) ACCOUNT="$2"; shift 2;;
    --partition) PARTITION="$2"; shift 2;;
    --time) TIME_LIMIT="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --mem) MEM="$2"; shift 2;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
    --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2;;
    --enforce-eager) ENFORCE_EAGER=1; shift;;
    --agent-client) AGENT_CLIENT=1; shift;;
    --enable-lora) ENABLE_LORA=1; shift;;
    --lora-modules) LORA_MODULES="$2"; shift 2;;
    --max-lora-rank) MAX_LORA_RANK="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# GRES tracks the FINAL TP unless explicitly set (env or --gres). Resolving it
# here, after flag parsing, means `--tp 2` alone requests 2 GPUs; previously the
# default was computed from the pre-flag TP and `--tp 2` still reserved 4.
if [ -z "${GRES}" ] || [ "${GRES}" = "gpu:" ]; then GRES="gpu:${TP}"; fi

if [ -z "${MODEL_KEY}" ] || [ -z "${MODEL_PATH}" ]; then
  echo "ERROR: MODEL_KEY and MODEL_PATH are required" >&2
  exit 2
fi

if [ -z "${ACCOUNT}" ] || [ -z "${PARTITION}" ]; then
  echo "ERROR: ACCOUNT and PARTITION are required (no default -- they are unique" >&2
  echo "       per user/PI). Pass --account/--partition or set the env vars." >&2
  exit 2
fi

# Per-job vLLM Slurm logs go under the caller's state dir when set (multi-tenant;
# matches ai_session.py/gateway.py _STATE), else next to the code (single-tenant).
STATE_BASE="${AISESSION_STATE_DIR:-${REPO}/ai-session}"
LOGDIR="${STATE_BASE}/logs/vllm"
# Resolve the writable caches under the per-user state dir unless the caller
# pointed them elsewhere. A path the running user cannot write would fail the
# job AFTER the GPU is reserved and floor-billing has begun.
HF_CACHE="${HF_CACHE:-${STATE_BASE}/hf_cache}"
INDUCTOR_CACHE="${INDUCTOR_CACHE:-${STATE_BASE}/torchinductor_cache}"
mkdir -p "${LOGDIR}" "${HF_CACHE}" "${INDUCTOR_CACHE}"

# Pick a free-ish port if not supplied (job-name carries it: 'model_key:port').
if [ -z "${PORT}" ]; then
  PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
fi

# Per-session backend API key. The vLLM /v1 endpoint binds --host 0.0.0.0 (the
# gateway on the login node must reach it, and compute<->compute is routable), and
# the node:port is discoverable (job-name, squeue). WITHOUT auth any co-tenant
# could curl the paid backend directly. So gate /v1 with a fresh random key passed
# to the engine as VLLM_API_KEY -- via the ENVIRONMENT, not argv, so it is not
# visible in `ps` to a co-tenant that lands on the same (possibly shared) node.
# It is NOT a serve flag, so bench_billing.py:build_serve_flags stays matched.
# The gateway learns this key from upstream.json (which ai_session writes 0600)
# and injects it when forwarding, so legitimate clients never handle it directly.
BACKEND_KEY=$(openssl rand -hex 16)

EAGER_FLAG=""
if [ "${ENFORCE_EAGER}" = "1" ]; then EAGER_FLAG="--enforce-eager"; fi

# Agent-client mode: enable server-side tool calling (opencode/Cline need it;
# Open WebUI and aider do NOT) and widen context. Pick the parser by model.
# NOTE: this changes the production serve config; throughput rates measured by
# bench_billing (batch config) don't transfer -- but agent sessions are
# interactive and floor-billed, so the token rate is moot for them.
AGENT_FLAGS=""
if [ "${AGENT_CLIENT}" = "1" ]; then
  case "${MODEL_KEY}" in
    qwen3*)  TOOL_PARSER="hermes" ;;   # qwen3coder parser also available
    qwen*)   TOOL_PARSER="hermes" ;;   # Qwen2.5 uses the hermes parser
    llama*)  TOOL_PARSER="llama3_json" ;;
    *)       TOOL_PARSER="hermes" ;;
  esac
  AGENT_FLAGS="--enable-auto-tool-choice --tool-call-parser ${TOOL_PARSER}"
  # widen context for agent harnesses, unless the caller set a non-default value
  if [ "${MAX_MODEL_LEN}" = "8192" ]; then MAX_MODEL_LEN=32768; fi
  echo "[launch] agent-client mode: tool-parser=${TOOL_PARSER} max-model-len=${MAX_MODEL_LEN}" >&2
fi

# LoRA serving: register fine-tuned adapters at launch so clients can select one
# by using its name as the request's model. Static registration only (adapters are
# fixed for the life of the session); no runtime add/remove endpoint is exposed.
# Adapter paths must be readable from the compute node (project storage, not $HOME
# scratch) and must not contain spaces (LORA_MODULES is word-split below).
# Like AGENT_FLAGS, this changes the serve config vs. the benchmarked one: rate
# records don't strictly transfer, but LoRA sessions are interactive and floor-
# billed in practice, so the token term is moot (same stance as agent mode).
LORA_FLAGS=""
if [ "${ENABLE_LORA}" = "1" ]; then
  LORA_FLAGS="--enable-lora --max-lora-rank ${MAX_LORA_RANK}"
  if [ -n "${LORA_MODULES}" ]; then
    LORA_FLAGS="${LORA_FLAGS} --lora-modules ${LORA_MODULES}"
  fi
  echo "[launch] LoRA enabled: max-rank=${MAX_LORA_RANK} adapters='${LORA_MODULES:-none registered}'" >&2
fi

# Reasoning parser. The Qwen3 family ships with "thinking" ON by default: the model
# emits its chain of thought inside <think>...</think> before the answer. Without a
# reasoning parser that raw block streams to the client AND the hidden reasoning bills
# as ordinary completion tokens. --reasoning-parser qwen3 splits it into a separate
# reasoning_content field (answer stays in content). Pick it by model key, the same
# way the tool-call parser is chosen above; ONLY Qwen3 gets it (Qwen2.5 / Llama do not
# think and the parser would be a no-op or worse). This applies whether or not
# --agent-client is set, since thinking is a property of the model, not the client.
REASONING_FLAG=""
case "${MODEL_KEY}" in
  qwen3*)
    REASONING_FLAG="--reasoning-parser qwen3"
    echo "[launch] Qwen3 reasoning parser enabled (reasoning split from the answer)" >&2
    ;;
esac

echo "[launch] submitting ${MODEL_KEY} TP=${TP} constraint=${CONSTRAINT} gres=${GRES} port=${PORT}" >&2

JID=$(
  sbatch --parsable \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --constraint="${CONSTRAINT}" \
    --gres="${GRES}" \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}" \
    --time="${TIME_LIMIT}" \
    --nodes=1 --ntasks=1 \
    --job-name "${MODEL_KEY}:${PORT}" \
    --output "${LOGDIR}/${MODEL_KEY}-%j.out" \
    --error  "${LOGDIR}/${MODEL_KEY}-%j.err" \
    --export=ALL,HF_HOME=${HF_CACHE},HUGGINGFACE_HUB_CACHE=${HF_CACHE},TORCHINDUCTOR_CACHE_DIR=${INDUCTOR_CACHE} \
    --wrap "$(cat <<EOF
set -euo pipefail
# Per-job TMPDIR hygiene: a stale TMPDIR from a previous (cancelled) job makes
# torch.compile's autotune cache write into a dead /scratch path and kills the
# engine on load.
unset TMPDIR SLURM_TMPDIR
export TMPDIR=/tmp/\${USER}_\${SLURM_JOB_ID}
mkdir -p "\$TMPDIR"

module load python/miniforge-25.3.0
eval "\$(mamba shell hook --shell bash)"
mamba activate ${ENV_PATH}

# Gate /v1 with the per-session key (env, not argv). vLLM's auth middleware only
# checks paths under /v1, so /metrics and /health stay open for the metering
# scrape and the readiness poll.
export VLLM_API_KEY="${BACKEND_KEY}"

# Production serve flags. A rate_table.json record is only valid for the exact serve
# flags AND vLLM version it was measured under, so these must stay consistent with
# benchmark/bench_billing.py:build_serve_flags. If you change these flags or upgrade
# vLLM, re-benchmark the affected (model,tier,TP) records -- see the "Upgrading vLLM
# (re-benchmark runbook)" section in ai-session/README.md. As a backstop, metering
# compares the running engine version against each record's provenance.vllm_version
# and bills floor-only on a mismatch. --host 0.0.0.0 so the gateway (login node) and
# the metering scrape can reach it; /v1 is protected by VLLM_API_KEY above. Stats stay
# ON so /metrics is populated.
vllm serve ${MODEL_PATH} \
  --served-model-name ${MODEL_KEY} \
  --host 0.0.0.0 \
  --port ${PORT} \
  --tensor-parallel-size ${TP} \
  --enable-prefix-caching \
  --trust-remote-code \
  --max-model-len ${MAX_MODEL_LEN} \
  --gpu-memory-utilization ${GPU_MEM_UTIL} \
  ${AGENT_FLAGS} \
  ${LORA_FLAGS} \
  ${REASONING_FLAG} \
  ${EAGER_FLAG}
EOF
)"
)

SERVER_LOG="${LOGDIR}/${MODEL_KEY}-${JID}.out"
echo "[launch] submitted jobid=${JID}" >&2

# Single machine-readable line on stdout for ai_session.py. Includes backend_key
# so ai_session can publish it to the gateway (upstream.json, 0600); ai_session
# does NOT persist it in the on-disk session file. This line is captured by
# ai_session (never echoed) and not written to any shared log.
python3 - "$JID" "$PORT" "$MODEL_KEY" "$MODEL_PATH" "$CONSTRAINT" "$TP" "$GRES" "$ACCOUNT" "$PARTITION" "$SERVER_LOG" "$ENFORCE_EAGER" "$BACKEND_KEY" <<'PY'
import json, sys
(_, jid, port, mk, mp, constraint, tp, gres, acct, part, log, eager, backend_key) = sys.argv
gpus = int(gres.split(":")[-1]) if ":" in gres else None
print(json.dumps({
    "jobid": jid, "port": int(port), "model_key": mk, "model_path": mp,
    "constraint": constraint, "tier_requested": constraint.lower(),
    "tp": int(tp), "gres": gres, "n_gpus_requested": gpus,
    "account": acct, "partition": part, "server_log": log,
    "enforce_eager": eager == "1", "backend_key": backend_key,
}))
PY
