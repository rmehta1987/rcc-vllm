#!/bin/bash
# run_coding_agent.sh -- ONE command to bring up (or tear down) the CODING client
# stack for the ai-session service: a local 72B you point a coding tool (aider) at
# to read + edit real code.
#
#     vLLM session   (compute node, SU-billed)
#        -> gateway      127.0.0.1:<GW_PORT>     (free, login-node async proxy)
#        -> aider        (YOU run it -- interactive REPL in your repo's terminal)
#
# Unlike run_browser_demo.sh (which backgrounds a long-running Open WebUI server),
# aider is an INTERACTIVE tool you drive by hand. So `up` brings the session +
# gateway up and then PRINTS the ready-to-run aider command; you run aider yourself
# in the directory of the git repo you want to edit (another terminal / same login
# node, or from your laptop via the printed SSH tunnel).
#
# Usage:
#     bash ai-session/run_coding_agent.sh up        # start session + gateway (default)
#     bash ai-session/run_coding_agent.sh down      # meter + scancel session, stop gateway
#     bash ai-session/run_coding_agent.sh status     # show what's running
#
# CODING DEFAULT = the code-specialized model: Qwen2.5-Coder-32B-Instruct.
#   * A100(80GB) TP=2  (default here) -- code-tuned, HALF the GPUs of the 72B, cheaper.
#   * Override to the general 72B for non-coding/mixed work, or H200 for throughput:
#         MODEL=qwen2.5_72B   TP=4 CONSTRAINT=A100 bash .../run_coding_agent.sh up
#         MODEL=qwen2.5_coder_32B TP=2 CONSTRAINT=H200 bash .../run_coding_agent.sh up
# Coding sessions serve at a WIDE 32K context (MAX_MODEL_LEN, vs the 8192 chat default)
# so aider can actually read repo files -- Qwen2.5 supports 32K natively (no YaRN).
# Still NO --agent-client: aider uses text-edit diffs, not native tool-calls (vLLM
# tool-parsing on a local model is fragile). Interactive coding on an exclusive node is
# floor-billed, so the wider context barely moves the bill (token term is dwarfed).
#
# MULTI-USER (any rcc-staff with read+write in /project/rcc): identical model to
# run_browser_demo.sh -- shared read-only venv/models/code, per-user WRITABLE state
# under AISESSION_STATE_DIR, per-user UID-derived GW_PORT. Override via env:
#     AISESSION_STATE_DIR  GW_PORT  MODEL  TP  CONSTRAINT  EDIT_FORMAT  MAX_MODEL_LEN  READY_TIMEOUT
#
# NOTE: `up` starts a Slurm GPU session -- it SPENDS SU (floor-billed for as long as
# it is up, busy or idle). Always run `down` the moment you stop coding. For a long
# session, run this inside tmux so an SSH drop doesn't kill the login-node gateway.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../ai-session  (shared, read-only)
PY=/project/rcc/mehta5/conda-envs/vllm-probe/bin/python       # shared vllm-probe env (no activation dance)
AIDER_ENV=/project/rcc/mehta5/aider-env                       # aider's OWN venv (NOT vllm-probe)
AIDER_BIN="$AIDER_ENV/bin/aider"
METADATA="$HERE/aider_model_metadata.json"                    # tells litellm the 8192 context (no warning)
U=$(whoami)
UID_NUM=$(id -u)

# Per-user WRITABLE state root (isolates multi-tenant runtime + billing logs).
# Threaded into gateway.py / ai_session.py / launch_ai_session.sh via this env var.
AISESSION_STATE_DIR=${AISESSION_STATE_DIR:-/project/rcc/mehta5/ai-session-state/$U}
export AISESSION_STATE_DIR
mkdir -p "$AISESSION_STATE_DIR/logs/gateway" "$AISESSION_STATE_DIR/run"

RUN_DIR="$AISESSION_STATE_DIR/run"
PIDFILE="$RUN_DIR/coding_agent.pids"
UPSTREAM="$AISESSION_STATE_DIR/logs/gateway/upstream.json"
USAGE_DIR="$AISESSION_STATE_DIR/logs/usage"      # `end` drops <user>_<jobid>_<ts>_summary.json here
KEYFILE="$AISESSION_STATE_DIR/logs/gateway/session_key"  # per-session gateway access key (mode 600)

# Coding wants the code-specialized model. Override via env.
MODEL=${MODEL:-qwen2.5_coder_32B}
TP=${TP:-2}
CONSTRAINT=${CONSTRAINT:-A100}         # uppercase A100 = 80GB nodes (32B@TP2 fits; 40GB 'a100' won't)
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}  # WIDE context for coding (chat default is 8192); Qwen2.5 native 32K
EDIT_FORMAT=${EDIT_FORMAT:-diff}       # token-efficient; the code model emits diffs well. EDIT_FORMAT=whole to rewrite files
# Native tool-calling for AGENT clients (opencode/Cline). aider does NOT need this and
# the default leaves it OFF (vLLM tool-parsing on a local model is fragile). AGENT_CLIENT=1
# starts the session with --agent-client; see ai-session/CODING_AGENTS.md (opencode section).
AGENT_CLIENT=${AGENT_CLIENT:-0}
# Session walltime (HH:MM:SS). Caps how long the GPU is held -- and thus the maximum
# floor charge; the session also ends the moment you run `down`. Flows to the launcher
# via TIME_LIMIT, which ai_session.py now respects when it is already exported (it no
# longer clobbers a caller-set value), so this knob actually takes effect.
TIME=${TIME:-02:00:00}
export TIME_LIMIT="$TIME"
# Per-user default gateway port (UID-derived) so two staff on one login node don't
# clash; override with GW_PORT. Same scheme as run_browser_demo.sh -- you run ONE
# stack at a time (a browser-demo gateway already on this port trips the busy check).
GW_PORT=${GW_PORT:-$((8400 + UID_NUM % 90))}
export GW_PORT
READY_TIMEOUT=${READY_TIMEOUT:-900}
ACTION=${1:-up}

# --- helpers ---------------------------------------------------------------- #
port_busy() { ss -ltn 2>/dev/null | grep -q ":$1 "; }       # is anything listening on :$1 ?

wait_gateway() {   # timeout_s -- poll the gateway's own health endpoint (200 == ready)
  local i=0
  until curl -sf "http://127.0.0.1:$GW_PORT/__gateway/health" >/dev/null 2>&1; do
    sleep 1; i=$((i+1))
    if [ "$i" -ge "$1" ]; then echo "ERROR: gateway health not OK on :$GW_PORT (waited ${1}s)" >&2; return 1; fi
  done
}

# --- optional systemd --user supervision (GATEWAY_SUPERVISE=1) --------------- #
# By default the gateway runs under nohup (below). With GATEWAY_SUPERVISE=1 AND a
# working user-systemd session it runs as a `systemd --user` service instead, so a
# crash is auto-restarted (Restart=on-failure). Login nodes without user systemd
# fall back to nohup, so this never breaks them.
GW_UNIT="ai-session-gateway.service"
have_user_systemd() { command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; }

start_gateway_supervised() {   # requires $KEY exported
  mkdir -p "$HOME/.config/ai-session" "$HOME/.config/systemd/user"
  ( umask 077; cat > "$HOME/.config/ai-session/gateway.env" <<EOF
PYBIN=$PY
GATEWAY_PY=$HERE/gateway.py
GW_PORT=$GW_PORT
AISESSION_STATE_DIR=$AISESSION_STATE_DIR
AISESSION_GATEWAY_KEY=$KEY
EOF
  )
  cp "$HERE/systemd/$GW_UNIT" "$HOME/.config/systemd/user/$GW_UNIT"
  systemctl --user daemon-reload
  systemctl --user restart "$GW_UNIT"
}

stop_gateway_supervised() {   # best-effort; no-op if the unit was never used
  have_user_systemd || return 1
  if systemctl --user is-active "$GW_UNIT" >/dev/null 2>&1; then
    systemctl --user stop "$GW_UNIT" 2>/dev/null || true
    echo "    stopped $GW_UNIT (systemd --user)"; return 0
  fi
  return 1
}

# --- up --------------------------------------------------------------------- #
do_up() {
  if [ ! -x "$AIDER_BIN" ]; then
    echo "ERROR: aider not found at $AIDER_BIN" >&2
    echo "Install it (login node has internet) into its OWN venv -- NOT vllm-probe:" >&2
    echo "    /software/python-3.11.9-el8-x86_64/bin/python3.11 -m venv $AIDER_ENV" >&2
    echo "    $AIDER_ENV/bin/python -m pip install aider-chat" >&2
    exit 1
  fi
  # Guard: don't submit a GPU job for a model that isn't fully on disk yet (e.g. a
  # download still in flight). Resolve the path from the registry and require a
  # config + weights with no leftover *.incomplete shards.
  MODEL_DIR=$($PY -c "import sys; sys.path.insert(0,'$HERE'); import server; print(server.model_path('$MODEL'))" 2>/dev/null || true)
  if [ -z "${MODEL_DIR:-}" ] || [ ! -f "$MODEL_DIR/config.json" ] \
     || ! ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 \
     || ls "$MODEL_DIR"/*.incomplete >/dev/null 2>&1 \
     || ls "$MODEL_DIR"/.cache/huggingface/download/*.incomplete >/dev/null 2>&1; then
    echo "ERROR: model '$MODEL' is not fully staged at: ${MODEL_DIR:-<unknown>}" >&2
    echo "       (missing config.json/*.safetensors, or a download is still in flight)." >&2
    echo "       Wait for staging to finish, or pick a staged model, e.g.:" >&2
    echo "           MODEL=qwen2.5_72B TP=4 bash $HERE/run_coding_agent.sh up" >&2
    exit 1
  fi
  if port_busy "$GW_PORT"; then
    echo "Something is already listening on :$GW_PORT (maybe a browser demo or another user)." >&2
    echo "Either '$HERE/run_coding_agent.sh down', or pick a free port:" >&2
    echo "    GW_PORT=8490 bash $HERE/run_coding_agent.sh up" >&2
    exit 1
  fi

  AGENT_FLAG=""
  if [ "$AGENT_CLIENT" = "1" ]; then AGENT_FLAG="--agent-client"; fi

  # Pre-flight SU estimate -- printed BEFORE the GPU job is submitted so you see
  # the reservation-floor cost before committing hardware. N=TP (single-node
  # session); the walltime is $TIME (exported as TIME_LIMIT), the same value
  # ai_session.py bills against. A whole-node reservation can bill a larger N;
  # token work only adds above the floor.
  echo "==> pre-flight: $($PY "$HERE/preflight_estimate.py" --constraint "$CONSTRAINT" --n "$TP" --time "$TIME")"

  echo "==> state dir : $AISESSION_STATE_DIR   (user $U)"
  echo "==> [1/2] starting vLLM session ($MODEL TP=$TP $CONSTRAINT, ctx=$MAX_MODEL_LEN${AGENT_FLAG:+, tool-calling}, walltime=$TIME) -- SU-billed; blocks until READY"
  # pipefail makes the pipeline fail if `start` fails even though tee succeeds.
  # $AGENT_FLAG is intentionally unquoted so an empty value expands to nothing.
  $PY "$HERE/ai_session.py" start \
      --model "$MODEL" --tp "$TP" --constraint "$CONSTRAINT" \
      --max-model-len "$MAX_MODEL_LEN" $AGENT_FLAG \
      --wait --ready-timeout "$READY_TIMEOUT" 2>&1 | tee "$RUN_DIR/start.log"
  if ! grep -q '"active": true' "$UPSTREAM" 2>/dev/null; then
    echo "ERROR: session did not publish an active backend ($UPSTREAM); aborting." >&2
    exit 1
  fi

  # Mint ONE per-session access key. The gateway will REQUIRE it (Bearer/API key).
  # Share it with your lab so they can use THIS session over their own tunnel; all
  # of their usage bills to YOU (the starter). Written 600 so only you can read it.
  KEY=$(openssl rand -hex 16)
  ( umask 077; printf '%s\n' "$KEY" > "$KEYFILE" )
  chmod 600 "$KEYFILE"
  export AISESSION_GATEWAY_KEY="$KEY"

  echo "==> [2/2] starting gateway on 127.0.0.1:$GW_PORT (API-key auth ENABLED)"
  if [ "${GATEWAY_SUPERVISE:-0}" = "1" ] && have_user_systemd; then
    start_gateway_supervised
    wait_gateway 60
    echo "    gateway healthy under systemd --user ($GW_UNIT, Restart=on-failure)"
    echo "    logs: journalctl --user -u $GW_UNIT -f"
    : > "$PIDFILE"                       # no PID to track; `down` stops the unit
  else
    [ "${GATEWAY_SUPERVISE:-0}" = "1" ] && \
      echo "    (GATEWAY_SUPERVISE=1 but no user systemd here -- using nohup)" >&2
    nohup "$PY" "$HERE/gateway.py" --host 127.0.0.1 --port "$GW_PORT" \
        > "$RUN_DIR/gateway.log" 2>&1 &
    GW_PID=$!
    wait_gateway 60
    echo "    gateway healthy (pid $GW_PID)  log: $RUN_DIR/gateway.log"
    echo "gateway $GW_PID" > "$PIDFILE"
  fi

  local login; login=$(hostname -s)
  cat <<EOF

================ READY -- code with the local ${MODEL} (ctx ${MAX_MODEL_LEN}) ================

  SESSION ACCESS KEY:  ${KEY}

  The gateway now REQUIRES this key. Share it with your lab so they can use THIS
  session over their OWN SSH tunnel to :${GW_PORT}; each member sets it as the
  OpenAI API key in their client (OPENAI_API_KEY / the client's API-key field).
  ALL of their usage bills to YOU ($U), the starter. Without the key the gateway
  refuses every request (401). Saved (mode 600, only you can read) at:
      ${KEYFILE}

The session + gateway are up. aider is INTERACTIVE -- run it yourself in the
git repo you want to edit. On THIS login node ($login), in your repo dir:

  cd /path/to/your/repo        # a git repo (aider needs one; 'git init' if new)
  OPENAI_API_BASE=http://localhost:${GW_PORT}/v1 OPENAI_API_KEY=${KEY} \\
    ${AIDER_BIN} \\
      --model openai/${MODEL} --weak-model openai/${MODEL} \\
      --model-metadata-file ${METADATA} \\
      --edit-format ${EDIT_FORMAT} --analytics-disable

One-shot (non-interactive) mode -- good for scripts/batch, no REPL:
  OPENAI_API_BASE=http://localhost:${GW_PORT}/v1 OPENAI_API_KEY=${KEY} \\
    ${AIDER_BIN} --model openai/${MODEL} --weak-model openai/${MODEL} \\
      --model-metadata-file ${METADATA} --edit-format ${EDIT_FORMAT} --analytics-disable \\
      --yes-always --no-auto-commit --message "add a docstring to foo() in bar.py"

Other clients can use the SAME endpoint (see ai-session/CODING_AGENTS.md):
  base URL  http://localhost:${GW_PORT}/v1     API key  ${KEY}     model  ${MODEL}

(If you'd rather run aider on your LAPTOP, first tunnel the gateway port:
  ssh -N -L ${GW_PORT}:localhost:${GW_PORT} -J ${U}@midway3.rcc.uchicago.edu ${U}@${login}
 then use the same command on your laptop against http://localhost:${GW_PORT}/v1.)

The SU clock is running. When done (frees the GPU, stops billing):

  bash $HERE/run_coding_agent.sh down
=======================================================================
EOF
}

# --- down ------------------------------------------------------------------- #
do_down() {
  local before after stopped=0 name pid pids
  # snapshot the newest billing receipt BEFORE end, so we can tell whether THIS
  # run actually billed a session (a new *_summary.json appears) vs. a no-op down.
  before=$(ls -t "$USAGE_DIR"/*_summary.json 2>/dev/null | head -1 || true)

  echo "==> ending session (meter + scancel + clear gateway)  [state: $AISESSION_STATE_DIR]"
  $PY "$HERE/ai_session.py" end || echo "    (no active session to end, or already ended)"

  echo "==> stopping gateway"
  # If it was supervised (GATEWAY_SUPERVISE=1), stop the unit FIRST so systemd does
  # not restart it when the port-owner kill below lands (SIGTERM -> non-clean exit).
  if stop_gateway_supervised; then stopped=1; fi
  if [ -f "$PIDFILE" ]; then
    while read -r name pid; do
      [ -n "${pid:-}" ] || continue
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        echo "    stopped $name (pid $pid)"; stopped=1
      fi
    done < "$PIDFILE"
    rm -f "$PIDFILE"
  fi
  # belt-and-suspenders: kill whatever still owns MY gateway port.
  # NB: by PORT OWNER via ss -- never `pkill -f`, whose pattern self-matches this shell.
  pids=$(ss -ltnp 2>/dev/null | grep ":$GW_PORT " | grep -oP 'pid=\K[0-9]+' | sort -u || true)
  for pid in $pids; do
    if kill "$pid" 2>/dev/null; then echo "    stopped :$GW_PORT owner (pid $pid)"; stopped=1; fi
  done
  [ "$stopped" -eq 1 ] || echo "    (nothing was running)"

  # remove the per-session access key -- it only applied to the session just ended.
  [ -f "$KEYFILE" ] && { rm -f "$KEYFILE"; echo "    removed session access key ($KEYFILE)"; }

  # --- the whole point of `down`: report the SU charge, LAST, so it can't scroll off ---
  # print_su_receipt.py renders the newest receipt only if it's newer than the
  # pre-`end` snapshot ($before) -- else it reports "none this run".
  "$PY" "$HERE/print_su_receipt.py" --usage-dir "$USAGE_DIR" --since "$before"
}

# --- status ----------------------------------------------------------------- #
do_status() {
  echo "== ai-session coding agent status  (user $U, gateway port $GW_PORT) =="
  echo "-- state dir: $AISESSION_STATE_DIR --"
  echo "-- aider: $([ -x "$AIDER_BIN" ] && echo "$AIDER_BIN" || echo 'NOT INSTALLED') --"
  echo "-- gateway upstream ($UPSTREAM) --"
  cat "$UPSTREAM" 2>/dev/null || echo "  (none)"
  echo
  echo "-- session access key ($KEYFILE) --"
  if [ -f "$KEYFILE" ]; then
    echo "  set: $(cut -c1-6 "$KEYFILE" 2>/dev/null)...  (first 6 chars only; shared with your lab, bills to you)"
  else
    echo "  (none -- keyless)"
  fi
  echo "-- listener on :$GW_PORT --"
  ss -ltn 2>/dev/null | grep -E ":$GW_PORT " || echo "  (none)"
  echo "-- saved pids --"
  cat "$PIDFILE" 2>/dev/null || echo "  (no pidfile)"
  echo "-- slurm jobs --"
  squeue -u "$U" 2>/dev/null || true
}

# --- dispatch --------------------------------------------------------------- #
case "$ACTION" in
  up)     do_up ;;
  down)   do_down ;;
  status) do_status ;;
  *) echo "usage: $(basename "$0") {up|down|status}" >&2; exit 2 ;;
esac
