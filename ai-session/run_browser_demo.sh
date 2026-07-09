#!/bin/bash
# run_browser_demo.sh -- ONE command to bring up (or tear down) the full
# "chat in your browser" stack for the ai-session service:
#
#     vLLM session   (compute node, SU-billed)
#        -> gateway      127.0.0.1:<GW_PORT>     (free, login-node async proxy)
#        -> Open WebUI   127.0.0.1:<OWUI_PORT>   (free, login-node chat UI)
#
# Then SSH-tunnel the UI port to your laptop and browse http://localhost:<OWUI_PORT>.
#
# Usage:
#     bash ai-session/run_browser_demo.sh up        # start everything (default)
#     bash ai-session/run_browser_demo.sh down      # meter+scancel session, stop gateway+UI
#     bash ai-session/run_browser_demo.sh status     # show what's running
#
# MULTI-USER (any rcc-staff with read+write in /project/rcc):
#   The venv, models and code are SHARED, read-only (group rcc-staff, setgid) --
#   you run mehta5's one managed install, you don't copy it. Your WRITABLE state
#   (session files, usage/billing logs, gateway pointer, Slurm logs) is isolated
#   under a PER-USER dir so colleagues never clobber each other:
#       AISESSION_STATE_DIR  (default: $HOME/.ai-session/state)
#   The OWUI chat DB lives privately in $HOME/.ai-session/openwebui-data (mode 700).
#   and each user gets PER-USER default ports (derived from your UID) so two people
#   on the same login node don't collide. Override any of these via env:
#       AISESSION_STATE_DIR  GW_PORT  OWUI_PORT  MODEL  TP  CONSTRAINT  READY_TIMEOUT
#
#   e.g. the big model:  MODEL=qwen2.5_72B TP=4 CONSTRAINT=A100 bash .../run_browser_demo.sh up
#
# NOTE: `up` starts a Slurm GPU session -- it SPENDS SU (floor-billed for as long
# as it is up). Always run `down` when you're done. For a long session, run this
# inside tmux so an SSH drop doesn't kill the login-node processes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../ai-session  (shared, read-only)
PY=/project/rcc/mehta5/conda-envs/vllm-probe/bin/python       # shared vllm-probe env (no activation dance)
U=$(whoami)
UID_NUM=$(id -u)

# Per-user WRITABLE state root (isolates multi-tenant runtime + billing logs).
# Threaded into gateway.py / ai_session.py / launch_ai_session.sh via this env var.
AISESSION_STATE_DIR=${AISESSION_STATE_DIR:-$HOME/.ai-session/state}
export AISESSION_STATE_DIR
mkdir -p "$AISESSION_STATE_DIR/logs/gateway" "$AISESSION_STATE_DIR/run"

RUN_DIR="$AISESSION_STATE_DIR/run"
PIDFILE="$RUN_DIR/browser_demo.pids"
UPSTREAM="$AISESSION_STATE_DIR/logs/gateway/upstream.json"
USAGE_DIR="$AISESSION_STATE_DIR/logs/usage"      # `end` drops <user>_<jobid>_<ts>_summary.json here
KEYFILE="$AISESSION_STATE_DIR/logs/gateway/session_key"  # per-session gateway access key (mode 600)

MODEL=${MODEL:-qwen3_4b}
TP=${TP:-1}
CONSTRAINT=${CONSTRAINT:-A100}
# Session walltime (HH:MM:SS). Caps how long the GPU is held -- and thus the maximum
# floor charge; the session also ends the moment you run `down`. Flows to the launcher
# via TIME_LIMIT, which ai_session.py now respects when it is already exported (it no
# longer clobbers a caller-set value), so this knob actually takes effect.
TIME=${TIME:-02:00:00}
export TIME_LIMIT="$TIME"
# Per-user default ports (UID-derived) so two staff on one login node don't clash;
# override with GW_PORT / OWUI_PORT. GW_PORT MUST be what run_openwebui.sh targets
# (it reads GW_PORT too -- we export it below so they always agree).
GW_PORT=${GW_PORT:-$((8400 + UID_NUM % 90))}
OWUI_PORT=${OWUI_PORT:-$((3000 + UID_NUM % 90))}
export GW_PORT
READY_TIMEOUT=${READY_TIMEOUT:-900}
ACTION=${1:-up}

# --- helpers ---------------------------------------------------------------- #
port_busy() { ss -ltn 2>/dev/null | grep -q ":$1 "; }       # is anything listening on :$1 ?

wait_port() {   # port  timeout_s  label
  local i=0
  while ! ss -ltn 2>/dev/null | grep -q ":$1 "; do
    sleep 1; i=$((i+1))
    if [ "$i" -ge "$2" ]; then echo "ERROR: $3 never bound :$1 (waited ${2}s)" >&2; return 1; fi
  done
}

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
  if port_busy "$GW_PORT" || port_busy "$OWUI_PORT"; then
    echo "Something is already listening on :$GW_PORT or :$OWUI_PORT (maybe another user on this node)." >&2
    echo "Either '$HERE/run_browser_demo.sh down', or pick free ports:" >&2
    echo "    GW_PORT=8490 OWUI_PORT=3090 bash $HERE/run_browser_demo.sh up" >&2
    exit 1
  fi

  # Pre-flight SU estimate -- printed BEFORE the GPU job is submitted so you see
  # the reservation-floor cost before committing hardware. N=TP (single-node
  # session); the walltime is $TIME (exported as TIME_LIMIT), the same value
  # ai_session.py bills against. A whole-node reservation can bill a larger N;
  # token work only adds above the floor.
  echo "==> pre-flight: $($PY "$HERE/preflight_estimate.py" --constraint "$CONSTRAINT" --n "$TP" --time "$TIME")"

  echo "==> state dir : $AISESSION_STATE_DIR   (user $U)"
  echo "==> [1/3] starting vLLM session ($MODEL TP=$TP $CONSTRAINT, walltime=$TIME) -- SU-billed; blocks until READY"
  # pipefail makes the pipeline fail if `start` fails even though tee succeeds.
  $PY "$HERE/ai_session.py" start \
      --model "$MODEL" --tp "$TP" --constraint "$CONSTRAINT" \
      --wait --ready-timeout "$READY_TIMEOUT" 2>&1 | tee "$RUN_DIR/start.log"
  if ! grep -q '"active": true' "$UPSTREAM" 2>/dev/null; then
    echo "ERROR: session did not publish an active backend ($UPSTREAM); aborting." >&2
    exit 1
  fi

  # Mint ONE per-session access key. The gateway will REQUIRE it (Bearer/API key)
  # and Open WebUI (started below) reads it from AISESSION_GATEWAY_KEY. Share it
  # with your lab so they can use THIS session over their own tunnel; all of their
  # usage bills to YOU (the starter). Written 600 so only you can read it.
  KEY=$(openssl rand -hex 16)
  ( umask 077; printf '%s\n' "$KEY" > "$KEYFILE" )
  chmod 600 "$KEYFILE"
  export AISESSION_GATEWAY_KEY="$KEY"

  echo "==> [2/3] starting gateway on 127.0.0.1:$GW_PORT (API-key auth ENABLED)"
  GW_SUPERVISED=0
  if [ "${GATEWAY_SUPERVISE:-0}" = "1" ] && have_user_systemd; then
    start_gateway_supervised
    wait_gateway 60
    echo "    gateway healthy under systemd --user ($GW_UNIT, Restart=on-failure)"
    echo "    logs: journalctl --user -u $GW_UNIT -f"
    GW_SUPERVISED=1
  else
    [ "${GATEWAY_SUPERVISE:-0}" = "1" ] && \
      echo "    (GATEWAY_SUPERVISE=1 but no user systemd here -- using nohup)" >&2
    nohup "$PY" "$HERE/gateway.py" --host 127.0.0.1 --port "$GW_PORT" \
        > "$RUN_DIR/gateway.log" 2>&1 &
    GW_PID=$!
    wait_gateway 60
    echo "    gateway healthy (pid $GW_PID)  log: $RUN_DIR/gateway.log"
  fi

  echo "==> [3/3] starting Open WebUI on 127.0.0.1:$OWUI_PORT (heavy imports -- ~30-60s)"
  # run_openwebui.sh reads GW_PORT (exported) + AISESSION_STATE_DIR, then `exec`s
  # open-webui, so $! IS the UI process.
  nohup bash "$HERE/run_openwebui.sh" "$OWUI_PORT" \
      > "$RUN_DIR/openwebui.log" 2>&1 &
  OWUI_PID=$!
  wait_port "$OWUI_PORT" 180 "Open WebUI"
  echo "    Open WebUI serving (pid $OWUI_PID)  log: $RUN_DIR/openwebui.log"

  {
    [ "$GW_SUPERVISED" = "1" ] || echo "gateway $GW_PID"   # supervised -> `down` stops the unit
    echo "openwebui $OWUI_PID"
  } > "$PIDFILE"

  local login; login=$(hostname -s)
  cat <<EOF

================ READY -- chat in your browser ================

  SESSION ACCESS KEY:  ${KEY}

  The gateway now REQUIRES this key. The Open WebUI started here already uses it,
  so YOUR browser tab works out of the box. To let your lab use THIS session,
  share this key: each member points their own client at the gateway (their own
  SSH tunnel to :${GW_PORT}) and sets this as the OpenAI API key. ALL of their
  usage bills to YOU ($U), the starter. Without the key the gateway refuses every
  request (401). Saved (mode 600, only you can read) at:
      ${KEYFILE}

On your LAPTOP, open the tunnel to THIS login node ($login) -- one login, -f backgrounds it:

  ssh -N -f -L ${OWUI_PORT}:localhost:${OWUI_PORT} ${U}@${login}.rcc.uchicago.edu

then browse:   http://localhost:${OWUI_PORT}      (pick model '${MODEL}')

The SU clock is running. When done (frees the GPU, stops billing):

  bash $HERE/run_browser_demo.sh down
==============================================================
EOF
}

# --- down ------------------------------------------------------------------- #
do_down() {
  local before after stopped=0 name pid port pids
  # snapshot the newest billing receipt BEFORE end, so we can tell whether THIS
  # run actually billed a session (a new *_summary.json appears) vs. a no-op down.
  before=$(ls -t "$USAGE_DIR"/*_summary.json 2>/dev/null | head -1 || true)

  echo "==> ending session (meter + scancel + clear gateway)  [state: $AISESSION_STATE_DIR]"
  $PY "$HERE/ai_session.py" end || echo "    (no active session to end, or already ended)"

  echo "==> stopping gateway + Open WebUI"
  # If the gateway was supervised (GATEWAY_SUPERVISE=1), stop the unit FIRST so
  # systemd does not restart it when the port-owner kill below lands (SIGTERM ->
  # non-clean exit -> Restart=on-failure would re-spawn it).
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
  # belt-and-suspenders: kill whatever still owns MY ports.
  # NB: by PORT OWNER via ss -- never `pkill -f`, whose pattern self-matches this shell.
  for port in "$GW_PORT" "$OWUI_PORT"; do
    pids=$(ss -ltnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | sort -u || true)
    for pid in $pids; do
      if kill "$pid" 2>/dev/null; then echo "    stopped :$port owner (pid $pid)"; stopped=1; fi
    done
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
  echo "== ai-session browser demo status  (user $U, ports $GW_PORT/$OWUI_PORT) =="
  echo "-- state dir: $AISESSION_STATE_DIR --"
  echo "-- gateway upstream ($UPSTREAM) --"
  cat "$UPSTREAM" 2>/dev/null || echo "  (none)"
  echo
  echo "-- session access key ($KEYFILE) --"
  if [ -f "$KEYFILE" ]; then
    echo "  set: $(cut -c1-6 "$KEYFILE" 2>/dev/null)...  (first 6 chars only; shared with your lab, bills to you)"
  else
    echo "  (none -- keyless)"
  fi
  echo "-- listeners on :$GW_PORT / :$OWUI_PORT --"
  ss -ltn 2>/dev/null | grep -E ":($GW_PORT|$OWUI_PORT) " || echo "  (none)"
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
