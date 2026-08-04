#!/bin/bash
# Autonomous driver for the coding-model refresh on the vLLM 0.26.0 stack.
# Pure orchestrator on --partition=build (the operator's routing): runs the headless `claude` CLI loop + the
# subagent gauntlet + git, downloads candidate weights IN THIS build allocation (internet + Xet disabled), and
# submits ALL gate-deciding compute as NESTED sbatch jobs -- GPU checks to --partition=test (A100/H200), CPU
# work to --partition=caslake. Never reserves a GPU here; never bills the SU floor; every nested GPU job is
# time-boxed. Survives build's 6h QOS wall by chaining a fresh job ~10 min before the kill; resumes losslessly
# because the driver commits after every step, re-orients from git (prompt §1), and re-attaches to in-flight
# `mrefresh-nest*` jobs by id.
#
# Drives prompts/_cluster_autonomous_model_refresh_loop.md, whose §10 emits the sentinels
# STAGE_LANDED | MODEL_REFRESH_COMPLETE | BLOCKED. The greps below MUST match those names.
#
# Submit from the repo root:  mkdir -p logs && sbatch scripts/run_model_refresh_loop.sbatch
# Stop:  scancel <jobid>  (also scancel any queued chained job and any mrefresh-nest* job;
#                          rm _scratch/.mrefresh_started for a fresh start)
# Right-size after run 1:  seff <jobid>   /   sacct -j <jobid> -o JobID,MaxRSS,Elapsed
#
#SBATCH --job-name=vllm-model-refresh
#SBATCH --output=logs/vllm-model-refresh-%j.log
#SBATCH --open-mode=append
#SBATCH --partition=build
#SBATCH --account=rcc-staff
#SBATCH --qos=build
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                       # headroom for an in-allocation download + the CPU arch dry-run import
#SBATCH --time=06:00:00                 # 6h = build's QOS MaxWall; the USR1 chain spans longer runs
#SBATCH --signal=B:USR1@600             # SIGUSR1 to the batch shell ~10 min before the kill

set -uo pipefail
set -m                                  # monitor mode: each `cmd &` gets its OWN process group, so the trap
                                        # can kill claude's whole tree (node + git + pytest/hf children)

REPO="$(cd "${SLURM_SUBMIT_DIR:-$PWD}" && git rev-parse --show-toplevel)"
SELF="$REPO/scripts/run_model_refresh_loop.sbatch"   # stable path -- NOT $0 (Slurm runs a spooled copy)
cd "$REPO"
mkdir -p logs _scratch

# Sanitize the launching shell's Claude Code session vars so the nested `claude -p` driver starts a FRESH
# headless session. This loop may be launched from WITHIN an interactive Claude Code session (which exports
# CLAUDE_CODE_SESSION_ID / CLAUDECODE / CLAUDE_PID / ...); inheriting them can misroute the child driver, let it
# touch the parent session's state, or (CLAUDE_PID) confuse the USR1 trap's process-group kill before the loop
# reassigns it. Auth is file-based (~/.claude/.credentials.json), so unsetting these does NOT affect login.
unset CLAUDECODE CLAUDE_CODE_SESSION_ID CLAUDE_CODE_CHILD_SESSION CLAUDE_CODE_ENTRYPOINT \
      CLAUDE_CODE_EXECPATH CLAUDE_PID CLAUDE_EFFORT AI_AGENT CLAUDE_CODE_SSE_PORT 2>/dev/null || true

# Branch guard: land edits on the milestone branch, NEVER main. The agent commits to HEAD and the sbatch may be
# launched from any branch -- force the branch up front and FAIL LOUD rather than committing to main.
BASELINE_BRANCH="milestone/model-refresh"
if [[ "$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" != "$BASELINE_BRANCH" ]]; then
  git checkout "$BASELINE_BRANCH" 2>/dev/null || git checkout -b "$BASELINE_BRANCH" 2>/dev/null || {
    echo "[FATAL] $(date) cannot switch to $BASELINE_BRANCH (dirty tree?) -- refusing to run on $(git rev-parse --abbrev-ref HEAD 2>/dev/null) to avoid committing to the wrong branch." >&2; exit 1; }
  echo "[branch] $(date) switched to $BASELINE_BRANCH"
fi

FF_FILE="$REPO/_scratch/.mrefresh_fastfail"
fastfail=$(cat "$FF_FILE" 2>/dev/null || echo 0); [[ "$fastfail" =~ ^[0-9]+$ ]] || fastfail=0   # load EARLY so the USR1 trap sees a fastfail storm

# --- nested-job cleanup: the orchestrator (a shell script) CAN scancel; the agent driver cannot. On a genuine
# terminal STOP, cancel any lingering time-boxed mrefresh-nest* GPU/CPU job so nothing runs orphaned on the
# paid test partition. NOT called on the USR1 chain (the successor re-attaches to the live nested job). ---
cancel_nested() {
  local ids; ids=$(squeue --me -h -o '%i %j' 2>/dev/null | awk '$2 ~ /^mrefresh-nest/ {print $1}' | tr '\n' ' ')
  if [[ -n "${ids// /}" ]]; then echo "[cleanup] $(date) scancel orphaned nested jobs: $ids"; scancel $ids 2>/dev/null || true; fi
}

# --- heartbeat: every ~30 min tail the latest per-iteration claude log into --output so a long no-commit
# iteration (an in-allocation download, a nested benchmark) is not mistaken for a stall. ---
HEARTBEAT_PID=""
heartbeat() {
  while :; do
    sleep 1800
    local latest; latest=$(ls -t logs/mrefresh-iter-*.log 2>/dev/null | head -1)
    if [[ -n "$latest" ]]; then echo "[heartbeat] $(date) -- tail of $latest:"; tail -n 25 "$latest" 2>/dev/null || true
    else echo "[heartbeat] $(date) -- no iter log yet (between iterations / nested job in flight)"; fi
  done
}
stop_heartbeat() { [[ -n "${HEARTBEAT_PID:-}" ]] && kill "$HEARTBEAT_PID" 2>/dev/null || true; HEARTBEAT_PID=""; }

# --- helpers (defined BEFORE the trap can fire) ---
stop_child() {
  stop_heartbeat
  [[ -n "${CLAUDE_PID:-}" ]] || return 0
  kill -TERM -"$CLAUDE_PID" 2>/dev/null || true
  wait "$CLAUDE_PID" 2>/dev/null || true
}
resubmit_self() {                       # chain a fresh job; retry a transient scheduler error before giving up.
  mkdir -p "$REPO/logs"                  # (build QOS = 1 job/user normally QUEUES the successor as PENDING, not
  local rc a                             #  reject, so a foreign build job just delays -- does not break -- the chain.)
  for a in 1 2 3; do
    sbatch "$SELF"; rc=$?
    if (( rc == 0 )); then echo "[chain] $(date) resubmitted OK (attempt $a)"; return 0; fi
    echo "[chain] $(date) resubmit attempt $a/3 failed rc=$rc -- retry in 30s" >&2; sleep 30
  done
  echo "[FATAL] $(date) sbatch resubmit FAILED after 3 attempts -- chain BROKEN (check build QOS / time limits)." >&2
  return 1                              # tell the caller the chain is dead so it can cancel orphaned nested jobs
}
chain_on_signal() {                     # USR1 ~10 min before the kill: stop child FIRST, then chain
  trap '' USR1
  echo "[chain] $(date) USR1 received -- stopping child before resubmit"
  stop_child
  if (( ${fastfail:-0} >= 5 )); then echo "[stop] $(date) fastfail storm -- NOT resubmitting"; cancel_nested; exit 1; fi
  # On a LIVE chain, do NOT cancel_nested: the successor re-attaches to the running nested job. But if the
  # resubmit truly FAILED (no successor), a live time-boxed GPU job would run orphaned -- so cancel it then.
  resubmit_self || { echo "[cleanup] $(date) chain dead -- cancelling nested jobs so none run orphaned" >&2; cancel_nested; }
  exit 0
}
trap chain_on_signal USR1

# Env for the ORCHESTRATOR: miniforge module for git/mamba tooling + the node `claude` CLI on PATH. The NESTED
# GPU serves go through tools/serve_cu129.sbatch (the vllm-serve-cu129 env); the in-allocation CPU arch dry-run
# uses the vllm-serve-cu129 python by ABSOLUTE path, so no activation here.
module load python/miniforge-25.3.0 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"    # the node-based `claude` CLI
command -v claude >/dev/null || { echo "[FATAL] $(date) claude CLI not on PATH ($HOME/.local/bin)" >&2; exit 1; }
export VLLM_SERVE_PY="/project/rcc/mehta5/conda-envs/vllm-serve-cu129/bin/python"
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HOME=/project/rcc/mehta5/hf-cache   # in-allocation downloads

heartbeat & HEARTBEAT_PID=$!
echo "[heartbeat] $(date) -- started (pid $HEARTBEAT_PID), tailing latest logs/mrefresh-iter-*.log every ~30 min"

PROMPT="prompts/_cluster_autonomous_model_refresh_loop.md"
[[ -r "$PROMPT" ]] || { echo "[FATAL] $(date) prompt not readable: $PROMPT" >&2; stop_heartbeat; exit 1; }

# Loop control -- health is judged by PROGRESS (a new git commit), not exit code or wall-clock. Counters
# persist across chained jobs, reset on any progress. Terminal: the §10 sentinels, a breaker, or the iter cap.
MAX_ITERS=200
MAX_THROTTLE=12                         # ~50 min of capped backoff before declaring the API down
MAX_NOPROGRESS=25                       # consecutive no-commit iterations before declaring a stall
TH_FILE="$REPO/_scratch/.mrefresh_throttle"
NP_FILE="$REPO/_scratch/.mrefresh_noprogress"
fastfail=$(cat "$FF_FILE" 2>/dev/null || echo 0); [[ "$fastfail" =~ ^[0-9]+$ ]] || fastfail=0
throttle=$(cat "$TH_FILE" 2>/dev/null || echo 0); [[ "$throttle" =~ ^[0-9]+$ ]] || throttle=0
noprogress=$(cat "$NP_FILE" 2>/dev/null || echo 0); [[ "$noprogress" =~ ^[0-9]+$ ]] || noprogress=0
reset_breakers(){ fastfail=0; throttle=0; echo 0 > "$FF_FILE"; echo 0 > "$TH_FILE"; }
# Broadened past the reference: the `claude` CLI renders a 529/429 as bare `API Error: Overloaded` / `API Error:
# 429` with NO status/http/code keyword, which the reference regex missed -> a recoverable overload was charged
# to the fastfail breaker and killed the run after 5 iters (gauntlet finding). Match bare `overloaded`, any
# `\b(429|529)\b`, and `api error ... (429|529)`. Over-matching only routes to the CAPPED throttle backoff.
is_throttle(){ grep -qiE 'temporarily limiting requests|rate.?limit(ed|ing)?|overloaded|"type" *: *"(overloaded|rate_limit)|too many requests|api error[^0-9]{0,12}(429|529)|\b(429|529)\b' "$1"; }

for ((i=1; i<=MAX_ITERS; i++)); do
  # If a NESTED mrefresh-nest* job is still in flight, WAIT (poll squeue, don't count it). SCOPED to the
  # `mrefresh-nest` prefix (a POSITIVE match on the job name) so this user's OTHER-project jobs -- phase0-5,
  # any amd/bigmem/caslake work unrelated to this loop -- are NEVER waited on, counted, or cancelled. CAPPED at
  # 2.5h so a stuck-PENDING nested job falls through to a diagnosing iteration instead of spinning. A squeue
  # ERROR (slurmctld hiccup) is NOT read as "idle" -- that would spawn an overlapping iteration / double-submit;
  # retry 3x, then fail-open (assume idle) only to avoid a permanent deadlock on a truly dead scheduler.
  wait_start=$SECONDS; sq_fail=0
  while [[ -n "${SLURM_JOB_ID:-}" ]]; do
    sq_raw=$(squeue --me -h -t RUNNING,PENDING -o '%i %j' 2>/dev/null); sq_rc=$?
    if (( sq_rc != 0 )); then
      sq_fail=$((sq_fail+1))
      if (( sq_fail >= 3 )); then echo "[wait] $(date) squeue errored ${sq_fail}x -- fail-open (assume no nested job) to avoid deadlock" >&2; break; fi
      echo "[wait] $(date) squeue hiccup ($sq_fail/3) -- retry in 15s, NOT assuming idle" >&2; sleep 15; continue
    fi
    sq_fail=0
    nested=$(printf '%s\n' "$sq_raw" | awk '$2 ~ /^mrefresh-nest/ {print $1}' | tr '\n' ' ')
    [[ -z "${nested// /}" ]] && break                                   # clean read, no mrefresh-nest job -> proceed
    if (( SECONDS - wait_start > 9000 )); then
      echo "[wait] $(date) nested job(s) [$nested] still in flight after 2.5h -- proceeding to a diagnosing iteration, NOT spinning" >&2; break
    fi
    echo "[wait] $(date) nested mrefresh-nest job(s) in flight: $nested -- polling every 120s (not a stall)"
    sleep 120
  done

  LOG="logs/mrefresh-iter-${SLURM_JOB_ID:-local}-$i.log"
  head_before=$(git rev-parse HEAD 2>/dev/null || echo none)
  if [[ -f _scratch/.mrefresh_started ]]; then
    claude -p --continue "Continue the loop per $PROMPT. Orient from git state (§1); resume the current stage." \
      --model claude-opus-4-8 --effort max --dangerously-skip-permissions --max-turns 400 > "$LOG" 2>&1 &
  else
    touch _scratch/.mrefresh_started
    claude -p "$(cat "$PROMPT")" \
      --model claude-opus-4-8 --effort max --dangerously-skip-permissions --max-turns 400 > "$LOG" 2>&1 &
  fi
  CLAUDE_PID=$!; wait "$CLAUDE_PID"; rc=$?

  # Sentinels scanned line-anchored over the WHOLE log. These names MUST match prompt §10.
  # Each terminal exit disarms the USR1 trap FIRST, so a signal landing in the exit window cannot resurrect a
  # run that already decided to stop (gauntlet finding). COMPLETE also sweeps any lingering nested job.
  if grep -qE '^=== LOOP: MODEL_REFRESH_COMPLETE ===$' "$LOG"; then trap '' USR1; echo "[done] $(date) model refresh complete"; stop_heartbeat; cancel_nested; rm -f "$FF_FILE" "$TH_FILE" "$NP_FILE"; exit 0; fi
  if grep -qE '^=== LOOP: BLOCKED ===$' "$LOG";                then trap '' USR1; echo "[stop] $(date) driver signalled BLOCKED"; stop_heartbeat; cancel_nested; exit 1; fi
  if grep -qE '^=== LOOP: STAGE_LANDED ===$' "$LOG"; then
    echo "[stage] $(date) STAGE_LANDED -- fresh session for the next stage"
    rm -f _scratch/.mrefresh_started; reset_breakers; noprogress=0; echo 0 > "$NP_FILE"; continue
  fi

  head_after=$(git rev-parse HEAD 2>/dev/null || echo none)
  if [[ "$head_after" != "$head_before" && "$head_after" != none ]]; then
    reset_breakers; noprogress=0; echo 0 > "$NP_FILE"; continue          # real progress
  fi
  noprogress=$((noprogress+1)); echo "$noprogress" > "$NP_FILE"
  if (( noprogress >= MAX_NOPROGRESS )); then
    trap '' USR1; echo "[stop] $(date) $noprogress iterations: no commit AND no nested job in flight -- genuinely stalled. BLOCKED" >&2; stop_heartbeat; cancel_nested; exit 1
  fi
  if (( rc == 0 )); then reset_breakers; continue; fi                    # clean exit, no commit yet: keep going (bounded)
  if is_throttle "$LOG"; then
    throttle=$((throttle+1)); echo "$throttle" > "$TH_FILE"
    if (( throttle > MAX_THROTTLE )); then trap '' USR1; echo "[stop] $(date) API throttled ${throttle}x with no progress. BLOCKED" >&2; stop_heartbeat; cancel_nested; exit 1; fi
    wait_s=$(( 60 * throttle )); (( wait_s > 300 )) && wait_s=300
    echo "[throttle] $(date) iter $i rate-limited (retry $throttle/$MAX_THROTTLE) -- sleeping ${wait_s}s"; sleep "$wait_s"; continue
  fi
  fastfail=$((fastfail+1)); echo "$fastfail" > "$FF_FILE"
  echo "[warn] $(date) iter $i failed (rc=$rc) with no git progress -- fastfail=$fastfail/5" >&2
  if (( fastfail >= 5 )); then trap '' USR1; echo "[stop] $(date) 5 failing iterations, no progress -- NOT resubmitting (check auth/env). BLOCKED" >&2; stop_heartbeat; cancel_nested; exit 1; fi
done

echo "[chain] $(date) iteration cap ($MAX_ITERS) reached, work unfinished -- refreshing process"
trap '' USR1                            # disable the trap so the tail resubmit can't double-fire
stop_heartbeat
resubmit_self || { echo "[cleanup] $(date) tail chain dead -- cancelling nested jobs" >&2; cancel_nested; }
exit 0
