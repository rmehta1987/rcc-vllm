#!/usr/bin/env python3
"""idle_reaper.py -- release the invoking user's idle ai-session GPU reservation.

An ai-session holds its GPUs exclusively and is floor-billed for the whole
reservation whether it is busy or idle (see BILLING_POLICY.md). A user who
forgets to run `end`/`down` keeps paying. This tool watches THIS user's running
ai-session job, reads the backend's request/token counters via the same
`/metrics` scrape the metering code uses, and if those counters have not advanced
for `--idle-min` minutes it warns and then runs the normal `end` path
(final metering + scancel) to free the GPUs and stop the charge.

It is a login-node tool (stdlib only; reuses ai-session/server.py + metering.py).
It only ever looks at the caller's OWN jobs -- `squeue --me` lists nothing else --
and reaps by calling `ai_session.py end --jobid <j>`, so it can never touch
another user's session.

Modes:
  --once        one pass then exit (cron-friendly; the cron cadence is the poll).
  (default)     loop forever, sleeping --poll-sec between passes (tmux / systemd).
  --dry-run     report only; writes nothing and reaps nothing (fully read-only).

Idle is judged across passes using a tiny per-job state file under
``$AISESSION_STATE_DIR/run/idle_reaper/``: each pass compares the live counters
to the last recorded ones; any increase resets the idle clock. The state file
persists the "last active" time so `--once` under cron works across invocations.

Run under the vllm-probe python (the `end` it invokes needs PyYAML):
    /project/rcc/mehta5/conda-envs/vllm-probe/bin/python \
        ai-session/idle_reaper.py --once --idle-min 30
Set AISESSION_STATE_DIR (the wrappers do) so it watches the right per-user state.
"""

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_BILLING = os.path.join(os.path.dirname(_HERE), "billing")
for _p in (_HERE, _BILLING):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import metering  # noqa: E402  (reuse: /metrics scrape)
import server    # noqa: E402  (reuse: MODEL_REGISTRY)

# Per-user writable state, matching ai_session.py / gateway.py.
_STATE = os.environ.get("AISESSION_STATE_DIR") or _HERE
REAPER_DIR = os.path.join(_STATE, "run", "idle_reaper")
AI_SESSION = os.path.join(_HERE, "ai_session.py")

# Counters that signal "the backend did work". Any increase counts as activity;
# tracking all three (prompt/generation/success) catches an in-flight generation
# that has not yet incremented the success counter, so a slow-but-busy session is
# never reaped.
_COUNTER_KEYS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
)


# --------------------------------------------------------------------------- #
# discovery + counters (reuse server.MODEL_REGISTRY + metering.scrape_metrics)
# --------------------------------------------------------------------------- #
def running_sessions() -> list:
    """This user's RUNNING ai-session jobs as dicts {jobid, model_key, node, port}.

    `squeue --me` is scoped to the caller, so other users' jobs are never seen.
    A job is an ai-session job when its name is `<model_key>:<port>` with
    model_key in MODEL_REGISTRY.
    """
    r = subprocess.run(
        ["squeue", "--me", "-h", "-o", "%i|%j|%T|%N"],
        capture_output=True, text=True,
    )
    out = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        jid, jobname, state, node = parts
        if state != "RUNNING" or not node or "[" in node:   # skip pending / multi-node
            continue
        model_key, _, port = jobname.partition(":")
        if model_key not in server.MODEL_REGISTRY:
            continue
        out.append({"jobid": jid, "model_key": model_key,
                    "node": node, "port": port or "8000"})
    return out


def read_counters(node: str, port: str):
    """Backend activity counters as a list [prompt, generation, success], or None.

    Uses the same scrape metering does; returns None if /metrics is unreachable
    (model still loading, or the node has gone away) -- the caller then declines
    to reap, because it cannot prove the session is idle.
    """
    m = metering.scrape_metrics(node, port, quiet=True)
    if not m:
        return None
    return [float(m.get(k, 0.0)) for k in _COUNTER_KEYS]


# --------------------------------------------------------------------------- #
# per-job idle-tracking state (survives across --once invocations)
# --------------------------------------------------------------------------- #
def _state_path(jobid: str) -> str:
    user = os.environ.get("USER", "unknown")
    return os.path.join(REAPER_DIR, f"{user}_{jobid}.json")


def load_state(jobid: str):
    try:
        with open(_state_path(jobid)) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_state(jobid: str, counters: list, last_active: float,
               first_seen: float, now: float) -> None:
    os.makedirs(REAPER_DIR, exist_ok=True)
    tmp = _state_path(jobid) + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"jobid": jobid, "counters": counters,
                   "last_active": last_active, "first_seen": first_seen,
                   "last_checked": now}, f, indent=2)
    os.replace(tmp, _state_path(jobid))


def clear_state(jobid: str) -> None:
    try:
        os.remove(_state_path(jobid))
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# notification + reap
# --------------------------------------------------------------------------- #
def _warn(msg: str) -> None:
    print(f"[idle-reaper] {msg}", file=sys.stderr)


def send_email(addr: str, subject: str, body: str) -> None:
    """Best-effort email via `mail` then `sendmail`; never raises."""
    if not addr:
        return
    try:
        subprocess.run(["mail", "-s", subject, addr],
                       input=body, text=True, timeout=15, check=False)
        return
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        pass
    try:
        msg = f"To: {addr}\nSubject: {subject}\n\n{body}\n"
        subprocess.run(["/usr/sbin/sendmail", "-t"],
                       input=msg, text=True, timeout=15, check=False)
        return
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        _warn(f"email to {addr} not sent (no mail/sendmail available)")


def reap(sess: dict, idle_min: float, email: str) -> None:
    """Warn, then run the existing `end` path (metering + scancel) for this job."""
    jid = sess["jobid"]
    subject = f"[ai-session] reaping idle job {jid} ({sess['model_key']})"
    body = (f"ai-session job {jid} ({sess['model_key']} on {sess['node']}) had no "
            f"backend activity for >= {idle_min:g} min and is being ended to release "
            f"its GPUs and stop the reservation charge.\n")
    _warn(subject.replace("[ai-session] ", ""))
    send_email(email, subject, body)
    # The existing end path: final /metrics scrape, SU charge, ledger, scancel.
    # sys.executable so `end` runs under the same (vllm-probe) python; env is
    # inherited so AISESSION_STATE_DIR resolves to this user's state.
    proc = subprocess.run([sys.executable, AI_SESSION, "end", "--jobid", jid],
                          env=os.environ.copy())
    if proc.returncode == 0:
        clear_state(jid)
        _warn(f"job {jid} ended (GPUs released).")
    else:
        _warn(f"`end` for job {jid} returned rc={proc.returncode}; state kept for retry.")


# --------------------------------------------------------------------------- #
# one pass
# --------------------------------------------------------------------------- #
def one_pass(args) -> int:
    """Evaluate every session once. Returns the number of sessions reaped."""
    sessions = running_sessions()
    now = time.time()
    idle_secs_limit = args.idle_min * 60.0
    if not sessions:
        print(f"[idle-reaper] no running ai-session jobs for "
              f"{os.environ.get('USER', 'this user')}.")
        return 0

    reaped = 0
    for s in sessions:
        jid = s["jobid"]
        cur = read_counters(s["node"], s["port"])
        st = load_state(jid)

        if cur is None:
            _warn(f"job {jid}: /metrics unreachable ({s['node']}:{s['port']}); "
                  "cannot confirm idle -- not reaping this pass.")
            continue

        if st is None:
            # First observation: establish a baseline; can't be idle yet.
            print(f"[idle-reaper] job {jid} ({s['model_key']}): baseline recorded "
                  f"(counters={_fmt(cur)}).")
            if not args.dry_run:
                save_state(jid, cur, last_active=now, first_seen=now, now=now)
            continue

        if cur != st.get("counters"):
            # Activity since last pass -> reset the idle clock.
            print(f"[idle-reaper] job {jid} ({s['model_key']}): active "
                  f"(counters advanced).")
            if not args.dry_run:
                save_state(jid, cur, last_active=now,
                           first_seen=st.get("first_seen", now), now=now)
            continue

        idle_secs = now - st.get("last_active", now)
        idle_min_now = idle_secs / 60.0
        if idle_secs >= idle_secs_limit:
            if args.dry_run:
                print(f"[idle-reaper] job {jid} ({s['model_key']}): idle "
                      f"{idle_min_now:.1f} min >= {args.idle_min:g} -> WOULD reap "
                      f"(dry-run, nothing done).")
            else:
                reap(s, args.idle_min, args.email)
                reaped += 1
        else:
            remaining = (idle_secs_limit - idle_secs) / 60.0
            print(f"[idle-reaper] job {jid} ({s['model_key']}): idle "
                  f"{idle_min_now:.1f} min (reap in ~{remaining:.1f} min).")
            if not args.dry_run:
                save_state(jid, cur, last_active=st.get("last_active", now),
                           first_seen=st.get("first_seen", now), now=now)
    return reaped


def _fmt(counters: list) -> str:
    return "prompt=%d gen=%d success=%d" % (counters[0], counters[1], counters[2])


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--idle-min", type=float, default=30.0,
                   help="reap after this many minutes with no backend activity (default 30)")
    p.add_argument("--poll-sec", type=int, default=300,
                   help="seconds between passes in loop mode (default 300)")
    p.add_argument("--once", action="store_true",
                   help="run one pass and exit (cron-friendly)")
    p.add_argument("--dry-run", action="store_true",
                   help="report only; write nothing and reap nothing (read-only)")
    p.add_argument("--email", default=os.environ.get("AISESSION_REAPER_EMAIL"),
                   help="optional address to notify before reaping (default $AISESSION_REAPER_EMAIL)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run:
        print("[idle-reaper] DRY-RUN: read-only; nothing will be ended or written.")
    if args.once:
        one_pass(args)
        return 0
    print(f"[idle-reaper] loop mode: idle-min={args.idle_min:g} poll-sec={args.poll_sec} "
          f"(Ctrl-C to stop).")
    try:
        while True:
            one_pass(args)
            time.sleep(args.poll_sec)
    except KeyboardInterrupt:
        print("\n[idle-reaper] stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
