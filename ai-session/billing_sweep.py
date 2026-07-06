#!/usr/bin/env python3
"""Staff sacct sweep: reconstruct each ai-session's authoritative FLOOR charge.

This is an RCC-staff tool run on a login node (cron-friendly). It reads Slurm
accounting (`sacct`), which users cannot edit, and writes one authoritative
reservation-floor record per completed ai-session job to the central billing
ledger. It runs independently of whether the user ever ran ``ai_session end``,
so the floor is captured even for sessions the user forgot to close.

Why the floor, not tokens: an ai-session holds its GPUs exclusively, so the
charge is ``max(token_su, w_gpu * N * reserved_wall_hours)`` and the floor term
dominates interactive use. The per-session ``end`` record (source="end") carries
the token detail; this sweep record (source="sweep") is the authoritative floor
reconstructed from accounting. Both may coexist for one job and are kept.

Run it under the vllm-probe python (it reuses the real billing formula, which
needs PyYAML)::

    /project/rcc/mehta5/conda-envs/vllm-probe/bin/python \
        ai-session/billing_sweep.py --since 2026-06-01

Single source of truth is reused, not reimplemented:
  * ``billing/su_formula.py`` -- w_gpu weight lookup and the floor computation.
  * ``ai-session/server.py``  -- MODEL_REGISTRY (what counts as an ai-session
                                 job) and resolve_node_tier (node -> GPU tier).
The reserved GPU count N is read from the job's AllocTRES (``gres/gpu=<n>``),
which is authoritative in accounting and survives after the controller has
purged the live job (so ``scontrol show job`` no longer works for it).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import socket
import subprocess
import sys

# --------------------------------------------------------------------------- #
# Reuse the single sources of truth (billing formula + model registry/tier).
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_BILLING = os.path.join(os.path.dirname(_HERE), "billing")
for _p in (_BILLING, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import su_formula as su  # noqa: E402  (billing/su_formula.py -- floor + weights)
import server  # noqa: E402          (ai-session/server.py -- MODEL_REGISTRY, tier)

SCHEMA = "ai-session-billing/1"
DEFAULT_BILLING_DIR = "/project/rcc/mehta5/ai-session-billing"

# Slurm states after which a job is done and its floor is final.
TERMINAL_STATES = {"COMPLETED", "CANCELLED", "TIMEOUT", "FAILED", "NODE_FAIL"}

# The columns requested from sacct, in order. --parsable2 emits '|'-separated
# fields with no padding; the %NN widths are ignored by parsable output but kept
# for documentation. AllocTRES is last so an internal comma never splits a field.
_SACCT_FORMAT = (
    "JobID,JobIDRaw,JobName%40,User,State,Elapsed,ElapsedRaw,NodeList,AllocTRES%80"
)
_N_FIELDS = 9

_GRES_GPU_RE = re.compile(r"gres/gpu=(\d+)")


# --------------------------------------------------------------------------- #
# sacct enumeration
# --------------------------------------------------------------------------- #
def run_sacct(since: str, user: str | None) -> str:
    """Return raw ``sacct --parsable2`` output for jobs since ``since``.

    ``-a`` requests all users; ``--user`` narrows to one when given.
    """
    cmd = [
        "sacct", "-a", "-S", since,
        "-o", _SACCT_FORMAT,
        "--parsable2",
    ]
    if user:
        cmd += ["--user", user]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"sacct failed (rc={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout


def parse_sacct_rows(raw: str):
    """Yield dicts for the main allocation rows (skip .batch/.extern steps).

    Rows with the wrong field count are skipped by the caller as malformed.
    """
    lines = raw.splitlines()
    if not lines:
        return
    header = lines[0].split("|")
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split("|")
        # Step rows (JobID contains a dot: '<id>.batch', '<id>.extern') are the
        # accounting steps, not the reservation; skip them.
        jobid = fields[0] if fields else ""
        if "." in jobid:
            continue
        yield {"__nfields__": len(fields), "__raw__": line,
               **dict(zip(header, fields))}


# --------------------------------------------------------------------------- #
# Per-job resolution
# --------------------------------------------------------------------------- #
def _first_node(nodelist: str) -> str | None:
    """First concrete node name from a NodeList (handles a bracketed range).

    'midway3-0605' -> 'midway3-0605';
    'midway3-[0602,0605]' -> 'midway3-0602';
    'None assigned' / '' -> None.
    """
    if not nodelist:
        return None
    nl = nodelist.strip()
    if not nl or nl.lower().startswith("none"):
        return None
    if "[" in nl:
        prefix, rest = nl.split("[", 1)
        first = re.split(r"[,\-\]]", rest, maxsplit=1)[0]
        return f"{prefix}{first}"
    return nl


def model_key_of(job_name: str) -> str | None:
    """model_key if ``job_name`` is an ai-session name '<model_key>:<port>'.

    Returns None for anything else (e.g. 'bench_billing', 'batch'), which is how
    benchmark and unrelated jobs are excluded from billing.
    """
    if not job_name or ":" not in job_name:
        return None
    key = job_name.split(":", 1)[0].strip()
    return key if key in server.MODEL_REGISTRY else None


def gpus_from_alloctres(alloctres: str) -> int | None:
    """Reserved GPU count N from AllocTRES ('...,gres/gpu=2,...'); None if absent."""
    if not alloctres:
        return None
    m = _GRES_GPU_RE.search(alloctres)
    return int(m.group(1)) if m else None


class Unresolved(Exception):
    """A job that is an ai-session job but whose tier/N/weight cannot be resolved."""


def resolve_job(row: dict, policy: "su.BillingPolicy") -> dict:
    """Resolve one main sacct row into a floor record (raises Unresolved/ValueError).

    Returns a dict with the resolved billing fields; the caller assembles the
    ledger record and decides write vs. skip.
    """
    job_name = (row.get("JobName") or "").strip()
    model_key = model_key_of(job_name)
    if model_key is None:
        raise ValueError("not an ai-session job")  # not billable here; caller filters

    state = (row.get("State") or "").strip()
    state_word = state.split()[0] if state.split() else ""
    if state_word not in TERMINAL_STATES:
        raise ValueError(f"non-terminal or empty state {state!r}")

    jobid = (row.get("JobID") or "").strip()
    user = (row.get("User") or "").strip() or "unknown"

    # reserved wall hours from ElapsedRaw (seconds); authoritative in accounting.
    raw_elapsed = (row.get("ElapsedRaw") or "").strip()
    try:
        elapsed_s = int(raw_elapsed)
    except ValueError as e:
        raise ValueError(f"unparseable ElapsedRaw {raw_elapsed!r}") from e
    reserved_wall_hours = elapsed_s / 3600.0

    # N from AllocTRES (survives controller purge, unlike scontrol show job).
    n_gpus = gpus_from_alloctres(row.get("AllocTRES") or "")
    if not n_gpus:
        raise Unresolved("no gres/gpu in AllocTRES")

    # tier from the node's hardware features (node persists after the job ends).
    node = _first_node(row.get("NodeList") or "")
    tier = server.resolve_node_tier(node) if node else None
    if not tier:
        raise Unresolved(f"could not resolve GPU tier from node {node!r}")

    # weight + floor via the single source of truth (billing/su_formula.py).
    try:
        w_gpu = policy.weight(tier)
    except KeyError as e:
        raise Unresolved(f"no w_gpu weight for tier {tier!r}: {e}") from e
    floor_su = su.reservation_floor_su(w_gpu, n_gpus, reserved_wall_hours)

    return {
        "jobid": jobid,
        "user": user,
        "model_key": model_key,
        "gpu_tier": tier,
        "n_gpus": int(n_gpus),
        "w_gpu": float(w_gpu),
        "reserved_wall_hours": reserved_wall_hours,
        "floor_su": floor_su,
        "state": state,
    }


def build_record(resolved: dict) -> dict:
    """Assemble the central-ledger record (source='sweep') from resolved fields."""
    return {
        "schema": SCHEMA,
        "jobid": resolved["jobid"],
        "user": resolved["user"],
        "model_key": resolved["model_key"],
        "gpu_tier": resolved["gpu_tier"],
        "n_gpus": resolved["n_gpus"],
        "w_gpu": resolved["w_gpu"],
        "reserved_wall_hours": resolved["reserved_wall_hours"],
        "floor_su": resolved["floor_su"],
        "token_su": None,          # sweep has no token detail (see the 'end' record)
        "billed_su": resolved["floor_su"],
        "basis": "floor",
        "source": "sweep",
        "written_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": socket.gethostname(),
    }


# --------------------------------------------------------------------------- #
# Ledger write (idempotent, mode 0640)
# --------------------------------------------------------------------------- #
def ledger_path(billing_dir: str, user: str, jobid: str) -> str:
    return os.path.join(billing_dir, f"{user}_{jobid}_sweep.json")


def write_record(path: str, record: dict) -> None:
    """Write ``record`` as JSON at 0640, atomically (temp + os.replace).

    The file's group is inherited from the setgid billing directory; we set only
    the file mode. (The setgid-bit gotcha applies to directories, not files.)
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o640)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(record, f, indent=2, sort_keys=False)
            f.write("\n")
        os.chmod(tmp, 0o640)   # defeat any umask that narrowed the create mode
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _default_since() -> str:
    d = _dt.datetime.now() - _dt.timedelta(days=7)
    return d.strftime("%Y-%m-%dT%H:%M:%S")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="billing_sweep.py",
        description="Staff sacct sweep: write authoritative floor charges to the "
                    "central ai-session billing ledger.",
    )
    p.add_argument("--since", default=_default_since(),
                   help="sacct start time (ISO or sacct time). Default: last 7 days.")
    p.add_argument("--user", default=None,
                   help="Restrict to one user. Default: all users.")
    p.add_argument("--billing-dir",
                   default=os.environ.get("AISESSION_BILLING_DIR", DEFAULT_BILLING_DIR),
                   help="Central ledger directory. Default: $AISESSION_BILLING_DIR "
                        f"or {DEFAULT_BILLING_DIR}.")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute and print; write nothing.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing <user>_<jobid>_sweep.json "
                        "(default: skip existing -- the sweep is idempotent).")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    policy = su.load_policy(su.DEFAULT_POLICY_PATH)

    try:
        raw = run_sacct(args.since, args.user)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not args.dry_run:
        created = not os.path.isdir(args.billing_dir)
        os.makedirs(args.billing_dir, exist_ok=True)
        if created:
            # Lock a freshly created ledger dir to rcc-staff (2770, no world bits).
            try:
                os.chmod(args.billing_dir, 0o2770)
            except OSError:
                pass

    n_written = n_skipped = n_unresolved = n_malformed = 0
    total_floor = 0.0
    print(f"# ai-session billing sweep  since={args.since}  "
          f"user={args.user or 'ALL'}  billing_dir={args.billing_dir}"
          f"{'  (dry-run)' if args.dry_run else ''}")
    print(f"# {'jobid':<12} {'user':<10} {'model_key':<20} {'tier':<6} "
          f"{'N':>2} {'hours':>8} {'floor_su':>10}  action")

    for row in parse_sacct_rows(raw):
        if row.get("__nfields__") != _N_FIELDS:
            n_malformed += 1
            print(f"# WARNING malformed sacct row (got {row.get('__nfields__')} "
                  f"fields, want {_N_FIELDS}): {row.get('__raw__')!r}", file=sys.stderr)
            continue

        # Fast filter: only ai-session-named, terminal-state jobs are billable.
        if model_key_of((row.get("JobName") or "").strip()) is None:
            continue
        state0 = (row.get("State") or "").strip().split()
        if not state0 or state0[0] not in TERMINAL_STATES:
            continue

        jobid = (row.get("JobID") or "").strip()
        user = (row.get("User") or "").strip() or "unknown"
        try:
            resolved = resolve_job(row, policy)
        except Unresolved as e:
            n_unresolved += 1
            print(f"  {jobid:<12} {user:<10} "
                  f"{(row.get('JobName') or '')[:20]:<20} {'?':<6} {'?':>2} "
                  f"{'?':>8} {'?':>10}  UNRESOLVED ({e})")
            continue
        except ValueError:
            # not an ai-session job / non-terminal after all -- already filtered,
            # but be defensive and skip quietly.
            continue
        except Exception as e:  # never let one bad job abort the whole sweep
            n_unresolved += 1
            print(f"  {jobid:<12} {user:<10} unexpected error: {e}  -> UNRESOLVED",
                  file=sys.stderr)
            continue

        record = build_record(resolved)
        path = ledger_path(args.billing_dir, resolved["user"], resolved["jobid"])
        exists = os.path.exists(path)

        if args.dry_run:
            action = ("would-overwrite" if exists and args.force
                      else "exists-skip" if exists else "would-write")
        elif exists and not args.force:
            action = "skipped"
            n_skipped += 1
        else:
            try:
                write_record(path, record)
            except OSError as e:
                n_unresolved += 1
                print(f"  {jobid:<12} {user:<10} write failed: {e}  -> UNRESOLVED",
                      file=sys.stderr)
                continue
            action = "overwritten" if exists else "written"
            n_written += 1

        total_floor += resolved["floor_su"]
        print(f"  {resolved['jobid']:<12} {resolved['user']:<10} "
              f"{resolved['model_key']:<20} {resolved['gpu_tier']:<6} "
              f"{resolved['n_gpus']:>2} {resolved['reserved_wall_hours']:>8.4f} "
              f"{resolved['floor_su']:>10.4f}  {action}")

    print(f"# totals: written={n_written} skipped={n_skipped} "
          f"unresolved={n_unresolved} malformed={n_malformed}  "
          f"sum_floor_su={total_floor:.4f}"
          f"{'  (dry-run: nothing written)' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
