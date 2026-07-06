#!/usr/bin/env python3
"""Pre-flight SU estimate, printed BEFORE any GPU is committed.

Given the requested GPU tier, the GPU count N, and the Slurm walltime, report the
per-hour reservation-floor cost (``w_gpu * N``) and the projected maximum over the
walltime (``w_gpu * N * hours``). This is the FLOOR: a session on an exclusively
held node is billed at least this much whether busy or idle, so it is the honest
number to show a user before they commit the hardware. Token-metered work only
raises the bill above the floor, which interactive use effectively never does.

The tier weight ``w_gpu`` comes from ``billing/billing_policy.yaml`` (the single
source of truth). If the requested tier is unknown or excluded, the estimate is
deliberately robust: it says the tier resolves at launch and states the floor
formula instead of an invented number. This helper must NEVER raise into a caller
that is about to launch a session -- ``main()`` swallows every error.

Usage:
    preflight_estimate.py --constraint A100 --n 2 --time 02:00:00
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BILLING = os.path.join(os.path.dirname(_HERE), "billing")
if _BILLING not in sys.path:
    sys.path.insert(0, _BILLING)
import su_formula as su  # noqa: E402


def parse_hours(time_limit: str) -> float:
    """Parse a Slurm time string ``[DD-]HH:MM:SS`` (or ``MM:SS``/``SS``) into hours."""
    s = str(time_limit).strip()
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3], parts[-2], parts[-1]
    return days * 24 + h + m / 60.0 + sec / 3600.0


def estimate_line(tier: str, n_gpus: int, hours: float,
                  policy=None, policy_path: str = None) -> str:
    """Return the human-readable pre-flight estimate line.

    Robust to an unknown/excluded/unweighted tier or an unreadable policy: in that
    case it reports that the tier resolves at launch and states the floor formula
    (``w_gpu*N`` per hour) rather than a number. Never raises.
    """
    norm = su._normalize_tier(tier)
    try:
        if policy is None:
            policy = su.load_policy(policy_path or su.DEFAULT_POLICY_PATH)
        w = policy.weight(tier)
    except Exception:
        # Any failure (unknown/excluded tier, missing policy, no yaml) -> safe
        # fallback. A pre-flight estimate must not block or crash a launch.
        return (f"Estimated cost: tier {norm!r} resolves at launch; "
                f"floor is w_gpu*N per hour (N={n_gpus} GPU over {hours:g} h walltime).")
    per_h = w * n_gpus
    proj = per_h * hours
    return (f"Estimated cost: {per_h:g} SU/h at tier {norm} (N={n_gpus} GPU); "
            f"projected max {proj:g} SU over the {hours:g} h walltime.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Pre-flight SU estimate for an ai-session.")
    p.add_argument("--constraint", required=True, help="requested GPU tier (A100/H200/H100/L40S/A40)")
    p.add_argument("--n", type=int, required=True, help="GPU count N (== TP for a single-node session)")
    p.add_argument("--time", default="02:00:00", help="Slurm walltime string; defaults to the 02:00:00 CLI default")
    p.add_argument("--policy", default=su.DEFAULT_POLICY_PATH)
    args = p.parse_args(argv)
    try:
        hours = parse_hours(args.time)
    except Exception:
        hours = 0.0
    print(estimate_line(args.constraint, args.n, hours, policy_path=args.policy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
