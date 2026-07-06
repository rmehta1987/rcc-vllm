#!/usr/bin/env python3
"""Print the SU-charge banner for an ai-session billing receipt.

`ai_session end` writes a `<user>_<jobid>_<ts>_summary.json` per session (a
`billing` block + `session` block). This renders that into a compact, human
banner -- it's what `run_browser_demo.sh down` prints last so the charge can't
scroll off. Kept as its own stdlib-only script (no third-party imports) so it's
portable and readable, and usable on its own to re-print any past receipt:

    # render a specific receipt
    python print_su_receipt.py logs/usage/mehta5_50412190_..._summary.json

    # render the NEWEST receipt in a usage dir
    python print_su_receipt.py --usage-dir /path/to/logs/usage

    # render the newest only if it's newer than a baseline (else say "none");
    # this is how `down` reports just *this* run's charge:
    python print_su_receipt.py --usage-dir .../logs/usage --since "$PREV_NEWEST"

Exit status is 0 for both "rendered a charge" and "nothing to bill" -- it's an
informational reporter, not a gate.
"""

import argparse
import glob
import json
import os
import sys

_BAR = "=" * 62


def newest_summary(usage_dir: str):
    """Return the most-recently-modified *_summary.json in usage_dir, or None."""
    files = glob.glob(os.path.join(usage_dir, "*_summary.json"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _same_path(a: str, b: str) -> bool:
    if not a or not b:
        return False
    try:
        return os.path.samefile(a, b)          # robust to ./ vs abs, symlinks
    except OSError:
        return os.path.abspath(a) == os.path.abspath(b)


class ReceiptError(Exception):
    """A user-facing problem with the receipt file (wrong kind / bad / empty JSON)."""


def load_summary(summary_path: str) -> dict:
    """Load + validate a *_summary.json, with friendly errors for the common
    mistakes: pointing at a per-request `.jsonl` usage log (JSON Lines), at a
    non-summary JSON, or at an empty/corrupt file."""
    with open(summary_path) as f:
        text = f.read()
    if not text.strip():
        raise ReceiptError(f"{summary_path}: file is empty.")
    try:
        d = json.loads(text)
    except json.JSONDecodeError as e:
        # The usual slip: passing a per-request / gateway usage log, which is
        # JSON Lines (one object per line) -> json.loads trips on "Extra data".
        body = text.strip()
        is_jsonl = "\n" in body
        try:
            json.loads(body.splitlines()[0])      # first line parses on its own?
        except json.JSONDecodeError:
            is_jsonl = False
        if is_jsonl:
            hint = "  Pass the matching *_summary.json instead."
            if summary_path.endswith(".jsonl"):
                sibling = summary_path[: -len(".jsonl")] + "_summary.json"
                if os.path.exists(sibling):
                    hint = f"  Did you mean its summary?  {sibling}"
            raise ReceiptError(
                f"{summary_path} looks like a per-request usage log (JSON Lines), "
                f"not a billing summary.\n{hint}")
        raise ReceiptError(f"{summary_path}: not valid JSON ({e}).")
    if not isinstance(d, dict) or "billing" not in d:
        raise ReceiptError(
            f"{summary_path}: not an ai-session billing summary (no 'billing' "
            f"block). Pass a *_summary.json.")
    return d


def render_banner(summary_path: str) -> None:
    """Print the SU banner for one summary JSON file."""
    d = load_summary(summary_path)
    b = d.get("billing", {})
    s = d.get("session", {})
    print()
    print(_BAR)
    print("  SU CHARGE -- this session")
    print(f"    BILLED : {b.get('billed_su', 0):.4f} SU      basis={b.get('basis', '?')}")
    print(f"    model  : {b.get('model_key', '?')} / {b.get('gpu_tier', '?')} / "
          f"TP={b.get('served_tp', '?')}   "
          f"(N={b.get('n_gpus_billed', '?')} GPU, w_gpu={b.get('w_gpu', '?')})")
    print(f"    usage  : reserved {b.get('reserved_wall_hours', '?')} h   "
          f"tokens in={b.get('total_input_tokens', 0)} "
          f"out={b.get('total_output_tokens', 0)} "
          f"({b.get('n_requests_billed', 0)} requests)")
    print(f"    job    : {s.get('jobid', '?')}")
    print(f"    receipt: {summary_path}")
    print(_BAR)


def _say_none() -> None:
    print()
    print("  SU CHARGE: none this run (no active session was billed).")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("summary", nargs="?", help="path to a *_summary.json to render")
    p.add_argument("--usage-dir", help="render the NEWEST *_summary.json in this dir")
    p.add_argument("--since", default="",
                   help="with --usage-dir: render only if the newest differs from this "
                        "baseline path; otherwise report 'none this run'")
    args = p.parse_args(argv)

    try:
        # 1) explicit file wins
        if args.summary:
            if not os.path.exists(args.summary):
                raise ReceiptError(f"no such receipt: {args.summary}")
            render_banner(args.summary)
            return 0

        # 2) newest-in-dir, optionally gated by --since
        if args.usage_dir:
            newest = newest_summary(args.usage_dir)
            if newest is None or _same_path(newest, args.since):
                _say_none()
            else:
                render_banner(newest)
            return 0
    except ReceiptError as e:
        print(f"print_su_receipt: {e}", file=sys.stderr)
        return 2

    p.error("give a summary file, or --usage-dir")


if __name__ == "__main__":
    raise SystemExit(main())
