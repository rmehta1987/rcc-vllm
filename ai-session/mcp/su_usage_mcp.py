#!/usr/bin/env python3
"""su_usage_mcp -- a read-only Service-Unit (SU) usage MCP server.

Answers "how many SU have I used?" for the invoking user by reading the billing
artifacts the ai-session stack already writes, over the MCP stdio protocol. It
opens files with the caller's own OS permissions and never bypasses them.

Tools:
  my_usage(since?)     total SU + session count, broken down by model and GPU tier
  my_sessions(since?)  one row per billed session (jobid, model, tier, SU, ...)

Both accept an optional 'since' = YYYY-MM-DD lower bound (inclusive).

This uses the official MCP SDK (the ``mcp`` package, FastMCP) from the dedicated
mcp-env; the protocol framing is the SDK's, only the tool bodies are ours.

Sources, in priority order (a session seen in more than one source is counted
once, keyed by job id):
  1. Per-user receipts:  <state-dir>/logs/usage/<user>_<jobid>_<ts>_summary.json
     Search dirs (first existing wins per location, all locations merged):
       $AISESSION_USAGE_DIR                              (explicit override)
       $AISESSION_STATE_DIR/logs/usage                  (multi-tenant layout)
       /project/rcc/mehta5/ai-session-state/<user>/logs/usage
       <repo>/ai-session/logs/usage                     (single-tenant default)
  2. Central ledger:  /project/rcc/mehta5/ai-session-billing/*_end.json
     (group rcc-staff; a normal user usually cannot read it -- PermissionError is
     caught and the ledger is simply skipped, receipts still answer the query.)

Security: read-only; rows are filtered to the invoking user's own username, so
even a receipt that happens to be group-readable for another user is never
surfaced by these tools; no shell, no external commands.
"""

import getpass
import glob
import json
import os
import pwd
import re
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import Field

SINCE_RE = re.compile(r"\A\d{4}-\d{2}-\d{2}\Z")

BILLING_DIR = os.environ.get("AISESSION_BILLING_DIR") \
    or "/project/rcc/mehta5/ai-session-billing"

# <repo>/ai-session/logs/usage, relative to this file (ai-session/mcp/).
_REPO_USAGE = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, "logs", "usage"))

mcp = FastMCP("su-usage-mcp", log_level="WARNING")


def _me():
    # Derive identity from the real uid, NOT env (getpass.getuser() is env-first
    # via LOGNAME/USER and thus spoofable). File reads still go through real-uid
    # OS permissions, but keying the row filter off the real user makes the
    # app-level filter a meaningful check rather than an env-trusting one.
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except (KeyError, OSError):
        try:
            return getpass.getuser()
        except Exception:
            return os.environ.get("USER") or "unknown"


def _usage_dirs():
    """Candidate per-user usage dirs, de-duplicated, most-specific first."""
    me = _me()
    cands = []
    if os.environ.get("AISESSION_USAGE_DIR"):
        cands.append(os.environ["AISESSION_USAGE_DIR"])
    if os.environ.get("AISESSION_STATE_DIR"):
        cands.append(os.path.join(os.environ["AISESSION_STATE_DIR"],
                                  "logs", "usage"))
    cands.append("/project/rcc/mehta5/ai-session-state/%s/logs/usage" % me)
    cands.append(_REPO_USAGE)
    seen, out = set(), []
    for d in cands:
        d = os.path.normpath(d)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _record_date(rec):
    """Best-effort YYYY-MM-DD for a normalized record (for 'since' filtering)."""
    for key in ("date",):
        v = rec.get(key)
        if isinstance(v, str) and len(v) >= 10:
            return v[:10]
    return ""


def _norm_summary(summary, path):
    """Normalize a per-user *_summary.json into a common session record."""
    b = summary.get("billing") or {}
    s = summary.get("session") or {}
    date = (summary.get("generated_at") or "")[:10]
    if not date:
        m = re.search(r"_(\d{8})T", os.path.basename(path))
        if m:
            g = m.group(1)
            date = "%s-%s-%s" % (g[0:4], g[4:6], g[6:8])
    return {
        "jobid": str(s.get("jobid") or b.get("jobid") or ""),
        "user": s.get("user") or "",
        "model_key": b.get("model_key"),
        "gpu_tier": b.get("gpu_tier"),
        "reserved_wall_hours": b.get("reserved_wall_hours"),
        "billed_su": b.get("billed_su"),
        "basis": b.get("basis"),
        "date": date,
        "source": "receipt",
    }


def _norm_ledger(rec):
    """Normalize a central ai-session-billing/1 record."""
    return {
        "jobid": str(rec.get("jobid") or ""),
        "user": rec.get("user") or "",
        "model_key": rec.get("model_key"),
        "gpu_tier": rec.get("gpu_tier"),
        "reserved_wall_hours": rec.get("reserved_wall_hours"),
        "billed_su": rec.get("billed_su"),
        "basis": rec.get("basis"),
        "date": (rec.get("written_at") or "")[:10],
        "source": "ledger",
    }


def _load_sessions(since):
    """Collect this user's sessions from receipts, then the central ledger.

    Returns (records, notes) where notes describes which sources were readable.
    A job id already seen in a receipt is not re-added from the ledger, so a
    session is counted once.
    """
    me = _me()
    by_job = {}
    notes = []

    receipt_dirs_read = 0
    for d in _usage_dirs():
        if not os.path.isdir(d):
            continue
        try:
            paths = glob.glob(os.path.join(d, "%s_*_summary.json" % me))
        except OSError:
            continue
        receipt_dirs_read += 1
        for p in sorted(paths):
            try:
                with open(p) as f:
                    summary = json.load(f)
            except (OSError, ValueError):
                continue
            rec = _norm_summary(summary, p)
            if rec["user"] and rec["user"] != me:
                continue  # not our row
            if not rec["jobid"]:
                rec["jobid"] = os.path.basename(p)
            if since and _record_date(rec) and _record_date(rec) < since:
                continue
            by_job.setdefault(rec["jobid"], rec)
    notes.append("receipt dirs read: %d" % receipt_dirs_read)

    # Central ledger -- optional, staff-readable only. Check access explicitly:
    # glob() silently returns [] on an unreadable dir instead of raising, so a
    # bare glob would hide a permission denial. os.access tells us up front.
    ledger_note = "central ledger: "
    if not os.path.isdir(BILLING_DIR):
        notes.append(ledger_note + "absent; using receipts only")
        records = sorted(by_job.values(),
                        key=lambda r: (r.get("date") or "", r.get("jobid") or ""))
        return records, notes
    if not os.access(BILLING_DIR, os.R_OK | os.X_OK):
        notes.append(ledger_note +
                     "not readable (staff-only); using receipts only")
        records = sorted(by_job.values(),
                        key=lambda r: (r.get("date") or "", r.get("jobid") or ""))
        return records, notes
    try:
        ledger_paths = glob.glob(os.path.join(BILLING_DIR,
                                              "%s_*_end.json" % me))
        added = 0
        for p in sorted(ledger_paths):
            try:
                with open(p) as f:
                    rec = _norm_ledger(json.load(f))
            except PermissionError:
                continue
            except (OSError, ValueError):
                continue
            if rec["user"] and rec["user"] != me:
                continue
            if since and _record_date(rec) and _record_date(rec) < since:
                continue
            if rec["jobid"] and rec["jobid"] not in by_job:
                by_job[rec["jobid"]] = rec
                added += 1
        ledger_note += "read (%d extra session(s))" % added
    except PermissionError:
        ledger_note += "not readable (staff-only); using receipts only"
    except OSError as exc:
        ledger_note += "unavailable (%s); using receipts only" % exc
    notes.append(ledger_note)

    records = sorted(by_job.values(),
                    key=lambda r: (r.get("date") or "", r.get("jobid") or ""))
    return records, notes


def _check_since(since):
    if since in (None, ""):
        return None
    if not SINCE_RE.match(since):
        raise ToolError("since must be a date YYYY-MM-DD; got %r" % (since,))
    return since


def _su(rec):
    v = rec.get("billed_su")
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


_SINCE_FIELD = Annotated[str, Field(
    description="Inclusive lower-bound date YYYY-MM-DD. Optional; omit for all "
                "time.")]


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #
@mcp.tool(
    description="Total Service Units (SU) the invoking user has been billed, "
                "with a breakdown by model and GPU tier. Reads local billing "
                "receipts and, if readable, the central ledger. Read-only.")
def my_usage(since: _SINCE_FIELD = "") -> str:
    since = _check_since(since)
    records, notes = _load_sessions(since)
    me = _me()
    total = sum(_su(r) for r in records)
    by_model, by_tier = {}, {}
    for r in records:
        by_model[r.get("model_key") or "?"] = \
            by_model.get(r.get("model_key") or "?", 0.0) + _su(r)
        by_tier[r.get("gpu_tier") or "?"] = \
            by_tier.get(r.get("gpu_tier") or "?", 0.0) + _su(r)

    lines = []
    scope = " since %s" % since if since else " (all time)"
    lines.append("SU usage for %s%s" % (me, scope))
    lines.append("  sessions: %d" % len(records))
    lines.append("  total billed: %.4f SU  (1 SU = 1 A100-GPU-hour)" % total)
    if by_model:
        lines.append("  by model:")
        for k in sorted(by_model, key=lambda x: -by_model[x]):
            lines.append("    %-22s %.4f SU" % (k, by_model[k]))
    if by_tier:
        lines.append("  by GPU tier:")
        for k in sorted(by_tier, key=lambda x: -by_tier[x]):
            lines.append("    %-22s %.4f SU" % (k, by_tier[k]))
    lines.append("  sources: %s" % "; ".join(notes))
    if not records:
        lines.append("  (no billed sessions found for you in the readable "
                     "billing artifacts)")
    return "\n".join(lines)


@mcp.tool(
    description="One row per billed session for the invoking user (job id, "
                "model, GPU tier, reserved wall-hours, SU, basis, date). "
                "Read-only.")
def my_sessions(since: _SINCE_FIELD = "") -> str:
    since = _check_since(since)
    records, notes = _load_sessions(since)
    me = _me()
    scope = " since %s" % since if since else " (all time)"
    lines = ["Billed sessions for %s%s (%d):" % (me, scope, len(records))]
    if records:
        lines.append("  %-14s %-20s %-8s %10s %8s %-7s %s" %
                     ("jobid", "model", "tier", "wall_hrs", "SU",
                      "basis", "date"))
        for r in records:
            wh = r.get("reserved_wall_hours")
            wh_s = "%.4f" % wh if isinstance(wh, (int, float)) else "?"
            lines.append("  %-14s %-20s %-8s %10s %8.4f %-7s %s" % (
                r.get("jobid") or "?",
                (r.get("model_key") or "?")[:20],
                (r.get("gpu_tier") or "?")[:8],
                wh_s,
                _su(r),
                (r.get("basis") or "?")[:7],
                r.get("date") or "?",
            ))
    lines.append("  sources: %s" % "; ".join(notes))
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
