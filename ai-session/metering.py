"""Metering for ai-session: /metrics scrape, SU computation, usage logging.

Single-user-per-session. Two token sources, in priority order:

  1. Per-request ``usage`` (prompt_tokens / completion_tokens) from the vLLM
     OpenAI responses, supplied as a JSONL the client appended to. AUTHORITATIVE
     -- every SU line item comes from this.
  2. vLLM's built-in ``/metrics`` endpoint (plain HTTP GET of Prometheus text,
     NO Prometheus server). Scraped at session start/end; its token delta is the
     billing source when no per-request log exists, and a cross-check when one
     does. If the scrape fails, billing proceeds from per-request usage alone --
     graceful fallback, not a separate mode.

The charge applies su_formula with the policy's w_gpu and the reserved GPU count
N, then the exclusive-node reservation floor.
"""

from __future__ import annotations

import datetime
import json
import os
import pwd
import re
import socket
import sys
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_BILLING = os.path.join(os.path.dirname(_HERE), "billing")
if _BILLING not in sys.path:
    sys.path.insert(0, _BILLING)
import su_formula as su  # noqa: E402


def _real_user() -> str:
    """Username from the REAL uid, never $USER (spoofable). Attribution only."""
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except (KeyError, OSError):
        return os.environ.get("USER") or "unknown"

# Central staff-only accounting ledger. One JSON record per (session, source)
# is written here in addition to the per-user usage summary, so staff can track
# every charge in one place. Overridable via AISESSION_BILLING_DIR, mirroring
# the AISESSION_STATE_DIR pattern in ai_session.py / gateway.py. The default is
# a staff-only dir (mode 2770, group rcc-staff, no world access); the write is
# best-effort and MUST NOT break `end` if the dir is missing or unwritable.
BILLING_DIR_DEFAULT = "/project/rcc/mehta5/ai-session-billing"


def _billing_dir() -> str:
    return os.environ.get("AISESSION_BILLING_DIR") or BILLING_DIR_DEFAULT

METRIC_KEYS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
)

# Tolerance for the per-request-vs-/metrics cross-check (relative).
CROSSCHECK_REL_TOL = 0.02


# --------------------------------------------------------------------------- #
# /metrics scrape
# --------------------------------------------------------------------------- #
def scrape_metrics(host: str, port, *, quiet: bool = False, log_path: str = None) -> dict:
    """GET http://host:port/metrics and sum the counters of interest.

    Returns {} on any failure (unreachable / parse error) -- caller falls back
    to per-request usage.

    While the model is still loading the endpoint isn't bound yet, so failures
    are EXPECTED and would spam the console once per poll. Pass ``quiet=True`` to
    suppress the stderr warning, and ``log_path`` to append the detail to a file
    (e.g. the session's .out) so it's still captured for debugging.
    """
    url = f"http://{host}:{port}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            text = r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        msg = f"[metering] /metrics scrape failed for {url}: {e}"
        if log_path:
            try:
                with open(log_path, "a") as lf:      # one atomic line per poll
                    lf.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
            except OSError:
                pass
        if not quiet:
            print(msg, file=sys.stderr)
        return {}
    totals = {k: 0.0 for k in METRIC_KEYS}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][^\s{]*)(\{[^}]*\})?\s+([0-9eE.+-]+)$", line)
        if not m:
            continue
        name, val = m.group(1), m.group(3)
        if name in totals:
            try:
                totals[name] += float(val)
            except ValueError:
                pass
    return totals


def scrape_version(host: str, port, *, timeout: float = 10.0) -> str:
    """GET http://host:port/version and return the running engine version, or None.

    vLLM serves ``{"version": "<x.y.z>"}`` at ``/version`` -- a root path, NOT under
    ``/v1``, so (like ``/metrics``) it answers without the per-session API key. Used by
    the rate-table version guard in :func:`compute_session_su`.

    Best-effort by design: any failure (unreachable, non-JSON, missing field) returns
    None so the guard fails OPEN -- an unreadable version never forces a session UNRATED
    and never breaks ``end``.
    """
    url = f"http://{host}:{port}/version"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, ConnectionError, OSError, ValueError) as e:
        print(f"[metering] /version scrape failed for {url}: {e}", file=sys.stderr)
        return None
    ver = data.get("version") if isinstance(data, dict) else None
    return str(ver) if ver else None


def metrics_delta(before: dict, after: dict) -> dict:
    """Token delta between two /metrics scrapes; {} if either is missing."""
    if not before or not after:
        return {}
    return {
        "prompt_tokens": after.get("vllm:prompt_tokens_total", 0.0) - before.get("vllm:prompt_tokens_total", 0.0),
        "generation_tokens": after.get("vllm:generation_tokens_total", 0.0) - before.get("vllm:generation_tokens_total", 0.0),
        "success_requests": after.get("vllm:request_success_total", 0.0) - before.get("vllm:request_success_total", 0.0),
    }


# --------------------------------------------------------------------------- #
# Per-request usage JSONL
# --------------------------------------------------------------------------- #
def read_usage_jsonl(path: str, since: float = None, until: float = None) -> list:
    """Read a JSONL of per-request usage (client- or gateway-written).

    Each line is a JSON object; recognized keys (with OpenAI aliases):
      prompt_tokens / input_tokens, completion_tokens / output_tokens,
      success (default True), ts (epoch, for windowing). Lines that are a full
      response with a nested ``usage`` object are also accepted.

    ``since``/``until`` filter by the line's ``ts`` (lines without a ts always
    pass, so hand-written logs are unaffected).
    """
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ts = obj.get("ts")
            if ts is not None:
                if since is not None and ts < since:
                    continue
                if until is not None and ts > until:
                    continue
            usage = obj.get("usage", obj)
            out.append({
                "prompt_tokens": int(usage.get("prompt_tokens", usage.get("input_tokens", 0))),
                "completion_tokens": int(usage.get("completion_tokens", usage.get("output_tokens", 0))),
                "success": bool(obj.get("success", usage.get("success", True))),
            })
    return out


def read_gateway_usage(gateway_dir: str, start_epoch: float, end_epoch: float) -> list:
    """Collect gateway usage lines (usage-YYYYMMDD.jsonl) within [start, end].

    Reads each day file the window touches. Returns [] if the gateway logged
    nothing (or was not used) for this session.
    """
    import glob
    if not start_epoch or not end_epoch:
        return []
    requests = []
    for path in sorted(glob.glob(os.path.join(gateway_dir, "usage-*.jsonl"))):
        # cheap day-level prefilter by filename
        day = os.path.basename(path)[len("usage-"):-len(".jsonl")]
        day_start = _day_epoch(day)
        if day_start is None or day_start > end_epoch or day_start + 86400 < start_epoch:
            continue
        try:
            requests.extend(read_usage_jsonl(path, since=start_epoch, until=end_epoch))
        except OSError:
            continue
    return requests


def _day_epoch(yyyymmdd: str):
    try:
        return time.mktime(time.strptime(yyyymmdd, "%Y%m%d"))
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# SU computation
# --------------------------------------------------------------------------- #
def compute_session_su(
    *,
    model_key: str,
    tier: str,
    served_tp: int,
    n_gpus_billed: int,
    reserved_wall_hours: float,
    policy: "su.BillingPolicy",
    rate_table: dict,
    requests: list = None,
    m_before: dict = None,
    m_after: dict = None,
    running_version: str = None,
) -> dict:
    """Compute the session SU charge.

    ``served_tp`` keys the rate-table lookup (throughput was measured at that TP);
    ``n_gpus_billed`` is the GPU count charged (== reserved gres; may exceed
    served_tp if the node is held whole).

    ``running_version`` is the vLLM version the session is actually served by (from
    :func:`scrape_version`). A rate record's throughput is only valid for the engine
    version it was measured under, so if the running version differs from the matched
    record's ``provenance.vllm_version`` the token term is dropped to floor-only
    (UNRATED) rather than billed against stale throughput. Pass None (or leave a
    record without a version) to skip the guard -- it FAILS OPEN: an unknown running
    version keeps the session rated, and a mismatch only removes the token term, never
    raising.
    """
    w_gpu = policy.weight(tier)
    rec = su.rate_record(rate_table, model_key, tier, served_tp)
    delta = metrics_delta(m_before or {}, m_after or {})

    # ---- rate-record match + version guard ------------------------------ #
    # Only bill the token term when a record exists AND the engine that served this
    # session matches the version the record was measured under. A confirmed version
    # mismatch (both versions known and unequal) drops to floor-only; an unknown
    # running version does NOT (fail-open, so a transient /version scrape miss cannot
    # spuriously unrate a session).
    rate_record_found = rec is not None
    rated_version = None
    version_mismatch = False
    if rate_record_found:
        rated_version = (rec.get("provenance") or {}).get("vllm_version")
        if running_version and rated_version and str(running_version) != str(rated_version):
            version_mismatch = True

    # ---- token totals + source selection -------------------------------- #
    per_req_in = per_req_out = None
    n_billed = 0
    if requests:
        per_req_in = sum(r["prompt_tokens"] for r in requests
                         if r.get("success", True) or policy.bill_failed)
        per_req_out = sum(r["completion_tokens"] for r in requests
                          if r.get("success", True) or policy.bill_failed)
        n_billed = sum(1 for r in requests if r.get("success", True) or policy.bill_failed)

    if requests:
        token_source = "per_request_usage"
        total_in, total_out = per_req_in, per_req_out
    elif delta:
        token_source = "metrics_delta"
        total_in = int(round(delta["prompt_tokens"]))
        total_out = int(round(delta["generation_tokens"]))
        n_billed = int(round(delta.get("success_requests", 0)))
    else:
        token_source = "none"
        total_in = total_out = 0

    # ---- token SU (needs a rate record whose version still matches) ------ #
    # ``rated`` folds in the version guard: a record must exist AND (if the running
    # version is known) match the version it was measured under. A confirmed
    # mismatch drops the token term to floor-only.
    rated = rate_record_found and not version_mismatch
    if rated:
        prefill_tps = float(rec["prefill_tps"])
        decode_tps = float(rec["decode_tps"])
        token_su = su.su_for_request(
            total_in, total_out, w_gpu, n_gpus_billed, prefill_tps, decode_tps
        )
    else:
        prefill_tps = decode_tps = None
        token_su = None

    # ---- floor + final charge ------------------------------------------- #
    floor_su = su.reservation_floor_su(w_gpu, n_gpus_billed, reserved_wall_hours)
    if rated:
        charge = su.su_for_session(
            token_su, w_gpu, n_gpus_billed, reserved_wall_hours,
            floor_enabled=policy.floor_enabled, n_requests=n_billed,
        )
        billed_su = charge.billed_su
        basis = charge.basis
    elif version_mismatch:
        # Rate record exists but the running engine version no longer matches the
        # version it was measured under. Bill the floor (dominant on an exclusive
        # node) and flag the token term unrated until the tier is re-benchmarked, so
        # we never bill against stale throughput numbers.
        billed_su = floor_su if policy.floor_enabled else None
        basis = ("floor (UNRATED: running vLLM {run} != rate_table record {rec} for "
                 "({mk},{tier},TP={tp}); re-benchmark this tier -- see "
                 "ai-session/README.md 'Upgrading vLLM')").format(
                     run=running_version, rec=rated_version, mk=model_key,
                     tier=tier, tp=served_tp)
    else:
        # No rate record: bill the floor (dominant on an exclusive node) and
        # flag that the token term is unrated until the benchmark runs.
        billed_su = floor_su if policy.floor_enabled else None
        basis = ("floor (UNRATED: rate_table has no record for "
                 f"({model_key},{tier},TP={served_tp}); run bench_billing.sbatch)")

    # ---- /metrics cross-check ------------------------------------------- #
    crosscheck = None
    if requests and delta:
        d_in = delta["prompt_tokens"]
        d_out = delta["generation_tokens"]
        in_ok = _within(per_req_in, d_in, CROSSCHECK_REL_TOL)
        out_ok = _within(per_req_out, d_out, CROSSCHECK_REL_TOL)
        crosscheck = {
            "per_request_input_tokens": per_req_in,
            "metrics_input_tokens_delta": d_in,
            "per_request_output_tokens": per_req_out,
            "metrics_output_tokens_delta": d_out,
            "input_within_tol": in_ok,
            "output_within_tol": out_ok,
            "rel_tol": CROSSCHECK_REL_TOL,
            "reconciled": bool(in_ok and out_ok),
            "note": ("metrics delta should be >= per-request sum (engine counts "
                     "warmup/probe requests too); large gaps suggest untracked traffic."),
        }

    return {
        "model_key": model_key,
        "gpu_tier": tier,
        "served_tp": served_tp,
        "n_gpus_billed": n_gpus_billed,
        "w_gpu": w_gpu,
        "reserved_wall_hours": round(reserved_wall_hours, 4),
        "token_source": token_source,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "n_requests_billed": n_billed,
        "rated": rated,
        "rate_record_found": rate_record_found,
        "engine_version": running_version,
        "rated_version": rated_version,
        "version_mismatch": version_mismatch,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "token_su": token_su,
        "floor_su": floor_su,
        "billed_su": billed_su,
        "basis": basis,
        "currency": "SU (1 SU = 1 A100-GPU-hour)",
        "crosscheck": crosscheck,
    }


def _within(a, b, rel_tol) -> bool:
    if a is None or b is None:
        return False
    if a == 0 and b == 0:
        return True
    denom = max(abs(a), abs(b), 1.0)
    return abs(a - b) / denom <= rel_tol


# --------------------------------------------------------------------------- #
# Usage log output
# --------------------------------------------------------------------------- #
def write_usage_log(session: dict, requests: list, summary: dict, out_dir: str) -> dict:
    """Write logs/usage/<user>_<jobid>_<ts>.jsonl + _summary.json.

    Returns {'jsonl': path|None, 'summary': path}.
    """
    os.makedirs(out_dir, exist_ok=True)
    user = session.get("user", _real_user())
    jobid = session.get("jobid", "nojob")
    ts = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    stem = os.path.join(out_dir, f"{user}_{jobid}_{ts}")

    jsonl_path = None
    if requests:
        jsonl_path = stem + ".jsonl"
        with open(jsonl_path, "w") as f:
            for r in requests:
                f.write(json.dumps(r) + "\n")

    summary_path = stem + "_summary.json"
    full = {
        "session": session,
        "billing": summary,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "per_request_log": jsonl_path,
    }
    with open(summary_path, "w") as f:
        json.dump(full, f, indent=2)
    return {"jsonl": jsonl_path, "summary": summary_path}


# --------------------------------------------------------------------------- #
# Central staff-only accounting ledger
# --------------------------------------------------------------------------- #
def _as_float(v):
    """None -> None; anything numeric-coercible -> float; else leave as-is."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def write_central_billing_record(session: dict, summary: dict,
                                  billing_dir: str = None) -> str:
    """Record this session's final charge to the central staff ledger.

    Writes one JSON file ``<user>_<jobid>_end.json`` (mode 0640) into
    ``billing_dir`` (default: AISESSION_BILLING_DIR env, else
    BILLING_DIR_DEFAULT), matching the shared ai-session-billing/1 record
    shape written by both `end` (source="end", with token detail) and the staff
    sacct sweep (source="sweep"). Reuses the numbers `end` already computed in
    ``summary`` -- it does NOT recompute billing.

    FAIL-SAFE: this must never break `end`. A missing/unwritable ledger dir (or
    any other error) is logged to stderr and swallowed; the caller continues to
    print the receipt and write the per-user summary. Returns the record path on
    success, or None if the write was skipped/failed.
    """
    try:
        out_dir = billing_dir or _billing_dir()
        user = session.get("user", _real_user())
        jobid = str(session.get("jobid", "nojob"))
        record = {
            "schema": "ai-session-billing/1",
            "jobid": jobid,
            "user": user,
            "model_key": summary.get("model_key"),
            "gpu_tier": summary.get("gpu_tier"),
            "n_gpus": int(summary["n_gpus_billed"]),
            "w_gpu": _as_float(summary.get("w_gpu")),
            "reserved_wall_hours": _as_float(summary.get("reserved_wall_hours")),
            "floor_su": _as_float(summary.get("floor_su")),
            "token_su": _as_float(summary.get("token_su")),
            "billed_su": _as_float(summary.get("billed_su")),
            "basis": summary.get("basis"),
            "source": "end",
            "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "host": socket.gethostname(),
        }
        if not os.path.isdir(out_dir):
            # If we have to create the ledger dir, lock it to rcc-staff (2770,
            # no world bits) rather than inherit a group-writable umask default.
            os.makedirs(out_dir, exist_ok=True)
            try:
                os.chmod(out_dir, 0o2770)
            except OSError:
                pass
        path = os.path.join(out_dir, f"{user}_{jobid}_end.json")
        # Write then chmod so the 0640 mode holds regardless of the umask.
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        os.chmod(path, 0o640)
        return path
    except Exception as e:  # never let central accounting break `end`
        print(f"[metering] WARNING: central billing write skipped: {e}",
              file=sys.stderr)
        return None
