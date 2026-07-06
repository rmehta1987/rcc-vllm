#!/usr/bin/env python3
"""Billing-grade throughput benchmark -> billing/rate_table.json.

Replaces the old bench-*.sbatch suite, which was unusable for billing (old SIF,
single fixed prompt, never set --tensor-parallel-size, inconsistent concurrency
across GPUs, 0.5B model = overhead-bound).

This script, run on a GPU compute node in the vllm-probe env:

  1. Launches ``vllm serve`` in the SAME config the production endpoint serves
     (so the rate table matches reality -- otherwise users are mis-charged).
  2. Detects the GPU tier from nvidia-smi.
  3. Runs ``vllm bench serve`` across three profiles with IDENTICAL sweep params
     on every tier:
       * prefill-heavy  -> aggregate prefill_tps (= total_input_tokens / duration)
       * decode-heavy   -> aggregate decode_tps  (= output_throughput)
       * balanced       -> ttft / tpot percentiles + alpha sanity check
  4. Scrapes the built-in /metrics endpoint before/after as a best-effort
     cross-check (no Prometheus server -- plain HTTP GET of Prometheus text).
  5. Computes alpha_empirical, illustrative su_per_1k_{in,out}, and merges one
     record per (model_key, gpu_tier, TP) into billing/rate_table.json.

The prefill/decode split is MEASURED here, not hand-set: decode is
memory-bandwidth-bound so decode_tps << prefill_tps, and that asymmetry is what
makes output tokens cost more GPU-time in su_formula.

Usage (typically from bench_billing.sbatch):
    python bench_billing.py --model-key qwen2.5_72B \
        --model-path /project/rcc/mehta5/vllm/models/Qwen2.5-72B-Instruct \
        --tp 4 [--tier a100] [--enforce-eager]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

# Import the pure formula module from ../billing.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BILLING = os.path.join(os.path.dirname(_HERE), "billing")
sys.path.insert(0, _BILLING)
import su_formula as su  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixed sweep -- IDENTICAL on every tier so rates are comparable. Overriding any
# of these breaks cross-tier comparability; the script warns if you do.
# --------------------------------------------------------------------------- #
@dataclass
class Profile:
    name: str
    input_len: int
    output_len: int
    num_prompts: int
    max_concurrency: int


REFERENCE_PROFILES = [
    # prefill-heavy: big prompt, tiny output -> wall-time dominated by prefill.
    Profile("prefill_heavy", input_len=2048, output_len=8, num_prompts=256, max_concurrency=64),
    # decode-heavy: tiny prompt, long output -> output_throughput == decode_tps.
    Profile("decode_heavy", input_len=64, output_len=1024, num_prompts=128, max_concurrency=64),
    # balanced: realistic mix for ttft/tpot percentiles + alpha sanity check.
    Profile("balanced", input_len=1024, output_len=512, num_prompts=256, max_concurrency=64),
]

REFERENCE_SEED = 1234

# /metrics counters reconciled at session start/end (all present in vllm 0.10.2).
METRIC_KEYS = [
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
]


# --------------------------------------------------------------------------- #
# GPU tier detection
# --------------------------------------------------------------------------- #
def detect_gpu_tier() -> tuple:
    """Return (normalized_tier, raw_name) from nvidia-smi."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    raw = out[0].strip() if out else "unknown"
    return su._normalize_tier(raw), raw


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #
def free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def build_serve_flags(args, port: int) -> list:
    """Canonical PRODUCTION serve flags. MUST stay in sync with
    ai-session/launch_ai_session.sh -- the rate table is only valid for the
    exact config it was measured under.
    """
    flags = [
        "vllm", "serve", args.model_path,
        "--served-model-name", args.model_key,
        "--port", str(port),
        "--tensor-parallel-size", str(args.tp),
        "--enable-prefix-caching",
        "--trust-remote-code",
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        # stats MUST stay enabled so /metrics is populated -- do NOT add
        # --disable-log-stats here.
    ]
    if args.enforce_eager:
        # Keep eager on BOTH benchmark and production if a tier shows
        # torch.compile instability; the rate-table provenance records it.
        flags.append("--enforce-eager")
    flags += list(args.extra_serve_args)
    return flags


def launch_server(args, port: int, log_path: str) -> subprocess.Popen:
    print(f"[bench] launching vLLM server on port {port} (log: {log_path})", flush=True)
    flags = build_serve_flags(args, port)
    print("[bench] serve flags: " + " ".join(flags), flush=True)
    logf = open(log_path, "w")
    proc = subprocess.Popen(flags, stdout=logf, stderr=subprocess.STDOUT)
    return proc


def wait_ready(port: int, proc: subprocess.Popen, timeout_s: int) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited early with code {proc.returncode}; check the server log."
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print("[bench] server ready.", flush=True)
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(5)
    raise TimeoutError(f"server not ready within {timeout_s}s")


# --------------------------------------------------------------------------- #
# /metrics scrape (best-effort cross-check)
# --------------------------------------------------------------------------- #
def scrape_metrics(port: int) -> dict:
    """Parse the Prometheus-text /metrics endpoint; sum counters across labels."""
    url = f"http://127.0.0.1:{port}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001 -- graceful: cross-check is optional
        print(f"[bench] /metrics scrape failed ({e}); continuing without it.", flush=True)
        return {}
    totals = {k: 0.0 for k in METRIC_KEYS}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"^([a-zA-Z_:][^\s{]*)(\{[^}]*\})?\s+([0-9eE.+-]+)$", line)
        if not m:
            continue
        name, _labels, val = m.group(1), m.group(2), m.group(3)
        if name in totals:
            try:
                totals[name] += float(val)
            except ValueError:
                pass
    return totals


# --------------------------------------------------------------------------- #
# Benchmark runs
# --------------------------------------------------------------------------- #
def run_bench(args, port: int, profile: Profile, results_dir: str) -> dict:
    out_file = os.path.join(results_dir, f"{args.model_key}_{args.tier}_tp{args.tp}_{profile.name}.json")
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--base-url", f"http://127.0.0.1:{port}",
        "--endpoint", "/v1/completions",
        "--model", args.model_key,            # == --served-model-name
        "--tokenizer", args.model_path,
        "--trust-remote-code",
        "--dataset-name", "random",
        "--random-input-len", str(profile.input_len),
        "--random-output-len", str(profile.output_len),
        "--num-prompts", str(profile.num_prompts),
        "--max-concurrency", str(profile.max_concurrency),
        "--seed", str(REFERENCE_SEED),
        "--ignore-eos",                        # honor output-len exactly
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,95,99",
        "--save-result", "--result-filename", out_file,
    ]
    print(f"\n[bench] === profile {profile.name} ===\n[bench] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    with open(out_file) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Record assembly + merge
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    """ISO-8601 timestamp. (Plain os/time -- no Date.now restriction here.)"""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def build_record(args, results: dict, m_before: dict, m_after: dict) -> dict:
    pre = results["prefill_heavy"]
    dec = results["decode_heavy"]
    bal = results["balanced"]

    # prefill_tps: aggregate prompt tokens processed per wall-second.
    prefill_tps = pre["total_input_tokens"] / pre["duration"]
    # decode_tps: aggregate output tokens generated per wall-second.
    decode_tps = dec["output_throughput"]

    alpha = su.alpha_empirical(prefill_tps, decode_tps)

    # Illustrative su_per_1k at N == TP (metering recomputes with the actual N).
    try:
        pol = su.load_policy(args.policy)
        w_gpu = pol.weight(args.tier)
        spk_in = su.su_per_1k_tokens(w_gpu, args.tp, prefill_tps)
        spk_out = su.su_per_1k_tokens(w_gpu, args.tp, decode_tps)
    except Exception as e:  # noqa: BLE001
        print(f"[bench] could not compute su_per_1k ({e}); leaving null.", flush=True)
        w_gpu, spk_in, spk_out = None, None, None

    # /metrics cross-check: engine token counters vs benchmark-reported totals.
    crosscheck = None
    if m_before and m_after:
        d_prompt = m_after.get("vllm:prompt_tokens_total", 0) - m_before.get("vllm:prompt_tokens_total", 0)
        d_gen = m_after.get("vllm:generation_tokens_total", 0) - m_before.get("vllm:generation_tokens_total", 0)
        bench_in = pre["total_input_tokens"] + dec["total_input_tokens"] + bal["total_input_tokens"]
        bench_out = pre["total_output_tokens"] + dec["total_output_tokens"] + bal["total_output_tokens"]
        crosscheck = {
            "metrics_prompt_tokens_delta": d_prompt,
            "metrics_generation_tokens_delta": d_gen,
            "bench_input_tokens_sum": bench_in,
            "bench_output_tokens_sum": bench_out,
            "note": "deltas >= bench sums (warmup/probe requests add a little)",
        }

    return {
        "model_key": args.model_key,
        "gpu_tier": args.tier,
        "tp": args.tp,
        "prefill_tps": round(prefill_tps, 2),
        "decode_tps": round(decode_tps, 2),
        "alpha_empirical": round(alpha, 4),
        "ttft_p50_ms": bal.get("p50_ttft_ms") or bal.get("median_ttft_ms"),
        "ttft_p95_ms": bal.get("p95_ttft_ms"),
        "tpot_p50_ms": bal.get("p50_tpot_ms") or bal.get("median_tpot_ms"),
        "tpot_p95_ms": bal.get("p95_tpot_ms"),
        "su_per_1k_in": spk_in,
        "su_per_1k_out": spk_out,
        "provenance": {
            "vllm_version": _vllm_version(),
            "dtype": "bfloat16",
            "w_gpu_at_measure": w_gpu,
            "serve_flags": build_serve_flags(args, 0)[3:],  # drop 'vllm serve <path>'
            "enforce_eager": args.enforce_eager,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "gpu_name": args.gpu_name,
            "node": socket.gethostname(),
            "profiles": {p.name: vars(p) for p in REFERENCE_PROFILES},
            "seed": REFERENCE_SEED,
            "metrics_crosscheck": crosscheck,
            "timestamp": now_iso(),
        },
    }


def _vllm_version() -> str:
    try:
        import vllm
        return vllm.__version__
    except Exception:  # noqa: BLE001
        return "unknown"


def merge_record(rate_table_path: str, record: dict) -> None:
    """Insert/replace the record keyed by (model_key, gpu_tier, tp)."""
    with open(rate_table_path) as f:
        rt = json.load(f)
    records = [
        r for r in rt.get("records", [])
        if not (r.get("model_key") == record["model_key"]
                and su._normalize_tier(r.get("gpu_tier", "")) == su._normalize_tier(record["gpu_tier"])
                and int(r.get("tp", -1)) == int(record["tp"]))
    ]
    records.append(record)
    records.sort(key=lambda r: (r["model_key"], r["gpu_tier"], r["tp"]))
    rt["records"] = records
    if rt.get("records"):
        rt["status"] = f"populated -- {len(records)} record(s); last updated {now_iso()}"
    tmp = rate_table_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rt, f, indent=2)
    os.replace(tmp, rate_table_path)
    print(f"[bench] merged record into {rate_table_path}", flush=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-key", required=True, help="registry key, e.g. qwen2.5_72B")
    p.add_argument("--model-path", required=True, help="local model dir")
    p.add_argument("--tp", type=int, required=True, help="tensor-parallel size")
    p.add_argument("--tier", default=None, help="GPU tier (auto-detected from nvidia-smi if omitted)")
    p.add_argument("--enforce-eager", action="store_true",
                   help="disable torch.compile (use ONLY if a tier shows compile instability; "
                        "then keep it on in production too)")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--extra-serve-args", nargs=argparse.REMAINDER, default=[],
                   help="extra flags passed verbatim to 'vllm serve' (after all other args)")
    p.add_argument("--ready-timeout", type=int, default=1800, help="seconds to wait for model load")
    p.add_argument("--results-dir", default=os.path.join(_HERE, "results"))
    p.add_argument("--rate-table", default=su.DEFAULT_RATE_TABLE_PATH)
    p.add_argument("--policy", default=su.DEFAULT_POLICY_PATH)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    tier, raw = detect_gpu_tier()
    args.gpu_name = raw
    if args.tier is None:
        args.tier = tier
        print(f"[bench] detected GPU tier: {args.tier} ({raw})", flush=True)
    elif su._normalize_tier(args.tier) != tier:
        print(f"[bench] WARNING: --tier {args.tier} but nvidia-smi reports {tier} ({raw})", flush=True)
        args.tier = su._normalize_tier(args.tier)
    else:
        args.tier = tier

    port = free_port()
    server_log = os.path.join(args.results_dir, f"serve_{args.model_key}_{args.tier}_tp{args.tp}.log")
    proc = launch_server(args, port, server_log)
    try:
        wait_ready(port, proc, args.ready_timeout)
        m_before = scrape_metrics(port)
        results = {}
        for profile in REFERENCE_PROFILES:
            results[profile.name] = run_bench(args, port, profile, args.results_dir)
        m_after = scrape_metrics(port)
    finally:
        print("[bench] terminating vLLM server...", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()

    record = build_record(args, results, m_before, m_after)
    print("\n[bench] RECORD:\n" + json.dumps(record, indent=2), flush=True)

    # Sanity: prefill must be faster than decode on every tier.
    if record["prefill_tps"] <= record["decode_tps"]:
        print("[bench] WARNING: prefill_tps <= decode_tps -- asymmetry missing; "
              "check the profiles before trusting this record.", flush=True)

    merge_record(args.rate_table, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
