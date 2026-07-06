"""Model registry + Slurm discovery + node->GPU-tier / job->GPU-count resolvers.

Adapted from decrypto/src/utils/server.py, extended so metering can resolve the
GPU tier ``g`` and the reserved GPU count ``N`` for a running session (both feed
the SU charge: w_gpu(g) * N * ...).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

# Reuse the single source of truth for tier normalization from the billing module.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BILLING = os.path.join(os.path.dirname(_HERE), "billing")
if _BILLING not in sys.path:
    sys.path.insert(0, _BILLING)
import su_formula as su  # noqa: E402

MODELS_ROOT = "/project/rcc/mehta5/vllm/models"

# model_key -> local path (or HF id). Keys are stable identifiers used in the
# Slurm job-name (`model_key:port`), the rate table, and usage logs.
MODEL_REGISTRY = {
    "qwen2.5_72B": f"{MODELS_ROOT}/Qwen2.5-72B-Instruct",   # Phase-1 production model (general chat)
    "qwen2.5_coder_32B": f"{MODELS_ROOT}/Qwen2.5-Coder-32B-Instruct",  # coding model (TP=2 footprint)
    "qwen3_4b": f"{MODELS_ROOT}/Qwen3-4B",                  # single-GPU benchmark anchor
    "llama3.1_70B": f"{MODELS_ROOT}/Meta-Llama-3.1-70B-Instruct",  # cross-check / optional
    "qwen2.5_0.5B": f"{MODELS_ROOT}/Qwen2.5-0.5B-Instruct", # smoke test only -- never a billing ref
}

# The only models actually served to users in Phase 1 (others are for benchmarking
# / smoke). ai_session.py rejects start requests for keys outside this set.
# qwen2.5_coder_32B is the coding-client default (code-specialized, half the GPUs of 72B).
PHASE1_SERVED = {"qwen2.5_72B", "qwen2.5_coder_32B", "qwen3_4b"}

KNOWN_TIERS = ("h200", "h100", "l40s", "l40", "a100", "a40", "v100", "rtx6000")


def model_path(model_key: str) -> str:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown model_key {model_key!r}; registered: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_key]


# --------------------------------------------------------------------------- #
# Discovery (squeue job-name == 'model_key:port')
# --------------------------------------------------------------------------- #
def _discover_servers_from_squeue() -> list:
    result = subprocess.run(
        ["squeue", "--me", "-o", '"%j, %N, %T, %i"'],
        capture_output=True, text=True,
    )
    lines = result.stdout.strip().split("\n")
    local_models = []
    for line in lines[1:]:
        line = line.strip('"')
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        full_job_name, nodelist, status, job_id = parts[0], parts[1], parts[2], parts[3]
        if "[" in nodelist:  # multi-node not supported
            continue
        if ":" in full_job_name:
            job_name, port = full_job_name.split(":", 1)
        else:
            job_name, port = full_job_name, "8000"
        if status != "RUNNING" or job_name not in MODEL_REGISTRY:
            continue
        server_address = f"http://{nodelist}:{port}/v1"
        existing = next((m for m in local_models if m["model_key"] == job_name), None)
        if existing:
            existing["urls"].append(server_address)
            existing["job_ids"].append(job_id)
            existing["nodes"].append(nodelist)
            existing["ports"].append(port)
        else:
            local_models.append({
                "model_key": job_name,
                "model_id": MODEL_REGISTRY[job_name],
                "urls": [server_address],
                "job_ids": [job_id],
                "nodes": [nodelist],
                "ports": [port],
            })
    return local_models


def get_available_servers() -> list:
    """Servers from a pre-built file (DECRYPTO/AISESSION_SERVERS_FILE) or squeue."""
    servers_file = os.environ.get("AISESSION_SERVERS_FILE") or os.environ.get("DECRYPTO_SERVERS_FILE")
    if servers_file and os.path.exists(servers_file):
        with open(servers_file) as f:
            return json.load(f)
    return _discover_servers_from_squeue()


# --------------------------------------------------------------------------- #
# node -> GPU tier (via scontrol features)
# --------------------------------------------------------------------------- #
def resolve_node_tier(node: str):
    """Return the normalized GPU tier for a node, or None if undeterminable.

    Reads ActiveFeatures/AvailableFeatures and the Gres line from
    ``scontrol show node`` and matches against known tier names.
    """
    if not node:
        return None
    try:
        out = subprocess.run(
            ["scontrol", "show", "node", node], capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    feats = ""
    for key in ("ActiveFeatures", "AvailableFeatures"):
        m = re.search(rf"{key}=(\S+)", out)
        if m:
            feats += "," + m.group(1)
    # Match features first (tier is usually a node feature like 'a100').
    for tok in feats.lower().replace(",", " ").split():
        for tier in KNOWN_TIERS:
            if tok.startswith(tier):
                return su._normalize_tier(tier)
    # Fall back to the Gres line, e.g. 'Gres=gpu:a100:4'.
    gm = re.search(r"Gres=gpu:([a-zA-Z0-9]+):", out)
    if gm:
        return su._normalize_tier(gm.group(1))
    return None


# --------------------------------------------------------------------------- #
# job -> reserved GPU count N (and whole-node detection)
# --------------------------------------------------------------------------- #
def resolve_job_gpus(job_id: str):
    """Return the number of GPUs allocated to a job, or None.

    Parses the TRES/Gres in ``scontrol show job``. This is the ``N`` in the SU
    charge. NOTE: if the partition holds nodes whole, this may exceed the TP the
    server was launched with -- the caller decides which to bill (see metering).
    """
    if not job_id:
        return None
    try:
        out = subprocess.run(
            ["scontrol", "show", "job", job_id], capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    # AllocTRES=...,gres/gpu=4  or  TresPerNode=gres:gpu:4  or  Gres=gpu:4
    for pat in (r"gres/gpu=(\d+)", r"gres:gpu:(\d+)", r"Gres=gpu:[a-zA-Z0-9]*:?(\d+)"):
        m = re.search(pat, out)
        if m:
            return int(m.group(1))
    return None


def reserved_wall_hours(job_id: str, fallback_start_epoch: float = None):
    """Reserved wall time in hours for the floor.

    Prefers sacct Elapsed (authoritative); falls back to (now - start) if a
    start epoch is given. Returns None if neither is available.
    """
    import time
    try:
        out = subprocess.run(
            ["sacct", "-j", str(job_id), "-o", "Elapsed", "-n", "-P", "-X"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()
        if out and out[0].strip():
            return _parse_elapsed(out[0].strip()) / 3600.0
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    if fallback_start_epoch is not None:
        return max(0.0, (time.time() - fallback_start_epoch) / 3600.0)
    return None


def _parse_elapsed(s: str) -> float:
    """Parse sacct Elapsed '[DD-]HH:MM:SS' into seconds."""
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3], parts[-2], parts[-1]
    return days * 86400 + h * 3600 + m * 60 + sec


if __name__ == "__main__":
    print(json.dumps(get_available_servers(), indent=2))
