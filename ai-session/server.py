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
    "qwen3_4b": f"{MODELS_ROOT}/Qwen3-4B",                  # single-GPU benchmark anchor
    "qwen3.5_122B": f"{MODELS_ROOT}/Qwen3.5-122B-A10B-FP8", # MoE, native FP8 -- serves TP=2 on 2xH200 (53069683) AND 2xH100 NVL (53538328)
    "qwen3.8_27B": f"{MODELS_ROOT}/Qwen3.8-27B",           # coding INCUMBENT -- dense 27.8B BF16, hybrid GDN+attn
    "gemma4_31B": f"{MODELS_ROOT}/Gemma-4-31B-it",          # SECOND coding option -- dense 30.7B BF16, Apache-2.0
    "qwen2.5_0.5B": f"{MODELS_ROOT}/Qwen2.5-0.5B-Instruct", # smoke test only -- never a billing ref
}

# The models served to users (others are for benchmarking / smoke). ai_session.py
# rejects start requests for keys outside this set. qwen3.8_27B is the coding-client
# default (see the note below) and is itself a thinking model, served with
# --reasoning-parser qwen3 and the qwen3_coder tool parser.
#
# RETIRED 2026-08-19: llama3.1_70B (Meta-Llama-3.1-70B-Instruct) removed and deleted from
# disk. It was served for months and recorded ZERO sessions in the central ledger, held no
# rate_table row (so it billed the floor), and was a 2024 general model whose role the
# newer qwen2.5_72B already fills. It was also the only licence-gated model here; the
# _LICENSE_GATED machinery in ai_session.py is kept, unused, for the next such model.
# NOTE: unlike every other model retired today, this one is NOT a free re-download --
# Meta gates the weights behind an accepted licence on Hugging Face.
#
# qwen3.5_122B is VALIDATED but deliberately NOT in PHASE1_SERVED yet -- what remains
# is a production cutover only the operator can decide, NOT a missing smoke test. The
# model-refresh loop (branch milestone/model-refresh, 2026-08-05) proved it end to end
# on the vllm-serve-cu129 env (vLLM 0.26.0, cu129): Gate-1 job 53069683 served it FP8
# TP=2 on 2xH200 (driver 535.216.03) with a clean code completion, and Gate-2 chose it
# as the Tier-B (H200) coding winner over qwen2.5_coder_32B on a frozen LiveCodeBench
# subset (45.0 vs 26.7 pass@1, +18.33 pts). It STILL cannot be added here safely because
# the production serve path (launch_ai_session.sh) hardcodes ENV_PATH to the 0.10.2
# vllm-probe env, where Qwen3_5MoeForConditionalGeneration does not load -- a bare add
# would let a user reserve 2xH200, fail on load, and floor-bill (the job name
# qwen3.5_122B:port is swept by billing_sweep.py::model_key_of). The remaining steps are
# operator decisions (runbook in prompts/60_model_refresh/verdict.md, flagged
# OPERATOR-DECISION-PENDING): (1) route production serving of this model to
# vllm-serve-cu129 (0.26.0); (2) measure a billing rate_table row with bench_billing.py
# on that env (the Stage-2 code-gen number is pass@1, NOT a prefill/decode throughput
# sweep, so it cannot fill a rate row); (3) then add qwen3.5_122B here. Until then, a
# staff smoke uses the SANCTIONED cu129 serve path -- NOT `ai_session.py start`, which
# routes through launch_ai_session.sh's hardcoded 0.10.2 env and would floor-bill this
# doomed load. Submit tools/serve_cu129.sbatch on H200 at the MEASURED TP=2 with a
# bench-* served-name (invisible to the billing sweep and to production discovery):
#   sbatch --job-name=mrefresh-nest-serve --constraint=H200 --gres=gpu:2 --time=00:30:00 \
#     --export=ALL,MODEL_DIR=/project/rcc/mehta5/vllm/models/Qwen3.5-122B-A10B-FP8,\
#   SERVED_NAME=bench-qwen35-122b,TP=2,PORT=8412 tools/serve_cu129.sbatch
#
# REMOVED 2026-08-19: glm5.2_753B (GLM-5.2-FP8, 704 GB) and DeepSeek-V4-Flash were
# deleted from disk. GLM never served a token and structurally could not: 755 GB of FP8
# weights exceed one 4xH200 node (564 GB), the multi-node launcher does not exist, and
# _discover_servers_from_squeue() skips multi-node nodelists anyway. V4-Flash was a
# measured Gate-1 NO-GO (job 53069684, Marlin FP4 repack vs driver 535). Both are
# ungated re-downloads; configs, index, and licenses are preserved under
# _scratch/tombstones/ with a restage recipe.
# qwen3.8_27B is the coding INCUMBENT as of 2026-08-19, replacing qwen2.5_coder_32B.
# Basis: frozen 60-problem LiveCodeBench subset, greedy, thinking-off, identical harness
# and serve env -- 50.00% (30/60) vs the incumbent's 26.67% (16/60), +23.33 pts (score job
# 53531932). Best of six candidates measured; beats the 122B (45.00%) at 44% of its
# footprint, though that 5-pt gap is inside the n=60 noise band. Serves BF16 TP=2 on
# H100 (jobs 53533204, 53534097) and H200 (53440093, 53496745) under vLLM 0.26.0.
# Tool calling VERIFIED working with --tool-call-parser qwen3_coder (job 53534097);
# hermes silently fails on it, so launch_ai_session.sh routes the parser by model key.
# The launcher also routes this key's ENV_PATH to vllm-serve-cu129 -- 0.10.2 cannot load
# Qwen3_5ForConditionalGeneration and a bare add here without that routing would
# floor-bill a doomed load.
# CAVEAT: no rate_table.json row yet, so sessions bill the reservation FLOOR
# (metering.py:322 "UNRATED"), same as qwen3_32B. A bench_billing.py run on the cu129
# env is owed before this model carries real token-metered billing.
# qwen2.5_coder_32B stays served (and stays the frozen benchmark baseline anchor).
# RETIRED 2026-08-19: qwen2.5_coder_32B and qwen3_32B removed from the registry and
# deleted from disk, superseded by qwen3.8_27B (50.00% vs 26.67% on the frozen LCB-60).
# qwen3.8_27B is now the SOLE coding model. Consequences, recorded deliberately:
#   - The frozen benchmark baseline's weights are gone. The 26.67% anchor survives only
#     as benchmark/frozen_baseline/stage2_score_bench-coder32b.json, which
#     score_stage2.sbatch still reuses for adjudication -- but it can never be
#     regenerated or re-run under a new condition (subset, decode, thinking mode).
#   - rate_table.json retains inert rows for both keys. They are measurement provenance;
#     nothing looks them up now that the keys are unserved. Left deliberately.
#   - qwen2.5_coder_32B was the only model an external user (ndtrung) had ever run, and
#     one of three models with an honest (non-floor) billing rate. qwen3.8_27B has no
#     rate row yet, so coding sessions now bill the reservation FLOOR until a
#     bench_billing.py run on the cu129 env fills one. That measurement is owed.
# gemma4_31B added 2026-08-20 as a SECOND coding option, NOT a replacement. qwen3.8_27B
# remains the `code` preset default; Gemma is reached with `--model gemma4_31B`.
# Basis: same frozen LCB-60 subset/decode/harness -- 66.67% (40/60) vs the 27B's 50.00%,
# Gate-2 PASS, score job 53554777. hard 14/30 vs 6/30. ~2.8 sigma, outside the n=60 noise
# band. It is NOT wired as the default because that comparison has a known asymmetry: the
# frozen decode pins enable_thinking:false, which is Gemma's NATIVE default but suppresses
# Qwen3.8's xhigh default, so the 27B's 50.00% is a lower bound. A thinking-on Qwen rerun
# is owed before any swap.
# The two differ in character, which is why offering both is useful:
#   qwen3.8_27B  thinks by default (reasoning_effort xhigh); has an MTP draft head (+53%
#                decode, unused); forced --enforce-eager by a CUDA-graph crash.
#   gemma4_31B   thinking OFF by default, opt-in via chat_template_kwargs.enable_thinking.
#                MEASURED cost of turning it on (job 53587542): 5-12x tokens AND wall time,
#                with no measurable answer improvement on two greedy prompts. Under GPU-time
#                billing that is a 5-12x cost multiplier, so leaving it off is the right
#                default. No MTP head in this checkpoint (verified on disk).
# Serves TP=2 on all four tiers: H100 NVL, A100 80GB, A100 40GB, A40 (jobs 53544338,
# 53544339, 53586878, 53586877). Tool calling verified with vLLM's gemma4 parser (53544725).
# CAVEAT: no rate_table row yet -> floor billing, same as qwen3.8_27B on A100.
PHASE1_SERVED = {"qwen2.5_72B", "qwen3_4b", "qwen3.8_27B", "gemma4_31B"}

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
