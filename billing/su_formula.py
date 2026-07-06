"""SU (Service Unit) billing formula for the ai-session vLLM service.

Pure, dependency-free functions. The only side-effecting helpers are the
``load_*`` loaders at the bottom, which lazily import PyYAML so that importing
this module (e.g. from unit tests) never requires yaml.

Currency convention (peer-anchored, NCSA Delta / Purdue Anvil):

    1 SU == 1 A100-GPU-hour.

Per-request charge (GPU-time form).  For a request served by ``model`` on GPU
tier ``g`` at tensor-parallel size ``N``::

    SU(request) = w_gpu(g) * N * ( T_in / prefill_tps + T_out / decode_tps ) / 3600

* ``T_in`` / ``T_out``      -- prompt / completion tokens (from vLLM ``usage``).
* ``prefill_tps``           -- benchmarked aggregate prefill throughput (tok/s).
* ``decode_tps``            -- benchmarked aggregate decode throughput (tok/s).
* ``w_gpu(g)``              -- GPU-tier cost multiplier (A100 = 1.0).
* ``N``                     -- GPUs the Slurm job reserves (== TP in the simple
                               case; the whole-node GPU count if the partition
                               holds nodes exclusively).

The output>input asymmetry is *measured*, not hand-set: decode is
memory-bandwidth-bound so ``decode_tps << prefill_tps`` and each output token
costs more GPU-time.  ``alpha_empirical = prefill_tps / decode_tps``.

Runtime form (precomputed rate table -- still a multiply-add)::

    su_per_1k_in (m,g,N) = w_gpu(g) * N * 1000 / (3600 * prefill_tps)
    su_per_1k_out(m,g,N) = w_gpu(g) * N * 1000 / (3600 * decode_tps)
    SU(request)          = su_per_1k_in * T_in/1000 + su_per_1k_out * T_out/1000

Session charge with the exclusive-node reservation floor::

    SU(session) = max( sum_requests SU(request),
                       w_gpu(g) * N * reserved_wall_hours )

The floor (pure ``w_gpu * N * hours``, no throughput) is what fixes the old
flat-rate perversion and is the dominant charge on an exclusively-held node.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

SECONDS_PER_HOUR = 3600.0


# --------------------------------------------------------------------------- #
# Core per-request charge
# --------------------------------------------------------------------------- #
def su_per_1k_tokens(w_gpu: float, n_gpus: int, tps: float) -> float:
    """SU charged per 1000 tokens processed at aggregate throughput ``tps``.

    Converts token-time into A100-GPU-hours::

        su_per_1k = w_gpu * n_gpus * 1000 / (3600 * tps)
    """
    if tps <= 0:
        raise ValueError(f"throughput must be > 0, got {tps!r}")
    if n_gpus <= 0:
        raise ValueError(f"n_gpus must be > 0, got {n_gpus!r}")
    if w_gpu < 0:
        raise ValueError(f"w_gpu must be >= 0, got {w_gpu!r}")
    return w_gpu * n_gpus * 1000.0 / (SECONDS_PER_HOUR * tps)


def su_for_request(
    t_in: int,
    t_out: int,
    w_gpu: float,
    n_gpus: int,
    prefill_tps: float,
    decode_tps: float,
) -> float:
    """SU for one request via the GPU-time form.

    Mathematically identical to :func:`su_for_request_from_rates` fed with
    :func:`su_per_1k_tokens` constants -- ``test_su_formula`` asserts the
    equivalence.
    """
    if t_in < 0 or t_out < 0:
        raise ValueError(f"token counts must be >= 0, got t_in={t_in}, t_out={t_out}")
    if prefill_tps <= 0 or decode_tps <= 0:
        raise ValueError(
            f"throughputs must be > 0, got prefill={prefill_tps}, decode={decode_tps}"
        )
    if n_gpus <= 0:
        raise ValueError(f"n_gpus must be > 0, got {n_gpus!r}")
    return (
        w_gpu
        * n_gpus
        / SECONDS_PER_HOUR
        * (t_in / prefill_tps + t_out / decode_tps)
    )


def su_for_request_from_rates(
    t_in: int,
    t_out: int,
    su_per_1k_in: float,
    su_per_1k_out: float,
) -> float:
    """SU for one request via the precomputed rate-table form (runtime path)."""
    if t_in < 0 or t_out < 0:
        raise ValueError(f"token counts must be >= 0, got t_in={t_in}, t_out={t_out}")
    return su_per_1k_in * t_in / 1000.0 + su_per_1k_out * t_out / 1000.0


def alpha_empirical(prefill_tps: float, decode_tps: float) -> float:
    """The measured input/output asymmetry ``prefill_tps / decode_tps``.

    Replaces the old hand-set ``alpha`` multiplier: each output token costs
    ``alpha`` times the GPU-time of an input token.
    """
    if decode_tps <= 0:
        raise ValueError(f"decode_tps must be > 0, got {decode_tps!r}")
    return prefill_tps / decode_tps


# --------------------------------------------------------------------------- #
# Session charge + exclusive-node reservation floor
# --------------------------------------------------------------------------- #
def reservation_floor_su(w_gpu: float, n_gpus: int, reserved_wall_hours: float) -> float:
    """The exclusive-node floor: pay for the GPUs held, in the same SU unit."""
    if reserved_wall_hours < 0:
        raise ValueError(f"reserved_wall_hours must be >= 0, got {reserved_wall_hours}")
    if n_gpus <= 0:
        raise ValueError(f"n_gpus must be > 0, got {n_gpus!r}")
    return w_gpu * n_gpus * reserved_wall_hours


@dataclass
class SessionCharge:
    """Result of :func:`su_for_session`.

    ``billed_su`` is what the user pays; ``basis`` says which term won.
    """

    token_su: float
    floor_su: float
    billed_su: float
    basis: str  # "floor" | "tokens" | "tokens (floor disabled)"
    n_requests: int = 0
    floor_enabled: bool = True

    def as_dict(self) -> dict:
        return {
            "token_su": self.token_su,
            "floor_su": self.floor_su,
            "billed_su": self.billed_su,
            "basis": self.basis,
            "n_requests": self.n_requests,
            "floor_enabled": self.floor_enabled,
        }


def su_for_session(
    token_su: float,
    w_gpu: float,
    n_gpus: int,
    reserved_wall_hours: float,
    *,
    floor_enabled: bool = True,
    n_requests: int = 0,
) -> SessionCharge:
    """Combine summed per-request SU with the reservation floor.

        SU(session) = max(token_su, w_gpu * N * reserved_wall_hours)

    With ``floor_enabled=False`` the session is billed purely on tokens (used
    only if a future deployment runs on shareable, non-exclusive nodes).
    """
    floor = reservation_floor_su(w_gpu, n_gpus, reserved_wall_hours)
    if not floor_enabled:
        return SessionCharge(
            token_su=token_su,
            floor_su=floor,
            billed_su=token_su,
            basis="tokens (floor disabled)",
            n_requests=n_requests,
            floor_enabled=False,
        )
    if token_su >= floor:
        basis = "tokens"
        billed = token_su
    else:
        basis = "floor"
        billed = floor
    return SessionCharge(
        token_su=token_su,
        floor_su=floor,
        billed_su=billed,
        basis=basis,
        n_requests=n_requests,
        floor_enabled=True,
    )


# --------------------------------------------------------------------------- #
# Aggregating a list of request records (edge cases live here, not in the math)
# --------------------------------------------------------------------------- #
def sum_request_su(
    requests,
    w_gpu: float,
    n_gpus: int,
    prefill_tps: float,
    decode_tps: float,
    *,
    bill_failed: bool = False,
) -> tuple:
    """Sum SU over an iterable of request dicts.

    Each request dict needs ``prompt_tokens`` and ``completion_tokens``; an
    optional ``success`` flag (default True) controls billing of failures.

    * Failed requests (``success=False``) are not billed unless
      ``bill_failed=True`` -- they have no ``usage`` from vLLM anyway.
    * Prefix-cache hits are NOT discounted: ``prompt_tokens`` from the
      authoritative ``usage`` field is billed in full (vLLM reports the full
      prompt regardless of cache hits; matches OpenAI).

    Returns ``(total_su, n_billed)``.
    """
    total = 0.0
    n_billed = 0
    for r in requests:
        if not r.get("success", True) and not bill_failed:
            continue
        t_in = int(r.get("prompt_tokens", 0))
        t_out = int(r.get("completion_tokens", 0))
        total += su_for_request(t_in, t_out, w_gpu, n_gpus, prefill_tps, decode_tps)
        n_billed += 1
    return total, n_billed


# --------------------------------------------------------------------------- #
# Policy + rate-table loaders (lazy yaml import -- pure math above needs none)
# --------------------------------------------------------------------------- #
@dataclass
class BillingPolicy:
    """Parsed ``billing_policy.yaml``."""

    su_per_a100_gpu_hour: float
    w_gpu: dict  # tier -> cost multiplier
    floor_enabled: bool
    bill_failed: bool
    prefix_cache_discount: bool
    excluded_tiers: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    def weight(self, tier: str) -> float:
        """Cost multiplier for a GPU tier (case-insensitive lookup).

        Raises if the tier is unknown or explicitly excluded from Phase 1.
        """
        key = _normalize_tier(tier)
        norm = {_normalize_tier(k): v for k, v in self.w_gpu.items()}
        if key in norm:
            return float(norm[key])
        excluded = {_normalize_tier(k) for k in self.excluded_tiers}
        if key in excluded:
            raise KeyError(
                f"GPU tier {tier!r} is excluded from Phase 1 (fp16-only); "
                f"no w_gpu weight. Excluded: {sorted(self.excluded_tiers)}"
            )
        raise KeyError(
            f"unknown GPU tier {tier!r}; known tiers: {sorted(self.w_gpu)}"
        )


def _normalize_tier(tier: str) -> str:
    """Canonicalize a GPU-tier string: lowercase, strip vendor/memory noise.

    ``"NVIDIA A100-SXM4-40GB"`` -> ``"a100"``, ``"H200"`` -> ``"h200"``.
    """
    t = str(tier).lower()
    for token in ("nvidia", "tesla", "-sxm4", "-sxm5", "-pcie", "_"):
        t = t.replace(token, " ")
    t = t.replace("-", " ").strip()
    # take the first whitespace token that looks like a tier name
    for tok in t.split():
        for known in ("h200", "h100", "l40s", "l40", "a100", "a40",
                      "v100", "rtx6000", "rtxa6000", "a6000"):
            if tok.startswith(known):
                return known
    return t.split()[0] if t.split() else t


def load_policy(path: str) -> BillingPolicy:
    """Load and validate ``billing_policy.yaml`` (lazy yaml import)."""
    import yaml  # noqa: PLC0415 -- lazy so pure-math import needs no yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    currency = data.get("currency", {})
    su_per = float(currency.get("su_per_a100_gpu_hour", 1.0))
    weights = data.get("w_gpu", {})
    if not weights:
        raise ValueError(f"{path}: 'w_gpu' table is empty")
    edge = data.get("edge_cases", {})
    return BillingPolicy(
        su_per_a100_gpu_hour=su_per,
        w_gpu=dict(weights),
        floor_enabled=bool(data.get("reservation_floor", {}).get("enabled", True)),
        bill_failed=bool(edge.get("bill_failed_requests", False)),
        prefix_cache_discount=bool(edge.get("prefix_cache_discount", False)),
        excluded_tiers=dict(data.get("excluded_tiers", {})),
        raw=data,
    )


def load_rate_table(path: str) -> dict:
    """Load ``rate_table.json``. Returns the parsed dict (may have empty records)."""
    with open(path) as f:
        return json.load(f)


def rate_record(rate_table: dict, model_key: str, tier: str, tp: int):
    """Find the record for ``(model_key, tier, TP)``; returns dict or None.

    Tier match is normalized (so ``"NVIDIA A100-SXM4-40GB"`` matches ``"a100"``).
    """
    want_tier = _normalize_tier(tier)
    for rec in rate_table.get("records", []):
        if (
            rec.get("model_key") == model_key
            and _normalize_tier(rec.get("gpu_tier", "")) == want_tier
            and int(rec.get("tp", -1)) == int(tp)
        ):
            return rec
    return None


# Default artifact locations (this file lives in billing/).
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POLICY_PATH = os.path.join(_HERE, "billing_policy.yaml")
DEFAULT_RATE_TABLE_PATH = os.path.join(_HERE, "rate_table.json")
