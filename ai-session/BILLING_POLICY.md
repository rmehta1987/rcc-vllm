# ai-session SU billing policy

This is the published policy for the **ai-session** LLM-serving service on the
UChicago RCC cluster. It states *what* you are charged and *why*. The machine
-readable source of these numbers is [`billing/billing_policy.yaml`](../billing/billing_policy.yaml);
the formula is implemented (and unit-tested) in [`billing/su_formula.py`](../billing/su_formula.py).

## Currency: 1 SU = 1 A100-GPU-hour

SUs (Service Units) are an internal allocation currency. We anchor them to the
published convention at **NCSA Delta** (UIUC / ACCESS — the closest analog to
UChicago RCC) and **Purdue Anvil**: **one A100-GPU-hour costs one SU.** No dollar
figure is required. (If RCC later wants dollar cost-recovery, peer data puts an
unsubsidized A100-GPU-hour at ≈ $0.50–1.00; OSC's subsidized $0.09 is a floor.)

## GPU-tier cost multiplier `w_gpu`

`w_gpu` is a **cost multiplier, not a performance index**. Because 1 A100-GPU-hour
= 1 SU, the A100 anchors at 1.0; `w_gpu(H200) = 3.0` means **one H200-GPU-hour
costs 3 SU**.

| GPU tier  | `w_gpu` | Basis / note |
|-----------|--------:|--------------|
| H200      | **3.0** | NCSA Delta exact; price-basis 4.75, dense-bf16 TFLOPS 3.17 |
| H100      | **2.0** | PSC Bridges-2 + TACC Lonestar6 SU convention (TFLOPS/price say 3.2–4.0; we use the HPC-SU value) |
| L40S      | **1.0** | TFLOPS 1.16 / price 1.13; Ada, bf16-capable |
| A100-40GB | **1.0** | reference anchor |
| A40       | **0.5** | NCSA Delta "discounted"; TFLOPS 0.48 / price 0.50 |

Synthesized from NCSA Delta, PSC Bridges-2, TACC Lonestar6, Harvard FASRC,
Caltech, and MSI, cross-checked against dense-BF16 TFLOPS and list price (which
agree closely except on Hopper, where price/scarcity exceeds raw FLOPS). These
values are RCC-staff-overridable in `billing_policy.yaml`.

**Excluded from Phase 1:** V100 (Volta) and RTX6000 (Turing) are **fp16-only**
(no bf16), so some Qwen models are numerically shakier and need `--dtype half`.
Phase 1 serves only **bf16-capable tiers (A40 and up)**. Indicative weights
(~0.35 / ~0.3) are recorded only in case a future phase re-includes them.

## What a request costs (token term)

A request served by `model` on GPU tier `g` at tensor-parallel size `N`:

```
SU(request) = w_gpu(g) × N × ( T_in / prefill_tps + T_out / decode_tps ) / 3600
```

* `T_in`, `T_out` — prompt and completion tokens, taken from the vLLM API
  `usage` field (authoritative).
* `prefill_tps`, `decode_tps` — **benchmarked** aggregate throughput for that
  exact `(model, tier, TP)`, measured in the *same vLLM config the endpoint
  serves* (see [`billing/rate_table.json`](../billing/rate_table.json)).
* `N` — the GPUs the Slurm job reserves.

This converts your tokens into **equivalent GPU-hours** and prices them in the
A100-hour currency. The **output-costs-more-than-input asymmetry is measured, not
hand-set**: decode is memory-bandwidth-bound so `decode_tps << prefill_tps`, and
each output token therefore costs `α = prefill_tps / decode_tps` times an input
token. (Commercial APIs charge output 5–6× input — the physics, not margin,
drives it.)

## What a session costs (the reservation floor)

A session's Slurm job holds its GPUs for its whole wall-clock lifetime, busy or
idle. **The allocation is GPU-exclusive, not node-exclusive.** The `test`
partition schedules consumable resources (`select/cons_tres`, `OverSubscribe=NO`,
`ExclusiveUser=NO`): a job is allocated exactly the GPUs, cores, and memory it
requests, and other jobs may run on the remainder of the node. This is measured,
not assumed — `sacct` over all eight GPU jobs this service has run to date
(50383949, 50384019, 50399550, 50400678, 50401160, 50433231, 50468311, 50705130;
checked 2026-07-03) shows `AllocTRES` equal to `ReqTRES` in every case. An
earlier draft of this policy justified the floor by whole-node exclusivity; the
correct justification is at the GPU level: **while your session runs, the N GPUs
it holds cannot be given to anyone else**, whether or not you keep them busy. So
a session is billed the **greater of work done or hardware held**, in the same
unit (charging held-but-idle hardware matches reservation billing at Delta /
NERSC / TACC / PSC):

```
SU(session) = max(  Σ_requests SU(request),
                    w_gpu(g) × N × reserved_wall_hours  )
```

The floor uses the same `w_gpu × N × hours = SU` definition, so it is coherent.

**Two honest consequences:**

1. **The floor is usually the whole bill.** A single
   request's token cost is tiny (~0.0007 SU); a multi-hour reservation is 8–12
   SU. The token term only *exceeds* the floor when you sustain higher
   concurrency than the benchmark reference. Token billing is a marginal top-up
   for heavy users — the main lever is **how long you hold the GPUs**, so release
   your session (`ai_session end`) when you're done.
2. **A more expensive GPU always costs more to hold** (H200 floor > H100 ≥
   L40S/A100 > A40, by pure `w_gpu × N`). This is the fix for the old flat-rate
   bug, where a faster GPU was *cheaper* for the same work. We do **not** claim
   the per-token rate is ordered across tiers — a fast-enough H200 can have a
   lower `su_per_1k` (throughput is in the denominator); the **floor**, not the
   token rate, is the cross-tier invariant.

## Measured rates

One row per record in [`billing/rate_table.json`](../billing/rate_table.json).
Every rate was benchmarked on the cluster (vLLM 0.10.2, bf16, prefix caching on,
concurrency 64) in the same serve configuration the endpoint uses; the Coder-32B
record was measured at its production 32,768-token context, the others at 8,192.
The `SU/1k` columns are stated at N = TP; metering recomputes them from the raw
throughputs with the N actually reserved. Hold SU/h = `w_gpu × N`. The `SU/1k`
values below were recomputed from the raw throughputs as
`w_gpu × N × 1000 / (tps × 3600)` and cross-checked against the stored table on
2026-07-03; all ten agree to within 3 parts per million (floating-point rounding).

| Model | Tier (card measured) | TP | `w_gpu` | Hold SU/h | SU/1k in | SU/1k out | Job | Date |
|---|---|--:|--:|--:|--:|--:|--|--|
| `qwen3_4b` | a100 (A100-PCIE-40GB) | 1 | 1.0 | 1.0 | 0.000014 | 0.000067 | 50383949 | 2026-06-02 |
| `qwen2.5_coder_32B` | a100 (A100 80GB PCIe) | 2 | 1.0 | 2.0 | 0.000116 | 0.000331 | 50705130 | 2026-06-10 |
| `qwen2.5_72B` | a100 (A100 80GB PCIe) | 4 | 1.0 | 4.0 | 0.000383 | 0.000990 | 50384019 | 2026-06-02 |
| `qwen2.5_72B` | h100 (H100 NVL 94GB) | 4 | 2.0 | 8.0 | 0.000587 | 0.001227 | 50401160 | 2026-06-02 |
| `qwen2.5_72B` | h200 (H200) | 2 | 3.0 | 6.0 | 0.000219 | 0.000716 | 50399550 | 2026-06-02 |

**Choosing a tier.** For the 72B model, the A100 (TP=4) has the lowest hold cost
(4.0 SU/h) and is the choice for interactive sessions, where the floor is the
bill. The H200 (TP=2) is the cheapest per output token (0.000716 SU/1k, about
0.72× the A100 rate) and wins for sustained throughput-bound work, where the
token term can matter. The H100 (TP=4) is dominated for this model — highest
hold cost (8.0 SU/h, four cards) *and* highest per-token cost (0.001227 SU/1k
out) — so it has no economical regime here. Across the production-scale models,
`qwen2.5_coder_32B` on two A100s is the cheapest per token of all (0.000331
SU/1k out) with the lowest hold cost (2.0 SU/h), which is why it is the coding
default.

**Unrated tiers.** A40 (`w_gpu` 0.5) and L40S (`w_gpu` 1.0) have cost weights
but no benchmark record. A session on those tiers is billed the **floor only**
— the token term is reported UNRATED — until a benchmark is run for that
`(model, tier, TP)`.

## Edge cases

| Case | Billing |
|------|---------|
| Failed / 5xx request | **Not billed** (no `usage`; the engine's `request_success_total` isn't incremented). |
| Cancelled streaming request | `T_in` billed in full (prefill happened) + tokens actually produced. Set `stream_options={"include_usage": true}` on the client, else we fall back to the `/metrics` session delta. |
| Prefix caching (`--enable-prefix-caching`) | Full `prompt_tokens` billed — **no cache discount in Phase 1** (vLLM reports the full prompt regardless of cache hits; matches OpenAI). The floor usually dominates anyway, so caching lets you do *more* per reservation, not pay less. A discount is a possible Phase-2 knob. |
| No rate-table entry for your `(model, tier, TP)` | Token term is reported as **UNRATED**; you are billed the **floor only** until the benchmark is run for that config. |

## Worked example (measured rates)

Qwen2.5-72B, one request `T_in=2000, T_out=500`, 2-hour reservation, using the
measured prefill/decode throughputs from
[`billing/rate_table.json`](../billing/rate_table.json) (jobs 50399550 and
50384019, 2026-06-02):

```
H200 (TP=2): token SU = 3.0 × 2 × (2000/7593.69 + 500/2328.72) / 3600 = 0.00080
A100 (TP=4): token SU = 1.0 × 4 × (2000/2901.16 + 500/1122.71) / 3600 = 0.00126
```

| Tier (TP)   | `w_gpu×N` | token SU (one request) | 2 h floor SU | **billed** |
|-------------|----------:|-----------------------:|-------------:|-----------:|
| H200 (TP=2) | 6.0       | 0.00080                | 12.0         | **12.0**   |
| A100 (TP=4) | 4.0       | 0.00126                | 8.0          | **8.0**    |

The floor is the charge; the token term is a rounding error for a single
request. H200 bills more than A100 because it costs more to *hold* — exactly the
intended behavior — even though its token term for this request is the smaller
of the two (two H200s process the request in less weighted GPU-time than four
A100s). That is the interactive-versus-throughput trade-off from the tier-choice
paragraph, visible in one request.

## How to read your usage summary

`ai_session end` writes `<user>_<jobid>_<ts>_summary.json` under the state dir
`/project/rcc/mehta5/ai-session-state/<user>/logs/usage/` when run via the
wrapper scripts (which set `AISESSION_STATE_DIR`), or under
`ai-session/logs/usage/` when the tools are run directly without that variable,
with: token source (per-request `usage` or `/metrics` delta), total tokens,
`token_su`, `floor_su`, the **`billed_su`** you pay, the `basis` (`floor` vs
`tokens`), and a `/metrics` reconciliation check when both sources are present.

## Central accounting

Every session's final charge is also recorded to a staff-only ledger for
tracking, in addition to the per-user summary above. On `ai_session end` a
single JSON record (schema `ai-session-billing/1`) is written to
`AISESSION_BILLING_DIR` (default `/project/rcc/mehta5/ai-session-billing`, a
directory with mode `2770`, group `rcc-staff`, and no world access) as
`<user>_<jobid>_end.json` at mode `0640`. It carries the same numbers `end`
reports -- `jobid`, `user`, `model_key`, `gpu_tier`, `n_gpus`, `w_gpu`,
`reserved_wall_hours`, `floor_su`, `token_su`, `billed_su`, and `basis` -- plus
`source` (`end` here; the staff sacct sweep writes `source=sweep` with the
authoritative reservation floor and `token_su=null`), `written_at`, and `host`.
The two sources may both exist for one job and are kept side by side. This write
is best-effort: if the ledger directory is missing or unwritable, `end` logs a
warning to stderr and continues -- it still prints the receipt and still writes
the per-user summary.
