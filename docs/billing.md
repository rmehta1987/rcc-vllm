# Billing and Service Units

This page is the canonical statement of what an ai-session costs. It defines the
Service Unit, gives the GPU-type multiplier table, states the charging formula,
lists the measured throughput rates the formula uses, and shows how to read the
charge you are billed. The machine-readable sources ship with the service
(`billing/billing_policy.yaml` for the policy numbers, `billing/rate_table.json`
for the measured rates); the published rationale is in the service's
`BILLING_POLICY.md`. How to start and stop sessions is documented on
[Getting Started](getting-started.md) (browser chat) and
[Coding Sessions](coding/overview.md); this page only explains what those
sessions cost.

## The unit: 1 SU = 1 A100-GPU-hour

A Service Unit (SU) is the internal accounting currency of the ai-session service:
**one A100-GPU-hour costs one SU**, following the published convention at NCSA Delta
and Purdue Anvil. SUs are fair-usage accounting, not dollars; no dollar figure is
attached to them in Phase 1 (`dollar_per_su: null` in the policy file).

!!! note "ai-session SUs are not RCC service units"
    The RCC user guide states that beagle3 and other private partitions do not
    consume RCC service units. The SUs on this page are the ai-session service's
    internal accounting, tracked by the service itself. They are separate from any
    RCC allocation balance and do not draw on it.

## GPU-type multiplier w_gpu

`w_gpu` is a cost multiplier, not a performance index. Because 1 A100-GPU-hour is
1 SU, the A100 anchors at 1.0, and `w_gpu(H200) = 3.0` means one H200-GPU-hour costs
3 SU. The values below are what the service charges today; the operators can
adjust them in the policy file.

| GPU type  | w_gpu | Basis |
|-----------|------:|-------|
| H200      | 3.0   | NCSA Delta exact; price ratio 4.75, dense-BF16 TFLOPS ratio 3.17 |
| H100      | 2.0   | PSC Bridges-2 and TACC Lonestar6 SU convention (TFLOPS and price ratios say 3.2-4.0; the HPC-SU value is used) |
| L40S      | 1.0   | TFLOPS ratio 1.16, price ratio 1.13; Ada generation, bf16-capable |
| A100-40GB | 1.0   | Reference anchor |
| A40       | 0.5   | NCSA Delta discounted tier; TFLOPS ratio 0.48, price ratio 0.50 |

The table is synthesized from the peer conventions at NCSA Delta, PSC Bridges-2,
TACC Lonestar6, Harvard FASRC, Caltech, and MSI, cross-checked against dense-BF16
TFLOPS and list price. The two checks agree closely except on the Hopper GPUs, where
price and scarcity exceed raw FLOPS.

V100 and RTX6000 are excluded from Phase 1: both are fp16-only (no bf16 support), so
some Qwen models are numerically shakier on them and would need `--dtype half`.
Phase 1 serves only bf16-capable types (A40 and up).

## What a session costs

A session reserves N GPUs for its whole wall-clock lifetime. The charge is the
greater of two terms computed in the same unit:

```text
token_su  = w_gpu(type) * N * (T_in / prefill_tps + T_out / decode_tps) / 3600
floor_su  = w_gpu(type) * N * hours_held
billed_su = max(token_su, floor_su)
```

- `T_in`, `T_out` — prompt and completion tokens, taken from the serving API's
  `usage` field (authoritative).
- `prefill_tps`, `decode_tps` — benchmarked aggregate throughput for that exact
  model-and-GPU configuration, from the [rate table below](#measured-rates).
- `N` — the number of GPUs the session reserves; each model's configuration fixes
  it (see the model tables on the launch pages).
- `hours_held` — how long the session held the GPUs, from the cluster's own
  accounting records.

The token term converts your tokens into equivalent GPU-hours and prices them in the
A100-hour currency. The floor term (the **reservation floor**) exists because the
GPUs your session reserves are held for its entire lifetime — idle GPUs inside a
running session cannot be given to anyone else — so you pay for the greater of the
work done and the hardware held, whether or not you were actively using it. This
matches how Delta, NERSC, TACC, and PSC bill reserved GPU time.

!!! warning "A running session consumes SU whether or not you send requests"
    The floor accrues for every hour the session holds its GPUs, idle or busy.
    Run `ai-session stop` as soon as you stop working.

Two honest consequences of the max():

1. **The floor is usually the whole bill for interactive use.** A single request's
   token cost is on the order of 0.001 SU (see the
   [worked example](#worked-example)); a multi-hour reservation is 8-12 SU. The
   token term exceeds the floor only if you sustain higher concurrency than the
   benchmark reference, so for most users token billing is a marginal top-up.
2. **A more expensive GPU always costs more to hold.** The floor is ordered by
   `w_gpu * N` across GPU types. The per-token rate is not ordered the same way —
   a fast enough H200 can have a lower per-1000-token cost, because throughput is
   in the denominator — so the floor, not the token rate, is the cross-type
   invariant.

The practical takeaway: the main cost lever is how long you hold the GPUs. Model
choice and prompt length are second-order; session duration is first-order.

**Output tokens cost more than input tokens, and the ratio is measured, not
hand-set.** Prefill (reading your prompt) is compute-bound and fast; decode
(generating the reply) is memory-bandwidth-bound and slow, so `decode_tps` is well
below `prefill_tps`. Each output token therefore costs
`alpha = prefill_tps / decode_tps` times an input token. The measured alpha ranges
from 2.1 (Qwen2.5-72B on H100) to 4.9 (Qwen3-4B on A100) across the benchmarked
configurations below. Commercial APIs price output 5-6 times input for the same
physical reason.

## Measured rates

The rate table holds one record per model-and-GPU configuration. Five records are
populated (`billing/rate_table.json`, last updated 2026-06-10). The floor column
is `w_gpu * N`, the SU cost per hour of holding that configuration.

| Model key           | GPU type | GPUs (N) | Prefill (tok/s) | Decode (tok/s) | Floor (SU/h) |
|---------------------|----------|---------:|----------------:|---------------:|-------------:|
| `qwen2.5_72B`       | a100     |        4 |         2901.16 |        1122.71 |          4.0 |
| `qwen2.5_72B`       | h100     |        4 |         3787.33 |        1810.50 |          8.0 |
| `qwen2.5_72B`       | h200     |        2 |         7593.69 |        2328.72 |          6.0 |
| `qwen2.5_coder_32B` | a100     |        2 |         4772.81 |        1679.03 |          2.0 |
| `qwen3_4b`          | a100     |        1 |        20063.28 |        4128.74 |          1.0 |

Provenance: all five records were benchmarked with vLLM 0.10.2 at dtype bfloat16, at
concurrency 64 over three request profiles (prefill-heavy, decode-heavy, balanced),
using the same serve flags production uses. Nodes and dates: the two a100 80 GB
records on midway3-0377 (Qwen2.5-72B on 2026-06-02, Qwen2.5-Coder-32B on
2026-06-10), h100 (H100 NVL) on midway3-0426 on 2026-06-02, h200 on midway3-0605 on
2026-06-02, and the qwen3_4b anchor on an A100-PCIE-40GB on midway3-0294 on
2026-06-02. Each record in the rate table carries the full provenance block (GPU
name, node, serve flags, profile parameters, a metrics cross-check, timestamp).

These are aggregate throughputs at concurrency 64, the basis for the token charge —
not the single-stream speed one interactive user perceives. Session start commands
and model choice are covered in [Coding Sessions](coding/overview.md) and
[Getting Started](getting-started.md).

A model-and-GPU configuration with no record bills the floor only; the token term
is reported as UNRATED in the billing summary.

## Worked example

One request with `T_in = 2000` prompt tokens and `T_out = 500` completion tokens to
Qwen2.5-72B, on a 2-hour reservation. Token terms are computed from the measured
rates in the table above.

| Configuration           | w_gpu x N | Token term (one request) | 2 h floor | Billed  |
|-------------------------|----------:|-------------------------:|----------:|--------:|
| Qwen2.5-72B, 2 x H200   |       6.0 |                0.0008 SU |   12.0 SU | 12.0 SU |
| Qwen2.5-72B, 4 x A100   |       4.0 |                0.0013 SU |    8.0 SU |  8.0 SU |

The floor is the charge; the token term is a rounding error, about one
ten-thousandth of the bill, for a single request. The H200 session costs more than
the A100 session for the same work because it costs more to hold — the intended
behavior of the type multiplier.

The everyday number: the default coding session (`qwen2.5_coder_32B` on two A100s)
costs **2.0 SU per hour held**. A three-hour afternoon of coding is 6.0 SU
regardless of how many requests you send, unless your request volume is high enough
for the token term to exceed the floor.

## Edge cases

| Case | Billed | Detail |
|------|--------|--------|
| Failed / 5xx request | No | The response carries no `usage` field and the engine's success counter is not incremented. |
| Cancelled streaming request | Partially | `T_in` is billed in full (prefill happened) plus the output tokens actually produced. The gateway asks the engine for per-request counts on streaming requests; without them, billing falls back to the engine's own session-total counters. |
| Prefix caching | Yes, in full | Full `prompt_tokens` are billed with no cache discount in Phase 1 — the engine reports the full prompt regardless of cache hits, matching OpenAI's practice. Since the floor dominates interactive use, caching lets you do more per reservation rather than pay less. A discount is a possible Phase-2 knob. |
| No rate record for the configuration | Floor only | The token term is reported UNRATED until the benchmark is run for that configuration; you are billed the reservation floor. |

## Reading your usage summary

`ai-session stop` meters the session and prints the itemized charge as its last
output, so it cannot scroll off:

```text
==============================================================
  SU CHARGE -- this session
    BILLED : <billed_su> SU      basis=<floor|tokens>
    model  : <model_key> on <n> x <GPU type> GPU   (weight <w> SU per GPU-hour)
    usage  : held <hours> h   tokens in=<T_in> out=<T_out> (<n> requests)
    session: <session id>
    receipt: <path to the summary JSON>
==============================================================
```

The same data is written to a `*_summary.json` receipt file under your state
directory. The summary contains:

| Field | Meaning |
|-------|---------|
| `token_source` | Where the token counts came from: per-request usage recorded by the gateway for the session window, or the engine's own session-total counters as the fallback. |
| `total_input_tokens`, `total_output_tokens` | The token counts billed. |
| `token_su` | The token term of the formula. |
| `floor_su` | The reservation floor. |
| `billed_su` | The charge: the maximum of the two terms. |
| `basis` | Which term set the charge: `floor` or `tokens`. For an unrated configuration the basis names the missing rate record. |
| `crosscheck` | Present when both per-request usage and the engine counters are available: the two are reconciled and a relative mismatch over 2% is flagged (`reconciled: false`). |

To re-print any past receipt, run this **on the login node**:

```bash
ai-session receipt
```

This renders your newest receipt; pass a specific summary-file path
(`ai-session receipt <file>`) to render an older one. If there are no receipts it
prints:

```text
  SU CHARGE: none this run (no active session was billed).
```

If a session fails before it is metered, or a charge looks wrong, see
[Troubleshooting](troubleshooting.md).

## Central accounting (for staff)

Your own receipt above is a convenience copy. Each session's charge is also
recorded to a staff-only ledger (`AISESSION_BILLING_DIR`), readable only by the
`rcc-staff` group. Two records may exist per session: one written when you end the
session (it carries the token detail), and one written by a staff sweep that
reconstructs the reservation-floor charge directly from the cluster scheduler's
own accounting records. The sweep is the authoritative floor because those
elapsed-time records cannot be edited by users, so a charge is recorded even for a
session that was never ended cleanly. Running the sweep and the ledger layout are
described in the operator guide.
