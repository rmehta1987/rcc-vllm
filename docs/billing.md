# Billing and Service Units

This page is the canonical statement of what an ai-session costs. It defines the
Service Unit, gives the GPU-tier multiplier table, states the charging formula, lists
the measured throughput rates the formula uses, and shows how to read the charge you
are billed. The machine-readable sources are
`/project/rcc/mehta5/vllm/billing/billing_policy.yaml` (policy numbers) and
`/project/rcc/mehta5/vllm/billing/rate_table.json` (measured rates); the published
rationale is `/project/rcc/mehta5/vllm/ai-session/BILLING_POLICY.md`. How to start
and stop sessions is documented on [Getting Started](getting-started.md) (browser
chat) and [Coding Sessions](coding/overview.md); this page only explains what those
sessions cost.

## The unit: 1 SU = 1 A100-GPU-hour

A Service Unit (SU) is the internal accounting currency of the ai-session service:
**one A100-GPU-hour costs one SU**, following the published convention at NCSA Delta
and Purdue Anvil. SUs are fair-usage accounting, not dollars; no dollar figure is
attached to them in Phase 1 (`dollar_per_su: null` in `billing_policy.yaml`).

!!! note "ai-session SUs are not RCC service units"
    The RCC user guide states that beagle3 and other private partitions do not
    consume RCC service units. The SUs on this page are the ai-session service's
    internal accounting, tracked by the service itself. They are separate from any
    RCC allocation balance and do not draw on it.

## GPU-tier multiplier w_gpu

`w_gpu` is a cost multiplier, not a performance index. Because 1 A100-GPU-hour is
1 SU, the A100 anchors at 1.0, and `w_gpu(H200) = 3.0` means one H200-GPU-hour costs
3 SU. The values below are what the service charges today; RCC staff can override
them in `/project/rcc/mehta5/vllm/billing/billing_policy.yaml`.

| GPU tier  | w_gpu | Basis |
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
Phase 1 serves only bf16-capable tiers (A40 and up).

## What a session costs

A session is a Slurm job that reserves N GPUs for its wall-clock lifetime. The
charge is the greater of two terms computed in the same unit:

```text
token_su  = w_gpu(tier) * N * (T_in / prefill_tps + T_out / decode_tps) / 3600
floor_su  = w_gpu(tier) * N * reserved_wall_hours
billed_su = max(token_su, floor_su)
```

- `T_in`, `T_out` — prompt and completion tokens, taken from the vLLM API `usage`
  field (authoritative).
- `prefill_tps`, `decode_tps` — benchmarked aggregate throughput for that exact
  (model, tier, TP) combination, from the [rate table below](#measured-rates).
- `N` — the number of GPUs the Slurm job reserves. For a single session this equals
  TP, the tensor-parallel size: the number of GPUs the model's weights are split
  across.
- `reserved_wall_hours` — how long the job held the GPUs, from `sacct` Elapsed.

The token term converts your tokens into equivalent GPU-hours and prices them in the
A100-hour currency. The floor term (the **reservation floor**) exists because the
GPUs your job reserves are held for its entire wall-clock lifetime — the scheduler
cannot reclaim idle GPUs from a running job — so you pay for the greater of the work
done and the hardware held. This matches how Delta, NERSC, TACC, and PSC bill
reserved GPU time.

!!! warning "A running session consumes SU whether or not you send requests"
    The floor accrues for every hour the session job holds its GPUs, idle or busy.
    End your session (`down` for the wrappers, `end` for the CLI) as soon as you
    stop working.

Two honest consequences of the max():

1. **The floor is usually the whole bill for interactive use.** A single request's
   token cost is on the order of 0.001 SU (see the
   [worked example](#worked-example)); a multi-hour reservation is 8-12 SU. The
   token term exceeds the floor only if you sustain higher concurrency than the
   benchmark reference, so for most users token billing is a marginal top-up.
2. **A more expensive GPU always costs more to hold.** The floor is ordered by
   `w_gpu * N` across tiers. The per-token rate is not ordered the same way — a
   fast enough H200 can have a lower per-1000-token cost, because throughput is in
   the denominator — so the floor, not the token rate, is the cross-tier invariant.

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

The rate table holds one record per (model, tier, TP) combination. Five records are
populated (`billing/rate_table.json`, last updated 2026-06-10). The floor column is
`w_gpu * TP`, the SU cost per hour of holding that configuration.

| Model key           | Tier | TP | Prefill (tok/s) | Decode (tok/s) | Floor (SU/h) |
|---------------------|------|---:|----------------:|---------------:|-------------:|
| `qwen2.5_72B`       | a100 |  4 |         2901.16 |        1122.71 |          4.0 |
| `qwen2.5_72B`       | h100 |  4 |         3787.33 |        1810.50 |          8.0 |
| `qwen2.5_72B`       | h200 |  2 |         7593.69 |        2328.72 |          6.0 |
| `qwen2.5_coder_32B` | a100 |  2 |         4772.81 |        1679.03 |          2.0 |
| `qwen3_4b`          | a100 |  1 |        20063.28 |        4128.74 |          1.0 |

Provenance: all five records were benchmarked with vLLM 0.10.2 at dtype bfloat16, at
concurrency 64 over three request profiles (prefill-heavy, decode-heavy, balanced),
using the same serve flags production uses. Nodes and dates: the two a100 80 GB
records on midway3-0377 (Qwen2.5-72B on 2026-06-02, Qwen2.5-Coder-32B on
2026-06-10), h100 (H100 NVL) on midway3-0426 on 2026-06-02, h200 on midway3-0605 on
2026-06-02, and the qwen3_4b anchor on an A100-PCIE-40GB on midway3-0294 on
2026-06-02. Each record in `rate_table.json` carries the full provenance block (GPU
name, node, serve flags, profile parameters, a /metrics cross-check, timestamp).

These are aggregate throughputs at concurrency 64, the basis for the token charge —
not the single-stream speed one interactive user perceives. Session start commands
and model choice are covered in [Coding Sessions](coding/overview.md) and
[Getting Started](getting-started.md).

A (model, tier, TP) combination with no record bills the floor only; the token term
is reported as UNRATED in the billing summary:

```text
  token SU      : UNRATED (no rate_table record -- run bench_billing.sbatch)
```

## Worked example

One request with `T_in = 2000` prompt tokens and `T_out = 500` completion tokens to
Qwen2.5-72B, on a 2-hour reservation. Token terms are computed from the measured
rates in the table above.

| Configuration            | w_gpu x N | Token term (one request) | 2 h floor | Billed  |
|--------------------------|----------:|-------------------------:|----------:|--------:|
| Qwen2.5-72B, H200, TP=2  |       6.0 |                0.0008 SU |   12.0 SU | 12.0 SU |
| Qwen2.5-72B, A100, TP=4  |       4.0 |                0.0013 SU |    8.0 SU |  8.0 SU |

The floor is the charge; the token term is a rounding error, about one
ten-thousandth of the bill, for a single request. The H200 session costs more than
the A100 session for the same work because it costs more to hold — the intended
behavior of the tier multiplier.

The everyday number: the default coding session (`qwen2.5_coder_32B`, TP=2, A100)
costs **2.0 SU per hour held**. A three-hour afternoon of coding is 6.0 SU
regardless of how many requests you send, unless your request volume is high enough
for the token term to exceed the floor.

## Edge cases

| Case | Billed | Detail |
|------|--------|--------|
| Failed / 5xx request | No | The response carries no `usage` field and the engine's `request_success_total` counter is not incremented. |
| Cancelled streaming request | Partially | `T_in` is billed in full (prefill happened) plus the output tokens actually produced. The gateway injects `stream_options.include_usage` into streaming requests so per-request counts exist; without them, billing falls back to the /metrics session delta. |
| Prefix caching | Yes, in full | Full `prompt_tokens` are billed with no cache discount in Phase 1 — vLLM reports the full prompt regardless of cache hits, matching OpenAI's practice. Since the floor dominates interactive use, caching lets you do more per reservation rather than pay less. A discount is a possible Phase-2 knob. |
| No rate record for the (model, tier, TP) | Floor only | The token term is reported UNRATED until the benchmark is run for that configuration; you are billed the reservation floor. |

## Reading your usage summary

Ending a session meters it and prints the itemized charge. For the wrapper scripts
this is the `down` subcommand (documented on their owning pages,
[Getting Started](getting-started.md) and [Coding Sessions](coding/overview.md));
for the CLI it is `ai_session.py end` (see the
[Command Reference](reference.md)). `status` and `down` themselves cost nothing —
only the running GPU session is billed.

The CLI prints a block that begins:

```text
=== ai-session billing summary ===
```

and the wrappers finish with a receipt banner so the charge cannot scroll off:

```text
==============================================================
  SU CHARGE -- this session
    BILLED : <billed_su> SU      basis=<floor|tokens>
    model  : <model_key> / <tier> / TP=<tp>   (N=<n> GPU, w_gpu=<w>)
    usage  : reserved <hours> h   tokens in=<T_in> out=<T_out> (<n> requests)
    job    : <jobid>
    receipt: <path to the summary JSON>
==============================================================
```

The same data is written to
`logs/usage/<user>_<jobid>_<ts>_summary.json` under your state directory
(`AISESSION_STATE_DIR`, default `/project/rcc/mehta5/ai-session-state/<user>`). The
summary contains:

| Field | Meaning |
|-------|---------|
| `token_source` | Where the token counts came from. Priority order: an explicit `--usage-jsonl` file, then the gateway's per-request usage log for the session window, then the /metrics delta from vLLM's own counters. |
| `total_input_tokens`, `total_output_tokens` | The token counts billed. |
| `token_su` | The token term of the formula. |
| `floor_su` | The reservation floor. |
| `billed_su` | The charge: the maximum of the two terms. |
| `basis` | Which term set the charge: `floor` or `tokens`. For an unrated configuration the basis names the missing (model, tier, TP) record. |
| `crosscheck` | Present when both per-request usage and the /metrics delta are available: the two are reconciled and a relative mismatch over 2% is flagged (`reconciled: false`). |

To re-print any past receipt, run this **on the login node** (the script is
stdlib-only Python):

```bash
/project/rcc/mehta5/conda-envs/vllm-probe/bin/python \
  /project/rcc/mehta5/vllm/ai-session/print_su_receipt.py \
  --usage-dir /project/rcc/mehta5/ai-session-state/<user>/logs/usage
```

- Replace `<user>` with your CNetID (the account you run sessions as).

This renders the newest `*_summary.json` in the directory; pass a specific summary
path instead of `--usage-dir` to render an older one. If the directory holds no
receipts it prints:

```text
  SU CHARGE: none this run (no active session was billed).
```

If a session fails before it is metered, or a charge looks wrong, see
[Troubleshooting](troubleshooting.md).

## Central accounting (for staff)

Your own receipt above is a convenience copy. Each session's charge is also recorded
to a staff-only ledger at `/project/rcc/mehta5/ai-session-billing/`
(`AISESSION_BILLING_DIR`), readable only by the `rcc-staff` group. Two records may
exist per session: one written when you end the session (it carries the token detail),
and one written by a staff sweep that reconstructs the reservation-floor charge
directly from Slurm's own accounting records. The sweep is the authoritative floor
because Slurm's elapsed-time records cannot be edited by users, so a charge is
recorded even for a session that was never ended cleanly. Running the sweep and the
ledger layout are described in the operator `README.md`.
