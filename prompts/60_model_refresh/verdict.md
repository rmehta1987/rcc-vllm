# Verdict — coding-model refresh on the vLLM 0.26.0 stack

Status: **COMPLETE (measured).** Branch `milestone/model-refresh`. Every candidate walked Stage D →
Stage 0 → Stage 1 → Stage 2 to a decided gate; each tier has a decided winner-or-kept-baseline;
Stage 3 wired the Tier-B winner branch-local with the one non-self-triggerable step handed to the
operator (see OPERATOR-DECISION-PENDING below). The gates and fail-branches are frozen in
[`session_start.md`](session_start.md) §2; the constants are frozen in [`prereg.md`](prereg.md) and
committed (`2bb927c`/`177bc4e`) before the first score.

## 1. Outcome per tier

| Tier | Hardware | Candidate(s) | Decision | Measured pass@1 (frozen 60-problem LiveCodeBench subset) |
|---|---|---|---|---|
| **A** | 1× A100 node (BF16, no FP8) | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | **NO-GO — keep baseline `qwen2.5_coder_32B`** | 25.00% (15/60) vs baseline 26.67% (16/60) → **−1.67 pts** (< +3.0 margin) |
| **B** | 1× H200 node (native FP8) | `Qwen3.5-122B-A10B-FP8` (winner), `deepseek-ai/DeepSeek-V4-Flash` (Gate-1 NO-GO) | **WINNER — `Qwen3.5-122B-A10B-FP8`** | 45.00% (27/60) vs baseline 26.67% (16/60) → **+18.33 pts** (≥ +3.0 margin) |

Baseline both tiers: `qwen2.5_coder_32B` (Qwen2.5-Coder-32B-Instruct, late-2024), the model this
refresh set out to replace. Its single shared generation was measured on A100/BF16.

## 2. Per-stage outcomes

**Stage D — download (build allocation, Xet disabled).** Both un-staged candidates verified byte-exact
against their safetensors index (Gate D): `Qwen3-Coder-30B-A3B-Instruct` 16 shards, 61,064,245,248 B ==
`total_size`, 0 `.incomplete`; `DeepSeek-V4-Flash` 46 shards, 159,609,485,896 B == `total_size`, 0
incomplete. `Qwen3.5-122B-A10B-FP8` was already staged ([[project_h200_staging]]).

**Stage 0 — load-safety arch dry-run (`tools/arch_dryrun.py`, CPU on the `vllm-serve-cu129` env).**
Gate 0 = exit 0. All three passed: `Qwen3MoeForCausalLM` (30B-A3B), `Qwen3_5MoeForConditionalGeneration`
(122B), `DeepseekV4ForCausalLM` (V4-Flash) each registered, imported, parsed config, and loaded a
tokenizer. Gate 0 proves the *architecture* loads on CPU; it does not prove the *quant kernel* runs on
the target GPU — that is the Stage-1 pre-flight, and it is exactly where V4-Flash failed.

**Stage 1 — serves (GPU, time-boxed, SERVE DISCIPLINE).** Served via `tools/serve_cu129.sbatch` on
`vllm-serve-cu129` (vLLM 0.26.0), job-named `mrefresh-nest-stage1`, served-model-name `bench-*` (never a
`MODEL_REGISTRY` key → no production discovery, no billing sweep).
- **Tier A `Qwen3-Coder-30B-A3B-Instruct`: Gate-1 PASS.** Job 53041215; BF16 TP=2 on 2× A100-PCIE-40GB;
  load 23.72 s + warmup 15.74 s; a text-only code prompt returned a clean compilable `two_sum`
  (HTTP 200, `finish_reason=stop`).
- **Tier B `Qwen3.5-122B-A10B-FP8`: Gate-1 PASS.** Job 53069683; FP8 TP=2 on 2× H200 (driver 535.216.03,
  cc 9.0); 58.24 GiB weight/GPU + 62.89 GiB KV; engine init 76.47 s; clean `two_sum`. **Measurement:
  FP8 fits TP=2 on 2× H200**, not the TP=4 the service had pinned — reconciled in Stage 3.
- **Tier B `DeepSeek-V4-Flash`: clean Gate-1 NO-GO** (pre-registered fail-branch, `session_start.md` §2).
  Job 53069684. Its NVFP4 experts resolve to the Marlin MXFP4 MoE backend on Hopper (sm_90, no FP4
  tensor cores), and the Marlin FP4 expert repack aborts with `cudaErrorUnsupportedPtxVersion`
  (`marlin_utils_fp4.py::_repack_marlin_experts`) — that kernel's PTX targets a newer toolchain than
  driver 535 can load. Deterministic hardware/driver incompatibility, NOT a §9 infra-retry: two H200
  reservations spent (53061901 config gap, 53069684 definitive), then stopped. Tier B decides on
  `Qwen3.5-122B-A10B-FP8` alone.

**Stage 2 — raw code-gen (GPU serve + CPU score).** Frozen LiveCodeBench subset (`prereg.md`: 60 problems,
`subset_sha256 b3c2b75377…`, window 2025-03-08…2025-04-06), identical DECODE (greedy, temperature 0,
top-p 1, n=1, `max_tokens` 8192, `enable_thinking:false`, read `content` not `reasoning`), both models
served under the SERVE DISCIPLINE on the **same** `vllm-serve-cu129`/0.26.0 env, scored by the same
harness on `caslake`. `prereg-check` PASS before every scoring run. **Gate 2** = tier winner's pass@1
beats baseline by ≥ +3.0 pts.

| Model | pass@1 | solved | hard (n=30) | medium (n=18) | easy (n=12) | Δ vs baseline | Gate 2 |
|---|---|---|---|---|---|---|---|
| baseline `qwen2.5_coder_32B` | 26.67% | 16/60 | 0/30 | 6/18 | 10/12 | — | — |
| **Tier B `Qwen3.5-122B-A10B-FP8`** | **45.00%** | 27/60 | 7/30 | 10/18 | 10/12 | **+18.33** | **PASS** |
| Tier A `Qwen3-Coder-30B-A3B-Instruct` | 25.00% | 15/60 | 1/30 | 5/18 | 9/12 | −1.67 | NO-GO |

Both candidate runs completed with `n_infra=0` (no problem dropped to infra failure). Tier B's +18.33
is driven by the medium/hard tail where the late-2024 32B solves 0/30 hard problems and the 2026 122B
solves 7/30. Tier A's smaller 30.5B/3.3B-active MoE does not beat the 32B dense incumbent → the
pre-registered "keep the baseline" NO-GO, a valid measured finding, not a blocker.

**Stage 3 — service wiring (CPU, Gate 3).** Branch-local wiring of the measured facts:
- `ai-session/server.py::MODEL_REGISTRY` — the `qwen3.5_122B` comment now records the validated state
  (serves FP8 TP=2 on 2× H200 under 0.26.0; Gate-2 Tier-B coding winner) and the precise reason it is
  not yet in `PHASE1_SERVED`; the inline TP note is corrected 4 → 2.
- `bin/ai-session::tp_for_model` — `qwen3.5_122B` moved to its own `echo 2` arm (measured); `qwen2.5_72B`
  and `llama3.1_70B` stay at TP=4. `constraint_for_model` (H200) unchanged.
- `docs/reference.md` — the 122B is no longer "smoke test pending"; validated on H200, Tier-B coding
  winner, cutover operator-pending (model table, prose, capability-frame marker).
- Tier A: `qwen2.5_coder_32B` kept as the Tier-A coding model; nothing rewired.

**Gate 3** = the registry/doc edits load in a fresh `ai_session.py` dry-parse and the gauntlet is clear.
`PHASE1_SERVED` and `billing/rate_table.json` are deliberately left untouched (next section).

## 3. OPERATOR-DECISION-PENDING — production cutover of `qwen3.5_122B`

The loop does not flip a production default outside its authority. Two steps remain, both requiring an
operator decision or a rate re-benchmark the loop must not self-trigger (`session_start.md` §8):

1. **Route production serving of `qwen3.5_122B` to the 0.26.0 serve env.** The live launcher
   `ai-session/launch_ai_session.sh` hardcodes `ENV_PATH=/project/rcc/mehta5/conda-envs/vllm-probe`
   (vLLM 0.10.2), where `Qwen3_5MoeForConditionalGeneration` does not load. Adding `qwen3.5_122B` to
   `server.py::PHASE1_SERVED` **before** this route exists would let a user reserve 2× H200 on 0.10.2,
   fail on load, and floor-bill (job name `qwen3.5_122B:port` is swept by
   `ai-session/billing_sweep.py::model_key_of`). Serving it means pointing production at
   `vllm-serve-cu129` for this model — a production-runtime change.
2. **Measure a billing `rate_table` row on the serve env.** Metering computes SU from
   `prefill_tps`/`decode_tps` measured by `benchmark/bench_billing.py` under the exact serve flags and
   vLLM version. The Stage-2 number is **pass@1** at concurrency 1, not a prefill/decode throughput
   sweep — it cannot fill a row, and inventing one would be a fabricated billing number. Run
   `bench_billing.py` against `qwen3.5_122B` on the 0.26.0 env (FP8, TP=2, H200) and add the resulting
   row (`gpu_tier: h200`, `tp: 2`, `provenance.vllm_version: 0.26.0`).

Only after (1) and (2): add `qwen3.5_122B` to `PHASE1_SERVED`. Note the metering backstop bills
floor-only when the running engine version does not match a record's `provenance.vllm_version`, so the
new row's version must match whatever env production actually serves on.

## 4. Compute-provenance — every `mrefresh-nest*` job

CPU work (`caslake`/build) never reserves a GPU; GPU work (`test`) only loads a model and checks it
serves/answers. Every GPU job time-boxed and self-terminating; served-model-names are `bench-*`, so
none was surfaced to production discovery or swept by billing.

| Job id | Stage · tier | Partition · constraint | Time box | What it proves |
|---|---|---|---|---|
| (in-allocation) | Stage 0 · A/B ×3 | build/caslake · CPU | ≲2 min | arch loads on CPU (`arch_dryrun.py` exit 0) — the anti-floor-bill gate |
| (build bg) | Stage D · A + B(V4) | build · CPU (internet) | background | shard-index byte-exact (Gate D) |
| 53041215 | Stage 1 · A cand | test · `A100\|a100`, TP=2 | 00:30:00 | 30B-A3B serves BF16 on 2× A100, clean `two_sum` |
| 53069683 | Stage 1 · B cand | test · `H200`, TP=2 | 00:30:00 | 122B serves FP8 on 2× H200, clean `two_sum`; **TP=2 measurement** |
| 53069684 | Stage 1 · B (V4) | test · `H200` | 00:30:00 | V4-Flash FP4 NO-GO (Marlin PTX vs driver 535) |
| 53057093 → 53083190 | Stage 2 · A gen | test · A100 | 02:00:00 | 30B-A3B LiveCodeBench generations (+ arc196_d resume, `n_infra=0`) |
| 53057094 | Stage 2 · baseline gen | test · A100 | 02:00:00 | `qwen2.5_coder_32B` baseline generations |
| 53073647 → 53080066 | Stage 2 · B gen | test · H200 | 02:00:00 | 122B generations (+ arc195/196 resume, `n_infra=0`) |
| 53088886 | Stage 2 · A score | caslake · CPU | time-boxed | Tier-A adjudication: −1.67 → NO-GO |
| 53083189 | Stage 2 · B score | caslake · CPU | time-boxed | Tier-B adjudication: +18.33 → PASS |

Infra precursors right-sized and re-submitted within the §9 ≤5 budget (not gate FAILs): Tier-A Stage-1
53030556 (context overflow, fixed `1e68c20`); Tier-B Stage-1 53061850 (home disk-quota, fixed
`0ced4b0`/`67e432e`); Tier-B V4 53061901 (kv-cache config gap). Retry counters at verdict: Tier-A 2/5,
Tier-B 122B 2/5, V4 2/5, Stage-2 Tier-A regen 2/5.

## 5. Fences honored + honest caveats

- **No floor-bill:** every candidate passed Gate 0 before any GPU reservation; no `start --force` on an
  unvalidated model; all GPU jobs on `test`/`rcc-staff`, `bench-*` served-names, `mrefresh-nest*` job
  names. V4-Flash's FP4-on-Hopper failure was taken as the pre-registered Gate-1 NO-GO after 2
  reservations, not retried 5×.
- **Production untouched:** all serves on `vllm-serve-cu129` (0.26.0) + `test`; the 0.10.2 `vllm-probe`
  env, the live `PHASE1_SERVED` models, and their `rate_table` rows were not touched. No production
  default flipped (the 122B cutover is the operator step above).
- **Same-env/same-decode comparison:** baseline and candidates served on identical env/version, identical
  frozen DECODE and subset, scored by one harness — an env/version/decode mismatch would silently game
  Gate 2; none was introduced.
- **No fudge:** the +18.33 / −1.67 are measured; no vendor self-report substituted; constants frozen and
  committed before the first score.
- **Caveats (also in `prereg.md` §5/§9 and the changelog):** n=60 greedy is a point estimate
  (SE ≈ 6 pts near p=0.4); Tier B's +18.33 is far outside that band, Tier A's −1.67 is comfortably a
  non-win. **Hardware/quant asymmetry (Tier B, architecturally forced):** the 122B is served on H200/FP8
  because it does not fit the A100/BF16 tier, while the shared baseline was measured on A100/BF16 — a
  deployment-realistic comparison, not identical silicon. Implausible as the driver of an 11-problem gap
  where the baseline solves 0/30 hard (a capability gap, not a cross-GPU numerics effect). Tier A is
  fully hardware-matched. An H200-served baseline was out of loop scope; an operator wanting
  silicon-matched Tier-B numbers can commission one.
