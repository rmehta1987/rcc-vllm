# Handoff — ai-session SU billing service (next session)

**Paste this whole file as your first message in the new session.**

You are continuing work on the **ai-session** HPC LLM-serving service with tiered
SU billing, under `/project/rcc/mehta5/vllm/` on the UChicago RCC cluster
(account `rcc-staff`, partition `test`). The service is built and proven; **all three
72B serving tiers (A100/H100/H200) are now benchmarked and the real-model e2e is
done.** Only OPTIONAL cross-checks (A40, Llama-3.1-70B, H100-NVL TP=2) remain.

## Orient first (read these, in order)
- Memory: `project_su_billing_impl.md` (done/pending split), `project_su_billing_design.md`
  (the billing formula + reservation floor), `project_hpc_llm_state.md`.
- Plain-language build+cost notes: `/project/rcc/mehta5/vllm/ai_session_notes.md`.
- Env is `/project/rcc/mehta5/conda-envs/vllm-probe` (vLLM 0.10.2). **Do NOT
  pip/conda install into it** — re-installing is the one thing that breaks it.
  Run pytest with `/software/python-anaconda-2020.11-el8-x86_64/bin/python -m pytest billing/`.

## STEP 1 — H200 TP=2 benchmark: DONE & GREEN (2026-06-02)
Job 50399550 (Qwen2.5-72B, **TP=2 on H200**, node midway3-0605) COMPLETED in
11m23s. Log: `benchmark/logs/bench_billing-50399550.out`. Record merged into
`billing/rate_table.json` as key `('qwen2.5_72B','h200',2)`:
- prefill_tps **7593.69**, decode_tps **2328.72** (α≈3.26); tier normalized to
  `h200` correctly, `w_gpu=3.0` applied. TTFT p50 2.14s, TPOT p50 39.6ms.
- su_per_1k_in **0.0002195**, su_per_1k_out **0.0007157**.
- `/metrics` cross-check reconciles (deltas ≥ bench sums); prefill > decode ✓.
Notes (`ai_session_notes.md`) + memory (`project_su_billing_impl.md`) already
updated for this run. **Nothing to do here — it's a finished record.**

Key cross-tier finding (72B): **H200 TP=2 vs A100 TP=4** —
- **Holding** costs MORE on H200: 6.0 SU/hr (3.0×2) vs 4.0 SU/hr (1.0×4).
- **Per output token** costs LESS on H200: 0.000716 vs 0.000990 SU/1k (~0.72×),
  because 2 H200s decode ~2.07× faster than 4 A100s, beating the 1.5× w_gpu bump.
- So: short/interactive use → A100 (lower floor); sustained high-throughput → H200.

## STEP 2 — H100 tier benchmark: DONE & GREEN (2026-06-02)
Job 50401160 (Qwen2.5-72B, **TP=4 on H100**, node midway3-0426 = H100 **NVL 94GB**)
COMPLETED in 13m48s (~14m queue wait for 4 free cards). Record in `rate_table.json`
as `('qwen2.5_72B','h100',4)`: prefill **3787.33**, decode **1810.5** (α≈2.09),
w_gpu=2.0, su_per_1k_in **0.000587** / su_per_1k_out **0.001227**, prefill>decode ✓,
crosscheck reconciles. **Used TP=4, not TP=2** — 72B's ~145GB weights don't fit two
80GB H100s (2×80×0.9≈144GB); they fit on H100 NVL at TP=2, but the H100 tier is
heterogeneous (some 80GB nodes), so TP=4 is the safe production config. Cross-tier
finding (now in `ai_session_notes.md` Tables 1–2): **for 72B the H100 tier is
dominated** — holds dearest (8.0 SU/hr, 4 cards) AND priciest per token (0.001227) →
A100 = value/interactive (floor 4.0/hr), H200 = throughput (cheapest/token).
**Nothing to do here — finished record.**

Remaining OPTIONAL benchmark cross-checks (each costs SU — get user OK):
- **A40** (lowercase `a40`, w_gpu=0.5) and a **Llama-3.1-70B** cross-check.
- An H100 **NVL**-only TP=2 record (idle 80GB H100 nodes won't fit 72B at TP=2).
  Same `bench_billing.sbatch`, override `--constraint`/`--gres`/`TP` (gres count == TP).

## STEP 3 — Real-model 72B e2e: DONE & GREEN (2026-06-02)
Session 50400678 (Qwen2.5-72B **TP=2 on H200** midway3-0605) ran the full service:
gateway on midway3-login3:8421 (env python), 14 mixed requests (9 normal, 5 streaming,
incl. two ~7390-tok prompts) via `ai-session/e2e_send_requests.py`. Gateway-captured
usage (17260 in / 1668 out) == `/metrics` delta **exactly** (`reconciled=True`, no
warmup drift). Billed **0.5667 SU**, basis=floor (token term RATED off the h200
record, dwarfed). Node freed (sacct CANCELLED, 00:05:41). The whole service is proven
start-to-finish on the real 72B + priciest tier with a rated token term.
**Nothing to do here.**

## Optional polish (cosmetic, non-blocking)
- ~~`gateway.py`: replace deprecated FastAPI `on_event` with lifespan handlers.~~
  **DONE 2026-06-03** — now uses an `@asynccontextmanager lifespan`; gateway starts
  with zero DeprecationWarnings (verified via hermetic MockTransport smoke + live start).
- `bench_billing.py` provenance `serve_flags` records a placeholder `--port 0`.

## Invariants / traps (do not trip these)
- Prod serve flags in `launch_ai_session.sh` MUST match
  `bench_billing.py:build_serve_flags` — the rate record is only valid for the
  measured serve config.
- `rate_table.json` stores raw `prefill_tps`/`decode_tps`; metering recomputes
  `su_per_1k` with the ACTUAL reserved N, so whole-node N≠TP works without
  regenerating the table.
- Case-sensitive features: `a100`(40GB)/`A100`(80GB), `a40`, `H100`/`H200`/`L40S`.
  72B needs 80GB cards on the A100 tier (TP=4).
- **H100 tier is heterogeneous**: midway3-0426 is H100 **NVL (94GB)**; other H100 nodes
  (e.g. midway3-0432) are 80GB. 72B (~145GB) needs **TP=4** to be safe on any H100
  (won't fit 2×80GB; fits 2×94GB NVL but you can't rely on landing on NVL). H200
  (141GB) and `A100`(80GB) tiers serve 72B at TP=2 / TP=4 respectively.
- Do **NOT** add `from __future__ import annotations` to `gateway.py` (breaks
  FastAPI `Request` annotation resolution → 422).
- Do **NOT** delete `vllm-v0.7.3.sif` or the Llama `original/` checkpoint without
  asking — user said keep them.
- `bench_billing.sbatch` `--output` dir must pre-exist (`benchmark/logs/` does now).

## Where things stand (as of 2026-06-02)
- DONE: billing math (22/22 pytest), gateway (httpx MockTransport tests), full CLI,
  launcher, clients decision (Open WebUI + aider; opencode via `--agent-client`).
- MEASURED rates in `rate_table.json` (**4 records**): `qwen3_4b/a100/1`,
  `qwen2.5_72B/a100/4`, `qwen2.5_72B/h100/4`, `qwen2.5_72B/h200/2` (all 2026-06-02).
- E2E dry run (Qwen3-4B) GREEN **and** real-model e2e (Qwen2.5-72B on H200) GREEN:
  gateway↔compute reachable, gateway usage == `/metrics` exactly, floor-billed, freed.
- PENDING: only OPTIONAL cross-checks — A40 tier, Llama-3.1-70B, H100-NVL TP=2.
- Cross-tier 72B economics (measured): hold SU/hr A100=4.0 < H200=6.0 < H100=8.0;
  per-1k-out SU H200=0.000716 < A100=0.000990 < H100=0.001227. A100=interactive value,
  H200=throughput value, H100=dominated for this model.
