# Frozen pre-registration — coding-model refresh on the vLLM 0.26.0 stack

Status: PRE-RESULT. The gate **formulas**, the candidate set, the partition routing, and the honest
fail-branches below are FROZEN. A pilot may set the numeric **constants** once (§Constants) and commit them as
its own pre-result commit; nothing here is back-fit after a gate scores. This document is authoritative for the
loop `prompts/_cluster_autonomous_model_refresh_loop.md`; do not re-derive its gates in the loop prompt.

## 0. Goal and the one load-bearing discipline

Replace the stale served coding model (`qwen2.5_coder_32B`, Qwen2.5-Coder-32B-Instruct, late-2024) with a
fresher model, chosen per serving tier on a **measured** raw-code-gen number, on the new `vllm-qwen35` env
(vLLM 0.26.0). Two tiers (ratified with the operator 2026-08-03):
- **Tier A — single A100 node** (4×A100-80GB, no FP8): candidate `Qwen/Qwen3-Coder-30B-A3B-Instruct` (30.5B/3.3B
  active MoE), BF16, TP=2. Deployable on today's hardware.
- **Tier B — single H200 node** (4×H200-141GB, native FP8): candidates `Qwen3.5-122B-A10B-FP8` (already staged,
  TP=2) and `deepseek-ai/DeepSeek-V4-Flash` (284B/13B active, native FP4+FP8, DP=4). The research pick to beat
  Qwen3.5-122B is DeepSeek-V4-Flash — see [[project_coding_model_landscape_2026]].

**THE PARTITION-ROUTING DISCIPLINE (operator-pinned, load-bearing — stated in every stage):**
- **`build` partition = the orchestrator + all downloads.** The headless `claude` driver, git, and the subagent
  gauntlet run in the `build` allocation; every model-weight download (internet-bearing, Xet disabled) runs
  **in that same `build` allocation** (NOT a second nested `build` job — the `build` QOS is one job per user).
- **GPU partition (`test`, A100/H200) = ONLY loading a vLLM model and checking it serves/answers.** The serve
  smoke test and the benchmark inference are the *only* work that reserves a GPU. Every GPU job is
  time-boxed, constraint spelled out, and **gated by a passing CPU load-safety dry-run first** (Gate 0).
- **CPU-only (`caslake`, or ≲2-min checks in the `build` allocation) = everything else.** The offline arch
  dry-run, weight-integrity verification, benchmark scoring/aggregation, rate-table math, registry/doc edits,
  and the gauntlet need no GPU and never reserve one.

## 1. Non-negotiable fences (do NOT relax)

- **NEVER floor-bill a doomed reservation.** A model reaches a GPU only after its **Gate 0 CPU arch dry-run
  passes** — `tools/arch_dryrun.py <dir>` returns **exit code 0** (equivalently `^RESULT .*: PASS`; not the
  bare string `RESULT: PASS`). `ai_session.py start --force` on an unvalidated model reserves nodes, fails on
  load, and bills the floor — forbidden. GPU jobs run on `--partition test --account rcc-staff` (high-priority
  test queue), NOT through the SU-billed production path. **The floor-bill mechanism is
  `billing_sweep.py::model_key_of`: it bills a job ONLY if its name is `<key>:<port>` with `<key>` in
  `MODEL_REGISTRY`.** Therefore every serve names its job `mrefresh-nest-<stage>` and its served-model-name
  `bench-<model>` (NOT a registry key) — this is what keeps even the baseline `qwen2.5_coder_32B` re-serve off
  the sweep and out of production discovery (see the SERVE DISCIPLINE below).
- **Every GPU nested job is time-boxed and self-terminating** (`--time=00:30:00` smoke, `--time=02:00:00`
  benchmark). The **driver cannot `scancel`** (classifier); the **orchestrator** is the only canceller and it
  scancels any `mrefresh-nest*` job **by name** on a terminal stop (job ids are persisted to `_scratch/` only so
  a chained orchestrator can re-attach). The loop must never depend on the driver cancelling a job.

- **DRIVER CONSTRAINT (measured 2026-08-03 — `cluster_provenance.md`).** The H200 test nodes run driver
  **535.216.03 (max CUDA 12.2)**. `vllm-qwen35` (vLLM 0.26.0 → torch 2.11/**cu130**/CUDA 13.0) **FAILS to init
  CUDA** on it ("driver too old") — it is the offline-arch-resolution reference ONLY, never a GPU serve env.
  CUDA-12.x torch works (cu128 proven). **PREREQUISITE (OPERATOR-DECISION-PENDING):** the GPU serve env must be
  a **CUDA-12.x vLLM rebuild** — newest version that registers `qwen3_5_moe`+`glm_moe_dsa` (≥0.17.0) AND ships
  `cu128`/`cu129` torch. Stage 1 cannot serve until that env exists (or the operator upgrades the driver to
  ≥R580); its `torch.cuda.is_available()` pre-flight NO-GOs every serve on `vllm-qwen35` until then.

**SERVE DISCIPLINE (every Stage-1/Stage-2 serve — candidate AND baseline; closes the launcher landmine).**
NEVER invoke `ai-session/launch_ai_session.sh` — it hardcodes `ENV_PATH=vllm-probe` (0.10.2, where the new
archs cannot load — serving there would DEFEAT Gate 0) and submits a job named `<key>:<port>` (floor-billed,
surfaced by production discovery, invisible to the `mrefresh-nest*` wait/cancel). Instead, write the nested
`sbatch` yourself: `mamba activate` the **CUDA-12.x serve env** (per the DRIVER CONSTRAINT above — NOT
`vllm-qwen35`); a bare `vllm serve` with `--served-model-name bench-<model>` (NOT a `MODEL_REGISTRY` key);
`--job-name mrefresh-nest-<stage>`. The baseline `qwen2.5_coder_32B` is served the SAME way (same serve env,
name `bench-coder32b`) so the Gate-2 comparison is same-version/same-env — an env or version mismatch silently
games Gate 2.
- **Do NOT disturb production.** Use `vllm-qwen35` (0.26.0) and the `test` partition only. Never touch the
  production `vllm-probe` (0.10.2) env, the live `PHASE1_SERVED` models, or their version-pinned `rate_table`
  records. Flipping a production default (adding a key to `PHASE1_SERVED`, a new `rate_table` row) happens ONLY
  behind Gate 3 + a cleared gauntlet.
- **Git:** branch `milestone/model-refresh` off `main`, NEVER commit to `main`. Push only to `origin` if ever;
  **NEVER push to `upstream`** (the rcc-uchicago org repo — operator pushes there). Loops leave the branch for
  operator review; **do not push, do not merge.** Stage files explicitly (never `git add -A`, never stage
  `_scratch/`, `models/`, `site/`, or any weights).
- **No fudge factors.** A gate that fails is a real result or a pre-registered fail-branch — never loosen a
  threshold after seeing the number. A self-reported vendor benchmark is NOT a substitute for a measured one.

## 2. Stages, each looped to its gate (dependency order)

- **Stage 0 — LOAD-SAFETY (CPU; the anti-floor-bill guard).** For each candidate, run
  `tools/arch_dryrun.py <model_dir>` in the `vllm-qwen35` env (CPU, in-allocation on `build` or a `caslake`
  nested job). **Gate 0: exit code 0** (arch registered + model class imports + config parses + tokenizer
  loads; the probe also prints `^RESULT .*: PASS`). A candidate that fails Gate 0 is recorded as blocked and
  **never reaches a GPU**. Weights must be on disk first — if absent, that candidate's Stage 0 waits on its
  download (Stage D). NOTE: Gate 0 proves the *architecture* loads on CPU; it does NOT prove the *quantization
  kernel* (FP8-block / NVFP4) runs on the target GPU — that is a Stage-1 pre-flight (the V4-Flash/FP4-on-Hopper
  trap below).
- **Stage D — DOWNLOAD (build allocation; internet).** Stage any un-staged candidate weights with the
  Xet-disabled recipe (`HF_HUB_DISABLE_XET=1`, retry loop, shard-index verify — mirror `sbatch_stage_glm52.sh`),
  run **in the `build` allocation** as a monitored background step. **Gate D:** every shard in the safetensors
  index present + non-zero, tensor-bytes = index `total_size`, no `.incomplete` chunks. Tier-A
  `Qwen3-Coder-30B-A3B-Instruct` (~61 GB) and Tier-B `DeepSeek-V4-Flash` (~148 GB) need staging; Qwen3.5-122B is
  already staged (see [[project_h200_staging]]).
- **Stage 1 — SERVES (GPU; time-boxed; per the SERVE DISCIPLINE above).** For each Gate-0-passing candidate,
  launch one time-boxed vLLM server on its tier — Tier A `--constraint "A100|a100" --tp 2` (Slurm features are
  case-sensitive; bare `A100` is only 4 contended nodes); Tier B Qwen3.5-122B `--constraint H200` (TP measured:
  FP8 fits TP=2 on 2×H200, but the live service pins TP=4 — Stage 1 confirms which serves cleanly, Stage 3
  reconciles `bin/ai-session::tp_for_model` + `server.py`); Tier B V4-Flash `--constraint H200` with
  data-parallel (hand-written — `launch_ai_session.sh` has no DP path). Poll `/health`, then send a fixed
  text-only code prompt to `/v1/chat/completions`. **Gate 1:** the server reaches ready and returns a coherent,
  compilable answer within the time box. **Pre-flight BEFORE the full load** (so a doomed reservation fails
  cheap): `torch.cuda.is_available()` + tiny fp tensor (torch-2.11/CUDA-13 driver gap) + arch registered + **the
  quant kernel is supported on this GPU's compute capability**. **V4-Flash / FP4-on-Hopper NO-GO:** NVFP4
  compute is Blackwell (sm_100+); these H200s are Hopper (sm_90). If vLLM 0.26.0 cannot run V4-Flash's FP4 on
  sm_90, that is a **clean Gate-1 NO-GO for V4-Flash — NOT a §9 infra-retry** (do not burn 5 H200 reservations
  on it); Tier B then decides on Qwen3.5-122B alone. **VLM caveat:** Qwen3.5-122B is a vision-language MoE served
  text-only — the serve may need multimodal-limit/processor config and a text chat template; the Gate-1
  accept-check sends text only and asserts a clean completion with no multimodal input.
- **Stage 2 — RAW CODE-GEN (GPU serve + CPU score).** Run a **pinned** LiveCodeBench subset (§Constants:
  version, problem set, decoding) **identically** against the candidate and the `qwen2.5_coder_32B` baseline —
  **both served under the SERVE DISCIPLINE on the SAME `vllm-qwen35`/0.26.0 env, same decode, same harness**
  (an env/version/decode mismatch silently games the gate); score on CPU. **Gate 2:** the tier winner's measured
  pass@1 exceeds the baseline by ≥ the pre-registered margin (§Constants). Report the honest measured number; a
  vendor self-report never substitutes. **Fail-branch:** if no candidate beats the baseline by the margin at a
  tier, that tier's finding is "keep the baseline; the fresh candidate did not clear the bar" — a valid NO-GO.
- **Stage 3 — SERVICE WIRING (CPU).** For each tier winner, add its key to `MODEL_REGISTRY`/`PHASE1_SERVED`
  (`ai-session/server.py`), add a `rate_table` row from the Stage-2 measured throughput, and update the docs
  (`docs/`, `CHANGELOG.md`). Gauntlet-gated (§5). **Gate 3:** the registry/rate-table/doc edits are internally
  consistent (a fresh `ai_session.py` dry-parse loads them) and the gauntlet is clear. Never flip a production
  default without this gate.

## 3. Constants — a SEPARATE pre-result file, committed BEFORE the first scoring run (anti-back-fit)

The pilot (an autonomous ≤2-min budget-bounded run — the loop sets these itself; no human needed) writes the
constants to a **separate `prompts/60_model_refresh/prereg.md`** (header `Status: PRE-RESULT`) and commits it as
its **OWN commit** whose message notes no gate has scored. Do NOT put constants in THIS frozen file — editing
`session_start.md` to hold a number seen later is the back-fit trap, and an "ancestor of HEAD" check cannot
detect it. **Before ANY Stage-1/Stage-2 `mrefresh-nest` job scores, assert:** (i) `prereg.md` exists with the
`PRE-RESULT` header; (ii) its commit **precedes the first Stage-1/2 nested-job submit timestamp** recorded in
`_scratch/` (not merely `--is-ancestor` of HEAD, which is trivially true); (iii) `git diff --quiet HEAD --
prompts/60_model_refresh/prereg.md` (unmodified since). If any check fails, that is a §9 blocker — never create
or edit `prereg.md` post-hoc. Constants to fix:
- `LCB_VERSION` = LiveCodeBench release + `LCB_SUBSET` = the pinned problem ids (time-boxable in ≤2 h/model).
- `DECODE` = temperature/top-p/max-tokens, identical for baseline and candidate.
- `WIN_MARGIN` = required pass@1 lead over baseline (pre-register, e.g. +2.0 points absolute).
- `SMOKE_PROMPT` = the fixed Gate-1 text-only code prompt + its accept check (compiles / passes a unit assertion).
- Time boxes: smoke `00:30:00`, benchmark `02:00:00`, download in-allocation (background).

## 4. Refuted framings — do NOT re-import

- "Qwen3.5-122B is obviously the best coding model" — it is a general vision-language MoE with only
  vendor-self-reported code numbers ([[project_coding_model_landscape_2026]]); it is a *candidate*, not the
  presumed winner.
- "GLM-5.2 is the tier leader" — refuted in research (0-3), and at 753B it does not fit a single H200 node; it
  is out of scope for both tiers here.
- "A code-specialized model must beat a general MoE" — false as of mid-2026; judge on the measured number only.

## 5. Gauntlet (after every commit and at verdict)

Spawn via `Agent`. Wave A (parallel): a **config/billing reviewer** (`general-purpose`) — checks every serve /
registry / rate-table edit for the billing-floor fence (no serve job named `<registry-key>:<port>`; no
`launch_ai_session.sh` invocation; served-model-name is `bench-*`, not a registry key), the production-safety
fence (no `vllm-probe` / `PHASE1_SERVED` / production-`rate_table` disturbance; no serve name that production
discovery would surface), and the time-box/scancel discipline; a **reference/citation checker** (`Explore`) —
every `path::symbol`, model id, and cited number resolves; an **adversary** (`general-purpose`, prompted to
break the result) — "the serve ran via `launch_ai_session.sh` (so on the 0.10.2 env, defeating Gate 0) or under
a `<key>:<port>` name (floor-billed / surfaced to production); the baseline and candidate were served on
DIFFERENT envs/versions or decodes (Gate 2 gamed); a self-report was substituted for a measurement; V4-Flash's
FP4 was retried 5× on Hopper instead of taken as a Gate-1 NO-GO; a Gate-0 pass was faked; constants were set
after seeing the number; an orphaned `mrefresh-nest*` job was left in `squeue`." Feed it the gate + the
gating-run log. **Resolve:** any open Critical or an un-refuted adversary strike re-enters the inner loop before
the stage's green floor advances.
