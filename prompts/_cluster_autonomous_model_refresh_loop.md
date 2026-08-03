# Autonomous cluster driver: coding-model refresh on the vLLM 0.26.0 stack

You are running **autonomously on the RCC/Midway3 cluster**. There is no operator in the loop. Your job: take the
stale served coding model (`qwen2.5_coder_32B`) and **replace it, per serving tier, with a fresher model chosen on
a MEASURED raw-code-gen number** on the new `vllm-qwen35` env (vLLM 0.26.0) — stage the candidate weights, prove
each loads safely, serve it, benchmark it against the baseline, and wire the tier winner into the service.
"Everything is done by you": you stage, run the cluster jobs, decide the pre-registered gate, commit, run the
review gauntlet, and write the verdict.

This is a **Ralph-style loop** (the sbatch re-feeds this SAME prompt each iteration; your prior work persists in
files + git history; you iterate until a terminal sentinel). It is modeled on the FlowDFE cluster loops
(`/project/rcc/mehta5/L2LGWAS_DFE/prompts/_TEMPLATE_cluster_autonomous_loop.md` — read once for the
implement→test→commit + gauntlet discipline) and inherits their safeguards, adapted to a **paid GPU service**
where the load-bearing new fences are: **never floor-bill a doomed reservation, never disturb the live service,
never orphan a GPU job.**

**The frozen plan is [`prompts/60_model_refresh/session_start.md`](60_model_refresh/session_start.md) §0–§5 —
read it from disk; its gates, candidate set, partition routing, and fail-branches are authoritative and frozen.
Do NOT re-derive or back-fit them.** The completion-honesty rule (§10) is absolute.

---

## 0. Authority, invariants, fences (read before anything that reserves a GPU or edits the service)

**Authority chain for THIS project** (no single spec file — verify against these, newest fact wins):
auto-memory `MEMORY.md` + the `project_*` slugs it indexes (especially [[project_h200_staging]],
[[project_coding_model_landscape_2026]], [[project_su_billing_impl]]) → `IMPLEMENTATION_ROADMAP.md` /
`PROGRESS.md` → `ai-session/README.md` + `ai-session/BILLING_POLICY.md` → `docs/`. Note: this repo's `AGENTS.md`
is the served-coding-model tool-tag workaround, **not** a governance doc — do not treat it as authority.

**Non-negotiable fences (do NOT relax) — full text in `session_start.md §1`:**
- **NEVER floor-bill a doomed reservation.** A model reaches a GPU only after its **Gate 0 CPU arch dry-run
  passes** — the gate is `tools/arch_dryrun.py <dir>` returning **exit code 0** (equivalently the log line
  matches `^RESULT .*: PASS`; do NOT grep the bare string `RESULT: PASS`, the probe pads the colon and appends
  a suffix). Never `ai_session.py start --force` an unvalidated model. GPU jobs run on `--partition test
  --account rcc-staff`, never the SU-billed production path.
- **Every GPU nested job is time-boxed** (`--time=00:30:00` smoke / `--time=02:00:00` benchmark) and
  self-terminating. **You (the driver) cannot `scancel`** — the orchestrator sbatch is the only canceller, and it
  cancels by **scancel-ing any `mrefresh-nest*` job by name on a terminal stop** (it does NOT read ids from
  `_scratch/`; you persist ids there only so a chained orchestrator can RE-ATTACH). This is exactly why the
  SERVE DISCIPLINE (§2) requires the `mrefresh-nest-<stage>` job name — a mis-named serve is neither cancelled
  nor re-attached. Never write a step that depends on you cancelling a job.
- **Do NOT disturb production.** `vllm-qwen35` (0.26.0) + `test` partition ONLY. Never touch `vllm-probe`
  (0.10.2), the live `PHASE1_SERVED` models, or their version-pinned `rate_table` rows. A production default
  flips ONLY behind Gate 3 + a cleared gauntlet.
- **Git:** branch `milestone/model-refresh`, **NEVER `main`**; **NEVER push to `upstream`** (rcc-uchicago org).
  Do not push or merge — leave the branch for operator review. Stage files explicitly (never `git add -A`;
  never stage `_scratch/`, `models/`, weights, `site/`).
- **No fudge factors** and **no self-report substitution**: a failed gate is a real result or a pre-registered
  fail-branch; a vendor benchmark number never stands in for a measured one.

**Citation discipline:** durable claims cite `path::symbol` (or a model id / `# NOTE` marker), never `path:line`.

## 0.5 Reference fence — verify on disk, do NOT assume from memory

At orient (§1), confirm present + readable before proceeding, then re-verify each acted-on claim:
- `tools/arch_dryrun.py` (the Gate-0 load-safety probe) and the `vllm-qwen35` env
  (`/project/rcc/mehta5/conda-envs/vllm-qwen35/bin/python` imports `vllm==0.26.0`).
- `ai-session/server.py::MODEL_REGISTRY` / `::PHASE1_SERVED` (the service registry you will edit at Gate 3;
  also the set `server.py::_discover_servers_from_squeue` scans — a serve job named `<registry-key>:<port>` is
  surfaced to production clients, so your serve jobs must NOT be named that, see §4);
  **`ai-session/launch_ai_session.sh` — DO NOT invoke it for your serves (§2 SERVE DISCIPLINE).** It hardcodes
  `ENV_PATH=/project/rcc/mehta5/conda-envs/vllm-probe` (a plain `set -u` assignment — un-overridable, so it
  serves on the FORBIDDEN production 0.10.2 env where the new archs cannot load) and submits its own sbatch
  named `<MODEL_KEY>:<PORT>` (invisible to the orchestrator's `mrefresh-nest*` wait/cancel, surfaced by
  production discovery, and — for the baseline `qwen2.5_coder_32B`, a live registry key — FLOOR-BILLED by
  `billing_sweep.py::model_key_of`). Read it only to copy the bare `vllm serve` flag list.
  `billing/rate_table.json` (the version-pinned records — do NOT edit a production row).
- The staging recipe `sbatch_stage_glm52.sh` (the Xet-disabled download pattern to mirror per candidate).
- The candidate weights under `models/` (Qwen3.5-122B already staged; `Qwen3-Coder-30B-A3B-Instruct` and
  `DeepSeek-V4-Flash` need Stage D).

**HALT condition:** if a load-bearing reference is absent (the dry-run probe, the env, the registry, the
launcher), STOP: write a one-line blocker (§9), fire a `PushNotification` naming the file, do NOT proceed from
memory.

---

## 1. Phase 0 — orient + (re)confirm the frozen gates

Each iteration, in parallel: `git status --short`; `git log --oneline -8`; confirm HEAD is on
`milestone/model-refresh` and carries `prompts/60_model_refresh/session_start.md`. Confirm the `vllm-qwen35`
env imports `vllm` and that `tools/arch_dryrun.py` runs. Run the reference fence (§0.5). **Sweep `squeue --me`
for any `mrefresh-nest*` job still in flight and RE-ATTACH to it** (read its id from `_scratch/`) rather than
resubmitting. Read `session_start.md §0–§5` from disk now — the gates are frozen there; do not re-derive them.

**The gates are FROZEN in `session_start.md §2`** (Gate 0 load-safety · Gate D download-integrity · Gate 1
serves · Gate 2 raw-code-gen-beats-baseline · Gate 3 service-wiring). If a result tempts you to move a
threshold, that is the back-fit trap — don't; take the pre-registered fail-branch. The Stage-2 `WIN_MARGIN`,
`LCB_SUBSET`, and `DECODE` constants are set ONCE by the pilot and committed as a **pre-result commit**
(`session_start.md §3`); assert that commit is an ancestor of HEAD and `session_start.md` is unmodified-since
before any Stage-2 score.

---

## 2. The deliverable — stages, each looped to its gate

Walk the stages in dependency order (full spec: `session_start.md §2`). Per candidate: **Stage D download**
(build allocation) → **Stage 0 load-safety** (CPU) → **Stage 1 serves** (GPU) → **Stage 2 raw-code-gen** (GPU
serve + CPU score) → **Stage 3 service wiring** (CPU, per tier winner). Candidates:
- **Tier A:** `Qwen/Qwen3-Coder-30B-A3B-Instruct` (BF16, A100 TP=2).
- **Tier B:** `Qwen3.5-122B-A10B-FP8` (staged, H200; TP per Stage-1 measurement — the service currently pins
  TP=4 in `bin/ai-session::tp_for_model` and `server.py:32`, while FP8 fits TP=2 on 2×H200; Stage 1 confirms,
  Stage 3 reconciles both files) and `deepseek-ai/DeepSeek-V4-Flash`. **V4-Flash caveat (pre-flight NO-GO):**
  its MoE experts are **NVFP4**, and FP4 tensor-core compute is **Blackwell (sm_100+)**; these H200s are
  **Hopper (sm_90)**. Before staging its ~148 GB, verify in the `vllm-qwen35` env that (i) `DeepseekV4ForCausalLM`
  is registered AND (ii) vLLM 0.26.0 actually runs its FP4 quant on sm_90 (not Blackwell-only). If FP4 is
  unsupported on Hopper, V4-Flash is a **clean Gate-1 NO-GO on this cluster — NOT a §9 infra-retry** (do not
  burn 5 H200 reservations on a quant the hardware can't run); record the finding and Tier B decides on
  Qwen3.5-122B alone. Also: `launch_ai_session.sh` has no data-parallel path, so any DP serving is hand-written.

**SERVE DISCIPLINE (every Stage-1/Stage-2 serve, candidate AND baseline — this closes four gauntlet findings):**
1. **Env:** serve from the **CUDA-12.x rebuild** (per the DRIVER CONSTRAINT below — NOT `vllm-qwen35`, whose
   torch-2.11/cu130 fails to init CUDA on the current driver 535; `cluster_provenance.md`), and NEVER
   `launch_ai_session.sh` (it forces the 0.10.2 `vllm-probe` env). The baseline `qwen2.5_coder_32B` is served on
   the SAME env, so the Gate-2 comparison is same-version/same-env (an env mismatch silently games Gate 2).
2. **Job name:** `--job-name mrefresh-nest-<stage>` so the orchestrator's wait/cancel sees it.
3. **Served-model-name:** a benchmark-only name that is **NOT a `MODEL_REGISTRY` key** (e.g. `bench-<model>`),
   so production discovery never surfaces it AND `billing_sweep.py::model_key_of` never floor-bills it. The
   baseline serve uses `bench-coder32b`, NOT `qwen2.5_coder_32B` (the live key — which WOULD be swept).
4. Reuse only the bare `vllm serve` flag list from `launch_ai_session.sh`; write the nested `sbatch` yourself (§4).

Baseline to beat at every tier: `qwen2.5_coder_32B`. "Reuse, don't rebuild": `tools/arch_dryrun.py` and the
`sbatch_stage_glm52.sh` download recipe already exist — extend them.

---

## 3. The inner loop (implement → test → gate → commit-after-green → reset-to-last-green on fail)
1. Each stage a `TaskCreate`; `in_progress` on start, `completed` on its gate.
2. Edit with `Edit`/`Write` (never `sed`). After a meaningful edit, run the fast local check for the touched
   file (e.g. `python -c 'import ast; ast.parse(open("ai-session/server.py").read())'`, or the arch dry-run for
   a probe change) in the `build` allocation (≲2 min). **GREEN ⇒ COMMIT** one logical change by explicit path
   (the new last-green floor). **RED ⇒ fix in place, ≤5 attempts, else `git reset --hard <last-green-commit>`**
   to discard the uncommitted WIP and re-implement simpler.
3. The gate-deciding run is a **nested `sbatch`** named `--job-name=mrefresh-nest-<stage>` (§4). After
   submitting, `squeue --me -h -o '%i %j'` to confirm the name landed, **persist the job id to `_scratch/`**,
   then **end the turn** and let the orchestrator's wait-loop ride it — do NOT `Monitor`-block for hours inside
   one turn. The gate keys off the pre-registered `session_start.md` metric — never a tolerance moved after
   seeing the number. VERIFY the gating number against the run log, not memory.
4. **Failure = a real result or a real finding** (a Gate-0 dry-run FAIL → that candidate never reaches a GPU;
   a Gate-1 driver/arch mismatch; a Gate-2 shortfall → the tier's "keep the baseline" NO-GO fail-branch). Fix
   the cause; never loosen, never fabricate, never substitute a self-report.
5. **Commit after green:** one logical commit per green stage (its gate passed or its fail-branch taken + the §5
   gauntlet cleared, adversary refuted) ⇒ the new last-green floor. Every service edit (registry / rate-table /
   launcher / docs) is gauntlet-gated (§5) before its commit counts green.
6. **Reset-to-last-green:** if a stage's WIP can't go green in ≤5 attempts, `git reset --hard <last-green>` and
   re-implement simpler. Never reset past a green commit, never force-push, never amend.

---

## 4. Cluster budget — orchestrator + downloads on `build`; GPU checks on `test`; CPU work on `caslake`
This is the operator's load-bearing routing (`session_start.md §0`). Mirror `scripts/run_model_refresh_loop.sbatch`:
- **`build` (orchestrator + downloads).** The `claude` driver, git, and the gauntlet run in the `build`
  allocation. **Stage-D downloads run IN THIS SAME ALLOCATION** (internet is here; Xet disabled;
  `HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60`) as a monitored background process — **not** a second nested
  `build` job (the `build` QOS is one job per user, which the orchestrator already holds). Poll the download,
  verify the shard-index (Gate D), commit a staged-manifest line. Only ≲2-min checks run in-allocation otherwise.
- **`test` (GPU — ONLY loading a model and checking it serves).** Stage-1 smoke and Stage-2 benchmark inference
  are the only GPU work. Every GPU nested `sbatch`: `--partition test --account rcc-staff`, constraint spelled
  out — **`--constraint "A100|a100"` for Tier A** (Slurm features are case-sensitive; bare `A100` matches only 4
  contended nodes, `a100` another ~50) and **`--constraint H200` for Tier B** (uppercase is correct for the
  midway3-0600–0606 H200 nodes) — TP/DP per `session_start.md §2`, **time-boxed** (`--time=00:30:00` smoke /
  `02:00:00` benchmark), env = the **CUDA-12.x serve rebuild** (per the §2 DRIVER CONSTRAINT — the cluster driver
  is 535/CUDA-12.2, so `vllm-qwen35`/cu130 fails; `cluster_provenance.md`), the **SERVE DISCIPLINE of §2** (name
  `mrefresh-nest-<stage>`, served-model-name `bench-<model>` NOT a registry key). Run a **fast pre-flight BEFORE
  the full model load** so a doomed reservation fails cheap: `torch.cuda.is_available()` + a tiny fp tensor
  (catches the driver/CUDA gap) + the arch in `ModelRegistry.get_supported_archs()` + **a quant-support probe for
  the exact quant** (FP8 block / NVFP4) on this GPU's compute capability — the arch being registered does NOT
  mean its quant kernel runs on Hopper (the V4-Flash/FP4 trap, §2). Persist the id; end the turn.
- **`caslake` (CPU-only — everything else).** The Gate-0 arch dry-run (if not run in-allocation), benchmark
  scoring/aggregation, and rate-table math go to a right-sized `caslake` nested job — `--partition caslake
  --account rcc-staff` with **`--qos=caslake` (or omit `--qos`); NEVER `--qos=test`** (caslake rejects it) —
  `mrefresh-nest-<stage>`, time-boxed, id persisted — never a GPU.
- **VLM-serve caveat (Qwen3.5-122B).** It is a vision-language MoE served text-only for code. Its Gate-1 serve
  may need multimodal limits/processor config (e.g. cap image inputs) and a text chat template; the Gate-1
  accept-check sends a **text-only** code prompt and asserts a clean completion with no multimodal input.
- **Scrub the inherited Slurm env before each nested submit** (`--export=NONE` or an `-u SLURM*` filter) and
  **name EVERY nested job `mrefresh-nest-<stage>`** (the orchestrator's wait-loop rides ONLY `mrefresh-nest*`,
  so your other-project jobs never pause this loop, and a mis-named job of THIS loop would spawn an overlapping
  iteration). Poll with `Monitor`/an until-loop on `squeue`; never foreground-`sleep`.
- **COEXISTENCE FENCE — this loop shares the cluster with the operator's unrelated jobs.** At any moment
  `squeue --me` may list foreign jobs (e.g. `phase0-5*` on amd/bigmem/caslake) that have NOTHING to do with this
  loop. **NEVER wait on, count, cancel, or reason about any job whose name does not start with `mrefresh-nest`.**
  Scope every `squeue --me` you run with a positive name match (`squeue --me -h -o '%i %j' | awk '$2 ~
  /^mrefresh-nest/'`) — never a bare `squeue --me` job count, never `scancel` by anything but a `mrefresh-nest*`
  id you persisted. A foreign job in the queue is not a signal about this loop's state.

---

## 5. Subagent gauntlet — after EVERY commit AND at verdict
Spawn via `Agent` (§`session_start.md §5` has the roster; a cheaper model for subagents is fine, driver stays
`claude-opus-4-8 --effort max`). Wave A (parallel):
- **config/billing reviewer** (`general-purpose`) — the launcher / registry / rate-table / doc edits against the
  three fences: billing-floor (no un-dry-run GPU reservation, no `--force` on an unvalidated model), production
  safety (no `vllm-probe` / `PHASE1_SERVED` / production-`rate_table` disturbance), and time-box/scancel
  discipline (every GPU job time-boxed, id persisted, no orphan).
- **reference/citation checker** (`Explore`) — every `path::symbol`, model id, cited benchmark number, and
  `[[memory]]` link resolves to disk or the run log.
- **adversary** (`general-purpose`, prompted to BREAK it) — "a GPU job wasn't time-boxed or would floor-bill; a
  Gate-0 pass was faked; the benchmark was gamed (different decode/harness for baseline vs candidate, or a
  self-report substituted); production was disturbed; a `mrefresh-nest*` job was orphaned in `squeue`." Feed it
  the gate + the gating-run log.

**Resolve:** any open Critical or an un-refuted adversary strike → re-enter §3 before the green floor advances.
`drift-auditor`-style reconciliation of `CHANGELOG.md` / memory / docs happens before the verdict lands. The
verdict never lands with a live adversary finding, a stale doc, or an orphaned job.

---

## 6. Branch + rollback
- Work on `milestone/model-refresh` (cut fresh off `main`; the sbatch forces it). **NEVER `main`.** One logical
  commit per green stage; the verdict its own commit. **Do not push, do not `--no-verify`, do not amend, do NOT
  push to `upstream`.** Stage files explicitly.
- **Rollback:** `git reset --hard <last-green-commit>` ONLY to discard uncommitted broken WIP. Never past a
  green commit, never force-push.
- **One green stage ⇒ a FRESH session for the next** (`--continue` within a stage; fresh at each boundary; the
  next session re-reads this prompt and orients from git, §1).
- Trailers on every commit: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` + `Changelog: <imperative>
  [model-refresh]`.

---

## 7. Bookkeeping (on verdict landing)
1. A JSONL line per gate to `_scratch/model_refresh_manifest.jsonl` (gitignored/scratch): `{stage, tier, model,
   metric, value, baseline, target, pass}` — Gate-D shard-verify, Gate-1 serves, Gate-2 measured pass@1 vs
   baseline + tok/s, Gate-3 wiring.
2. `CHANGELOG.md` under `## Unreleased` (newest first): the per-tier winner (or the "kept baseline" NO-GO), the
   measured pass@1 and the baseline delta, the serve config (env, partition, TP/DP), and the measured tok/s.
3. Memory: update [[project_coding_model_landscape_2026]] with the MEASURED head-to-head (it currently records
   the *inferred* research pick) + a one-line `MEMORY.md` pointer leading with the measured winner.
4. `prompts/60_model_refresh/verdict.md`: per-stage outcomes, the gates, the honest measured numbers, and a
   compute-provenance table (every `mrefresh-nest*` job: id · partition · constraint · time · what it proves).
5. Run the §5 gauntlet on the verdict.
Follow this project's doc style ([[feedback_doc_style]]): scientist-to-scientist prose, no emoji/checkmarks,
exact commands, tables, measured numbers.

---

## 8. Escalation protocol (autonomous)
You may serve models on `test` and edit the service registry/rate-table/docs on the branch. You may **not** merge
to `main`, push, or flip a production default outside Gate 3. For anything the loop must not self-trigger (a
production cutover, a `vllm-probe` upgrade, a rate re-benchmark of a live model), flag it in `CHANGELOG.md`
"Why" + `verdict.md` as `OPERATOR-DECISION-PENDING`, fire a `PushNotification`, and proceed with the
branch-local interpretation.

---

## 9. Blocker protocol
A genuine blocker — a gate fails after a clean ≤5-attempt re-implementation; the gauntlet surfaces a Critical you
cannot resolve in scope; a required reference/env/weight is absent (§0.5); a GPU driver gap makes the torch-2.11
stack unrunnable on the H200 (rebuild-with-cu128 is an operator decision) — **STOP that stage.** Write
`prompts/60_model_refresh/blocker.md` with `path::symbol` citations + the failing assertion, commit only that
(`debug: blocker on <stage>`), fire a `PushNotification`, and either continue to an **independent** stage (Tier A
and Tier B are independent; the two Tier-B candidates are independent) or stop if all remaining stages depend on
the blocked one. **A nested GPU/CPU job that fails to COMPLETE** (OOM / wall-timeout / crash before a metric — NOT
a cleanly-decided gate FAIL) is an **infra** failure: right-size and re-submit **≤5 times total** (track the count
in `_scratch/`); on the 5th, treat as a blocker. **A Gate-2 shortfall handled via the pre-registered "keep the
baseline" fail-branch is a valid finding, NOT a blocker.** Never fabricate a pass, skip a gate, loosen a
threshold, or substitute a self-report.

---

## 10. Loop control & completion signals
Inner loop (§3) iterates within a stage until its gate; the outer loop walks the stages in dependency order.
Re-enter at §1 orient each iteration. When waiting on a nested Slurm job, poll with `Monitor` (an until-loop on
`squeue`/`sacct`) — never foreground-`sleep`; prefer ending the turn so the orchestrator's wait-loop rides it.

**Terminal signal — emit EXACTLY `=== LOOP: <NAME> ===` as a line of its own**, NAME ∈
`STAGE_LANDED` | `MODEL_REFRESH_COMPLETE` | `BLOCKED`, written **nowhere else** (never in a plan, TODO, code
block, quote, or sentence — a stray literal line is an unrecoverable premature exit; when you *discuss* one,
name it in words like "the COMPLETE sentinel"). At most one per iteration:
- **`=== LOOP: STAGE_LANDED ===`** — a stage's gate passed (or its pre-registered fail-branch was taken), its
  commit + bookkeeping landed on `milestone/model-refresh`, the §5 gauntlet cleared (no open Critical, adversary
  refuted, no orphaned `mrefresh-nest*` job), and a new last-green floor is set → a FRESH session for the next
  stage.
- **`=== LOOP: MODEL_REFRESH_COMPLETE ===`** — every candidate has walked Stage D → Stage 0 → Stage 1 → Stage 2
  to a decided gate (PASS or its pre-registered fail-branch), each tier has a decided winner-or-kept-baseline,
  Stage 3 has wired every winner, `verdict.md` records the honest measured numbers, `CHANGELOG.md` + memory
  reflect it, **and `squeue --me -h -o '%j' | grep '^mrefresh-nest'` is EMPTY (no serve/benchmark job left
  running)** — never emit COMPLETE with a live nested job. Stop.
- **`=== LOOP: BLOCKED ===`** — §9 fired and no independent stage remains. Stop.
- Otherwise (mid-work) emit no sentinel; the loop re-fires and you resume from the last green floor.

> **THE COMPLETION-HONESTY RULE (absolute).** A terminal sentinel is a promise the machinery trusts. Emit
> `MODEL_REFRESH_COMPLETE` / `BLOCKED` / `STAGE_LANDED` **only when it is completely and unequivocally TRUE** —
> NEVER to escape the loop because you feel stuck or over-budget. If stuck, the honest exit is
> `=== LOOP: BLOCKED ===` **with** a real `blocker.md` (§9), never a faked COMPLETE. A false completion ships an
> unfinished, possibly production-affecting change as done — the worst failure mode of an autonomous loop.
