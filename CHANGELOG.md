# Changelog

## Unreleased — model-refresh (branch milestone/model-refresh)

### Gemma-4-31B-it added as a SECOND coding option (2026-08-20)

Not a replacement. `qwen3.8_27B` remains the `code` preset default; Gemma is reached with
`ai-session code --model gemma4_31B`. Four models served.

**Why it is not the default despite scoring higher.** Gemma took Gate-2 at 66.67% (40/60)
against the 27B's 50.00% on the identical frozen LCB-60 subset, decode and harness — but the
frozen decode pins `enable_thinking:false`, which is Gemma's NATIVE default and a suppression
of Qwen3.8's `xhigh` default. The 27B's 50.00% is therefore a lower bound and the 16.67-point
gap is not like-for-like. A thinking-on Qwen rerun is owed before any swap is justified.

**Wiring.** `gemma4_31B` in `MODEL_REGISTRY` + `PHASE1_SERVED`; TP=2; tool parser `gemma4`;
reasoning parser `gemma4`; serving env routed to `vllm-serve-cu129` (0.10.2 predates Gemma 4
entirely); `READY_TIMEOUT` 1800.

**Tier pin is `"a40|a100"` — lowercase deliberately.** Uppercase `A100` is midway3's 80GB
cards, which sit in PI-owned partitions; per the node-ownership rule in CLAUDE.md those must
not be a default target. Lowercase `a40`/`a100` are beagle3 consortium hardware, 22 nodes each,
owned by nobody and reachable by ordinary users.

**A40 is both the cheaper AND the roomier tier for this model** — the reverse of the usual
ordering, because Gemma is heavy (58.25 GiB, 30.38 GiB/GPU at TP=2) and an A40 card (46 GiB)
is larger than a 40GB A100. MEASURED KV headroom at TP=2:
  a40  (46 GiB)  9.17 GiB KV = 51,325 tokens, 3.13x concurrency @16K, tier weight 0.5
  a100 (40 GiB)  4.71 GiB KV = 26,342 tokens, 1.61x concurrency @16K, tier weight 1.0
The a100 figure is tight: 1.61x means barely more than one full-context request at a time.

**Rate row measured on a40** (job 53710675): prefill 2301.17, decode 348.67, alpha 6.60,
su_per_1k_out 0.000797. Decode is modest — A40 is slow silicon, running eager, with limited
batching — but the COST is the best in the fleet: ~36% cheaper per output token than
`qwen3.8_27B` on h100 (0.000797 vs 0.001242) despite decoding 2.6x slower, because the tier
weight is 0.5 against 2.0. Floor is 1.0 SU/h vs 4.0. An a100 row is still measuring.

**Thinking measured, not assumed** (job 53587542, new `tools/thinking_probe.{py,sbatch}`).
The `gemma4` reasoning parser correctly separates the chain of thought into `reasoning`,
leaving the answer alone in `content` — verified, not inferred; a broken parser would have
silently contaminated every downstream score. Cost of enabling thinking: **10.4x tokens and
11.8x wall time on an easy prompt, 5.1x/5.3x on a hard one**, with content length essentially
unchanged (618 vs 616 chars on the hard prompt). Thinking spends roughly constant absolute
effort regardless of difficulty, so the multiplier is worst on trivial requests -- the same
pathology as Qwen's `xhigh`, though far smaller in absolute terms. Under GPU-time billing
that is a 5-12x cost multiplier, so off-by-default is correct and the launcher now prints
the measured cost when a user starts the model.

**Hardware coverage complete for both coding models.** Gemma passes TP=2 on H100 NVL, A100
80GB, A100 40GB and A40 (jobs 53544338, 53544339, 53586878, 53586877); Qwen3.8-27B passes the
same four. Neither is restricted to hardware ordinary users cannot reach.

Docs: both coding models in the model/hardware/licence tables, plus a "Choosing between the
two coding models" section giving the honest trade — Qwen thinks by default and suits hard
problems; Gemma is cheaper and faster by default and scored higher on our benchmark, with the
comparison caveat stated.

### Fleet consolidation, hardware coverage, and a new measured leader (2026-08-19, later)

Follow-on to the coding cutover earlier the same day. Everything below is MEASURED.

**Gemma-4-31B-it: Gate-2 PASS at 66.67% (40/60), +40.00 pts over the frozen baseline and
+16.67 over the Qwen3.8-27B incumbent adopted hours earlier.** Same frozen LCB-60 subset,
SHA and content fingerprint, same greedy `enable_thinking:false` decode, same harness and
serve env. By difficulty hard 14/30 (vs the 27B's 6/30), medium 14/18, easy 12/12 — the only
model to sweep easy. Generation 53546905 plus tail resume 53554776 (`resume_kept=56`), score
53554777. At ~2.8 sigma this is outside the n=60 noise band, unlike the 27B-vs-122B gap.
Staged from `google/gemma-4-31B-it` (Apache-2.0, UNGATED), 58.25 GiB BF16, Gate-D byte-exact.
Serves TP=2 on H100 (51.89 GiB KV) and A100 80GB (40.49 GiB KV, 226,696 tokens, 13.8x
concurrency). Tool calling works with vLLM's `gemma4` parser; `functiongemma` returns nothing.
Caveat: the frozen decode is Gemma's NATIVE default but suppresses Qwen3.8's `xhigh`, so the
27B's 50.00% is a lower bound and a thinking-on rerun is owed before this is called final.

**Qwen3.8-27B now verified on every GPU tier users can reach.** H100 NVL, A100 80GB (SXM4 and
PCIe), A100 40GB (beagle3), and A40 — all PASS at TP=2. `constraint_for_model` moved from an
H100 pin to `"A100|a100"`: H100 exists ONLY in PI-owned partitions, so the H100 pin made the
default coding model unstartable for most users. A100 is reachable via beagle3 (22 nodes) and
the open `gpu` partition, and halves the floor (tier weight 1.0 vs 2.0). A40 also passes, at
0.5. NOTE: the rate row is still h100-only, so an A100 session bills the floor until an a100
row is measured — still half the H100 floor, so a net saving either way.

**Single-env consolidation measured.** All three served models now have vLLM 0.26.0 rate rows,
and 0.26.0 is faster on both re-measured models: qwen3_4b a100 20063/4129 -> 22167/5114
(+10.5% prefill, +23.9% decode); qwen2.5_72B a100 2901/1123 -> 2914/1191. Two needed a pinned
flag to get there, both now mirrored into the launcher so production matches the measured row:
  - `qwen3.8_27B` (and the whole qwen3.5 family) forced to `--enforce-eager`. With CUDA graphs
    on, vLLM auto-selects the FlashInfer `mnnvl` allreduce and `trtllm_mnnvl_allreduce_fusion`
    dies with an illegal memory access inside `profile_cudagraph_memory` (job 53537347). The
    engine never becomes ready, so a user would hold GPUs through the readiness timeout and be
    floor-billed for a server that never served. Costs decode (894.71 tok/s, alpha 10.02 —
    out of family); `pass_config.fuse_allreduce_rms=false` is the untested recovery path.
  - `qwen2.5_72B` pinned to `--gpu-memory-utilization 0.86`. At 0.90 it OOMs during cudagraph
    capture on 0.26.0 by 76 MiB (job 53539312) where 0.10.2 was fine; 0.86 clears it with ~3 GiB
    of margin and measured slightly FASTER than the old row.

**Models deleted (total 1.48 TB reclaimed today, 1.79 TB -> 315 GB).** This entry adds
Qwen3.6-35B-A3B (67G, unregistered and superseded), Meta-Llama-3.1-70B (263G: 132G of
unreferenced `original/*.pth` duplicates plus the 132G model). Llama had ZERO recorded sessions
in the central ledger, no rate row, and was the only licence-gated model here; note it is also
the only deletion that is NOT a free re-download, since Meta gates the weights. Its removal
touched the launcher tool-parser arm, the browser-demo timeout arm, the whole Llama 3.1 licence
section in the docs, and four model tables. `_LICENSE_GATED` is retained, unused, for the next
restrictively-licensed model.

**H200 removed as a serving tier.** The h200 rate row is dropped and no model may pin that
tier. `qwen3.5_122B` is re-pinned to `"H100|H200"` — it is native FP8 (E4M3, block-wise
128x128, 96.8% of weights) and FP8 needs Hopper tensor cores, so it must never fall through to
the preset's A100 default and floor-bill on a doomed load. It is validated at TP=2 on BOTH
Hopper tiers (H200 job 53069683; H100 NVL job 53538328, 20.85 GiB KV = 1,016,978 tokens) but
stays unserved pending a rate row. Docs now state plainly that only groups owning Hopper
hardware can ever start it, and that `qwen3.8_27B` is the better choice anyway.

**Production breakage found and fixed.** `bin/ai-session::do_code` still exported
`MODEL=qwen2.5_coder_32B` after that model was deleted — `ai-session code` would have failed
outright. Four further stale references in the same sweep: the usage text, `run_coding_agent.sh`,
`run_browser_demo.sh`'s READY_TIMEOUT arm, and a `server.py` comment.

**Harness hardening, each from a failure that cost a reservation.**
  - Persistent Triton cache at `/project/rcc/mehta5/.serve-cache/triton` (was per-job `$TMPDIR`,
    so every job recompiled from scratch). A cold sm80 cache blew vLLM's 600s engine-ready
    timeout and killed the first Qwen A100 smoke (53539504); `VLLM_ENGINE_READY_TIMEOUT_S`
    raised to 2400 alongside. The retry then passed in 7m54s.
  - `$TMPDIR` namespaced to `/tmp/$USER-vllmjob/$SLURM_JOB_ID`. The node epilog sweeps
    `/tmp/$USER*`, so a concurrent job of ours finishing on the same node deleted a running
    benchmark's tmpdir; FlashAttention died mid-run (`flash_attn.py:1014`) after 27 of 60
    problems, leaving 33 HTTP 500s that would have scored as wrong answers.
  - GPU-visibility guard: abort before loading weights if `nvidia-smi` device count != TP.
    Slurm reported `gres/gpu=2` while the cgroup exposed 1 on beagle3-0029 (job 53539505).
  - `bench_billing.sbatch` routes the env by model key, with `FORCE_CU129=1` to measure a
    0.10.2-era model against 0.26.0; `toolcall_probe.sbatch` parameterised by model/parsers.
  - `serve_cu129.sbatch` gained a `gemma4` reasoning-parser arm.

**CLAUDE.md: node-ownership rule.** The H200 restriction generalised — a staff-accessible
partition spans most of the cluster, so a bare `--constraint` can silently land on another
group's hardware. Adds the `scontrol show node | grep Partitions=` pre-submit check, a table of
known per-node ownership, and a note not to use an elevated QOS (`beagle3-prio`, priority
100,000,000) to jump the queue on shared hardware.

**RTX 6000 nodes settled:** `Quadro RTX 6000, 24576 MiB, compute capability 7.5` (job 53544135)
— Turing, not the RTX PRO 6000 Blackwell. fp16-only, no bf16, no FlashAttention 2/3, 24 GB.
Unusable for every model here, confirming the existing `excluded_tiers` entry. The open `gpu`
partition therefore offers exactly ONE usable node (midway3-0294, A100 40GB) to this service.

### Coding incumbent cutover — Qwen3.8-27B replaces Qwen2.5-Coder-32B (2026-08-19)

Measured on the frozen 60-problem LiveCodeBench subset (`subset_sha256=b3c2b753…7021b`,
greedy, `enable_thinking:false`, `max_tokens` 8192, concurrency 1), scored by the same
harness on `caslake`, `prereg-check` PASS before scoring. Score job 53531932.

- **Qwen3.8-27B: 50.00% (30/60) vs baseline 26.67% (16/60), +23.33 pts.** By difficulty
  hard 6/30, medium 13/18, easy 11/12. Best of six candidates; beats `qwen3.5_122B`
  (45.00%) at 44% of its footprint, though that 5-pt gap is inside the n=60 noise band
  (SE ≈ 6 pts near p=0.5). Generation job 53496745.
- Also measured this round, previously unrecorded: Qwen3.6-35B-A3B 43.33%,
  Qwen3-Coder-Next 36.67% (both Gate-2 PASS).

**Tool calling — a defect found and fixed before it shipped.** `launch_ai_session.sh`
mapped `qwen3*` to the `hermes` tool parser. Qwen3.8-27B emits tool calls in XML
(`<tool_call><function=NAME><parameter=K>v</parameter></function></tool_call>`); hermes
`json.loads()` the text between the tags and structurally cannot parse it, which would
have reproduced the silent empty-`tool_calls` failure that broke opencode on
Qwen2.5-Coder. MEASURED job 53534097 across three parsers: `hermes`
MODEL_EMITS_PARSER_MISMATCH (0 calls, raw XML in content); `qwen3_coder` and `qwen3_xml`
PARSER_WORKS (correct name+args, no spurious call on a no-tool prompt). The glob is now
split so the 3.5/3.6/3.8 family gets `qwen3_coder` while `qwen3_32B`, whose template
emits JSON, kept `hermes` — folding them together would have broken a served model.
The `AGENTS.md` tool-tag workaround is obsolete for this model and documented as such.

**Serving env now follows the model.** `launch_ai_session.sh` hardcoded the vLLM 0.10.2
`vllm-probe` env, which cannot load `Qwen3_5ForConditionalGeneration`. Serving the 27B
there would have reserved GPUs, failed on load, and floor-billed. The launcher now routes
the Qwen3.5-family to `vllm-serve-cu129` (0.26.0); every other model keeps 0.10.2.

**MTP speculative decoding verified available** (not enabled). vLLM 0.26.0 resolves
`Qwen3_5MTP` from our checkpoint's in-tree draft head (`mtp_num_hidden_layers: 1`).
A/B on 2×H100, job 53533204: 90.36% draft acceptance (375/415), 23.22 vs 15.17 tok/s
(+53%) under `--enforce-eager`. NOT enabled for benchmarking — it perturbed greedy output
on 1 of 3 prompts, and the benchmark's value is that it is controlled by construction.
Its place is the serving path, where a ~35% GPU-time reduction is a direct SU saving;
that needs a rate measurement and a check against vLLM #46249 (MTP + tool calls) first.

**Models deleted from disk (1.18 TB reclaimed, 1.79 TB → 644 GB).** GLM-5.2-FP8 (704 GB,
never servable: 755 GB exceeds one 4×H200 node and the multi-node launcher does not
exist), DeepSeek-V4-Flash (149 GB, Gate-1 NO-GO), Qwen3-Coder-Next (149 GB),
Qwen2.5-Coder-32B (62 GB), Qwen3-32B (62 GB), Qwen3-Coder-30B-A3B (57 GB). Configs,
index files, and licenses for the two never-served checkpoints are preserved under
`_scratch/tombstones/` with a re-staging recipe.

**Consequences recorded, not papered over:**
- `qwen3.8_27B` has **no `rate_table.json` row**, so coding sessions now bill the
  reservation FLOOR with no token-metered component. `qwen2.5_coder_32B` held one of only
  three honest rate rows. A `bench_billing.py` run on the cu129 env is owed and is now
  the highest-priority open item.
- The frozen benchmark baseline's weights are gone. The 26.67% anchor survives only as
  `benchmark/frozen_baseline/stage2_score_bench-coder32b.json`, which
  `score_stage2.sbatch` still reuses for adjudication — but it can never be regenerated
  or re-run under a new condition. Benchmark evidence was moved out of gitignored
  `_scratch/` into tracked `benchmark/frozen_baseline/` before any weights were deleted.
- `tools/stage2_score.py:26` still reads `BASELINE_KEY = "qwen2.5_coder_32B"` and was
  deliberately NOT edited: `prereg-check` verifies harness constants against the frozen
  `prereg.md`, and editing it in place is exactly the post-hoc change that gate exists to
  catch. Promoting 50.00% to the baseline needs a NEW pre-registration for the next round.
- 50.00% was measured with thinking OFF, which is **not** this model's default mode
  (`reasoning_effort: xhigh`). The figure is a lower bound; a thinking-on run is unrun.
- `qwen2.5_coder_32B` was the only model an external user (ndtrung) had ever run.

Docs updated across 15 pages, including inverting the tool-calling guidance that told
users to avoid the coding model for agent work. `mkdocs build --strict` passes.

### Stage 3 (service wiring, Gate 3) — Tier-B winner wired branch-local; production cutover deferred to operator (2026-08-05)

Wires the measured Tier-B result into the service on the branch and hands the one step the
loop must not self-trigger to the operator. No production default is flipped.

- **Registry / TP reconciliation.** `server.py`'s `qwen3.5_122B` comment now records the
  validated state (Gate-1 job 53069683: serves FP8 **TP=2** on 2×H200 under vLLM 0.26.0;
  Gate-2 Tier-B coding winner) in place of the stale "not smoke-tested" note, and the TP
  pin is corrected from 4 to the measured **2** in `bin/ai-session::tp_for_model` and the
  inline `MODEL_REGISTRY` comment. FP8 fits TP=2 on 2×H200, so a staff smoke reserves two
  GPUs, not four. No production-served model's config changes — only the not-yet-served
  122B (`qwen2.5_72B` and `llama3.1_70B` stay at TP=4).
- **Docs.** `docs/reference.md` no longer lists the 122B as "smoke test pending"; it is
  validated on H200 and the measured Tier-B coding winner, with the production cutover
  marked operator-pending (model table, prose, and capability-frame marker).
- **Kept baseline at Tier A.** Per the Stage-2 NO-GO, `qwen2.5_coder_32B` remains the
  Tier-A (A100) coding model; nothing is rewired there.
- **Verdict.** `prompts/60_model_refresh/verdict.md` records the per-stage outcomes, the
  honest measured numbers, and a compute-provenance table for every `mrefresh-nest*` job.

**OPERATOR-DECISION-PENDING — production cutover of `qwen3.5_122B` (why the loop stops here).**
Adding `qwen3.5_122B` to `PHASE1_SERVED` and giving it a billing `rate_table` row is
deliberately NOT done — both would be unsafe or dishonest from inside the loop:
  1. The production launcher (`launch_ai_session.sh`) hardcodes the 0.10.2 `vllm-probe`
     serving env, where `Qwen3_5MoeForConditionalGeneration` does not load. Adding the key
     to `PHASE1_SERVED` today would let a user reserve 2×H200, fail on load, and **floor-bill**
     (the job name `qwen3.5_122B:port` is swept by `billing_sweep.py::model_key_of`). A safe
     flip first requires routing production serving of this model to the `vllm-serve-cu129`
     (0.26.0) env — a production-runtime change outside the loop's authority.
  2. A `rate_table` row is the billing source of truth (prefill/decode throughput measured
     by `bench_billing.py` on the serve env). The Stage-2 measurement is **pass@1** at
     concurrency 1, not a throughput sweep, so it cannot fill a row — inventing one would be
     a fabricated billing number (forbidden). A valid row needs a `bench_billing.py` run on
     the 0.26.0 serve env (a rate re-benchmark), which the loop must not self-trigger.
The operator runbook is in `prompts/60_model_refresh/verdict.md`. Until it is done, the
branch leaves `PHASE1_SERVED` and `rate_table.json` untouched.

### Tier A & B Stage 2 (raw code-gen, Gate 2) — measured 2026-08-05, frozen 60-problem LiveCodeBench subset

Greedy pass@1 head-to-head against the incumbent `qwen2.5_coder_32B` (Qwen2.5-Coder-32B-Instruct,
the model this refresh replaces). Every model — baseline and both candidates — served on
`vllm-serve-cu129` (vLLM 0.26.0) under the identical frozen DECODE (`prereg.md` §3: greedy,
`enable_thinking:false`, `max_tokens` 8192) and scored by the same harness on `caslake`;
`prereg-check` PASS before every scoring run (subset SHA, decode, and the +3.0-pt margin were
frozen and committed before the first score). Baseline measured 26.67% (16/60) on this subset.

- **Tier B — Qwen3.5-122B-A10B-FP8: Gate-2 PASS, +18.33 pts (45.00% vs 26.67%).** 27/60 vs
  16/60; by difficulty hard 7/30 vs 0/30, medium 10/18 vs 6/18, easy 10/12 vs 10/12 (tie).
  Clears the +3.0 margin decisively and coherently — the 2026 122B pulls ahead across the
  medium/hard tail where the late-2024 32B solves none of the 30 hard problems, and ties on
  easy. Both runs complete (n_infra=0). Score job `mrefresh-nest-score` 53083189; generations
  53080066 (122B, concurrency-1 resume that completed the arc195/arc196 tail) and 53057094
  (baseline). **Gate-2 outcome: adopt Qwen3.5-122B as the Tier-B (H200) served coding model —
  Stage 3 wires it (registry/rate-table/docs, TP reconciliation), gauntlet-gated.**
- **Tier A — Qwen3-Coder-30B-A3B-Instruct: Gate-2 NO-GO, −1.67 pts (25.00% vs 26.67%).** 15/60
  vs 16/60; hard 1/30 vs 0/30, medium 5/18 vs 6/18, easy 9/12 vs 10/12. The smaller
  30.5B/3.3B-active MoE does not beat the 32B dense incumbent on raw code-gen, so the
  pre-registered fail-branch applies (`session_start.md` §2): **keep the baseline
  `qwen2.5_coder_32B` at Tier A.** A valid measured finding, not a blocker. Both runs complete
  (n_infra=0). Score job 53088886; generations 53057093 + arc196_d resume 53083190
  (resumed_kept=59), baseline 53057094.

Honest caveat (disclosed in `prereg.md` §5/§9): n=60 greedy is a point estimate; a 3-pt gap is
within binomial noise (SE ≈ 6 pts near p=0.4). Tier B's +18.33 is far outside that band; Tier A's
−1.67 is comfortably a non-win. The window (2025-03/04) post-dates the baseline's training cutoff
(fair to the baseline) and skews hard, so absolute pass rates are low for all models — but the
COMPARISON is controlled by construction (same subset, decode, harness, and serve env/version).

Hardware/quant asymmetry (Tier B only, architecturally forced — not in `prereg.md`, disclosed here):
the Tier-B candidate is served on 2×H200 (FP8) because Qwen3.5-122B does not fit the A100/BF16 tier,
while the single shared incumbent baseline was measured on A100 (BF16). So the Tier-B head-to-head
compares each model on the hardware it would actually be served on (deployment-realistic), not on
identical silicon; Tier A is fully hardware-matched (candidate and baseline both A100/BF16). This is
implausible as the driver of the +18.33 result — the gap is 11 problems with the baseline solving
0/30 hard (a capability gap), far beyond any cross-GPU/quant numerics effect, which shifts at most
~1 greedy problem. An H200-served baseline was not separately measured (out of loop scope; an
operator wanting silicon-matched Tier-B numbers can commission one).

### Tier B Stage 1 (serves) — measured 2026-08-05, vLLM 0.26.0 (vllm-serve-cu129), H200 driver 535.216.03

- **Qwen3.5-122B-A10B-FP8: Gate-1 PASS.** Serves on 2×H200 at TP=2 (FP8), 58.24 GiB
  weight/GPU plus 62.89 GiB KV cache, engine init 76 s; a text-only code prompt returns
  a clean compilable `two_sum` (HTTP 200, finish_reason=stop). Job mrefresh-nest-stage1
  53069683, served-name `bench-qwen35-122b` (a benchmark-only name, not a MODEL_REGISTRY
  key — no production discovery, no billing sweep). Measurement: FP8 needs only TP=2 on
  2×H200, not the TP=4 the service currently pins (`bin/ai-session::tp_for_model`, the case
  arm for `qwen3.5_122B`; `server.py`'s MODEL_REGISTRY entry only mirrors it in a comment);
  Stage 3 reconciles. Raw-code-gen vs the baseline is the next gate.
- **DeepSeek-V4-Flash: clean Gate-1 NO-GO on this cluster** (the pre-registered
  fail-branch, `session_start.md` §2). With `--kv-cache-dtype fp8` (its fp8_ds_mla layout
  requires it) the load clears arch and kv-cache and reaches expert quantization, where
  the NVFP4 experts resolve to the Marlin MXFP4 MoE backend (Hopper has no FP4 tensor
  cores) and the Marlin FP4 repack aborts with `cudaErrorUnsupportedPtxVersion`
  (`marlin_utils_fp4.py::_repack_marlin_experts`): that kernel's PTX targets a newer CUDA
  toolchain than driver 535 can load. Deterministic — not a §9 infra-retry. Tier B
  proceeds on Qwen3.5-122B alone. Job 53069684; two H200 reservations spent (53061901
  config gap, 53069684 definitive), no further retries. A speculative Hopper-viable path
  remains — force the Triton MXFP4 MoE backend (`--kernel-config`) or upgrade the cluster
  driver/CUDA toolkit — but it is unproven (may hit the same PTX wall, and V4's expert
  config is untested there) and needs a harness change plus a GPU validation, so it cannot
  overturn this NO-GO; recorded for a future operator decision.

### Serve-env hardening (enabling the above)

- `tools/serve_cu129.sbatch`: redirect all serve-time caches off the quota-limited home
  fileset (vLLM cache to /project; XDG/Triton/Inductor/Torch/FlashInfer to node-local
  $TMPDIR) after a full home crashed a 122B load with `[Errno 122] Disk quota exceeded`;
  auto-add `--kv-cache-dtype fp8` for DeepSeek-V4* model dirs. Frozen DECODE unchanged;
  billing-floor, production, and time-box fences intact (commits 0ced4b0, 67e432e).

## 2026-07-09 — consistency-audit fixes (four-way adversarial review)

### Correctness (Tier 1)

- `--agent` now works on `chat` and `fast`, not only `code`: `bin/ai-session`
  exports `AGENT_CLIENT=1` and `run_browser_demo.sh` passes `--agent-client`
  through to `ai_session.py start`. This makes the documented autonomous
  reference-tool use in browser chat real; the tools docs (`getting-started.md`,
  `faq.md`, `reference.md`) now state which tools are UI-orchestrated (web
  search, URL fetch — any model) and which need `--agent` plus a tool-calling
  model (the model placing reference-tool calls itself).
- mcpo lifecycle hardening: `run_browser_demo.sh` writes its pidfile
  incrementally (a UI that never binds no longer leaves `stop` on the wrong
  teardown branch), adds `MCPO_PORT` to the `down` port-owner backstop, and
  tears the whole stack down if `up` fails after the GPU session is active
  (trap installed only once OUR session is confirmed, so a refused `start`
  can never end a pre-existing session). `run_openwebui.sh` uses a writable
  standalone `RUN_DIR` (`~/.ai-session/state/run`, mkdir'ed), refuses to point
  the UI at a foreign listener on `MCPO_PORT`, and liveness-checks mcpo before
  exporting `TOOL_SERVER_CONNECTIONS`.
- `server.py`: the `qwen3.5_122B` comment now gives the real smoke-test
  command (`--constraint H200` spelled out — `start --force` alone would land
  FP8 on an A100 and floor-bill a failed load).
- `aider_model_metadata.json`: added `qwen3_4b` and `qwen3_32B` entries
  (32768-token context, both key forms) so the documented
  `code --model qwen3_*` paths do not trip litellm's unknown-context warning;
  fixed the stale "8192 context" comment in `run_coding_agent.sh`.

### Pre-existing items (Tier 4, user-approved)

- `run_browser_demo.sh` gains the partial-staging guard `run_coding_agent.sh`
  already had (a half-downloaded `--model` no longer submits and floor-bills).
- `su_usage_mcp.py` usage-dir fallbacks now include the wrappers' default
  `$HOME/.ai-session/state/logs/usage`, so an MCP server launched without
  `AISESSION_STATE_DIR` finds the receipts.
- `billing.md` w_gpu table: the A100 row no longer reads "A100-40GB" — one tier
  for both memory sizes, with a note that sessions run on 80 GB nodes while the
  `qwen3_4b` rate record was measured on a 40 GB PCIe card (policy yaml comment
  aligned; no charged value changed).
- Minor doc drifts: opencode page now distinguishes the config file from the
  one workaround file (matching the FAQ); the front-page support section now
  matches the FAQ's single RCC ticket channel with an `ai-session` routing
  hint; the FAQ's "vision model on the roadmap" line replaced with
  "text-only today; ask the operators". Not done (deliberately): GLM-5.2 stays
  out of `MODEL_REGISTRY` until a serving path exists; the billing-sweep
  crontab stays uninstalled; the dated "verified" lines stay.

### Docs (Tiers 2–3)

- Doc↔code fixes: `-J` tunnel described as fallback (not "equivalent");
  `qwen3.5_122B` described as staged (weights on disk, awaiting H200
  validation) instead of "coming"; H200 described as present hardware;
  `qwen3_32B` no longer "being staged" on the opencode page; the front-page
  data-location statement now points at the `AISESSION_TOOLS` opt-in
  exception; session key length corrected to 32 hex characters;
  `ai_session.py connect` prints the single-hop tunnel with the full
  `.rcc.uchicago.edu` host.
- Prose pass to the house style: frame-of-reference table rewritten with plain
  comparatives (no "frontier-adjacent", `MoE`/BFCL-V4 expanded, one caveat
  instead of three, dropped the self-contradicting cost sentence); Llama
  phrasing now cites the Community License instead of "free to run";
  removed "natural fit" / "three practical advantages" / "answer better";
  "data residency" → "Data location".

## 2026-07-07 — LoRA serving, `ai-session mcp`, `module load opencode`, de-jargon pass 2

### LoRA adapter serving (new)

- `ai-session chat|code|fast --lora NAME=PATH` (repeatable) serves a user's
  PEFT/LoRA adapter alongside the base model; requests with `model=NAME` get the
  adapter, `model=<base key>` the base. Every adapter is validated on the login
  node before any GPU is reserved (`adapter_config.json` present, absolute
  path, rank <= 256, warn on base-model mismatch); `MAX_LORA_RANK` is computed
  from the adapters. `launch_ai_session.sh` gains env-gated `ENABLE_LORA` /
  `LORA_MODULES` / `MAX_LORA_RANK` (flags `--enable-lora` / `--lora-modules` /
  `--max-lora-rank`) and passes `--enable-lora ...` to `vllm serve`. Static
  registration only; billing stance matches agent mode (floor-billed,
  rate-table records unchanged). New user page `docs/lora.md`; operator section
  in `ai-session/README.md`; training-path design in
  `ai-session/LORA_TRAINING_DESIGN.md` (recommendation: recipe on the user's
  own allocation first, managed `tune` verb only if demand shows).
- Launcher fix: standalone `--tp N` now sizes the GPU request (`gpu:N`);
  previously GRES was resolved before flag parsing, so `--tp 2` still reserved
  4 GPUs. The `ai_session.py` production path always set GRES explicitly and
  was not affected.

### MCP access without paths (new)

- `ai-session mcp run jobs|usage` execs the two read-only MCP servers with the
  right interpreter and state directory; `ai-session mcp config` prints the
  ready-to-paste opencode block. Agent configurations no longer contain any
  install path or per-server environment block; `docs/coding/mcp.md` rewritten
  accordingly.

### opencode as a module (new)

- Shared install `/project/rcc/mehta5/opencode/1.14.41/bin/opencode` (the
  verified version) + repo modulefile `modulefiles/opencode/1.14.41`, deployed
  as a symlink under `/project/rcc/mehta5/modulefiles/opencode/`. On the
  cluster the docs now say `module load opencode`; the laptop install path
  remains documented.

### Docs

- De-jargon pass 2 over the user pages: "gateway" occurrences roughly halved
  and defined in passing where kept ("the small always-on relay on the login
  node"); "reverse proxy" removed; "OpenAI-compatible" reduced to one defining
  use per page; the serving software's name removed from user-flow prose;
  remaining scheduler vocabulary cleared outside quoted output. New pages:
  `docs/lora.md`; FAQ gains a support-contact entry; command reference gains
  `--lora` and the `mcp` verbs.
- `IMPLEMENTATION_ROADMAP.md`: status update + next-stage tiers (A: smoke/
  reaper/Tier-3 install; B: embeddings, FIM, JSON mode, LoRA recipe; C:
  reasoning/vision catalog; D: decision-gated).

## 2026-07-06 — module packaging, `ai-session` CLI, jargon-free user docs

Implemented the plan in `HANDOFF_UX_DOCS.md`: packaging first, then the
documentation rewrite against the new commands.

### Packaging (new)

- `bin/ai-session`: user-facing dispatcher. Verbs: `chat` (Qwen2.5 72B, browser
  UI), `code` (Qwen2.5 Coder 32B, aider/opencode/Continue; `--agent` enables
  tool calling), `fast` (Qwen3 4B), `status`, `connect`, `env`, `models`,
  `receipt`, `stop`. Start verbs accept `--time HH:MM:SS` and `--model KEY`
  (the GPU configuration is chosen from the model). Operator env overrides
  (`MODEL`/`TP`/`CONSTRAINT`/`TIME`/`GW_PORT`/...) pass through to the wrappers
  unchanged. `status` reads the gateway's keyless `/status` route and reports
  READY / STARTING / none, plus the access-key state and session uptime.
- `bin/aider`: symlink to the shared aider install, so plain `aider` works after
  `module load ai-session`.
- `modulefiles/ai-session/1.0`: the Tcl modulefile is now version-controlled in
  the repository; the deployed copy at
  `/project/rcc/mehta5/modulefiles/ai-session/1.0` is a symlink to it (no
  drift). Verified end to end: `module use /project/rcc/mehta5/modulefiles &&
  module load ai-session && ai-session status`.
- Endpoint via environment, mirroring Sherlock's ollama pattern: session start,
  `connect`, and `env` write `~/.ai-session/env` (mode 600) with
  `AISESSION_BASE_URL`, `AISESSION_API_KEY`, `AISESSION_MODEL`;
  `eval "$(ai-session env)"` loads them. `ai-session/opencode.example.json` now
  uses `{env:AISESSION_BASE_URL}` / `{env:AISESSION_API_KEY}` (no hand-editing);
  the documented aider and Open WebUI commands use the same variables. The
  existing `examples/agent_pydantic.py` already read exactly these two
  variables.

### Documentation (rewritten against the new commands)

- All twelve pages under `docs/` rewritten or edited so a user meets only
  `module load ai-session`, the `ai-session` verbs, and plain language:
  "session", "model server", "GPU node", "session time limit", "GPU type",
  "hours held". Slurm, sbatch/squeue/scancel/sacct, tensor-parallel/TP, gres,
  partition, constraint, and walltime no longer appear in user-flow pages, and
  no user page contains the install path.
- Remaining scheduler mentions are confined to `docs/coding/mcp.md`, which
  documents the job-queue MCP server and is explicitly framed as such: the
  server file name `slurm_mcp.py`, the read-only query commands it wraps
  (`squeue`, `sacct`, `sinfo`, `scontrol show`), the `scancel` string inside an
  input-validation example, and three literal script paths inside `opencode.json`
  command arrays (JSON cannot expand shell variables).
- `docs/index.md` gained a "first five minutes" quickstart;
  `docs/getting-started.md` and `docs/coding/overview.md` now begin with the
  module-load step; `docs/reference.md` was restructured as the `ai-session`
  reference with a "For administrators" pointer; the raw launcher and wrapper
  machinery lives in `ai-session/README.md` (operator guide), which gained a
  "User packaging" section documenting the module, dispatcher, env file, and
  install symlink.
- `ai-session/CODING_AGENTS.md` snippets modernized to the same pattern
  (`ai-session code`, `eval "$(ai-session env)"`, `$AISESSION_*` variables);
  the 2026-07-03 verification record in section 8.1 kept verbatim.
- `mkdocs.yml` site description no longer says "Slurm-launched".

### Behavior fixes (inconsistencies found while implementing)

- `ai_session.py connect` defaulted `--gateway-port` to 8080, which never
  matched the wrappers' UID-derived port (`8400 + UID % 90`), so `connect`
  printed a URL no gateway was listening on. It now defaults to `$GW_PORT`,
  else the derived per-user port; the dispatcher also passes the port
  explicitly.
- `print_su_receipt.py` banner de-jargoned: the model line now reads
  `<model> on <n> x <type> GPU (weight <w> SU per GPU-hour)` instead of
  `/ TP=<tp> (N=.. w_gpu=..)`, `reserved` is now `held`, and `job:` is now
  `session:`. The receipt examples in the docs match the real output (verified
  against an actual receipt).
- The handoff described `opencode.example.json` as carrying a `<GW_PORT>`
  placeholder; the file actually had the verification user's literal port 8450
  hard-coded. Superseded by the env-based config, which removed the hazard.
- The interim modulefile's help text advertised commands that did not exist
  (`bin/` was not built). The commands now exist and the help text lists the
  full verb set.
- `docs/licenses.md` showed a `--force` start example that the new user CLI
  cannot execute (the dispatcher never passes `--force`); the page now
  describes the gate (`ACCEPT_LLAMA_LICENSE=1`) and points operators at the
  advanced launcher instead of showing a command that would be refused.

### Verification

- `pytest billing/` — 22 passed (before and after).
- `mkdocs build --strict` — exit 0.
- Gateway import check (`gateway.build_app(require_key='x')`) — passes.
- `module load` + every read-only verb (`help`, `models`, `status`, `env`,
  `connect`, `receipt`) exercised on the login node. The start verbs were not
  run: they submit GPU jobs, which requires explicit approval.
- `squeue -u mehta5` — no stray jobs at session start or end.

### Not done (needs a decision or GPU time)

- Goal 2 extras: wiring the idle reaper into cron/systemd (start command needs
  operator approval), a one-command laptop connect.
- Goal 3: embeddings endpoint (#7), FIM autocomplete (#13), structured-output
  verification (#8), reasoning/vision models (#18/#19) — each needs a GPU
  benchmark and SU approval; literal LoRA fine-tuning (#24) needs a design
  decision.
- Tier 3 module install: ask RCC to symlink the modulefile into
  `/software/modulefiles`.
