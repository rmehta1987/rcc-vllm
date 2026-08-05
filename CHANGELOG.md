# Changelog

## Unreleased — model-refresh (branch milestone/model-refresh)

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
