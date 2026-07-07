# Changelog

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
