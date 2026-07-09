# Handoff: consistency fixes + extensions for ai-session

Status as of 2026-07-09. This is a work prompt for a new session. Part 1 is a
consistency-audit backlog (found by four independent adversarial reviewers of the
docs, code, and framework). Parts 2–3 carry the open project items and a set of
proposed extensions / research tooling to discuss and build. Read the standing
constraints first; they are not optional.

## How to use this document

Line numbers drift — verify every `file:line` against the current tree before
acting. Fix Part 1 in tier order (1 → 4). Part 2 is prior-handoff carryover. Part 3
is open-ended: propose, discuss, and prototype; nothing there is settled. Keep
`mkdocs build --strict` green throughout and match the house prose style.

## Standing constraints (do not violate)

- **Never submit a Slurm job** (`sbatch`/`srun`/`salloc`, or `ai-session
  chat|code|fast`, or a wrapper `up`) without the user's explicit go-ahead. A GPU
  session costs SU. The auto-mode classifier blocks these and blocks `scancel`; a
  prior agent still orphaned a ~4-GPU billing job. Use read-only checks (`sacct`,
  `squeue`, `sinfo` are fine) and stubs. If any agent/workflow runs, check `squeue
  -u $USER` afterward.
- **The `/software` module symlinks the working tree** (`/software/ai-session-0.1-x86_64/repo`
  -> `/project/rcc/mehta5/vllm`). Editing a file under `ai-session/` or `bin/` is
  **live immediately** for `module load ai-session` users. Commit to record, not to
  deploy — test edits with care.
- **Do not modify** `conda-envs/vllm-probe` (serving env) or `mcp-env`. New Python
  deps go in a dedicated venv — `tools-env` was created this way (mcpo +
  paper-search-mcp). `openwebui-env` is the Open WebUI venv.
- **`pkill -f` self-matches your own shell** — it has bitten this work repeatedly
  (exit 144). Kill by explicit PID (capture first) or by port owner (`ss -ltnp |
  grep :PORT`), never by a `-f` pattern that appears in your own command line.
- **Large downloads:** the login node has a per-user memory cgroup that SIGKILLs
  heavy processes (exit 137) — three concurrent HF downloads triggered it. Stage on
  the **`build` partition** (`--account rcc-staff --qos build`; internet + 184 GB
  RAM). Build QOS `MaxWall` is 6 h; the user has `scontrol` rights to extend a
  running job past it (`scontrol update jobid=<id> TimeLimit=14:00:00`).
- Keep `mkdocs build --strict` green (Pages deploy); build with
  `/project/rcc/mehta5/mkdocs-env/bin/mkdocs`.
- **Docs style:** scientist-to-scientist prose; no marketing, hype, emoji, or
  jargon/acronym soup; measured numbers; exact commands. Match the voice in the
  user's own committed edits (`git show` commits `da379e7`, `9b8f081`, `2c6247c`,
  `daeddb3`). Bold sparingly (the user uses at most one emphasis per section).
- Do not delete `vllm-v0.7.3.sif` or the Llama `original/` checkpoint.

## Current state (what is live, 2026-07-09)

- **Central module live:** `module load ai-session` works cluster-wide, no `module
  use`. Built via a `noop` build in the `midway3-software` repo (cloned at
  `/scratch/midway3/mehta5/midway3-software`) that RCC deployed to
  `/software/repo/modulefiles/ai-session/`.
- **Served models** (`ai-session/server.py` `PHASE1_SERVED`): `qwen2.5_72B` (chat),
  `qwen2.5_coder_32B` (code), `qwen3_4b` (fast), `qwen3_32B` (thinking, A100 TP=2 —
  smoke-tested OK), `llama3.1_70B` (any user, one-time `ACCEPT_LLAMA_LICENSE=1`).
- **Staged, NOT served:** `qwen3.5_122B` (Qwen3.5-122B-A10B-FP8, 127 GB, in
  `MODEL_REGISTRY` + `constraint_for_model`=H200, TP=4) — flip into `PHASE1_SERVED`
  only after an H200 smoke test passes. GLM-5.2-FP8 (755 GB) is on disk but not in
  the registry; it needs multi-node H200 serving that is not built.
- **H200 hardware** exists: `midway3-0600..0606`, 4× H200 each, in the `test`
  partition. FP8 needs Hopper (A100 cannot run it).
- **Open WebUI tools** (opt-in `AISESSION_TOOLS=1`): web search (DuckDuckGo default),
  URL fetch, and academic reference search (arXiv/bioRxiv/PubMed/… via
  `paper-search-mcp` behind `mcpo`, from `tools-env`). Verified at the component
  level; the in-browser click-through is untested.
- **Tunnel:** single-hop `ssh -N -f -L …@<login-node>.rcc.uchicago.edu` is now the
  documented default; `-J` is a fallback.
- Recent commits: `cdde563` (tools + frame-of-reference), `5bdcda1` (tunnel),
  `ae3e60f` (qwen3 wiring), `daeddb3` (module/llama/reasoning docs). See memory
  `project_central_module`, `project_h200_staging`.

---

# Part 1 — Consistency-audit backlog

Found by four cold, independent adversarial reviewers. Ranked by tier.

## Tier 1 — correctness (most from recent additions)

**1a. Browser-chat tool-calling is documented but impossible.** `docs/faq.md` and
`docs/getting-started.md` tell users `AISESSION_TOOLS=1 ai-session chat --model
qwen3_32B` gives autonomous tool-calling. But `do_chat` in `bin/ai-session` never
exports `AGENT_CLIENT=1` (only `do_code` does, `bin/ai-session:277`), so the vLLM
server starts without `--enable-auto-tool-choice --tool-call-parser`
(`launch_ai_session.sh`), and `parse_start_opts` accepts `--agent` on `chat`/`fast`
then silently discards it (`bin/ai-session:141`). **Decision + fix:** either (a) let
`do_chat`/`do_fast` honor `--agent` (small code change — makes the docs true and lets
Open WebUI use the paper-search tools via native tool-calling), and/or (b) correct
the docs: web search + URL fetch work with any model (UI-orchestrated); the
reference tools are driven by Open WebUI's own function-calling, not by `--agent`.
Recommended: do both — accept `--agent` on chat, and reword the docs to not imply
`--agent` is what enables it. Acceptance: the documented command does what it says,
or the docs describe only what the code does.

**1b. mcpo can leak and isn't robust** (`ai-session/run_openwebui.sh` +
`run_browser_demo.sh`, added this session). Problems: the browser pidfile is written
only after `wait_port` succeeds, so a UI that never binds aborts `do_up` under
`set -e` with no pidfile — then `ai-session stop` picks the *coding* teardown branch
(`bin/ai-session:382`) which reaps only `GW_PORT`, leaking mcpo + open-webui;
`MCPO_PORT` is absent from the port-owner backstop in `do_down`; mcpo start is not
liveness-checked and `MCPO_PORT` is not `port_busy`-checked (risk of pointing Open
WebUI at another user's mcpo); standalone `run_openwebui.sh` `RUN_DIR` falls back to
the unwritable install dir and is never `mkdir`ed; `mpid` in `do_down` is not
`local`. **Fix:** write the mcpo pid before Open WebUI is waited on (or record
mcpo into `browser_demo.pids`); add `MCPO_PORT` to the `do_down` port loop; check
the port + verify mcpo came up; `mkdir -p "$RUN_DIR"`; `local mpid`. Acceptance: a
failed `up` leaves nothing running; `stop` always reaps mcpo.

**1c. The documented `qwen3.5_122B` test path lands FP8-on-A100.**
`ai-session/server.py` comment says "launcher TP=4/H200," but the H200 pin is in
`bin/ai-session` `constraint_for_model` (not the launcher), and the only `--force`
route is `ai_session.py start`, whose default is `--constraint A100`
(`ai_session.py:619`). So the naive test reserves an A100 (no FP8), fails on load,
and floor-bills. **Fix:** correct the server.py comment and any doc/handoff smoke-test
line to the actual working command:
`… ai_session.py start --model qwen3.5_122B --force --tp 4 --constraint H200
--account rcc-staff --partition test --time 00:30:00 --wait`. Acceptance: the
documented test command targets H200.

**1d. `aider_model_metadata.json` covers only `qwen2.5_coder_32B` + `qwen2.5_72B`.**
But `run_coding_agent.sh:228` passes `--model-metadata-file` for any `MODEL`, and the
docs now recommend `code --model qwen3_4b` / `qwen3_32B`, which will hit litellm's
"unknown context window" warning (the very thing the file prevents). Its comment
(`run_coding_agent.sh:46`) also says "8192 context" while the file is sized 32K.
**Fix:** add `qwen3_4b` and `qwen3_32B` entries (32768 ctx); correct the comment.

## Tier 2 — doc↔code / doc-internal contradictions

- `docs/coding/overview.md:~178` still calls the `-J` jump-host tunnel "equivalent";
  it now authenticates twice and is a fallback. Reword to match the single-hop
  default (and confirm what the scripts actually print).
- `docs/index.md` calls `qwen3.5_122B` "roadmap/coming"; code + `reference.md` have it
  "staged" (registered, weights on disk). Make index say "staged, not yet servable".
- `docs/coding/opencode.md:~240` "a larger Qwen3 is being staged" — `qwen3_32B` is
  served now; update.
- Front-page "nothing leaves RCC" (`index.md`) never mentions the `AISESSION_TOOLS`
  opt-in exception; the getting-started/faq caveats do. Add a one-line pointer.
- `docs/getting-started.md:~112` says a "64-hex" session key; the wrappers mint 32
  (`openssl rand -hex 16`). Fix the number.
- `ai_session.py:~559` (`connect`) prints a tunnel with a bare `os.uname().nodename`
  (no `.rcc.uchicago.edu`) and the gateway port — inconsistent with every doc
  example and with the single-hop form. Align it.
- H200 framed as "coming online" in `index.md`/`reference.md` though `billing.md`
  charges H200 today and `overview.md` offers a 72B-on-H200 config. Reword: H200
  exists; what is pending is multi-node serving for the 755 GB models.

## Tier 3 — prose (no marketing / no jargon; match the user's voice)

Rewrite plainly, in the user's committed style:
- `docs/reference.md` frame-of-reference: drop the self-contradicting "no dollar cost
  … a sixth the token cost" sentence; replace the coinage "frontier-adjacent" and the
  horse-race "leads/trails" register with plain comparatives; collapse the three
  stacked caveats into one; expand or remove unexplained `MoE`/`BFCL-V4`.
- `docs/getting-started.md` + `faq.md` tools section: remove the heavy bolding, the
  "`paper-search-mcp` behind `mcpo`" plumbing (users don't need it), and "and more".
- "**data residency**" → the user's term "Data location" (in `getting-started.md`,
  `coding/continue.md`, `coding/aider.md`).
- "Llama 3.1 is free to run for research on university hardware" (`index.md`,
  `licenses.md`) → "the Llama 3.1 Community License permits this use; see Model
  licenses" (the license imposes conditions).
- `index.md`: "three practical advantages", "answer better", bolded roadmap product
  names; `overview.md`: "natural fit" slogan.

## Tier 4 — pre-existing (not from recent edits; confirm with user before changing)

- `docs/billing.md` w_gpu table lists only `A100-40GB` (1.0 anchor) though everything
  bills 80 GB cards and `qwen3_4b` was benchmarked on a 40 GB PCIe node. Reconcile
  the table with what is charged.
- `run_browser_demo.sh` `do_up` lacks the partial-staging guard that
  `run_coding_agent.sh:141` has (a half-downloaded `--model` submits + floor-bills).
- `ai-session/mcp/su_usage_mcp.py:79-83` fallback paths are stale — they omit the
  current default `$HOME/.ai-session/state/logs/usage`, so an MCP server launched
  without `AISESSION_STATE_DIR` finds no receipts.
- Minor: opencode "workaround file" count (1 vs 2); support-channel framing
  (`index.md` two channels vs `faq.md` one); vision model mentioned only in `faq.md`;
  GLM-5.2 weights on disk but absent from the registry and docs.
- The dated "verified 2026-07-03 / benchmarked 2026-06-02" lines in
  opencode/troubleshooting/continue: the user asked to avoid job-verification
  sections — leave these unless the user says otherwise.

---

# Part 2 — Open items from prior handoffs

See `HANDOFF_MULTIUSER_READINESS.md` for full detail. Still open:

- **Item 2 (ports):** per-user gateway/UI ports are UID-derived (`8400 + UID%90`),
  predictable and collision-prone; a co-tenant can squat the port and harvest the
  shared session key. Move to ephemeral free ports recorded in `~/.ai-session/env`.
- **Item 3 (billing sweep) — verified, NOT scheduled.** `billing_sweep.py` works
  (read-only `sacct`, idempotent) but no cron/timer runs it, so non-staff sessions
  never reach the central ledger. A ready crontab is staged at
  `scratchpad/ai-session.crontab` (needs `PATH` for the Slurm bins). Install was
  blocked pending the user's go-ahead.
- **Item 5 (spend cap):** billing is accounting-only; no ceiling, no central reaper.
  Decide whether Slurm enforces per-account GPU-hour limits; if not, add a max
  `--time`, a concurrent-session cap, and/or a staff-run reaper.
- **Item 6 (durability):** code/venvs/models/ledger live under one person's
  allocation; ~19 files hardcode `/project/rcc/mehta5`. Relocate to a service account
  or `/software` (the module is registered but its `repo` handle is still a symlink
  to the personal tree). The user has accepted this gap for now.

---

# Part 3 — Extensions, ideas, and research tools (open discussion)

Nothing here is committed; propose and discuss with the user. Curated, not
exhaustive.

**Research / reference tooling (build on the tools work):**
- **Self-hosted SearXNG** as the default `WEB_SEARCH_ENGINE` so web-search queries go
  through a proxy the service controls (narrows the data-residency exposure vs. an
  external API). Weigh the cost of running the SearXNG service on the login node.
- **Local embedding model for RAG** so Open WebUI "knowledge" (document upload,
  project-file RAG) works without the `OFFLINE_MODE=False` outbound HuggingFace
  download — stage an embedding checkpoint and point `RAG_EMBEDDING_*` at it.
- **First-class `ai-session mcp` verbs** for the research tools (mirror the existing
  `su_usage`/`slurm` MCP servers) so aider/opencode agents — not just Open WebUI —
  can search papers. `paper-search-mcp` already speaks MCP stdio.
- **Citation / bibliography helpers:** Crossref/DOI resolution, BibTeX export,
  Zotero. `paper-search-mcp` already covers Crossref/OpenAlex/Unpaywall.

**Serving / models:**
- **Multi-node vLLM for GLM-5.2-FP8** (755 GB, 2× H200): Ray cluster or TP×PP across
  nodes, 2-node reservations, a launcher path that isn't single-node `srun`. This is
  the real blocker to serving the frontier-adjacent model already on disk.
- **Flip `qwen3.5_122B` to served** once the H200 smoke test passes (single node,
  TP=4).
- **A vision model** (mentioned as roadmap in the FAQ but nowhere else): pick one,
  wire multimodal serving + an Open WebUI image path.
- **Reasoning visibility toggle** (`REASONING_INLINE=1`) to inline `<think>` for
  browser chat where the clean split isn't wanted (documented earlier as an option).

**Framework / operability:**
- **Make `do_chat` honor `--agent`** (Tier 1a) so browser chat can do reliable
  tool-calling — this unlocks agentic Open WebUI use.
- **Auto-generated model catalog page** from `MODEL_REGISTRY` + `rate_table.json`, so
  the docs model tables can't drift from code (this audit found several drifts).
- **Spend guardrails** (Item 5) and the **scheduled billing sweep** (Item 3).
- **Usage dashboard** from the central ledger (per-user SU, model mix).

**Quality process:**
- The four-way adversarial audit that produced Part 1 is worth re-running after big
  changes (docs↔code, docs-internal, prose, code/framework). Cheap and it caught real
  bugs in freshly written code.

---

# Suggested order

1. Part 1 Tier 1 (correctness — mostly in recently-shipped code).
2. Part 1 Tier 2 + Tier 3 (doc↔code + prose) in one docs pass; keep `--strict` green.
3. Part 1 Tier 4 + Part 2 items as the user directs.
4. Part 3: discuss and pick an extension to prototype.

Pointers: memory index at
`/home/mehta5/.claude/projects/-project-rcc-mehta5-vllm/memory/MEMORY.md`; prior
handoffs `HANDOFF_MULTIUSER_READINESS.md` and the `docs: handoff …` commits; key
paths `ai-session/`, `bin/ai-session`, `docs/`, `billing/`, venvs under
`/project/rcc/mehta5/{conda-envs/vllm-probe,mcp-env,tools-env,openwebui-env}`.
