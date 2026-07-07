# Handoff — ai-session: UX polish, natural docs, pipeline continuation

**STATUS 2026-07-07: LARGELY DONE — see `IMPLEMENTATION_ROADMAP.md` ("Status
update and next-stage tiers") for the current plan.** Delivered since this was
written: the `bin/ai-session` dispatcher + in-repo modulefile (2026-07-06), the
jargon-free docs rewrite plus a second de-jargon pass (2026-07-07),
`module load opencode`, `ai-session mcp {config,run}`, and LoRA adapter serving
(`--lora`). Still open from this file: the Tier-3 central module install (RCC
action), scheduling the idle reaper, and Goal-3 pipeline items (embeddings, FIM,
JSON mode, LoRA training recipe).

**Paste this whole file as your first message in the new session.**

You are continuing work on the **ai-session** service under `/project/rcc/mehta5/vllm`
on the UChicago RCC cluster: Slurm-launched vLLM sessions serving open Qwen/Llama
models behind a stable per-user OpenAI-compatible gateway, with tiered Service-Unit
(SU) billing. The service is built, hardened, documented, and published. This phase
has three goals, in priority order:

1. **Rewrite the user-facing documentation so it reads naturally and hides the
   HPC/Slurm plumbing.** Users should think in terms of "start a session, get a URL,
   use it, stop it" — not Slurm jobs, tensor-parallel degree, partitions, or GPU
   constraints. This is the top priority and is mostly mechanical + editorial.
2. **Make the service more intuitive for users** — simpler commands, friendlier model
   choices, clearer status, less setup friction.
3. **Continue building and fine-tuning the vLLM pipeline** — the remaining roadmap
   capabilities and, if the user wants literal model fine-tuning, a LoRA path.

Cross-cutting requirement underpinning goals 1 and 2: **users must reach the service
with `module load ai-session`, never by typing `/project/rcc/mehta5/vllm/...`.** A
hard-coded project path in a user page is the same kind of leak as a Slurm term. Build
the module and clean commands FIRST (see "Packaging" below), then rewrite the docs
against them.

## Orient first (read these, in order)

- Auto-memory (loaded each session): `project_gap_analysis` (the ranked backlog and
  what is implemented), `project_opencode_mkdocs`, `project_deep_research_novel_ideas`,
  `project_su_billing_impl`, and the two feedback notes `feedback_doc_style` and
  `feedback_workflow_no_slurm`.
- `IMPLEMENTATION_ROADMAP.md` (repo root) — the ranked 26-item backlog with what is
  done vs. gated. `PROGRESS.md` — plain-language project summary.
- `ai-session/README.md` — operator guide (this is where Slurm detail BELONGS).
- The live user docs: <https://rmehta1987.github.io/rcc-vllm/>. Source under `docs/`.

## Where things stand (done, do not redo)

- Core service works end to end: gateway (`ai-session/gateway.py`), SU billing
  (`billing/`), metering, and three proven clients — Open WebUI (browser), aider
  (coding), opencode (tool-calling, with the AGENTS.md workaround).
- Security/privacy hardening done: private per-user chat history (in `$HOME`), a shared
  per-session gateway key plus a per-session backend key, `upstream.json` locked to the
  owner, a central staff-only billing ledger (`/project/rcc/mehta5/ai-session-billing`),
  and a `billing_sweep.py` that rebuilds the un-forgeable floor charge from Slurm records.
- Zero-SU roadmap items shipped: Qwen3 reasoning parser, the `TIME_LIMIT`/`TIME=` fix,
  a pre-flight SU estimate at `up`, telemetry disabled, model-license surfacing + a
  Llama license gate, an agent-responsibilities + MCP-security docs page, two read-only
  MCP servers (`ai-session/mcp/slurm_mcp.py`, `su_usage_mcp.py`), a PydanticAI agent
  example (`examples/agent_pydantic.py`), an idle-session reaper (`idle_reaper.py`),
  gateway supervision + a `/status` route, gateway rate limiting, and a rate-table
  version guard.
- Docs site is on **GitHub Pages** and auto-deploys on every push to `main` via
  `.github/workflows/deploy-docs.yml` (it enables Pages itself via `configure-pages`).
  Repo moved off the RCC GitLab to **`git@github.com:rmehta1987/rcc-vllm.git`** (public).
  GitHub SSH works with the default key; `git push` from this directory just works.
- Module packaging is STARTED but not finished: an interim Tcl modulefile exists and
  loads cleanly (`/project/rcc/mehta5/modulefiles/ai-session/1.0`, verified with
  `module use` + `module load`). It sets `AISESSION_HOME` and prepends `<install>/bin`
  to `PATH`. The `bin/` dispatcher it points at is NOT built yet, and the modulefile
  currently lives OUTSIDE the repo (not version-controlled). The access-tier decision is
  recorded under Packaging (Tier 1 + Tier 3, no Tier 2).

## Environment, invariants, and traps (do not trip these)

- **Do not modify** `/project/rcc/mehta5/conda-envs/vllm-probe` (the production vLLM
  env). Creating new separate venvs under `/project/rcc/mehta5/` is fine (that is how
  `mkdocs-env`, `aider-env`, `openwebui-env` were made).
- **Never submit Slurm jobs** (`sbatch`/`srun`/`salloc`/`ai_session.py start`/wrapper
  `up`) without the user's explicit go-ahead. A build agent once orphaned a 4-GPU job
  that billed ~8 SU before it was caught. Read-only Slurm queries are always fine. After
  any workflow that touches the launcher/wrappers, run `squeue -u mehta5` and check for
  stray `<model>:<port>` jobs. The auto-mode classifier will block YOU from `scancel`ing
  a job you did not create — surface the job id and command to the user instead.
- **Doc style (hard user rule):** scientist-to-scientist plain prose; no emoji, no
  checkmark/cross glyphs, no marketing adjectives; numbered steps; exact copy-pasteable
  commands; tables carry real values. See `feedback_doc_style`.
- **Coder-32B tool-calling caveat:** the vLLM `hermes` parser does not populate tool
  calls for Qwen2.5-Coder-32B (vLLM issue #29192). Default every agent/MCP/tool-calling
  example to `qwen2.5_72B` or `qwen3_4b`, and say why.
- Pythons: `/project/rcc/mehta5/conda-envs/vllm-probe/bin/python` (py3.12, fastapi +
  httpx + yaml, no pytest); `/software/python-anaconda-2020.11-el8-x86_64/bin/python`
  (py3.8, pytest — the billing suite); `/project/rcc/mehta5/mkdocs-env/bin/mkdocs`
  (docs); `/software/python-3.11.9-el8-x86_64/bin/python3.11` (build new venvs, full path).
- **Keep `mkdocs build --strict` green** — if it fails, the GitHub Pages deploy fails and
  the live site stops updating. Verify locally before every push that touches `docs/`.

## Packaging — `module load ai-session`, not full project paths (build before the docs rewrite)

The user's requirement: people should not type `/project/rcc/mehta5/vllm/...`; they
should load a module and get clean commands, the way Stanford Sherlock exposes ollama
(`ml ollama`; then `ollama serve`, `ollama run`) and opencode (`ml opencode`; then
`opencode`). Reference pages:
<https://www.sherlock.stanford.edu/docs/software/using/ollama/> and
<https://www.sherlock.stanford.edu/docs/software/ai/coding-agents/>.

RCC module system, checked this cycle: **Environment Modules 4.6.1 — Tcl-based, NOT
Lmod**; `MODULEPATH=/software/modulefiles` (RCC-staff managed); no personal modulefile
directory exists yet. So write a **Tcl** modulefile, not an Lmod `.lua`.

What to build:

- **Clean entry points on `PATH`.** A small `bin/` with an `ai-session` dispatcher
  wrapping the existing scripts: `ai-session code` / `ai-session chat` /
  `ai-session status` / `ai-session stop` / `ai-session connect` in place of
  `bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh up`. Users never see the
  install path.
- **A Tcl modulefile** at `/project/rcc/mehta5/modulefiles/ai-session/1.0` — already
  created and verified this cycle (it does `setenv AISESSION_HOME` and
  `prepend-path PATH <install>/bin`; dir 755, file 644, group `rcc-staff`, so colleagues
  can load it). The access model is intentionally **two-tier, with no per-user shell
  edits**:
  - **Tier 1 — testing phase (now):** testers run, in each shell,
    `module use /project/rcc/mehta5/modulefiles` then `module load ai-session`. This is
    the interim mechanism while the service is being validated by a few rcc-staff.
  - **Tier 3 — production goal:** ask RCC to place the modulefile (or a symlink) in
    `/software/modulefiles` (already on everyone's default `MODULEPATH`) so any user runs
    plain `module load ai-session` with zero setup and no `module use`.
  - **Explicitly NOT Tier 2:** do not tell users to add `module use` to their `~/.bashrc`.
    Testers use Tier 1 by hand; everyone else waits for the Tier 3 central install.
  The remaining build is the `<install>/bin` dispatcher the modulefile points at — until
  it exists, loading the module sets the environment but there is no `ai-session` command
  to run yet.
- **Endpoint via environment, mirroring Sherlock.** Sherlock's server writes its address
  to `~/.ollama_server`; the user runs `export OLLAMA_BASE_URL=http://$OLLAMA_HOST/v1`;
  and `opencode.json` references it as `{env:OLLAMA_BASE_URL}`. Do the analogue: on `up`/
  `connect`, write the gateway base URL and the session key to a well-known file (for
  example `~/.ai-session/env`), and have `connect` print `export AISESSION_BASE_URL=...`
  and `export AISESSION_API_KEY=...`. Then change `ai-session/opencode.example.json` to
  use `{env:AISESSION_BASE_URL}` and `{env:AISESSION_API_KEY}` instead of the current
  `<GW_PORT>` / `<SESSION_KEY>` placeholders, and do the same for the aider and Continue
  snippets (they already read `OPENAI_API_BASE` / `OPENAI_API_KEY`). This removes all
  hand-editing and makes the copy-paste setup identical in spirit to Sherlock's.
- Keep opencode config **project-local**; never touch the user's personal
  `~/.config/opencode/opencode.json`. `opencode auth login` is theirs to run.
- **Version-control the modulefile.** It currently exists only at
  `/project/rcc/mehta5/modulefiles/ai-session/1.0` (outside the repo). Move the source
  into the repo (e.g. `modulefiles/ai-session/1.0`) and treat the paths under
  `/project/rcc/mehta5/modulefiles` (Tier 1) and, later, `/software/modulefiles` (Tier 3)
  as install targets — copy or symlink from the repo so the deployed copy and the source
  never drift. Do the same for the `bin/` dispatcher (source in repo, on `PATH` via the
  module).

Note we can go further than Sherlock on hiding HPC: Sherlock still has users request the
GPU themselves (`sh_dev -g 1`, `salloc`) before `ollama serve`. Our wrapper already
submits and waits on the GPU job inside `up`, so a user's whole flow can be
`module load ai-session; ai-session code`, with no Slurm ever visible.

## Goal 1 — natural, jargon-free user documentation (do this first)

The user's exact complaint: the docs reference Slurm jobs and other HPC internals that
are not useful for actual users. Measured leakage in the user-facing `docs/` tree
(Slurm/HPC-term hit counts, as of this handoff):

    docs/reference.md         35      docs/troubleshooting.md   13
    docs/billing.md           21      docs/coding/mcp.md        13
    docs/coding/overview.md   19      docs/coding/opencode.md    7
    docs/getting-started.md   15      docs/coding/continue.md    3
    docs/index.md             13      docs/faq.md                1

Terms leaking into user pages: `Slurm`, `sbatch`, `squeue`, `scancel`, `sacct`,
`walltime`, `partition`, `tensor-parallel`/`TP`, `gres`, `compute node`, `constraint`,
`reservation floor`.

**Principle: separate the user's mental model from the operator's.** A user starts a
*session*, waits for it to be *ready*, gets a *web address*, connects a *client*, and
*stops* it when done. The fact that a session is a Slurm job on a GPU node with a
tensor-parallel vLLM server is an implementation detail they should not need.

**What to do:**

- **Move, don't delete, the machinery.** Slurm/tensor-parallel/partition detail belongs
  in `ai-session/README.md` (operator guide), not in `docs/`. If a user page genuinely
  needs a "how it works under the hood" note, put it in one clearly labelled
  *"For administrators"* section or page, not woven through every page.
- **Reframe the recurring terms** in user pages roughly as:
  - "Slurm GPU job" / "submits the Slurm job" -> "starts a model server on a GPU (this
    can take a few minutes while the model loads)".
  - `TP` / "tensor-parallel degree" / "holds four A100s" -> hide it. Larger models use
    more GPUs; that is chosen for the user. If a knob must be exposed, describe it as
    "which model" not "how many GPUs / how they are split".
  - `CONSTRAINT=A100` / "Slurm feature constraint" -> "GPU type", selected by a plain
    name, ideally via a friendly preset (see Goal 2).
  - `walltime` -> "session time limit" (and the point that matters: an idle session still
    costs, so stop it when done).
  - `squeue` / `scancel` -> the wrapper's `status` and `down` (or the friendly CLI).
  - "reservation floor" in `billing.md` -> keep the billing concept but explain it in
    plain terms ("you pay for the time you hold the GPUs, whether or not you are actively
    using them"); the word "floor" can stay if defined once.
- **Legitimate exceptions:** the Slurm-query MCP server page (`docs/coding/mcp.md`) is
  *about* querying the queue, so a minimal, well-framed mention of the queue is fine
  there. `troubleshooting.md` may reference `status`/`down` behavior. Use judgment; the
  test is "would a first-time user be confused or feel they must learn HPC?".
- **Naturalness pass:** read each page as prose and smooth it. Many pages were assembled
  by separate agents and can feel stitched together. Keep the house style.

**Definition of done for Goal 1:**
- `grep -riE 'slurm|sbatch|squeue|scancel|sacct|tensor.parallel|\bTP\b|\bgres\b|
  compute node|constraint|walltime' docs/` returns essentially nothing in user-flow
  pages (a handful of justified mentions on the MCP page is acceptable and should be
  listed explicitly in your summary).
- `grep -rn '/project/rcc/mehta5/vllm' docs/` is empty — user pages use
  `module load ai-session` and the clean commands, not the install path.
- `/project/rcc/mehta5/mkdocs-env/bin/mkdocs build --strict` passes; the site
  auto-deploys; spot-check a few pages at the live URL.
- A colleague-readability check: someone with a laptop and no HPC background could get
  from the home page to a working chat or coding session without meeting a Slurm term.

## Goal 2 — make the service more intuitive

Candidate work (confirm scope/priority with the user; some is login-node-only, some is
UX-in-docs):

- **Friendly model presets.** Replace user-facing `MODEL/TP/CONSTRAINT` with intent
  names: e.g. `code` (Coder-32B), `chat` (72B), `fast` (Qwen3-4B), and later `reason`,
  `vision`. A thin wrapper maps the preset to the real serve config so users never set
  tensor-parallel or GPU constraints. Keep the raw knobs available for operators.
- **One-command connect.** `ai_session.py connect` already prints per-client config and
  the key; make it copy-paste perfect for each client and, ideally, a single command a
  user runs on their laptop. Reduce the SSH-tunnel friction (the number-one confusion).
- **Clear "session is loading" feedback.** The `/status` route exists (live/loading/gone)
  — surface it so users see "model still loading, about N minutes" instead of raw errors.
- **A friendlier CLI verb set** wrapping the current scripts (e.g. `ai-session chat`,
  `ai-session code`, `ai-session stop`) if the user wants it.
- **Wire in the idle reaper.** `idle_reaper.py` is built but not scheduled; enabling it
  (cron or a systemd --user timer) stops users being billed for a forgotten session —
  the exact failure that orphaned a job this cycle.
- **A short "first five minutes" quickstart** at the top of the docs.

## Goal 3 — continue building / fine-tuning the pipeline

Clarify with the user which sense of "fine-tuning" they mean:

- **Refining the pipeline (most likely near-term):** pursue the remaining roadmap items.
  Highest leverage next is **#7, a served embeddings endpoint** (unblocks Continue
  `@codebase`, document RAG, and the docs-RAG MCP server), then **#13 FIM autocomplete**,
  **#8 JSON/structured-output verification**, and the **#18/#19 reasoning and vision
  models** for the catalog. Each needs a short GPU benchmark into `billing/rate_table.json`
  — get explicit SU approval first (see the no-orphan-job rule).
- **Literal model fine-tuning (roadmap #24, bigger, decision-gated):** a path for a
  researcher to LoRA-tune a base model on their own data and have it served. vLLM
  supports serving LoRA adapters at runtime (`--enable-lora`, per-request adapter
  selection), so the serving side is feasible; the training side, storage, cost model,
  and access policy need design and an RCC decision. Treat this as a research/design task
  before any build.
- **Pipeline hygiene:** keep `bench_billing.py:build_serve_flags` in sync with
  `launch_ai_session.sh` (the rate table is only valid for the measured serve config),
  and re-benchmark when the serve flags or vLLM version change (the version guard already
  drops to floor-only billing on a mismatch).

## Verification (run before declaring anything done)

    cd /project/rcc/mehta5/vllm
    /software/python-anaconda-2020.11-el8-x86_64/bin/python -m pytest billing/ -q        # expect 22 passed
    /project/rcc/mehta5/mkdocs-env/bin/mkdocs build --strict                             # expect exit 0
    AISESSION_STATE_DIR=$(mktemp -d) /project/rcc/mehta5/conda-envs/vllm-probe/bin/python \
      -c "import sys;sys.path.insert(0,'ai-session');import gateway;gateway.build_app(require_key='x')"

Then commit and push to `main`; the docs deploy runs automatically. Confirm the live
site with a couple of `curl -sL -o /dev/null -w '%{http_code}'` checks against
`https://rmehta1987.github.io/rcc-vllm/` and a few sub-pages.

## Suggested first tasks for the new session

1. Read the memory files and `IMPLEMENTATION_ROADMAP.md`; skim the live docs site.
2. Build the Packaging layer first (it is the foundation the docs rewrite depends on):
   the `ai-session` CLI dispatcher and `bin/`, the Tcl modulefile under
   `/project/rcc/mehta5/modulefiles/ai-session/`, and the env-based endpoint/key file;
   then switch `ai-session/opencode.example.json` and the aider/Continue snippets to
   `{env:AISESSION_BASE_URL}` / `{env:AISESSION_API_KEY}`. Verify end to end (login-node
   only, no GPU): `module use /project/rcc/mehta5/modulefiles && module load ai-session
   && ai-session status`.
3. Do the Goal-1 de-jargon rewrite against the new commands, page by page, worst
   offenders first (`reference.md`, `billing.md`, `coding/overview.md`,
   `getting-started.md`, `index.md`). Move HPC detail into `ai-session/README.md`. Keep
   `--strict` green.
4. Run both Goal-1 definition-of-done greps (no Slurm terms, no install paths) and a
   readability pass; push and verify the live site.
5. Bring Goal 2 options to the user, implement the agreed subset (friendly model presets
   and the loading-status feedback are high value and low risk).
6. For Goal 3, confirm the "fine-tuning" scope, then either take the embeddings endpoint
   (with SU approval) or write the LoRA design note.
