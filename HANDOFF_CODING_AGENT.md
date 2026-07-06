# Handoff — stand up the CODING-AGENT client (next session)

**Paste this whole file as your first message in the new session.**

You are continuing the **ai-session** HPC LLM-serving service (tiered SU billing) under
`/project/rcc/mehta5/vllm/` on UChicago RCC (account `rcc-staff`, partition `test`).
The service, billing, and gateway are built and proven; all three 72B serving tiers
(A100/H100/H200) are benchmarked; and the **browser chat client (Open WebUI) is DONE**,
runnable in one command. **Your task: stand up the CODING-AGENT client the same way** —
a developer points a coding tool at the gateway and edits real code with the local 72B.

## Orient first (read these, in order)
- Memory: `project_su_billing_impl.md` (done/pending split **+ the ratified clients
  decision**), `project_su_billing_design.md` (billing formula + reservation floor),
  `project_hpc_llm_state.md`.
- `ai-session/README.md` — the **Clients** table (§4) is the spec: **aider** = coding
  client (robust with local models, text-edit format, no tool-calling); **opencode/Cline**
  = autonomous agents needing native tool-calling (`--agent-client`).
- **Reuse the pattern just built** (this is your template, don't reinvent it):
  - `ai-session/run_browser_demo.sh` — one-shot `{up|down|status}`, multi-user via
    `AISESSION_STATE_DIR`, per-user UID-derived ports, prints the SU receipt on `down`.
  - `ai-session/run_openwebui.sh` — how a client tool gets its OWN separate venv.
  - `ai-session/print_su_receipt.py` — stdlib-only SU-charge banner from a summary JSON.
  - `ai-session/ai_session.py {start|connect|end}` — `connect` prints the per-session
    gateway URL + SSH tunnel + per-client config.
- Env: shared read-only `vllm-probe` at `/project/rcc/mehta5/conda-envs/vllm-probe`. Call
  its python by **absolute path** — no `module load` needed on the login node (the GPU job
  activates itself inside `launch_ai_session.sh`). **Do NOT pip/conda install into vllm-probe.**

## Definition of done
A developer can, in ~2 commands: (1) bring up a 72B session + gateway, (2) run **aider**
pointed at the gateway, (3) have it read+edit files in a real git repo with the round-trip
proven and metered, (4) tear down and see the SU charge. Ideally wrapped in a one-command
helper `ai-session/run_coding_agent.sh` mirroring `run_browser_demo.sh`.

> **Scope: aider only.** opencode / Cline (native tool-calling agents) are **NOT planned** —
> a local 72B with vLLM tool-parsing is too fragile to depend on. The `--agent-client` flag
> stays in the code as a latent capability but is **not** a target of this task.

## Plan

### 0. Pick model + tier — costs SU, GET USER OK before submitting any GPU job
Coding wants the big model: **Qwen2.5-72B-Instruct** (already staged under `models/`).
Measured tiers (from `billing/rate_table.json`):
- **A100 TP=4** → interactive/value (floor **4.0 SU/hr**) — recommended default for hands-on coding.
- **H200 TP=2** → sustained throughput (cheapest per token) — better for long agent runs.
- **H100 is dominated for 72B — skip it.**
No Qwen-**Coder** model is staged. If one is wanted, download it on the **login node**
(it has internet) into `models/` first — compute nodes are offline.

### 1. aider — the coding client (text-edit diffs; no tool-calling; robust with a local 72B)
- Install in its OWN venv (like Open WebUI), e.g. `/project/rcc/mehta5/aider-env`, built with
  the full python path `/software/python-3.11.9-el8-x86_64/bin/python3.11 -m venv …`
  (`module load python/3.11.9` does NOT reorder PATH). **NOT** vllm-probe.
- Start the session in the **DEFAULT serve config** (no `--agent-client` — aider uses
  text-edit diffs, not native tool-calls). That config matches the benchmarked rate record,
  so the **token term is exact**, not just the floor.
- Point aider at the gateway (use the per-user gateway port the helper / `connect` prints —
  NOT a hardcoded 8421; the served model id == the model_key):
  ```bash
  OPENAI_API_BASE=http://localhost:<gw_port>/v1 OPENAI_API_KEY=ai-session \
    aider --model openai/qwen2.5_72B
  ```
- **LIKELY SNAG:** aider/litellm doesn't know a custom model's context window → it warns
  and/or mis-sizes prompts. Supply a model-metadata file (`--model-metadata-file`) declaring
  `max_input_tokens` = **8192** (the default serve length). Budget time for this.
- **Prove it:** in a scratch git repo, ask aider to make a small edit; confirm the diff
  applies and the gateway meters the tokens (`logs/gateway/usage-*.jsonl` under the state dir).

### 2. One-command helper `ai-session/run_coding_agent.sh`
Mirror `run_browser_demo.sh`: `{up|down|status}`; export `AISESSION_STATE_DIR` (default
`/project/rcc/mehta5/ai-session-state/<user>`) + per-user gateway port; `up` starts
session+gateway and prints the ready-to-run aider command (+ tunnel if needed); `down` runs
`ai_session end` then prints the SU receipt via `print_su_receipt.py`. Keep teardown **by
PID / port-owner (ss), never `pkill -f`** (its pattern self-matches the wrapper shell → exit 144).

## Invariants / traps (do not trip these)
- Start aider sessions in the **default serve config** (no `--agent-client`) so the
  benchmarked rate record applies exactly. (`--agent-client` would change context length and
  add a tool parser → token rate only approximate; not needed for aider, and opencode/Cline
  aren't planned.)
- **One venv per client tool** — never install aider/opencode into `vllm-probe`.
- Prod serve flags in `launch_ai_session.sh` must stay in sync with
  `bench_billing.py:build_serve_flags` for the NON-agent config (rate-record validity).
- Do **NOT** add `from __future__ import annotations` to `gateway.py` (breaks FastAPI
  `Request` resolution → 422).
- Do **NOT** delete `vllm-v0.7.3.sif` or the Llama `original/` checkpoint without asking.
- Multi-user writable state (incl. **billing JSONs**) lives at
  `/project/rcc/mehta5/ai-session-state/<user>/logs/...` (setgid `rcc-staff`, group-readable
  for central billing). When fixing perms, `chmod 2775` **LAST** — `chgrp` clears setgid.
- Login node HAS internet (PyPI + model downloads OK); compute nodes do NOT.
- Serving + tool tokens cost SU — get user OK before any GPU job; `end` the moment you stop
  (exclusive node = floor-billed regardless of how little you send).

## Where things stand (as of 2026-06-03)
- DONE: billing math (22/22 pytest); gateway (lifespan handlers, multi-user via
  `AISESSION_STATE_DIR`); full CLI (start/status/connect/end); **4 measured rate records**
  (`qwen3_4b/a100/1`, `qwen2.5_72B/{a100/4, h100/4, h200/2}`); **browser client (Open WebUI)
  GREEN** via `run_browser_demo.sh`; per-`down` SU receipt (`print_su_receipt.py`).
- Multi-user proven in the wild: colleague `ndtrung` already ran `run_browser_demo.sh` and got
  billed 0.1981 SU into their own state subdir.
- Clients decision (ratified): Open WebUI (done) + **aider (THIS task)**. opencode/Cline are
  **NOT planned** — aider only (see the scope note up top).
- NEXT (this handoff): aider against the gateway on the 72B with a proven edit round-trip,
  wrapped in `run_coding_agent.sh`.
