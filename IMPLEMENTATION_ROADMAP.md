# ai-session — implementation roadmap

_Compiled 2026-07-05 from the peer gap analysis (`wf_ad79e9b3-b31`) and the deep-research
report (`wf_3263349a-1a7`). This is a planning document: a ranked backlog, a workflow
prompt to build the code-able items, a staff test plan, and a user-facing FAQ._

## Already delivered (do not re-do)

- Private browser-chat history (moved to `$HOME/.ai-session/openwebui-data`, mode 700).
- Shared per-session gateway key (bill-to-starter) plus a per-session backend key so the
  vLLM server cannot be reached directly; `upstream.json` is now owner-only.
- Central staff billing ledger (`/project/rcc/mehta5/ai-session-billing`, mode 2770) that
  every `end` writes to, plus a staff `billing_sweep.py` that reconstructs the
  un-forgeable floor charge from Slurm accounting.
- opencode proven against the service (Coder-32B tool-tag workaround); mkdocs user site;
  fractional-allocation billing correction.

## Ranked backlog

Effort key: **S** = hours, login-node only, no SU. **M** = days, or needs one short GPU
benchmark (small SU). **L** = needs an RCC/policy decision or real infrastructure.
"Decision" = cannot be built until a named person decides something.

| # | Item | What it delivers | Effort | SU | Decision? | Depends on |
|--:|------|------------------|:------:|:--:|:---------:|-----------|
| 1 | Reasoning parser for Qwen3-4B | Stops streaming raw `<think>` traces to clients and stops billing hidden reasoning as output; `--reasoning-parser qwen3` in the launcher | S | 0 | no | — |
| 2 | Fix `TIME_LIMIT` clobber + duration knob | `TIME=08:00:00` is silently capped at 2 h today; long refactors die mid-edit. Honour the caller's walltime, print it at `up` | S | 0 | no | — |
| 3 | Pre-flight SU estimate at `up` | Print "this tier costs W SU/h, projected max Y SU over Z h" before GPUs are committed | S | 0 | no | — |
| 4 | Disable client telemetry + scope the residency claim | Turn off Continue's `allowAnonymousTelemetry`, add `--analytics-disable` to aider; reword the absolute "nothing leaves RCC" to cover the serving path only | S | 0 | no | — |
| 5 | Surface model licenses + gate `--force` Llama | License column in the model table; record an acknowledgment before serving Llama (Meta community license) or Qwen2.5-72B (Tongyi license needs attribution) | S | 0 | no | — |
| 6 | Agent-responsibility + MCP-security page | One page: you own what your agent does; an agent runs with your full cluster permissions and can read a labmate's files through shared dirs; prompt-injection risk | S | 0 | no | — |
| 7 | Served embeddings endpoint (`/v1/embeddings`) | Unblocks Continue `@codebase`, document RAG, and the docs-RAG MCP server. Stage a small embedding model, serve `vllm --runner pooling` on an A40 (w_gpu 0.5), benchmark it into the rate table | M | small | no | — |
| 8 | Structured-output (JSON) mode: verify + document | Confirm `response_format`/guided decoding per served model; the load-bearing feature for dataset extraction and LLM-judge pipelines | M | small | no | live session |
| 9 | Slurm-query MCP server | Natural-language "what is in my queue, why is my job pending" for opencode/Cline; showcases MCP with zero model risk | S | 0 | no | — |
| 10 | SU-usage MCP server | "How many SUs has my group burned, on which model" answered from the central ledger; borrowed from NRP's accounting-MCP pattern | S | 0 | no | central ledger (done) |
| 11 | Agent example on the gateway (PydanticAI) | A working, documented "build an agent on ai-session" script; turns "we serve models" into "we are an agent platform" (AMD ROCm blueprint) | S | 0 | no | key (done) |
| 12 | Docs-RAG MCP server over the mkdocs site | The agent answers "how do I start a session, what does an H200 cost" from our own docs | M | 0 | no | #7 embeddings |
| 13 | FIM autocomplete model | Ghost-text completion needs a fill-in-the-middle model; today the docs mis-advise pointing autocomplete at a chat model. Stage Coder-1.5B/7B on an A40 | M | small | no | — |
| 14 | Idle-session reaper + orphan detection | Warn then auto-`end` a session with no traffic for N minutes; a dropped SSH currently bills idle GPUs to the walltime cap | M | 0 | no | — |
| 15 | Supervise the gateway + status/liveness endpoint | Run the gateway under `systemd --user` (restart on failure) instead of `nohup`; a `/status` route so clients see live/loading/gone instead of raw 502s | M | 0 | no | — |
| 16 | Gateway rate limits + per-user session cap | Token-bucket + max body size (429), one-session-per-user check; stop a runaway agent loop from hammering the login node | M | 0 | no | — |
| 17 | Rate-table version guard + upgrade runbook | Assert the running `vllm --version` matches the rate-record provenance; mark UNRATED on mismatch so an upgraded engine cannot bill against stale throughput numbers | S | 0 | no | — |
| 18 | Reasoning model in the catalog | Stage and serve DeepSeek-R1-Distill-Llama-70B (SDSU precedent); benchmark into the rate table | M | med | no | — |
| 19 | Vision model in the catalog | Stage and serve a multimodal model (Qwen2.5-VL or Llama-4); benchmark | M | med | no | — |
| 20 | Per-user state-dir isolation | Session files and usage logs still sit in group-writable (2775) dirs; move to owner-only or ACLs so a staff peer cannot tamper. Touches the RCC setgid convention | L | 0 | **yes** | RCC storage decision |
| 21 | Data-classification ceiling + AUP + acknowledgment gate | Decide what data is allowed; publish an acceptable-use page; add a first-run "type yes" gate that writes a timestamped acceptance record. Every peer ships this; a Phase-2 blocker | L | 0 | **yes** | RCC/ISO decision |
| 22 | Full server-side billing enforcement | A root-owned append-only collector, or wire SU into a Slurm QOS (NCSA `QOSGrpBillingMinutes` precedent), so charges cannot be edited even by the owner | L | 0 | **yes** | RCC decision |
| 23 | Managed RAG service | Embeddings + vector store + retrieval as one offering; greenfield (no peer packages it) | L | med | maybe | #7 embeddings |
| 24 | Self-service LoRA-adapter hosting | User submits data, gets a served adapter; greenfield | L | high | **yes** | — |
| 25 | Slurm-native backfill scheduling | Let inference jobs opportunistically fill idle Slurm windows (GWDG design) to raise GPU utilisation | L | 0 | **yes** | scheduler policy |
| 26 | Phase-2 governance | Institutional-identity login, expiring API keys, per-PI token minting (NRP/LLNL precedent) for opening past the trusted group | L | 0 | **yes** | #21 |

### How the ranking was made

Items are ordered by **(value to a real user now) × (readiness) ÷ (effort)**, with two
overrides: things that close an integrity or privacy hole are pulled up, and things that
merely follow fashion are pushed down. The top band (1–6) is cheap, needs no decision, and
either fixes a live defect or is a prerequisite for the interesting work. The middle band
(7–17) is the capability payload — embeddings, MCP, agents, autocomplete, operational
robustness — mostly cheap or one small benchmark. The catalog additions (18–19) cost real
GPU time and are demand-driven. The bottom band (20–26) cannot start until a named person
decides a policy or storage question; two of them (per-user SU billing, managed RAG) are
places where we would be leading peers rather than catching up.

## What is genuinely novel versus catching up

The deep research found **no peer** doing per-user SU billing for inference (they use only
rate limits), a turnkey managed RAG service, self-service LoRA hosting, or a published MCP
security model. Items 22–24 and the MCP-security half of item 6 are therefore greenfield:
building them puts us ahead, not level. The rest (embeddings, reasoning/vision catalog,
agent platform, governance) is well-trodden by NRP, GWDG, SDSU, LLNL, LBL, Isambard, and
Duke, so we can copy known-good designs.

---

## Implementation workflow prompt

Feed this to the Workflow tool. It is split so the zero-SU code lands first and safely; the
GPU-benchmark and decision-gated items are deliberately excluded until you approve SU spend
or make the policy calls (items 18–26).

> **Scope:** implement backlog items 1–6, 9, 10, 11, 14, 15, 16, 17 (all zero-SU,
> login-node code and docs). Hold items 7, 8, 12, 13 for a second SU-approved run; do not
> touch items 18–26 (they need decisions or GPU staging).
>
> **Hard constraints:** never modify `/project/rcc/mehta5/conda-envs/vllm-probe`; never
> submit Slurm jobs; no git commits; no `from __future__ import annotations` in
> `gateway.py` and keep fastapi/httpx lazily imported; billing suite
> (`/software/python-anaconda-2020.11-el8-x86_64/bin/python -m pytest billing/ -q`) must
> stay 22-green; docs build with `/project/rcc/mehta5/mkdocs-env/bin/mkdocs build --strict`;
> doc style is plain scientist-to-scientist prose, no emoji, no marketing, exact commands.
>
> **Phase 1 — build (parallel tracks, one agent each):**
> - Launcher track (items 1, 2, 17): in `launch_ai_session.sh` add `--reasoning-parser
>   qwen3` for Qwen3 models; stop `ai_session.py` clobbering a caller-set `TIME_LIMIT` and
>   add a `TIME=` knob to both wrappers, printed at `up`; add a rate-table version guard
>   that marks a config UNRATED when the running `vllm --version` differs from the record
>   provenance.
> - Cost-and-trust track (items 3, 4, 5): print a pre-flight SU estimate at `up`; set
>   Continue `allowAnonymousTelemetry:false` and add `--analytics-disable` to every printed
>   aider command; add a License column and a per-model license page, and gate the
>   `--force` Llama path on a recorded acknowledgment.
> - MCP track (items 9, 10): write two stdlib MCP servers under `ai-session/mcp/` — a
>   Slurm-query server (wraps `squeue`/`sacct` read-only) and an SU-usage server (reads the
>   central ledger); document enabling them in a project-local `opencode.json`.
> - Agent-platform track (items 6, 11): write the agent-responsibility + MCP-security docs
>   page, and a runnable PydanticAI example (`examples/agent_pydantic.py`) pointed at the
>   gateway URL with the session key, using Qwen3 or the 72B (not Coder-32B — the hermes
>   parser fails to populate tool calls for the Coder variant, vLLM #29192).
> - Operations track (items 14, 15, 16): an idle-session reaper (poll `squeue` + the
>   backend request-counter, warn, then `end`); a `systemd --user` unit and a `/status`
>   route on the gateway; per-client rate-limit + max-body middleware and a
>   one-session-per-user check.
>
> **Phase 2 — verify (one adversarial agent per track):** security review of the MCP servers
> (they run with the user's permissions — confirm they cannot be coerced into writing or
> into reading outside their remit) and the rate-limit/reaper logic; correctness review of
> the launcher and cost tracks; re-run the billing suite and the strict docs build.
>
> **Phase 3 — repair** any confirmed defect, then a **final gate**: billing suite green,
> strict docs build, `bash -n`/`py_compile` on every changed file, gateway still imports,
> no emoji/marketing in docs, and the staff test plan below passes its login-node checks.

A second, SU-approved workflow would then add items 7, 8, 12, 13 (embeddings + FIM
benchmarks, JSON-mode live verification, docs-RAG server), each with a single short GPU run
and a rate-table record.

---

## Staff test plan (rcc-staff)

Run these on a login node after the workflow lands. No GPU is needed except the marked
end-to-end smoke.

**A. Static checks (no GPU):**

1. Billing math unchanged: `cd /project/rcc/mehta5/vllm && /software/python-anaconda-2020.11-el8-x86_64/bin/python -m pytest billing/ -q` → 22 passed.
2. Docs build: `/project/rcc/mehta5/mkdocs-env/bin/mkdocs build --strict` → exit 0.
3. Scripts parse: `bash -n` on every `ai-session/*.sh`; `py_compile` on every changed `.py`.
4. Gateway invariants: `grep -n "0.0.0.0" ai-session/gateway.py` empty as a default; `grep from __future__ ai-session/gateway.py` empty; gateway builds under the vllm-probe python.
5. Version guard: temporarily point a rate record's provenance at a bogus version and confirm metering reports UNRATED (floor-only), then revert.
6. MCP servers: run each `ai-session/mcp/*.py` with a canned request on stdin and confirm it returns only the intended read-only data; confirm neither can write.
7. Rate limit: fire N+1 rapid requests at a stub gateway and confirm the (N+1)th gets HTTP 429.

**B. End-to-end smoke (one short GPU session, ~0.5 SU):**

8. `TIME=00:30:00 AGENT_CLIENT=1 MODEL=qwen3_4b TP=1 CONSTRAINT=A100 bash ai-session/run_coding_agent.sh up` and confirm: the printed walltime is 00:30:00 (not 2 h); the pre-flight SU estimate prints; a session key prints.
9. Reasoning parser: send a chat request to Qwen3-4B and confirm the response separates reasoning from the answer (no raw `<think>` in the content) and that the usage block accounts for reasoning tokens.
10. Agent example: run `examples/agent_pydantic.py` against the session (key in `api_key`) and confirm a tool call completes.
11. Idle reaper: leave the session idle past the reaper threshold and confirm it warns then ends the job; check `squeue -u <you>` is empty and a ledger record was written.
12. `down` and confirm the receipt, the central-ledger `end` record, and a clean teardown (no gateway port listening, key file removed).

**C. Billing integrity:**

13. `billing_sweep.py --dry-run --user <you> --since <today>` lists the smoke session with a floor that matches `w_gpu × N × elapsed`.

---

## User FAQ

**Access and cost**

- _How do I get access?_ You need an RCC account with an allocation on the `rcc-staff`
  (Phase 1) account. Start a session with the wrapper scripts; there is no separate signup
  yet.
- _What will it cost?_ Sessions bill in SUs (1 SU = one A100-GPU-hour). You pay the greater
  of the GPUs you hold (`w_gpu × N × hours`) or the tokens you process; for interactive use
  the hold cost dominates. The default coding session (Coder-32B on two A100s) is 2.0 SU/h.
  See `docs/billing.md`.
- _How do I keep the cost down?_ Run `down` the moment you stop working — an idle session
  still bills. Pick A100 for interactive work (lowest hold cost); use H200 only for
  sustained high-throughput generation.
- _Is my data private?_ Prompts and completions are served entirely on the cluster and are
  not sent to any outside provider. Your browser-chat history is stored in your private home
  directory. Note that coding-tool telemetry is a separate matter — the service disables it
  where it can, but verify your own client settings.

**Connecting and sharing**

- _How do I connect from my laptop?_ Open an SSH tunnel to the login node the session runs
  on, then point your client at `http://localhost:<port>`. The exact command is printed by
  `connect` and in `docs/getting-started.md`.
- _Can my lab share one session?_ Yes. The person who starts it gets an access key; hand it
  to your labmates, who each open their own tunnel and use the key. All usage bills to the
  starter. Anyone without the key is refused.
- _Which model should I use?_ Coder-32B for code, the 72B for mixed prose-and-code or the
  largest general model, Qwen3-4B for quick/cheap tasks, and (once staged) the reasoning
  model for math/planning and the vision model for images.

**Coding agents, MCP, and building agents**

- _Which coding tool should I use?_ aider is the dependable default for editing files.
  opencode is supported for full tool-calling agents but needs the `AGENTS.md` workaround
  (see `docs/coding/opencode.md`). Continue is for in-editor use.
- _My agent said it did something but nothing changed. Why?_ The Coder-32B model does not
  emit the tool-call markers vLLM expects, so tool calls silently fail. Use the `AGENTS.md`
  workaround, or use the 72B / Qwen3 for agent work.
- _How do I add an MCP tool?_ Add an `mcp` block to a project-local `opencode.json` (not
  your personal config). The service offers a Slurm-query and an SU-usage MCP server; you
  can add your own, but see the agent-responsibility page first — an MCP server runs with
  your full cluster permissions.
- _Can I build my own agent on these models?_ Yes. Point any OpenAI-compatible framework
  (PydanticAI, LangGraph, smolagents, the OpenAI Agents SDK) at the gateway URL with your
  session key as the API key. See `examples/agent_pydantic.py`.

## Blockers and issues by audience

**Staff (operators)**

- Compute nodes have no internet, so anything requiring a download (new model weights, a pip
  install) must be staged from a login node first. This blocks self-service model or package
  additions.
- The `billing_sweep.py` all-users scan is slow because Slurm accounting history is large;
  schedule it with `--user` or a tight `--since` for cron.
- Per-user state directories are still group-writable (item 20); until that is resolved, a
  staff peer can in principle read another user's session files. The billing ledger and chat
  history are already locked down.
- Opening the service past the trusted group is blocked on the data-classification/AUP
  decision (item 21) and on identity/governance (item 26).

**Students**

- GPU queue waits: a session may sit pending until cards free up; the launcher aborts if it
  waits too long. Choose a less-contended tier or a smaller model.
- SSH tunnels are the most common point of confusion — the port and login node must match
  what `connect` printed; a tunnel to the wrong login node yields a dead URL.
- Home-directory quota: browser-chat history now lives in your home directory, which has a
  smaller quota than project space; clear old chats if it fills.
- Idle billing: forgetting `down` (or dropping your SSH session) keeps the GPUs billing;
  the idle reaper (item 14) will mitigate this once it lands.

**Faculty (PIs)**

- Cost attribution: shared-lab sessions bill entirely to the member who started them, not
  per person. Per-person accounting needs the deferred keys-with-owners work.
- Data policy: there is not yet a published acceptable-use or data-classification statement,
  so do not put regulated data (PHI, FERPA, export-controlled) into the service until item
  21 is decided.
- Model licenses: serving Llama-3.1-70B or Qwen2.5-72B to your group carries attribution and
  use-policy obligations (item 5) that the service will make explicit.
- Allocation model: Phase 1 runs on the `rcc-staff` account; a per-PI allocation and quota
  model (item 26) is a Phase-2 decision.
