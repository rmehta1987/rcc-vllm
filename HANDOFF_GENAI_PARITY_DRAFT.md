# Expanding ai-session toward a Purdue-GenAI-Studio-class service

> **DRAFT, 2026-10-01. Not started.** Written after a full crawl of docs.rcac.purdue.edu/services/genai/. Implementation is waiting on review.

## Context

Purdue RCAC's GenAI Studio (docs.rcac.purdue.edu/services/genai/, crawled in full: overview, chat-interface, workspace, models, groups, api, tool-calling, mcp-integration, the AnvilGPT twin, agentic-ai, the 2024/2025 news posts, and the PurdueRCAC/genai-studio GitHub repo) runs this stack:

- Open WebUI with CILogon SSO and Postgres+pgvector
- LiteLLM, which meters per user
- vLLM (through KubeAI) and Ollama, on Kubernetes, always on

Users can:

- build **Knowledge Bases**: upload documents, embed them with EmbeddingGemma-300M, then pull one into a chat by typing `#name`
- build **custom models**: base model + system prompt + Knowledge Bases + tools
- write **Workspace Tools**: a Python `Tools` class, called natively by the vLLM models
- use **MCP servers**, which only admins register, over Streamable HTTP
- use an **OpenAI-style API** with keys (20 requests/min), audio, and structured output
- share through **groups**, which RCAC staff create

Our ai-session is a different shape:

- Each user starts their own Slurm job running vLLM.
- A gateway on the login node provides the OpenAI and Anthropic APIs.
- Open WebUI 0.9.6 runs per user, with `WEBUI_AUTH=False` and its data in `~/.ai-session/openwebui-data`.
- Knowledge/RAG is unconfigured. The only embedder is the built-in MiniLM, run on the login-node CPU.
- No embeddings endpoint is served.
- Tools are limited to opt-in web search plus paper search through mcpo.
- MCP is limited to two read-only stdio servers (Slurm jobs, usage), reachable only from opencode.

**Key point:** Open WebUI's per-user data directory is persistent on disk, even though the GPU job is not. Knowledge bases, tools, and custom models created in one session are therefore still there in the next. We can deliver Purdue's tools, knowledge-base and assistant features **inside the current per-user architecture**, without first building an always-on shared service. Shared groups and SSO do need that service, so they are a separate, decision-gated phase.

### Where we stand against Purdue

| Purdue feature | Us today | Plan |
|---|---|---|
| Chat UI, history, per-chat settings | Have | — |
| Knowledge Bases (`#kb` in chat) | UI exists but embedding uses MiniLM on the login node | Phase A |
| Embeddings | None | Phase A (also roadmap #7) |
| Custom models / assistants | UI exists, undocumented | Phase B3 |
| Python Workspace Tools | UI exists, untested; chat sessions lack a tool-call parser | Phase B |
| MCP tool servers | stdio only, opencode only | Phase B2 |
| API with keys, tool calling, structured output | Have (session key, rate limiting); undocumented | Phase C (docs) |
| Vision models | gemma4 is multimodal, untested in the UI | Phase C |
| Web search (SearXNG), Docling parsing | DuckDuckGo opt-in | Phase C |
| Speech-to-text / text-to-speech | None | Deferred (low value per GPU) |
| Side-by-side multi-model compare | One model per session | Deferred |
| SSO, groups, shared KBs, always-on endpoint | None | Phase D (decision-gated) |
| **We have, Purdue doesn't** | Your own GPU and context, coding agents (Claude Code, opencode, aider), LoRA serving, Slurm and usage MCP, SU accounting, data in your own home dir, bulk ingest from cluster storage (Phase A3) | |

## Phase A — Knowledge Bases (embeddings first)

### A1. Serve an embedding model inside the user's GPU job

Today embedding runs on the login-node CPU, which is too slow and anti-social for a 100 MB PDF.

1. Stage **Qwen3-Embedding-0.6B** (and optionally **bge-reranker-v2-m3**) into `/project/rcc/mehta5/hf_cache/hub`. Use the build-partition download recipe with Xet disabled (see the H200-staging memory). Add both to the registry in `ai-session/server.py`, marked as auxiliary rather than chat-served.
2. In `ai-session/launch_ai_session.sh`, after the main vLLM comes up, start a second vLLM process from the 0.26.0 cu129 env: `--runner pooling`, served name `embed`, on GPU 0, with `--gpu-memory-utilization 0.05`. Lower the main model's utilization by the same amount. Write its port into `upstream.json` as an `embed` entry.
   - For `--cpu` sessions, run the embedder on CPU instead.
3. In `ai-session/gateway.py`, route `/v1/embeddings` and `/v1/rerank` (or any body whose `model` is `embed`) to that upstream. Everything else goes to the main backend as now. Keep key checking, rate limits and the usage JSONL unchanged.
   - Bonus: this gives users `/v1/embeddings` for their own code and Continue `@codebase`, and closes roadmap #7.

### A2. Point Open WebUI at it (`ai-session/run_openwebui.sh`)

1. Set:
   - `RAG_EMBEDDING_ENGINE=openai`
   - `RAG_OPENAI_API_BASE_URL=<gateway>/v1`
   - `RAG_OPENAI_API_KEY=<session key>`
   - `RAG_EMBEDDING_MODEL=embed`
   - `VECTOR_DB=chroma` (already installed; per user, in `DATA_DIR`)
   - `CHUNK_SIZE`/`CHUNK_OVERLAP`, `RAG_TOP_K`
   - `RAG_FILE_MAX_SIZE=100`
   - `ENABLE_RAG_HYBRID_SEARCH=True`, plus `RAG_RERANKING_MODEL` if the reranker is staged
2. **Pin the embedding model.** Changing it later makes every stored vector stale. Write the model ID into `DATA_DIR` on first use. If a later start finds a different ID, warn and offer `ai-session kb reindex`.
3. Keep MiniLM as a fallback for UI-only use when no GPU embedder is up.

### A3. CLI ingest (our differentiator: the data is already on /project)

Add `ai-session kb {list,create,add,reindex}` to `bin/ai-session`, implemented in a new `ai-session/kb.py`. It drives Open WebUI's REST API:

- `POST /api/v1/knowledge/create`
- `POST /api/v1/files/`
- `POST /api/v1/knowledge/{id}/file/add`

Authentication uses a local admin API key, enabled with `ENABLE_API_KEYS`. Example: `ai-session kb add mylab-papers ~/papers/*.pdf` bulk-loads a directory, skipping files it has already loaded (tracked by hash).

### A4. Seed a knowledge base

Build an "RCC docs" knowledge base from our own `docs/` (mkdocs) plus the RCC user guide. This also satisfies roadmap #12.

## Phase B — Tools

### B1. Workspace Tools work end to end

1. Turn on the tool-call parser by default for `chat` and `fast` (today it needs `--agent`), so Native function calling works. The parser for Qwen2.5-72B is `hermes`; it fails only on Coder-32B, which is retired.
2. Set `ENABLE_PIP_INSTALL_FRONTMATTER_REQUIREMENTS=False`. Login nodes have restricted internet, so a tool's `requirements:` line would hang. Document that tools may import only what is in the openwebui-env venv, and publish that list.
3. Ship a curated starter library in `ai-session/owui-tools/` (each a Python `Tools` class). Seed them idempotently on first start through `POST /api/v1/tools/create`. Proposed set:
   - `slurm_jobs`: reuses the argument-checked squeue/sacct logic in `ai-session/mcp/slurm_mcp.py`. Factor that logic into an importable module rather than copying it.
   - `su_usage`: reuses `ai-session/mcp/su_usage_mcp.py`.
   - `rcc_docs_lookup`
   - a units/constants calculator, the equivalent of Purdue's temperature example
   - a `python_sandbox` stub only if B4 is approved
4. Security note for the docs: tools run in **your own** Open WebUI process as **you**, unlike Purdue's shared host. The blast radius is your own account, which is still worth warning about given prompt injection (point to the existing `docs/coding/agents.md`).

### B2. MCP inside Open WebUI

Open WebUI 0.9.6 supports `type: mcp` connections natively (see `utils/middleware.py`).

1. Run `slurm_mcp` and `su_usage_mcp` with FastMCP's streamable-http transport on localhost ports.
2. Register them in `TOOL_SERVER_CONNECTIONS` next to the existing paper-search mcpo entry.
3. **Require a bearer token on every localhost MCP or mcpo port.** Ports derived from the UID are predictable on a shared login node; this is the co-tenant risk in `HANDOFF_MULTIUSER_READINESS.md`. Reuse the session-key pattern from `run_browser_demo.sh:217`.
4. Add `ai-session mcp run papers` so opencode and Claude Code agents get paper search too (`HANDOFF_AUDIT_FIXES_AND_EXTENSIONS.md:217`).

### B3. Custom models (assistants)

Seed one example: "RCC Helper" = qwen2.5_72B + a system prompt + the "RCC docs" knowledge base + the `slurm_jobs` tool. Document how to build your own (base model, prompt, knowledge, tools, Native mode).

### B4. Code interpreter (optional, needs your approval)

`ENABLE_CODE_INTERPRETER` with the Jupyter engine pointed at a kernel inside the user's Slurm job, not on the login node.

## Phase C — Documentation and smaller parity items

1. New docs pages, added to `mkdocs.yml` nav and written in the house style (numbered steps, exact commands, measured numbers):
   - `docs/knowledge.md`: UI steps mirroring Purdue's, `#name` usage, `ai-session kb`, limits, what to do when a scanned PDF times out
   - `docs/tools.md`: writing a `Tools` class (type hints and docstrings become the schema), enabling it from the Integrations menu, the starter library, MCP, security
   - `docs/assistants.md`: custom models
   - an API section in `docs/reference.md`: a tool-calling loop, `response_format`/`json_schema` structured output (vLLM already supports these), embeddings, images to gemma4
2. Test image input with gemma4_31B in the UI and API, and document it.
3. Optional: self-hosted SearXNG to replace DuckDuckGo (`HANDOFF_AUDIT_FIXES_AND_EXTENSIONS.md:212`), and Docling as `CONTENT_EXTRACTION_ENGINE` for scanned and tabular PDFs.

## Phase D — Shared, always-on service (decision-gated; not implemented by this plan)

Purdue's groups, SSO, shared knowledge bases, a persistent API URL and per-user keys all need:

- an always-on Open WebUI with Postgres+pgvector on a service VM, not a login node
- institutional login through Globus or CILogon OIDC (`OAUTH_*` envs)
- LiteLLM, or our gateway extended to per-user keys and metering (roadmap #26)
- at least one persistent model, or a scheduler that starts Slurm jobs on demand

It is blocked on the items in `HANDOFF_MULTIUSER_READINESS.md` and on roadmap #20–22 and #26. I can write it up as a separate design note when you want to raise it with RCC.

## Files touched (Phases A–C)

- **Edited:**
  - `ai-session/launch_ai_session.sh` (embedder process, parser default)
  - `ai-session/gateway.py` (multiple upstreams, `/v1/embeddings`)
  - `ai-session/run_openwebui.sh` (RAG and tool environment, seeding, MCP connections, tokens)
  - `ai-session/run_browser_demo.sh` (start and stop the extra processes)
  - `bin/ai-session` (`kb` verb, `mcp run papers`)
  - `ai-session/server.py` (embed and rerank registry entries)
  - `ai-session/mcp/slurm_mcp.py`, `su_usage_mcp.py` (factor out the logic, add an HTTP transport)
  - `mkdocs.yml`, `docs/*`
  - `IMPLEMENTATION_ROADMAP.md` (mark #7 and #12)
- **New:** `ai-session/kb.py`, `ai-session/owui-tools/*.py`, `ai-session/owui_seed.py`, `docs/knowledge.md`, `docs/tools.md`, `docs/assistants.md`
- **Billing:** no formula change. The embedder runs inside the user's already-billed GPU allocation. Embedding tokens go to the usage JSONL for visibility only.

## Verification

1. **Unit and offline checks:**
   - gateway routing test (embed model goes to the embed upstream; a missing key gets 401)
   - `kb.py` against a CPU-only Open WebUI with a stub embedder
   - tool-seeding idempotence: run it twice and get no duplicates
   - `mkdocs build --strict`
2. **GPU smoke on pedramh-gpu (midway3-0423) only:**
   - Settings:
     - job name `mrefresh-nest-kbtools`
     - served name `bench-…`
     - `--exclude=midway3-[0600-0606]`
   - The launcher's job name is floor-billed, so first add an override for the job and served names, used only in tests.
   - Check that:
     1. `curl /v1/embeddings` returns vectors
     2. `ai-session kb add` loads 20 PDFs, and a `#kb` question cites them
     3. a seeded tool (`slurm_jobs`) and the MCP `su_usage` are called natively by the 72B model
     4. the knowledge base still exists after stopping and starting a new session
     5. an unauthenticated request to an MCP port is rejected
3. **Afterwards:** `squeue -u $USER` to confirm no GPU job was left running. Commit any measured results out of `_scratch/`.
