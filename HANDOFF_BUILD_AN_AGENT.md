# Handoff: building an agent on an ai-session model

Status 2026-07-08. A work prompt + worked example for building a custom agent
whose reasoning engine is a model you serve through ai-session. It also settles a
common confusion about who does what.

## Who does what (the confusion worth clearing up first)

The intuition "vLLM is the tool caller, the agent is the orchestrator" is half
right. The precise roles:

- **The model, served by vLLM — DECIDES, does not execute.** Given the
  conversation plus the list of available tools (their names and JSON schemas),
  the model decides whether to use a tool and, if so, emits a *structured
  request*: a function name and JSON arguments. vLLM's job is to serve the model
  and PARSE that emission into the OpenAI `tool_calls` field (the `hermes`
  parser, enabled by `--agent`). vLLM never runs the tool. It only conveys the
  model's *request* to run one.
- **The agent — ORCHESTRATES and EXECUTES.** The agent is the loop that actually
  calls the tools. It: (1) sends the conversation + tool schemas to the model;
  (2) receives the tool-call request; (3) RUNS the tool (executes your Python
  function, hits an API, reads a file); (4) appends the result to the
  conversation; (5) sends it back to the model; (6) repeats until the model
  returns a final answer instead of another tool call.
- **The tools — plain functions/APIs** the agent runs on the model's behalf.

So the correction: the **agent is the tool caller (executor) AND the
orchestrator**; the **model (via vLLM) is the tool-call decider/emitter**, not
the caller. Mental model: *the model proposes, the agent disposes.*

Why this matters: because the agent executes the tools, they run in YOUR process
with YOUR permissions — the model never touches your files, it only asks the
agent to. That is exactly the accountability point in `docs/coding/agents.md`.

## The loop, concretely (one tool round-trip)

```
agent → model:   messages=[system, user "how many words in '...'?"]
                 tools=[count_words(text: str) -> int]        # schema advertised
model → agent:   finish_reason="tool_calls"
                 tool_calls=[{id, function:{name:"count_words",
                                            arguments:'{"text":"..."}'}}]
                                                              # model DECIDED
agent:           result = count_words(text="...")  → 9        # agent EXECUTED
agent → model:   append {role:"tool", tool_call_id, content:"9"}, resend
model → agent:   finish_reason="stop", content="There are 9 words."   # final
```

The model emits the middle step; the agent runs `count_words` and feeds `9` back.
Everything is the standard OpenAI chat-completions + tools protocol, so any
framework that speaks it works unchanged against the gateway.

## The runnable example that already ships

`examples/agent_pydantic.py` is a complete, self-contained PydanticAI agent:

- It wraps the gateway's OpenAI-compatible endpoint as a model
  (`OpenAIChatModel(MODEL, provider=OpenAIProvider(base_url=..., api_key=...))`).
- It registers one read-only local tool (`count_words`) with `@agent.tool_plain`
  — PydanticAI advertises the function's name, docstring, and typed signature to
  the model as a tool schema.
- It asks one question that forces a tool call and prints the final answer after
  the round-trip (`agent.run_sync(prompt).output`).
- If `pydantic-ai` is not installed it prints setup instructions and exits 0
  (installs nothing).

Read that file end to end before building anything new — it is the reference.

## How to run it (needs a live GPU session — costs SU)

```bash
# 1. Start a TOOL-CALLING session on a login node, in tmux/screen. Use the 72B
#    or the 4B, NEVER the coder default -- see the model note below.
module load ai-session
ai-session code --agent --model qwen2.5_72B      # or --model qwen3_4b (cheaper)

# 2. Your own venv with the framework (login node has internet):
python -m venv ~/agent-venv && source ~/agent-venv/bin/activate
pip install pydantic-ai

# 3. Load the session URL + key into the shell, then run:
eval "$(ai-session env)"        # sets AISESSION_BASE_URL / AISESSION_API_KEY / AISESSION_MODEL
python /project/rcc/mehta5/vllm/examples/agent_pydantic.py
```

`eval "$(ai-session env)"` is the modern way to get the base URL and key; the
example's docstring still shows the older `run_coding_agent.sh up` +
`8400+UID%90` port form (note: the per-user port derivation is being replaced by
ephemeral ports — see `HANDOFF_MULTIUSER_READINESS.md` item 2 — so always read the
URL from `ai-session env`/`connect`, never recompute it).

## Model choice (a hard gotcha)

Use `qwen2.5_72B` or `qwen3_4b` for ANY tool-calling agent. The served
`qwen2.5_coder_32B` checkpoint does not emit the tool-call marker tokens vLLM's
`hermes` parser matches (vLLM #29192), so `tool_calls` comes back empty and the
agent silently does nothing — no error on either side. The coder model is only
for aider (which drives edits without native tool calling) or opencode with the
AGENTS.md workaround. See `docs/coding/agents.md`.

## Extending it — add your own tool

A tool is just a typed function with a docstring. Replace `count_words` with real
capability, keeping it read-only and side-effect-free unless you have reviewed the
risks in `docs/coding/agents.md`:

```python
@agent.tool_plain
def read_file_head(path: str, n: int = 20) -> str:
    """Return the first n lines of a UTF-8 text file at `path`."""
    with open(path) as f:
        return "".join(f.readlines()[:n])
```

The model now sees `read_file_head` in its tool list and can decide to call it;
the agent (your process, your permissions) executes it. Multi-tool agents just
register several; the model picks per turn. Multi-step tasks fall out of the loop
automatically — the agent keeps round-tripping until the model stops calling
tools.

## Suggested future work (the actual handoff task)

Build a second, richer example that goes beyond one trivial tool, and verify it
live. Options, roughly increasing ambition:

1. **Multi-tool read-only agent** — e.g. `list_dir`, `read_file_head`,
   `grep_repo` — that answers "how is X implemented in this repo?" end to end.
   Shows tool selection and multi-step reasoning.
2. **Agent that uses the shipped MCP servers as tools.** The service already
   provides two read-only MCP servers (`ai-session mcp run jobs|usage`). Wire
   them into an agent (PydanticAI/LangGraph both support MCP stdio clients) so
   the agent can answer "is my session still running and how much have I spent?"
   This demonstrates the model orchestrating EXTERNAL tools, not just in-process
   functions, and reuses the isolated `mcp-env`.
3. **A LangGraph or OpenAI-Agents-SDK variant** of the same task, to show the
   protocol is framework-agnostic (all target the same gateway endpoint).

Deliverable: the new example under `examples/`, a short docs page (or an addition
to `docs/coding/agents.md`) walking through it, and a note in `CHANGELOG.md`.

### Verification (needs the user's go-ahead — costs SU)

The example can only be proven against a LIVE tool-calling session, which
reserves a GPU and bills SU. A `qwen3_4b` session (`ai-session fast`-class, ~1.0
SU/h) is the cheap way to smoke-test tool calling; budget ~0.5 SU for a short
run. DO NOT start it without explicit approval — a prior build agent orphaned a
billed job. Until then, the code can be written and `python -c`-import-checked,
but "it emits and executes tool calls" must be verified on hardware.

## Standing constraints

- Never submit a Slurm job / start a session (`ai-session chat|code|fast`, any
  `up`/`start`, `sbatch`/`srun`/`salloc`) without explicit user approval.
- Do not install into `/project/rcc/mehta5/conda-envs/vllm-probe`. Agent
  frameworks go in the USER's own venv (the example already documents this).
- Keep `mkdocs build --strict` green (`/project/rcc/mehta5/mkdocs-env/bin/mkdocs`).
- Docs style: scientist-to-scientist prose, no emoji/marketing, exact commands.
