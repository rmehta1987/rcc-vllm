# Frequently asked questions

This page collects the questions new users ask most often. If your question is about a
specific client, the [Coding Sessions](coding/overview.md) pages and
[Getting Started](getting-started.md) go into more depth; for charges see
[Billing and Service Units](billing.md); for failures see
[Troubleshooting](troubleshooting.md).

## Access and cost

### Who runs this, and how do I get help?

The service is run by RCC staff. For access requests, problems the
[Troubleshooting](troubleshooting.md) page does not solve, or requests that need an
operator (a new model, a larger fine-tuned adapter, a longer time limit), open a
ticket through the standard RCC support channel and mention `ai-session`.

### How do I get access?

You need an RCC account (no special group membership is required) and a Slurm
account and GPU partition to run the GPU job under — the latter are unique to
you and your PI, so you supply them the
first time you start a session (`--account` and `--partition`, then remembered).
Once you have those, you load the module and start a session as described in
[Getting Started](getting-started.md). There is no separate signup step.

### What will a session cost?

Sessions are billed in Service Units (SU), where 1 SU is one A100-GPU-hour. You are charged
the greater of two things: the GPUs you hold (`w_gpu × N × hours`) or the tokens you
process. For interactive work the hold cost almost always dominates. The default coding
session (Qwen2.5-Coder-32B on two A100 cards) costs 2.0 SU per hour. The full formula and
the measured per-model rates are in [Billing and Service Units](billing.md).

### How do I keep the cost down?

Run `ai-session stop` the moment you stop working. An idle session still holds its GPUs
and still bills. For interactive use the standard A100 configurations have the lowest
hold cost; the faster H200 configuration only pays for itself under sustained,
high-throughput generation.

### Is my data private?

Your prompts and the model's completions are served entirely on the cluster and are not
sent to any outside provider. Your browser-chat history is stored in a private directory in
your home folder that other users cannot read. One caveat: coding tools can have their own
telemetry that is separate from the model service. The service disables that telemetry where
it controls the configuration, but you should confirm the settings in any client you install
yourself. A second caveat, opt-in: if you start browser chat with `AISESSION_TOOLS=1` (web
search, URL fetch, reference lookup — see below), those specific tool requests send your query
terms to services outside RCC. They are off unless you set that flag.

## Connecting and sharing

### How do I connect from my laptop?

Open an SSH tunnel from your laptop to the login node your session is running on, then point
your client at `http://localhost:<port>`. The exact tunnel command, with the right port and
login node filled in, is printed when you run `ai-session connect` and is shown in
[Getting Started](getting-started.md).

### Can my lab share one session?

Yes. The person who starts the session receives an access key. Share that key with your
labmates; each of them opens their own SSH tunnel to the same login node and uses the key as
their API key. Anyone without the key is refused. All usage bills to the person who started
the session, so coordinate within the group on who runs it.

### Which model should I use?

Use Qwen2.5-Coder-32B for code, the Qwen2.5-72B general model for mixed prose-and-code work
or when you specifically want the largest general model, and Qwen3-4B for quick or
low-cost tasks. For math and multi-step planning, the Qwen3 thinking models reason
before answering — Qwen3-32B (`--model qwen3_32B`) on two A100s, or the small Qwen3-4B.
A vision model (for images) is still on the roadmap.

Whichever you end up on, start small and scale up: get your prompts or agent setup working
against the small model first — it loads faster, spends less time waiting for free GPUs, and
costs the least per hour — then switch to a larger model without changing any client
configuration. If you need larger, ask; new models are staged on request. For a rough sense of
how these open models compare to closed "frontier" models, see the
[capability frame of reference](reference.md#rough-capability-frame-of-reference).

### Can the browser chat search the web or find papers?

Yes, opt-in. Start browser chat with `AISESSION_TOOLS=1 ai-session chat` to add three tools you
enable per conversation in Open WebUI: web search, URL fetch, and academic reference search
(arXiv, bioRxiv, medRxiv, PubMed, Semantic Scholar, via the open-source `paper-search-mcp`).
Web search works with any model; for the model to *autonomously* call the reference tools, start
a tool-calling model — `AISESSION_TOOLS=1 ai-session chat --model qwen3_32B`. These tools reach
outside RCC (see [Is my data private?](#is-my-data-private) and
[Getting Started](getting-started.md#web-search-and-reference-tools-opt-in)); they are off by
default.

## Coding agents, MCP, and building agents

### Which coding tool should I use?

aider is the dependable default for editing files and does not need tool-calling. opencode
is supported for full tool-calling agents but requires a small workaround file (see
[opencode and Cline](coding/opencode.md)). Continue is the choice for in-editor use inside
VS Code or JetBrains.

### My agent said it made a change, but nothing happened. Why?

The Qwen2.5-Coder-32B model does not emit the tool-call markers that the model server's parser expects,
so tool calls can fail silently: the agent reports success but no file is edited. Use the
`AGENTS.md` workaround documented on the [opencode](coding/opencode.md) page, or switch to
the Qwen2.5-72B or Qwen3 model for agent work, which do not have this problem.

### How do I add an MCP tool to my agent?

Add an `mcp` block to a project-local `opencode.json` in your working directory, not to your
personal configuration; `ai-session mcp config` prints a ready-to-paste block for the two
built-in tool servers (see [MCP Servers](coding/mcp.md)). Before you enable a server, read
the agent-responsibility guidance:
an MCP server runs with your full cluster permissions and can reach any file you can reach,
including a labmate's files through shared project directories.

### Can I build my own agent on these models?

Yes. Point any agent framework that speaks the standard OpenAI API format — for example
PydanticAI, LangGraph, smolagents, or the OpenAI Agents SDK — at the session URL, using
your session key as the API key. For reliable tool-calling, use the Qwen2.5-72B or Qwen3 model rather than the Coder
model.

## Common problems

### My session sits in the queue and never starts.

The GPUs you requested may be busy. A session waits for cards to free up, and the start
command gives up if it waits too long. Try a smaller model (`ai-session fast`), or try
again later.

### My connection worked and then stopped.

If the login node was rebooted, or your SSH session dropped, the gateway — the connection
point on the login node — can stop while the GPU session keeps running. Check with
`ai-session status`; if the connection point is gone but a server is still listed, run
`ai-session stop` to release the GPUs and start again.

### My home directory filled up.

Browser-chat history is stored in your home directory, which has a smaller quota than
project space. Clear old conversations in the chat interface, or remove old files under
`$HOME/.ai-session/`.

### I forgot to stop my session and was billed for idle time.

An unused session bills its GPUs until its time limit expires. Always run
`ai-session stop` when you finish. If you routinely forget, ask the operators whether
the idle-session reaper is enabled, which warns and then ends sessions that have gone
quiet.
