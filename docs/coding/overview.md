# Coding Sessions

A coding session serves a code-specialized language model on an RCC GPU node and
connects it to a coding tool that reads and edits source code in your git
repository. No prompt, file content, or completion leaves the cluster. The
command-line tools (aider, opencode) run in the terminal where your code already
lives, over SSH, with no graphical display needed and no copy of the repository
on your laptop. The cost is
GPU reservation time, charged in Service Units (SU); 1 SU is 1 A100-GPU-hour (see
[Billing and Service Units](../billing.md)).

The command is `ai-session code`, the coding counterpart to the browser-chat
`ai-session chat` on [Getting Started](../getting-started.md). The difference in
shape: a coding tool is an interactive program you drive by hand, not a background
server, so `ai-session code` starts the session and the gateway — the small
always-on connection point the service runs on the login node — and then prints
the ready-to-run client command, which you run yourself in the repository you want
to edit. The default client is aider; [Continue](continue.md) and
[opencode and Cline](opencode.md) connect to the same endpoint. The stack is the
same three pieces described on [AI Sessions on RCC](../index.md): a model server
on a GPU node, the gateway (which gives every session one stable web address in
the standard OpenAI API format that most AI tools can talk to, and records token
counts for billing), and your client.

## Quick Start

| Step | Description | Command | Run on |
|---|---|---|---|
| 0 | Put `ai-session` on your PATH (once per shell) | `module load ai-session` | Login node |
| 1 | Start the session; wait for the READY block and the printed client command | `ai-session code` (first run: add `--account <acct> --partition <part>`) | Login node |
| 2 | Run the printed client command inside the git repository you want to edit | printed by `ai-session code` (aider by default) | Login node or Local machine |
| 3 | Check what is running at any time; costs nothing | `ai-session status` | Login node |
| 4 | Stop the session, free the GPUs, print the SU charge | `ai-session stop` | Login node |

## Step 1: Start the session and gateway

Run this **on the login node**, inside `tmux` or `screen`, so that an SSH
disconnect does not take the connection point down with it:

```bash
ai-session code
```

The command starts the model server on cluster GPUs, waits until the model is
loaded (typically several minutes for the default 32B model), starts the gateway,
and prints a block containing the session's port, the exact aider command, and the
SSH tunnel command for laptop access. Leave this terminal running.

Options: `--account NAME` and `--partition NAME` name the Slurm account and GPU
partition to run under — required on your first session and then remembered in
`~/.ai-session/config` (see [Getting Started](../getting-started.md#step-1-start-the-session));
`--time HH:MM:SS` sets the session time limit (default `02:00:00`; the session
ends after this even if you forget to stop it, which caps the maximum charge);
`--model KEY` serves a different registered model; `--agent` enables native tool
calling, required by [opencode and Cline](opencode.md) but not by aider or
Continue.

!!! warning "A running session consumes SU whether or not you send requests"
    `ai-session code` starts a billed GPU reservation. Stop it as soon as you
    finish: `ai-session stop`

Verification — the command ends with a READY banner of this form:

```
==> [2/2] starting gateway on 127.0.0.1:<GW_PORT>
    gateway healthy (pid <PID>)  log: .../run/gateway.log

================ READY -- code with the local qwen2.5_coder_32B (ctx 32768) ================
```

followed by the aider command to copy, the connection parameters for other
clients, and the tunnel command. If it exits with an error instead, see
[Troubleshooting](../troubleshooting.md).

## Step 2: Run the coding tool

Open a second terminal, change into the git repository you want to edit, and run
the command printed in Step 1. aider requires the working directory to be inside a
git repository; run `git init` first if necessary. Per-client setup and usage:

- [aider](aider.md) — the default; terminal REPL, edits files as text diffs.
- [Continue (VS Code and JetBrains)](continue.md) — in-editor chat and edit/apply.
- [opencode and Cline (Tool-Calling Agents)](opencode.md) — autonomous agents;
  require the session to be started with `ai-session code --agent`.

Before running any autonomous agent, read
[Agent responsibilities and risks](agents.md): an agent acts with your full
cluster permissions, can read anything you can read (including shared project
directories), and its actions and SU are your responsibility.

To run the tool on your laptop instead of the login node, first open the SSH
tunnel (next section).

Verification — **on the login node**, confirm the connection point answers
before starting the client (exits 0 when healthy):

```bash
curl -sf "http://127.0.0.1:<GW_PORT>/__gateway/health"
```

- Replace `<GW_PORT>` with the port printed at start.

## Step 3: Stop the session

Run this **on the login node** the moment you stop working:

```bash
ai-session stop
```

It meters the session, releases the GPUs (stopping the clock), shuts down the
connection point, and prints the itemized SU charge for the run as its last
output. The same summary is written as a receipt file under your state
directory. `ai-session stop` and
`ai-session status` themselves cost nothing; only the running GPU session does.

Verification — `ai-session status` afterwards reports no session running and no
access key set.

## Connection parameters

Every client uses the same four values. After the session is up, one command loads
them into your shell:

```bash
eval "$(ai-session env)"
```

This sets `AISESSION_BASE_URL`, `AISESSION_API_KEY`, and `AISESSION_MODEL` (also
written to `~/.ai-session/env`, mode 600). `ai-session connect` prints the literal
values and ready-to-paste setup for every client.

| Parameter | Value |
|---|---|
| Base URL | `$AISESSION_BASE_URL` — `http://localhost:<GW_PORT>/v1`; the port is derived from your numeric user ID (`8400 + UID % 90`), so it differs per user |
| API key | `$AISESSION_API_KEY` — the session access key (see below) |
| Model name | `qwen2.5_coder_32B` (default) or `qwen2.5_72B`; must equal the model you started |
| Context window | 32768 tokens for coding sessions (8192 for chat sessions) |

The model name is the identifier the server exposes. Clients that route through
litellm (aider, Continue) prefix it with `openai/` to select the standard
format, for example `openai/qwen2.5_coder_32B`.

### The session access key

Starting a session mints a random access key, and every request to the session
must carry it. `ai-session code` prints it in the READY block and saves it,
readable only by you, at `<state-dir>/logs/gateway/session_key` (mode 600);
`ai-session connect` and `ai-session status` (first six characters only) also show
it. Use it as the API key in every client below.

Because the connection point binds to `127.0.0.1` and accepts only requests
carrying the key, no one else on the shared login node can use your session by
accident, and you can deliberately share it with your lab: give a labmate the
key, have them open their own
tunnel to your `GW_PORT`, and set the key as the API key in their client. All of
their usage bills to you, the starter — one key per session, no per-person split. A
request without the key is refused with HTTP 401. `ai-session stop` deletes the key
file, so the key stops working when the session ends and the next start mints a
fresh one. Only a gateway RCC staff start by hand with no key configured is
keyless, in which case any non-empty string works.

## Remote access from your laptop

The connection point listens on `127.0.0.1:<GW_PORT>` on the login node where you
started the session, so a client on your laptop needs a forwarded port. Run this
**on your local machine** and leave it running:

```bash
ssh -N -L <GW_PORT>:localhost:<GW_PORT> <cnetid>@<login-node>.rcc.uchicago.edu
```

- Replace `<GW_PORT>` with the port printed at start (both occurrences).
- Replace `<cnetid>` with your CNetID.
- Replace `<login-node>` with the login node named in the start output; the
  tunnel must target that node, not an arbitrary one.

`ai-session code` prints a ready-made tunnel command with the node filled in — the
same single-connection form, with `-f` added so the tunnel backgrounds itself once
connected. Only if your network cannot reach the named login node directly, jump
through the round-robin alias as a fallback (this authenticates twice):
`ssh -N -L <GW_PORT>:localhost:<GW_PORT> -J <cnetid>@midway3.rcc.uchicago.edu <cnetid>@<login-node>`.
The client on your laptop then uses `http://localhost:<GW_PORT>/v1` as
its base URL. When the client runs on a login node, no tunnel is needed; use the
same `localhost` URL directly. For what an SSH tunnel is and how to debug one, see
[Getting Started](../getting-started.md).

## Choosing a model

The default is `qwen2.5_coder_32B` (Qwen2.5-Coder-32B-Instruct). To serve the
general 72B model instead, run `ai-session code --model qwen2.5_72B`; the right
GPU configuration is chosen for you. For light work — debugging, quick questions,
or simple edits where you want the cheapest session — serve the small `qwen3_4b`
on a single GPU with `ai-session code --model qwen3_4b`; it holds one GPU at the
1.0 SU/h floor and, unlike the coder model, emits native tool calls reliably, so
it is also the small option for `--agent` clients and MCP tools (see
[opencode and Cline](opencode.md) and [Agent responsibilities](agents.md)).
For a coding model that reasons before answering, serve `qwen3_32B`
(`ai-session code --model qwen3_32B`): a Qwen3 thinking model on two A100s whose
chain of thought is returned separately from the answer — visible in opencode with
`--thinking` (see [opencode](opencode.md#seeing-the-models-reasoning-qwen3-only)).

| Model key | Parameters | GPUs it runs on | Prefill (tok/s) | Decode (tok/s) | Reservation floor |
|---|---|---|---:|---:|---:|
| `qwen3_4b` (cheapest; debugging, simple edits, tool calling) | 4B | 1 x A100-80GB | — | — | 1.0 SU/h |
| `qwen2.5_coder_32B` (default) | 32B | 2 x A100-80GB | 4773 | 1679 | 2.0 SU/h |
| `qwen3_32B` (thinking; reasons before editing) | 32B | 2 x A100-80GB | — | — | 2.0 SU/h |
| `qwen2.5_72B` | 72B | 4 x A100-80GB | 2901 | 1123 | 4.0 SU/h |
| `qwen2.5_72B` (H200 option) | 72B | 2 x H200 | 7594 | 2329 | 6.0 SU/h |

Throughput figures are aggregate, measured by the billing benchmark at concurrency
64 over prefill-heavy, decode-heavy, and balanced request mixes.
They are the basis of the per-token charge, not the latency one user perceives
(next section). The reservation floor is the minimum charge for a session — the
GPU-type weight times the number of GPUs times the hours held; a session bills the
larger of the metered token work and this floor. The canonical rate table and the
full formula are on [Billing and Service Units](../billing.md).

The coder model is the default because it is specialized for code, uses half the
GPUs of the 72B, and has a lower measured per-token cost than the 72B at every
benchmarked configuration. The 72B is preferable for mixed code-and-prose work or
when the larger general model is specifically wanted. The H200 configuration
(faster, higher floor) and other advanced serving overrides are handled by
RCC staff; ask them, or see the staff guide in the repository.

!!! warning "Each of these starts a billed GPU reservation"
    The floors above apply from the moment the session starts. Stop with
    `ai-session stop`.

## Context window and prompt sizing

Coding sessions serve a 32768-token context. This is the native context length of
the Qwen2.5 models; no rope/YaRN scaling is applied. Chat sessions default
to 8192.

The aider metadata file declares the split as 28000 input tokens and 4096 output
tokens, leaving headroom so that prompt plus generated tokens cannot exceed
32768. If your added files exceed the input budget, aider reports the prompt as
too long; see [Troubleshooting](../troubleshooting.md).

Single-stream generation latency for the 32B is approximately 66 ms per output
token (measured median time-per-output-token, same benchmark as above), i.e. on
the order of 15 tokens per second as perceived by one interactive user. The
aggregate throughput figures in the table above are higher because they sum
across 64 concurrent requests.

## Build your own agent

The listed clients are not the only way to use a session. Because every session
exposes the OpenAI chat-completions protocol at the session URL, you can write
your own agent in any framework that speaks the standard format (PydanticAI,
LangGraph, smolagents, or the OpenAI Agents SDK), pointing it at
`$AISESSION_BASE_URL` with
`$AISESSION_API_KEY` as the API key and a served model name (`qwen2.5_72B` or
`qwen3_4b` for tool calling). A runnable, self-contained PydanticAI example ships
with the service; the [Build your own agent](agents.md#build-your-own-agent)
section walks through it.

Use `qwen2.5_72B` or `qwen3_4b`, not the coder model, for any tool-calling
agent: the server's tool-call parser does not populate `tool_calls` for
Qwen2.5-Coder-32B (model-server bug #29192), so a custom agent pointed at the
coder model runs but never calls your tools. Before running an autonomous agent, read
[Agent responsibilities and risks](agents.md): it acts with your full cluster
permissions and its actions are your responsibility.
