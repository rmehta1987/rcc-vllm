# opencode and Cline (Tool-Calling Agents)

opencode and Cline are autonomous coding agents. They differ from [aider](aider.md)
in how they drive the model: instead of asking for edits as plain text, they use
native function calling (tool calling), where the model returns a
structured call — a function name plus JSON arguments — that the agent then executes
(read a file, apply an edit, run a command (grep), run a tool (squeue)). opencode is a
supported client; it needs the provider configuration (`opencode.json`) described in Step 2,
and occasionally a retry. aider remains the default and recommended client: it performs the
same edits through the chat-completions API without function calling and needs no
per-repository configuration at all.

Because these agents need function calling, the session must be started with tool
calling enabled (`ai-session code --agent`); a session started for aider or
[Continue](continue.md) will not accept tool calls. For how sessions, the gateway
(the connection point on the login node), and SSH tunnels fit together, see
[Coding Sessions](overview.md).

## Quick Start

| Step | Description | Command | Run on |
|---|---|---|---|
| 1 | Start a session with tool calling enabled | `ai-session code --agent` | Login node |
| 2 | Get opencode, then create `opencode.json` in your repository (Step 2 below) | `module load opencode` (login node) or `curl -fsSL https://opencode.ai/install \| bash` (laptop) | Wherever opencode runs |
| 3 | Run opencode inside your git repository | `opencode` | Laptop or login node |
| 4 | Stop the session when finished | `ai-session stop` | Login node |

## Step 1: Start the session with tool calling enabled

Run this **on the login node**, inside `tmux` or `screen` so an SSH disconnect does
not terminate the gateway:

```bash
ai-session code --agent
```

!!! warning "A running session consumes SU whether or not you send requests"
    The reservation floor for the default configuration (Qwen3.8-27B, 2 A100
    GPUs) is 2.0 SU per hour; see [Billing](../billing.md). Stop with
    `ai-session stop` as soon as you finish.

`--agent` starts the model server with tool calling enabled, selecting the tool-call
parser that matches the model you are serving. This switch controls only tool calling; the
served context length is independent of it and stays at the coding default of 32768 tokens.

The command blocks until the model is loaded (typically several minutes for the
27B model) and then prints a block containing the session's port (`GW_PORT`), the
connection parameters, and the SSH tunnel command. Note the port; you need it in
Step 2. Verify at any time (no cost):

```bash
ai-session status
```

## Step 2: Configure opencode

On the cluster, opencode is a central module, the same as `ai-session` itself:
`module load opencode` works from any login node with no `module use` line and no
special group membership. **On the login node:**

```bash
module load opencode
opencode --version   # the service currently provides 1.14.41, the verified version
```

If opencode runs on your laptop instead, install it there with the official
script, `curl -fsSL https://opencode.ai/install | bash` (or
`npm install -g opencode-ai`), and open the SSH tunnel printed at start first so
`localhost:<GW_PORT>` reaches the session (see
[Coding Sessions](overview.md)); on a login node no tunnel is needed.

One file must be placed in the repository you are editing: `opencode.json` (the
provider configuration). A project-local `opencode.json` is merged over your personal
`~/.config/opencode/opencode.json`, so nothing personal is modified.

### opencode.json

The verified example file ships with the service. If your repository is on the
cluster, copy it (**on the login node**, after `module load ai-session`), and
load the endpoint and key into your shell — the file references them as
environment variables, so there is nothing to edit for the connection:

```bash
cd /path/to/your/repo
cp "$AISESSION_HOME/ai-session/opencode.example.json" ./opencode.json
eval "$(ai-session env)"
```

If your repository is on your laptop, create `opencode.json` in the repository
root with exactly the following content, which reproduces the example file:

```json title="opencode.json"
{
  "$schema": "https://opencode.ai/config.json",
  "model": "rcc/qwen3.8_27B",
  "small_model": "rcc/qwen3.8_27B",
  "share": "disabled",
  "autoupdate": false,
  "enabled_providers": ["rcc"],
  "mcp": {
    "my-personal-server": {
      "type": "local",
      "enabled": false,
      "command": ["true"]
    }
  },
  "provider": {
    "rcc": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "RCC local vLLM",
      "options": {
        "baseURL": "{env:AISESSION_BASE_URL}",
        "apiKey": "{env:AISESSION_API_KEY}"
      },
      "models": {
        "qwen3.8_27B": {
          "name": "Qwen3.8-27B (local)",
          "limit": {
            "context": 32768,
            "output": 8192
          }
        }
      }
    }
  }
}
```

- The `{env:...}` references resolve from opencode's environment. On the login
  node, `eval "$(ai-session env)"` sets both variables. On your laptop, export
  them yourself with the values `ai-session connect` prints (the base URL is the
  same `http://localhost:<GW_PORT>/v1` once the tunnel is open):
  `export AISESSION_BASE_URL=... AISESSION_API_KEY=...`. The key is required;
  a request without it is refused with HTTP 401. See
  [Coding Sessions](overview.md#the-session-access-key) for sharing it with your lab.
- The `mcp` block is the only part of the file a user ever edits. Replace
  `my-personal-server` with the name of each MCP server in your personal
  `~/.config/opencode/opencode.json`, one disabled entry per server; delete the
  `mcp` block if you have none. The connection is never configured by editing
  the file: the URL and key always arrive through the two environment
  variables, so the same file works unchanged for every session, port, and key.

What the entries do: the `rcc` provider block routes requests through the generic
OpenAI-compatible adapter (`@ai-sdk/openai-compatible`) to the session URL with the
session access key; `model` and `small_model` both point at the local model, so no
request leaves the cluster (opencode's default `small_model`, used for session
titles, is an externally hosted model); `enabled_providers` makes the local provider
the only selectable one; `share` is disabled and `autoupdate` is off, so the tool
does not contact opencode's external services while you work; the `limit` block
declares the served 32768-token context and an 8192-token output cap so opencode
sizes its prompts correctly.

The `mcp` block matters more than it looks: MCP servers from your personal
configuration are advertised to the model as extra tools and inflate every prompt.

### If you have an `AGENTS.md` workaround file, delete it

Earlier versions of these instructions required an `AGENTS.md` rules file in your repository
root that told the model to spell out `<tool_call>` tags character by character. **That file
is now counterproductive: delete it.** Every served model emits tool calls natively, and the
launcher selects the parser that matches the model — `qwen3_coder` for `qwen3.8_27B`, whose
calls come back in an XML form:

```
<tool_call>
<function=read_file>
<parameter=path>
/etc/hostname
</parameter>
</function>
</tool_call>
```

An `AGENTS.md` file that asks for a hand-written format instead tells the model to produce
something that is not its native output. `AGENTS.md` remains useful for ordinary project
instructions; it is only the tool-call workaround that must go.

### Run

Sanity-check the configuration before spending tokens, then run opencode inside
your git repository:

```bash
opencode models   # must list exactly one model: rcc/qwen3.8_27B
opencode
```

If opencode occasionally prints tool-call JSON as ordinary chat text instead of acting on
it, re-issue the instruction. If it happens on every turn, check that the session was
started with `--agent` and that you have no leftover `AGENTS.md` workaround file; switch to
[aider](aider.md) if it persists. See also [Troubleshooting](../troubleshooting.md).

### Seeing the model's reasoning

The Qwen3-family models think before answering. Served with `--reasoning-parser qwen3`,
their chain of thought comes back in a separate `reasoning_content` field, kept
out of the answer (see [Command Reference](../reference.md)). opencode reads that
field and displays it as a **Thinking** block, but the display is off by default.

To use it, start a thinking session and point opencode at that model:

```bash
ai-session code --model qwen3_4b --agent           # serve a thinking model
# in opencode.json set "model": "rcc/qwen3_4b" and add it under the provider's models
opencode run --thinking --model rcc/qwen3_4b "…"   # prints a "Thinking: …" block, then the answer
```

The interactive TUI shows the thinking block inline above each reply. `qwen3_4b` and the
coding default `qwen3.8_27B` reason this way by default. `gemma4_31B` can, but only when you
ask for it per request; the Qwen2.5 models do not think at all, so `--thinking` has no
effect with them. aider, by contrast, does not surface `reasoning_content` against this
endpoint — it shows only the answer.

## Cline

Cline is a VS Code extension in the same class of tool: an autonomous agent driven
by native tool calling. Configure it with the same three values — base URL
`http://localhost:<GW_PORT>/v1`, API key the session access key, model
`qwen3.8_27B` — against a session started with `ai-session code --agent`.
Cline has not been **tested** against this service; nothing is known to be wrong with it,
but nothing has been verified either. aider is the fallback.

## Step 3: Stop the session

!!! warning "Stop the session as soon as you stop working"
    A session is billed at least its reservation floor — GPU-type weight times GPU
    count times hours held — regardless of request volume. Run `ai-session stop`
    immediately when you finish:

```bash
ai-session stop
```

This meters the session, releases the GPUs, stops the gateway, and prints the SU
charge for the run.
