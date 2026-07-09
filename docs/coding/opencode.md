# opencode and Cline (Tool-Calling Agents)

opencode and Cline are autonomous coding agents. They differ from [aider](aider.md)
in how they drive the model: instead of asking for edits as plain text, they use
native function calling (tool calling), where the model returns a
structured call — a function name plus JSON arguments — that the agent then executes
(read a file, apply an edit, run a command (grep), run a tool (squeue)). opencode is a supported client;
it needs the two files described in Step 2 — a provider configuration
(`opencode.json`) and a workaround rules file (`AGENTS.md`) — and occasional
retries even with them. aider remains
the default and recommended client: it performs the same edits through the
chat-completions API without function calling and needs no workaround.

Because these agents need function calling, the session must be started with tool
calling enabled (`ai-session code --agent`); a session started for aider or
[Continue](continue.md) will not accept tool calls. For how sessions, the gateway
(the connection point on the login node), and SSH tunnels fit together, see
[Coding Sessions](overview.md).

## Quick Start

| Step | Description | Command | Run on |
|---|---|---|---|
| 1 | Start a session with tool calling enabled | `ai-session code --agent` | Login node |
| 2 | Get opencode, then create `opencode.json` and `AGENTS.md` in your repository (Step 2 below) | `module load opencode` (login node) or `curl -fsSL https://opencode.ai/install \| bash` (laptop) | Wherever opencode runs |
| 3 | Run opencode inside your git repository | `opencode` | Laptop or login node |
| 4 | Stop the session when finished | `ai-session stop` | Login node |

## Step 1: Start the session with tool calling enabled

Run this **on the login node**, inside `tmux` or `screen` so an SSH disconnect does
not terminate the gateway:

```bash
ai-session code --agent
```

!!! warning "A running session consumes SU whether or not you send requests"
    The reservation floor for the default configuration (Qwen2.5-Coder-32B, 2 A100
    GPUs) is 2.0 SU per hour; see [Billing](../billing.md). Stop with
    `ai-session stop` as soon as you finish.

`--agent` starts the model server with tool calling enabled (the `hermes` parser,
which extracts tool calls from Qwen-model output). This switch controls only tool
calling; the served context length is independent of it and stays at the coding
default of 32768 tokens.

The command blocks until the model is loaded (typically several minutes for the
32B model) and then prints a block containing the session's port (`GW_PORT`), the
connection parameters, and the SSH tunnel command. Note the port; you need it in
Step 2. Verify at any time (no cost):

```bash
ai-session status
```

## Step 2: Configure opencode

On the cluster, opencode is provided as a module — the same mechanism as
`ai-session` itself. **On the login node:**

```bash
module load opencode
opencode --version   # the service currently provides 1.14.41, the verified version
```

If opencode runs on your laptop instead, install it there with the official
script, `curl -fsSL https://opencode.ai/install | bash` (or
`npm install -g opencode-ai`), and open the SSH tunnel printed at start first so
`localhost:<GW_PORT>` reaches the session (see
[Coding Sessions](overview.md)); on a login node no tunnel is needed.

Two files must be placed in the repository you are editing: `opencode.json` (the
provider configuration) and `AGENTS.md` (a rules file without which the model does
not emit parseable tool calls — 0 of 10 task runs passed without it in the
2026-07-03 verification). A project-local `opencode.json` is merged over your
personal `~/.config/opencode/opencode.json`, so nothing personal is modified.

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
  "model": "rcc/qwen2.5_coder_32B",
  "small_model": "rcc/qwen2.5_coder_32B",
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
        "qwen2.5_coder_32B": {
          "name": "Qwen2.5 Coder 32B (local)",
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

### AGENTS.md 

This is mainly needed for **opencode** .

Create `AGENTS.md` in the repository root with exactly the following content.
Reason: the served Qwen2.5-Coder-32B-Instruct checkpoint does not generate the
`<tool_call>` / `</tool_call>` marker tokens (vocabulary ids 151657 and 151658)
that the model server's `hermes` parser matches — at temperature 0 the tags are simply
omitted — so without this file the tool JSON streams back as plain text that
opencode ignores. The rules file instructs the model to write the tags as ordinary
characters, spelled out piece by piece; written that way, the tags survive
generation and the parser matches them after decoding. The file deliberately never
contains the literal tag string, which would tokenize to the very marker token the
model cannot reproduce.

```markdown title="AGENTS.md"
# Rules

CRITICAL tool-call format rule. When you invoke a tool, output the tool-call JSON
wrapped in XML-style tags built EXACTLY as follows.

- Opening tag: the character '<' + the word 'tool' + the characters '_call' + '>'.
- Closing tag: the characters '</' + the word 'tool' + the characters '_call' + '>'.

So every tool invocation looks like (spelling the tags out):
OPENTAG newline JSON newline CLOSETAG, where OPENTAG is '<'+'tool'+'_call'+'>' and
CLOSETAG is '</'+'tool'+'_call'+'>'. Write the tags as literal text characters.

IMPORTANT: the opening tag ENDS with the '>' character. The exact character
sequence is: less-than sign, t, o, o, l, underscore, c, a, l, l, greater-than
sign. The '>' must come IMMEDIATELY after the second 'l', with no newline and
no space before it. Only AFTER the '>' do you start a new line with the JSON.
Emit the opening tag exactly once per call, never twice.

Never output a bare JSON object without these tags. Never use any other tag name.
The JSON has the form {"name": "<name of tool>", "arguments": { ... }}.
One JSON object per tag pair. Output nothing else around the call.

Common mistakes you MUST avoid, on EVERY step of the conversation (including
after you have received tool results):
- WRONG: writing the opening tag but stopping before the '>' (e.g. a line that
  ends after the second 'l'). The tag is INVALID without its final '>'.
- WRONG: writing the opening tag twice in a row.
- WRONG: outputting the JSON with no tags at all.
If your previous message in this conversation contains one of these mistakes,
do NOT imitate it. Emit the correct, complete tags every single time.
```

The same content is kept in section 8.3 of the coding-agents guide that ships
with the service (`$AISESSION_HOME/ai-session/CODING_AGENTS.md`).

### Run

Sanity-check the configuration before spending tokens, then run opencode inside
your git repository:

```bash
opencode models   # must list exactly one model: rcc/qwen2.5_coder_32B
opencode
```

If opencode prints tool-call JSON as ordinary chat text instead of acting on it,
re-issue the instruction (in the verification this occurred in 1 of 4 task runs);
if it recurs persistently, check that `AGENTS.md` is present and switch to
[aider](aider.md) if needed. See also [Troubleshooting](../troubleshooting.md).

### Seeing the model's reasoning (Qwen3 only)

The Qwen3 models think before answering. Served with `--reasoning-parser qwen3`,
their chain of thought comes back in a separate `reasoning_content` field, kept
out of the answer (see [Command Reference](../reference.md)). opencode reads that
field and displays it as a **Thinking** block, but the display is off by default.

To use it, start a Qwen3 session and point opencode at that model:

```bash
ai-session code --model qwen3_4b --agent           # serve a thinking model
# in opencode.json set "model": "rcc/qwen3_4b" and add it under the provider's models
opencode run --thinking --model rcc/qwen3_4b "…"   # prints a "Thinking: …" block, then the answer
```

The interactive TUI shows the thinking block inline above each reply. Only the
Qwen3 models reason this way (`qwen3_4b` and the larger `qwen3_32B`, both served);
the default `qwen2.5_coder_32B` and the Qwen2.5/Llama models do not think, so
`--thinking` has no effect with them. aider, by contrast, does not surface
`reasoning_content` against this endpoint — it shows only the answer.

## Cline

Cline is a VS Code extension in the same class of tool: an autonomous agent driven
by native tool calling. Configure it with the same three values — base URL
`http://localhost:<GW_PORT>/v1`, API key the session access key, model
`qwen2.5_coder_32B` — against a session started with `ai-session code --agent`.
Cline has not been **tested** yet. The missing `<tool_call>` marker
tokens (see Step 2) are a property of the served model, not of opencode, so Cline
is expected to need an equivalent rules file; its mechanism is `.clinerules`
rather than `AGENTS.md`. aider is the fallback.

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
