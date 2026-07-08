# MCP Servers (Job Queue and SU Usage)

The Model Context Protocol (MCP) is a standard way for a coding agent to expose
extra tools to the language model it drives. An MCP server is a small program the
agent launches; it advertises a fixed set of named tools with typed arguments, and
when the model decides to call one, the agent runs it and feeds the result back
into the conversation. The transport used here is stdio: the agent starts the
server as a subprocess and exchanges newline-delimited JSON-RPC messages with it on
standard input and output. Because the agent starts the server as one of your
processes, an MCP server runs with your own cluster permissions and sees exactly
what you would see from a login-node shell.

This service ships two read-only MCP servers so an agent can answer operational
questions about your work without your having to leave the editor:

| Server | Started as | Tools | Answers |
|---|---|---|---|
| Job queue | `ai-session mcp run jobs` | `my_jobs`, `job_detail`, `partition_info` | Is my session (or any job of mine) running, queued, or finished? What happened to job N? Which parts of the cluster are free? |
| SU usage | `ai-session mcp run usage` | `my_usage`, `my_sessions` | How many Service Units have I been billed, by model and GPU type? What were my individual billed sessions? |

The job-queue server queries the cluster's job scheduler (Slurm) on your behalf —
that is its purpose, so the scheduler's read-only query commands appear by name in
the tables below. You do not need to know or run those commands yourself; the
agent does, through the server.

Both servers are built on the official MCP SDK and already installed;
`module load ai-session` is the only setup. `ai-session mcp run ...` starts them
with the right interpreter and the right settings, so an agent's configuration
never contains an install path.

## What each server exposes

Job-queue server, all queries scoped to you:

| Tool | Argument | Underlying query |
|---|---|---|
| `my_jobs` | `states` (optional, e.g. `RUNNING,PENDING`) | `squeue --me` |
| `job_detail` | `job_id` (required, e.g. `12345` or array task `12345_6`) | `sacct -j <id>`, after verifying you own the job |
| `partition_info` | `partition` (optional, e.g. `gpu` or `gpu,test`) | `sinfo` |

SU-usage server, reading the billing artifacts:

| Tool | Argument | Reads |
|---|---|---|
| `my_usage` | `since` (optional, `YYYY-MM-DD`) | your `*_summary.json` receipts, plus the central ledger if you can read it |
| `my_sessions` | `since` (optional, `YYYY-MM-DD`) | same sources, one row per billed session |

`my_usage` returns your total billed SU with a breakdown by model and by GPU type;
`my_sessions` lists each session's id, model, GPU type, hours held, SU, and
billing basis. Both draw from your per-session receipts and, only if the
filesystem lets you read it, the staff-only central billing ledger. A normal user
cannot read that ledger; the server detects this and answers from your receipts
alone rather than failing. A session that appears in both sources is counted once.

## Security model

These servers are deliberately narrow:

- They run only read-only queries. No job-cancelling, job-submitting, or
  state-changing command is reachable. There is no general "run this command"
  tool.
- No argument is ever placed into a shell string. Every command is built as an
  argument list and executed without a shell, so shell metacharacters have no
  effect.
- Arguments are validated against fixed patterns before anything runs. A `job_id`
  must match `^[0-9_]+$`; a value such as `1; cancel 999` fails the pattern and the
  call is refused before any process starts. A `since` date must match
  `^\d{4}-\d{2}-\d{2}$`.
- `job_detail` verifies that the job's accounting owner is you before returning any
  detail; a job you do not own is reported as not found.
- The SU tools filter rows to your own username and rely on filesystem permissions
  for everything else; they never read another user's data that the OS would not
  already let you read.

## Enable them in opencode

`ai-session mcp config` prints the block below at any time. Add it to your
project-local `opencode.json` — the same file described on
[opencode and Cline](opencode.md); the entries replace the placeholder
`my-personal-server` entry shown there.

```json title="opencode.json (mcp block)"
{
  "mcp": {
    "jobs": {
      "type": "local",
      "enabled": true,
      "command": ["ai-session", "mcp", "run", "jobs"]
    },
    "usage": {
      "type": "local",
      "enabled": true,
      "command": ["ai-session", "mcp", "run", "usage"]
    }
  }
}
```

Start opencode from a shell in which you have run `module load ai-session`, so
that the `ai-session` command is found. There is nothing else to configure: the
servers locate your billing receipts automatically, including when your sessions
use a custom state directory.

Other MCP-capable agents (for example Cline, or Claude Code) take the same two
facts — command `ai-session`, arguments `mcp run jobs` (or `mcp run usage`) — in
whatever form their configuration uses to declare a local stdio server.

## Serve a model that can call tools

An MCP tool is invoked through the model's native tool calling, so the session
must serve a model that emits tool calls reliably. Use `qwen2.5_72B` or
`qwen3_4b`, not the default coder model: the served Qwen2.5-Coder-32B checkpoint
does not emit the tool-call markers the server expects (a known upstream issue),
so a coder session will not reliably trigger these tools even though the
configuration is correct. Start the session with tool calling enabled and the 72B
model:

```bash
ai-session code --agent --model qwen2.5_72B
```

## Check a server by hand

You can drive either server over stdio without an agent to confirm it responds.
The following completes the initialize handshake and lists the usage tools. The
trailing `sleep` keeps the input pipe open a moment so the server flushes its last
reply before the pipe closes (a normal MCP client holds the connection open, so it
never needs this):

```bash
{ printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"probe","version":"1"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'; sleep 1; } \
  | ai-session mcp run usage
```

Replace `usage` with `jobs` to check the job-queue server. Each line of output is
one JSON-RPC response; the `tools/list` response enumerates the tools the agent
will see. (The `notifications/initialized` line is required: the server answers
`tools/list` only after the initialize handshake is complete.)

## Responsibilities

An agent that can query your jobs and usage is still an agent taking actions on your
behalf. Before you enable any MCP server, read
[Agent responsibilities and risks](agents.md), which covers what the model is and is
not allowed to do and how to keep an agent session under control. For how the
billed numbers these tools report are computed, see
[Billing and Service Units](../billing.md).
