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

| Server | File | Tools | Answers |
|---|---|---|---|
| Job queue | `ai-session/mcp/slurm_mcp.py` | `my_jobs`, `job_detail`, `partition_info` | What are my jobs doing? What happened to job N (that I own)? Which parts of the cluster are free? |
| SU usage | `ai-session/mcp/su_usage_mcp.py` | `my_usage`, `my_sessions` | How many Service Units have I been billed, by model and GPU type? What were my individual billed sessions? |

This page is necessarily about the cluster's job scheduler (Slurm): the first
server exists precisely to let an agent query it for you, so the scheduler's own
query commands appear below by name.

Both servers are pure Python standard library (the `mcp` package is not installed
on the cluster and no installs are permitted), so they have no dependencies beyond
a Python interpreter. Run them with the same interpreter the rest of the stack
uses: `/project/rcc/mehta5/conda-envs/vllm-probe/bin/python`.

## What each server exposes

Job-queue server (`slurm_mcp.py`), all queries scoped to you:

| Tool | Argument | Underlying command |
|---|---|---|
| `my_jobs` | `states` (optional, e.g. `RUNNING,PENDING`) | `squeue --me` |
| `job_detail` | `job_id` (required, e.g. `12345` or array task `12345_6`) | `sacct -j <id>`, after verifying you own the job |
| `partition_info` | `partition` (optional, e.g. `gpu` or `gpu,test`) | `sinfo` |

SU-usage server (`su_usage_mcp.py`), reading the billing artifacts:

| Tool | Argument | Reads |
|---|---|---|
| `my_usage` | `since` (optional, `YYYY-MM-DD`) | your `*_summary.json` receipts, plus the central ledger if you can read it |
| `my_sessions` | `since` (optional, `YYYY-MM-DD`) | same sources, one row per billed session |

`my_usage` returns your total billed SU with a breakdown by model and by GPU type;
`my_sessions` lists each session's id, model, GPU type, hours held, SU, and
billing basis. Both draw from your per-session receipts under
`<state-dir>/logs/usage/` and, only if the filesystem lets you read it, the
staff-only central billing ledger. A normal user
cannot read that ledger; the server detects this and answers from your receipts
alone rather than failing. A session that appears in both sources is counted once.

## Security model

These servers are deliberately narrow:

- They run only read-only scheduler queries (`squeue`, `sacct`, `sinfo`,
  `scontrol show`). No job-cancelling, job-submitting, or state-changing command
  is reachable. There is no general "run this command" tool.
- No argument is ever placed into a shell string. Every command is built as an
  argument list and executed without a shell, so shell metacharacters have no
  effect.
- Arguments are validated against fixed patterns before anything runs. A `job_id`
  must match `^[0-9_]+$`; a value such as `1; scancel 999` fails the pattern and the
  call is refused before any process starts. A `since` date must match
  `^\d{4}-\d{2}-\d{2}$`.
- `job_detail` verifies that the job's accounting owner is you before returning any
  detail; a job you do not own is reported as not found.
- The SU tools filter rows to your own username and rely on filesystem permissions
  for everything else; they never read another user's data that the OS would not
  already let you read.

## Enable them in opencode

Add an `mcp` block to your project-local `opencode.json`. This is the same file
described on [opencode and Cline](opencode.md); the entries below replace the
placeholder `flytetest` entry shown there.

```json title="opencode.json (mcp block)"
{
  "mcp": {
    "slurm": {
      "type": "local",
      "enabled": true,
      "command": [
        "/project/rcc/mehta5/conda-envs/vllm-probe/bin/python",
        "/project/rcc/mehta5/vllm/ai-session/mcp/slurm_mcp.py"
      ]
    },
    "su-usage": {
      "type": "local",
      "enabled": true,
      "command": [
        "/project/rcc/mehta5/conda-envs/vllm-probe/bin/python",
        "/project/rcc/mehta5/vllm/ai-session/mcp/su_usage_mcp.py"
      ]
    }
  }
}
```

The servers find your billing receipts automatically for the standard single-user
layout. If you run sessions with a custom state directory (`AISESSION_STATE_DIR`),
pass it through so the SU server looks in the right place, using opencode's
per-server `environment` map:

```json title="opencode.json (su-usage with a custom state dir)"
    "su-usage": {
      "type": "local",
      "enabled": true,
      "command": [
        "/project/rcc/mehta5/conda-envs/vllm-probe/bin/python",
        "/project/rcc/mehta5/vllm/ai-session/mcp/su_usage_mcp.py"
      ],
      "environment": { "AISESSION_STATE_DIR": "/project/rcc/mehta5/ai-session-state/<cnetid>" }
    }
```

Other MCP-capable agents (for example Cline) take the same two values — the
interpreter path and the server script path — in whatever form their configuration
uses to declare a local stdio server.

## Serve a model that can call tools

An MCP tool is invoked through the model's native tool calling, so the session must
serve a model whose tool calls the server parses correctly. Use `qwen2.5_72B` or
`qwen3_4b`, not `qwen2.5_coder_32B`: the server's tool-call parser does not
populate `tool_calls` for the served Qwen2.5-Coder-32B checkpoint (vLLM issue
#29192), so a coder session will not reliably trigger these tools even though the
config is correct. Start the session with tool calling enabled and the 72B model:

```bash
ai-session code --agent --model qwen2.5_72B
```

## Check a server by hand

You can drive either server over stdio without an agent to confirm it responds. The
following sends the initialize handshake and lists the SU tools:

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | /project/rcc/mehta5/conda-envs/vllm-probe/bin/python \
      "$AISESSION_HOME/ai-session/mcp/su_usage_mcp.py"
```

Replace the last two lines with `slurm_mcp.py` to check the Slurm server. Each line
of output is one JSON-RPC response; the `tools/list` response enumerates the tools
the agent will see.

## Responsibilities

An agent that can query your jobs and usage is still an agent taking actions on your
behalf. Before you enable any MCP server, read
[Agent responsibilities and risks](agents.md), which covers what the model is and is
not allowed to do and how to keep an agent session under control. For how the
billed numbers these tools report are computed, see
[Billing and Service Units](../billing.md).
