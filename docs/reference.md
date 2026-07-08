# Command Reference

This page collects every user-facing command of the ai-session service in one
place: the `ai-session` command and its verbs, the values clients need, and
scripted access to the API. The everyday paths are covered end to end on
[Getting Started](getting-started.md) (browser chat) and
[Coding Sessions](coding/overview.md); come here when you need the full option
list or want to script against the service.

All commands on this page run **on the login node**. A session serves one model
on cluster GPUs; the gateway — the small always-on relay the service runs on the
login node at a fixed per-user port — forwards requests to wherever the current
session is running, so clients keep one stable address. Charges are counted in
Service Units (SU), where 1 SU = 1 A100-GPU-hour; the formula and rates are on
[Billing and Service Units](billing.md).

## Setup

Once per shell:

```bash
module use /project/rcc/mehta5/modulefiles
module load ai-session
```

This puts `ai-session` on your PATH and sets `AISESSION_HOME` to the shared
install. The `module use` line is needed during the current testing phase; once
RCC installs the module centrally, plain `module load ai-session` will suffice.

## Command summary

| Task | Command | Cost |
|---|---|---|
| Start a general chat session (browser UI) | `ai-session chat` | 4.0 SU/h floor |
| Start a coding session (aider/Continue/opencode) | `ai-session code` | 2.0 SU/h floor |
| Start a small, cheap chat session | `ai-session fast` | 1.0 SU/h floor |
| Is the session ready, loading, or stopped? | `ai-session status` | free |
| Print client setup (URL, key, per-client commands) | `ai-session connect` | free |
| Export `AISESSION_*` variables for clients | `eval "$(ai-session env)"` | free |
| List the model presets | `ai-session models` | free |
| Re-print the newest billing receipt | `ai-session receipt` | free |
| Print the agent tool-server (MCP) config block | `ai-session mcp config` | free |
| Run a built-in tool server (agents call this) | `ai-session mcp run jobs` / `ai-session mcp run usage` | free |
| Stop the session, free the GPUs, print the charge | `ai-session stop` | free (ends the billing) |

The start verbs accept:

| Option | Default | Purpose |
|---|---|---|
| `--account NAME` | none — required once | Your Slurm account. Required on the first session, then remembered in `~/.ai-session/config`; there is no default because the account is unique per user/PI. |
| `--partition NAME` | none — required once | The GPU partition to run in. Required on the first session, then remembered. |
| `--time HH:MM:SS` | `02:00:00` | Session time limit. The session ends when it expires even if you forget `stop`, capping the maximum floor charge. |
| `--model KEY` | the preset's model | Serve a different registered model (table below); the GPU configuration is chosen for you. |
| `--agent` | off | `code` only: enable native tool calling, required by [opencode and Cline](coding/opencode.md), not by aider or Continue. |
| `--lora NAME=PATH` | none | Also serve your own fine-tuned adapter under the name `NAME`; repeatable. Validated before anything is reserved. See [Your Own Fine-Tuned Model](lora.md). |

!!! warning "A running session consumes SU whether or not you send requests"
    Every start verb reserves GPUs billed at least the reservation floor until you
    run `ai-session stop`. The floor rate is printed before the session starts.

## Where session state lives

The install is multi-tenant: code, model weights, and the serving environment are
shared and read-only, while everything a session writes goes to a per-user
directory.

- `AISESSION_STATE_DIR` is the per-user writable root — `~/.ai-session/state`
  under your own home directory by default, so a session needs no write access to
  the shared install. It holds session records, the gateway pointer, per-request
  usage capture, billing receipts, and server logs. Set it to a scratch path if
  your home quota is tight.
- `~/.ai-session/env` (mode 600) holds the current session's client settings —
  written by the start verbs and refreshed by `ai-session env` and
  `ai-session connect`.
- Default ports are derived from your numeric user ID so two users on one login
  node do not collide: the gateway listens on `GW_PORT = 8400 + UID % 90` and the
  browser UI on `3000 + UID % 90`. Print yours with
  `echo $((8400 + $(id -u) % 90)) $((3000 + $(id -u) % 90))`.

## Models

`--model` takes a registry key:

| Key | Model | Available | License |
|---|---|---|---|
| `qwen2.5_72B` | Qwen2.5-72B-Instruct | Yes (`chat` preset) | Qwen (Tongyi) community license |
| `qwen2.5_coder_32B` | Qwen2.5-Coder-32B-Instruct | Yes (`code` preset) | Apache-2.0 |
| `qwen3_4b` | Qwen3-4B | Yes (`fast` preset; also `code --model qwen3_4b` for cheap coding) | Apache-2.0 |
| `llama3.1_70B` | Meta-Llama-3.1-70B-Instruct | Operators only | Llama 3.1 Community License + Acceptable Use Policy |
| `qwen2.5_0.5B` | Qwen2.5-0.5B-Instruct | Operators only (smoke tests) | Apache-2.0 |

The license terms and the obligations that apply when you serve these models to
other people are set out on [Model licenses](licenses.md). Serving Llama 3.1
additionally requires a one-time recorded acknowledgment, described there; ask
the operators.

To serve a model you fine-tuned yourself alongside its base model, add
`--lora NAME=PATH` to the start verb; requests whose model is `NAME` are answered
by your fine-tune. Requirements and examples are on
[Your Own Fine-Tuned Model](lora.md).

Qwen3 sessions serve with a reasoning parser, so the model's chain of thought is
returned in a separate `reasoning_content` field and the answer stays in
`content` — the raw `<think>…</think>` block is not mixed into the reply. Qwen2.5
and Llama models do not think and are served without it.

## The session access key

Each session start mints a random access key; the gateway requires it on every
request and refuses requests without it (HTTP 401). The start verbs print it in
the READY block and save it readable only by you; `ai-session connect` re-prints
it, `ai-session env` exports it as `AISESSION_API_KEY`, and `ai-session status`
shows its first six characters. Share it with your lab to let them use your
session over their own tunnel — all of their usage bills to you, the starter.
`ai-session stop` deletes the key, and the next start mints a fresh one.

## Scripted access with curl and Python

Any client that speaks the standard OpenAI API format works against the session
URL. After `eval "$(ai-session env)"`, the base URL is `$AISESSION_BASE_URL`
(`http://localhost:<GW_PORT>/v1`) on the login node where you started the
session; from your laptop, tunnel the port first:

```bash
ssh -N -L <GW_PORT>:localhost:<GW_PORT> <cnetid>@<login-node>.rcc.uchicago.edu
```

- Replace `<GW_PORT>` with your session port (`echo $((8400 + $(id -u) % 90))`).
- Replace `<cnetid>` with your CNetID.
- Replace `<login-node>` with the login node where you started the session (the
  start verbs print it; `hostname -s` on that node shows it).

List the served model — run this **on the login node** (or on your laptop through
the tunnel, with the two variables set to the values `ai-session connect` prints):

```bash
curl -s "$AISESSION_BASE_URL/models" -H "Authorization: Bearer $AISESSION_API_KEY"
```

Expected output (trimmed):

```
{"object": "list", "data": [{"id": "qwen2.5_coder_32B", "object": "model", ...}]}
```

A minimal chat completion with the `openai` Python package (install it in your
own environment):

```python title="chat_example.py"
import os
from openai import OpenAI

client = OpenAI(base_url=os.environ["AISESSION_BASE_URL"],
                api_key=os.environ["AISESSION_API_KEY"])
resp = client.chat.completions.create(
    model=os.environ["AISESSION_MODEL"],
    messages=[{"role": "user", "content": "Write a one-line docstring for a matrix transpose function."}],
)
print(resp.choices[0].message.content)
print(resp.usage)
```

The gateway records per-request token usage automatically: every chat/completions
response's `usage` object is appended to a per-day usage log under your state
directory, and for streaming requests the gateway asks the engine to report usage
in the final stream chunk. `ai-session stop` consumes this log as the billing
source, so scripted clients need no billing instrumentation. Server-side tool
calling for agent frameworks requires a session started with
`ai-session code --agent`; opencode support was verified against the live service
on 2026-07-03; see the [coding agents guide](coding/opencode.md) for caveats.

??? question "What does the gateway do with paths other than /v1?"
    The gateway proxies `/v1`, `/metrics`, `/health`, `/version`, `/ping`,
    `/tokenize`, `/detokenize`, and `/pooling` to the current backend; other paths
    return 404, and the bare `/` returns a JSON hint. Its own health check is
    `GET /__gateway/health`, which reports gateway liveness and whether a backend
    is published (`{"gateway":"ok","backend_active":true|false}`) but not the
    backend's internal address — that endpoint needs no key, so the address is
    withheld. A keyless structured-status route, `GET /status`, answers
    `ready` / `loading` / `no_backend`, which is what `ai-session status` shows.
    When no session is active, proxied requests return 503 with
    `"type": "no_backend"`; see [Troubleshooting](troubleshooting.md).

## Checking a charge

`ai-session stop` prints the itemized charge; `ai-session receipt` re-prints the
newest receipt, and `ai-session receipt <file>` renders an older one. How the
bill is computed — token sources, the floor, cross-checks, and the fallbacks for
unrated configurations — is on [Billing and Service Units](billing.md).

## For administrators

The advanced launcher (per-flag control of the serving configuration, GPU type
selection, serving context length, memory utilization, alternative accounts and
partitions), the raw wrapper scripts and their environment variables, gateway
internals, the billing benchmark, and rate-table maintenance are documented in
the operator guide, `ai-session/README.md` in the service repository. The billing
policy and rate table are editable by RCC staff only. Users never need these; if
a preset does not fit your case, ask the operators.
