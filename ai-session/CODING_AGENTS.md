# Using the local code model from a coding tool

This document describes how to connect a coding tool (aider, Continue, opencode) to
the on-cluster language-model service and use it to read and edit source code. The
model runs on an RCC compute node; no source code, prompt, or completion is sent off
the cluster. The cost is GPU reservation time, charged in Service Units (SU); see
[Billing](#billing).

This is the end-user document. Operator/runtime details are in
[`README.md`](README.md); the charging policy is in [`BILLING_POLICY.md`](BILLING_POLICY.md).

## Contents

1. [Architecture](#1-architecture)
2. [Prerequisites](#2-prerequisites)
3. [Procedure: start, use, stop](#3-procedure-start-use-stop)
4. [Connection parameters](#4-connection-parameters)
5. [Remote access over SSH](#5-remote-access-over-ssh)
6. [aider](#6-aider)
7. [Continue (VS Code / JetBrains)](#7-continue-vs-code--jetbrains)
8. [opencode](#8-opencode)
9. [Model selection](#9-model-selection)
10. [Context window and prompt sizing](#10-context-window-and-prompt-sizing)
11. [Billing](#11-billing)
12. [Data residency](#12-data-residency)
13. [Troubleshooting](#13-troubleshooting)
14. [Design notes](#14-design-notes)

---

## 1. Architecture

Three processes are involved:

```
coding tool            gateway (login node)           vLLM server (compute node)
your laptop/login  ->  http://localhost:<port>/v1  -> http://<node>:<port>/v1
(aider/Continue/...)   stable URL, reverse proxy      ephemeral, GPU-backed, SU-billed
```

- The **vLLM server** runs as a Slurm job on a GPU node and serves an
  OpenAI-compatible HTTP API. Its node and port change every session, and the
  compute node has no inbound network route from your laptop.
- The **gateway** is a reverse proxy on a login node at a fixed port. It forwards to
  whatever vLLM server the current session is using, so the client sees one URL that
  does not change between sessions. It also records per-request token counts for
  billing.
- The **coding tool** is configured once against the gateway URL.

The helper script `ai-session/run_coding_agent.sh` starts the vLLM server and the
gateway together (`up`), and stops both and reports the charge (`down`).

## 2. Prerequisites

- An account in the `rcc-staff` Slurm account with access to the project tree
  `/project/rcc/mehta5`. The shared environment, model weights, and scripts are
  read-only to the group; you do not install anything.
- The coding tool itself:
  - aider is already installed at `/project/rcc/mehta5/aider-env/bin/aider`.
  - Continue is a VS Code / JetBrains extension you install in your own editor.
  - opencode you install yourself on the machine where it runs (laptop or login
    node): `curl -fsSL https://opencode.ai/install | bash` (installs to
    `~/.opencode/bin/`; npm alternative: `npm install -g opencode-ai`). It is
    optional and not the default.
- A git repository to edit. aider requires the working directory to be inside a git
  repository; run `git init` first if necessary.

No environment activation is required to run `run_coding_agent.sh`; it calls the
shared Python environment by absolute path.

## 3. Procedure: start, use, stop

Run these from a login node, inside `tmux` or `screen` so that an SSH disconnect does
not terminate the gateway.

Step 1 — start the session and gateway:

```bash
bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh up
```

This submits the Slurm job, waits until the model is loaded (typically several
minutes for the 32B model), starts the gateway, and prints a block containing the
gateway port, the exact aider command, and the SSH tunnel command. Leave this
terminal running.

Step 2 — run the coding tool. For aider, open a second terminal, change into your
repository, and run the command printed in Step 1 (see [Section 6](#6-aider) for the
full form and an explanation of each flag).

Step 3 — stop the session when finished:

```bash
bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh down
```

This meters the session, cancels the Slurm job (releasing the GPU), stops the
gateway, and prints the SU charge for the run.

To check what is running at any time (no cost):

```bash
bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh status
```

The GPU is reserved exclusively for the lifetime of the session and is billed for
that wall-clock time whether or not requests are in flight. Run `down` as soon as you
stop working.

## 4. Connection parameters

Every client uses the same four values. The port is derived from your user ID and is
printed by `up` and `status`.

| Parameter | Value |
|---|---|
| Base URL | `http://localhost:<GW_PORT>/v1` |
| API key | the session access key that `up` printed (see below) |
| Model name | `qwen2.5_coder_32B` (default) or `qwen2.5_72B`; must equal the model you started |
| Context window | 32768 tokens for coding sessions (8192 for the default chat sessions) |

The model name is the identifier vLLM serves under (`--served-model-name`). For
OpenAI-compatible clients that route through litellm (aider, Continue), prefix it
with `openai/`, for example `openai/qwen2.5_coder_32B`.

### The session access key

`up` mints a random access key for the session and the gateway requires it on every
request. `up` prints it in the READY block and saves it, readable only by you, at
`<state-dir>/logs/gateway/session_key` (mode 600); `connect` and `status` (first six
characters only) also show it. Use it as the API key (the `<SESSION_KEY>` placeholder)
in every client below. A request without it is refused with HTTP 401.

The key is what lets you share one session with your lab. The gateway binds to
`127.0.0.1` and accepts only requests carrying the key, so no one else on the shared
login node can use your session by accident. To share it, give a labmate the key, have
them open their own tunnel to your `GW_PORT`, and set the key as the API key in their
client. All of their usage bills to you, the starter — one key per session, no
per-person split. `down` deletes the key file, so the key stops working when the
session ends and the next `up` mints a fresh one. Only a gateway you start by hand with
no `AISESSION_GATEWAY_KEY` is keyless, in which case any non-empty string works.

## 5. Remote access over SSH

The gateway listens on `127.0.0.1:<GW_PORT>` on the login node where you ran `up`. To
reach it from a coding tool on your laptop, forward that port. The exact command,
with your login node filled in, is printed by `up`. Its form is:

```bash
ssh -N -L <GW_PORT>:localhost:<GW_PORT> -J <user>@midway3.rcc.uchicago.edu <user>@<login-node>
```

Leave this running. The client on your laptop then uses
`http://localhost:<GW_PORT>/v1` as its base URL. On a login node you can skip the
tunnel and use the same `localhost` URL directly.

## 6. aider

aider drives the model through the chat-completions API and applies edits as text
(unified diffs or whole-file replacements). It does not depend on the model's
function-calling support. It is the default client.

The command printed by `up`, in full:

```bash
cd /path/to/your/repo
OPENAI_API_BASE=http://localhost:<GW_PORT>/v1 \
OPENAI_API_KEY=<SESSION_KEY> \
/project/rcc/mehta5/aider-env/bin/aider \
  --model openai/qwen2.5_coder_32B \
  --weak-model openai/qwen2.5_coder_32B \
  --model-metadata-file /project/rcc/mehta5/vllm/ai-session/aider_model_metadata.json \
  --edit-format diff --analytics-disable
```

Flags:

| Flag | Purpose |
|---|---|
| `--model openai/<key>` | Selects the served model. The `openai/` prefix selects the OpenAI-compatible protocol in litellm. |
| `--weak-model openai/<key>` | Routes aider's auxiliary requests (commit messages, history summarization) to the same local model rather than to `api.openai.com`. |
| `--model-metadata-file <path>` | Declares the model's context window (32768) and zero token cost to litellm. Without it, litellm cannot size prompts and prints `Unknown context window size`. |
| `--edit-format diff` | Requests unified-diff edits instead of full-file rewrites. Set `EDIT_FORMAT=whole` on `up` to switch to whole-file edits if diffs are rejected for a given file. |
| `--analytics-disable` | Permanently disables aider's own usage telemetry. This is a property of the client, separate from the model traffic (which never leaves RCC); disable it in your client as a matter of course. See [Section 12](#12-data-residency). |

The two `OPENAI_API_*` variables are read by litellm; the API base must include the
`/v1` suffix.

### In-session commands

| Command | Effect |
|---|---|
| `/add <path>` | Add a file to the editable context. |
| `/drop <path>` | Remove a file from the context. |
| `/ask <question>` | Ask a question without editing files. |
| `/tokens` | Report current context token usage. |
| `/clear` | Clear conversation history; added files remain. |
| `/run <cmd>` | Run a shell command and optionally add its output to the context. |

aider maintains a repository map so the model has structural awareness of files you
have not explicitly added; add files with `/add` only when you intend to edit them.

### Non-interactive use

To make a single edit and exit, for example from a batch script:

```bash
OPENAI_API_BASE=http://localhost:<GW_PORT>/v1 \
OPENAI_API_KEY=<SESSION_KEY> \
/project/rcc/mehta5/aider-env/bin/aider \
  --model openai/qwen2.5_coder_32B \
  --weak-model openai/qwen2.5_coder_32B \
  --model-metadata-file /project/rcc/mehta5/vllm/ai-session/aider_model_metadata.json \
  --edit-format diff --analytics-disable \
  --yes-always --no-auto-commit \
  --message "add type hints to the public functions in utils.py"
```

Standard input can be piped in, for example to analyze a job log:

```bash
cat slurm-${SLURM_JOB_ID}.out | \
OPENAI_API_BASE=http://localhost:<GW_PORT>/v1 OPENAI_API_KEY=<SESSION_KEY> \
/project/rcc/mehta5/aider-env/bin/aider --model openai/qwen2.5_coder_32B \
  --weak-model openai/qwen2.5_coder_32B \
  --model-metadata-file /project/rcc/mehta5/vllm/ai-session/aider_model_metadata.json \
  --analytics-disable \
  --message "explain the traceback in this log and propose a fix"
```

## 7. Continue (VS Code / JetBrains)

Continue is an editor extension. Install it from the VS Code or JetBrains
marketplace, open the SSH tunnel ([Section 5](#5-remote-access-over-ssh)), and add a
model definition pointing at the gateway.

`~/.continue/config.yaml`. The `allowAnonymousTelemetry: false` line turns off
Continue's own usage telemetry, which is a client concern separate from the model
traffic (that never leaves RCC; see [Section 12](#12-data-residency)):

```yaml
allowAnonymousTelemetry: false
models:
  - name: Qwen2.5-Coder-32B (RCC)
    provider: openai
    model: qwen2.5_coder_32B
    apiBase: http://localhost:<GW_PORT>/v1
    apiKey: <SESSION_KEY>
    roles: [chat, edit, apply]
```

Older Continue versions use `~/.continue/config.json`:

```json
{
  "allowAnonymousTelemetry": false,
  "models": [
    {
      "title": "Qwen2.5-Coder-32B (RCC)",
      "provider": "openai",
      "model": "qwen2.5_coder_32B",
      "apiBase": "http://localhost:<GW_PORT>/v1",
      "apiKey": "<SESSION_KEY>"
    }
  ]
}
```

Use the chat and edit/apply features. Tab-autocomplete requires low latency and is
not suitable for the 32B model on a shared session; leave it disabled, or run a
separate small-model session (for example `qwen3_4b`) for autocomplete only.

## 8. opencode

opencode is an autonomous agent that drives the model through native function/tool
calling: the model returns a structured call (a function name plus JSON arguments)
that the agent executes — read a file, apply an edit, run a command. It is a
supported client, verified end-to-end against this service on 2026-07-03. It is not
the default client: it works only with the two workaround files described below, and
occasional retries are still needed even with them. For routine editing, aider
(Section 6) performs the same edits without function calling and needs no
workarounds.

### 8.1 Verified configuration and measured reliability

The verification of 2026-07-03 (Slurm job 51391003) ran opencode 1.14.41
non-interactively (`opencode run`) against a session started with:

```bash
AGENT_CLIENT=1 MODEL=qwen2.5_coder_32B TP=2 CONSTRAINT=A100 \
  bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh up
```

that is, Qwen2.5-Coder-32B-Instruct on two A100-80GB, served by vLLM 0.10.2 with
`--enable-auto-tool-choice --tool-call-parser hermes`. Three graded tasks were run
in a scratch git repository: create a new file, modify an existing function, and
read a file then edit it based on the value read (the third passes only if a read
tool call actually preceded the edit).

| Quantity | Measured value (2026-07-03, job 51391003) |
|---|---|
| Graded tasks passed | 3 of 3, all through native tool calls (write; edit; read then edit) |
| Task runs without the `AGENTS.md` workaround | 0 of 10 passed |
| Task runs with both workaround files in place | 4 total: 3 passed on the first attempt, 1 needed one retry |
| Tool-call parser exceptions in the vLLM server log | 0 |
| Gateway requests | 52 (249,551 prompt tokens, 3,289 completion tokens) |
| Session wall time and charge | 28 min 53 s; 0.9622 SU (reservation floor, 2.0 SU/h) |

Failures do not appear as server-side parse errors; they appear client-side as the
tool-call JSON streaming back as ordinary assistant text, which opencode does not
execute and does not report as an error. One passing edit also left a harmless
duplicated dead-code fragment inside the new function. Review diffs as you would
with any client, and re-issue the instruction if opencode prints JSON instead of
acting.

### 8.2 Why two workaround files are required

Out of the box (session correctly started with `AGENT_CLIENT=1`, provider block
correctly configured), opencode failed 10 of 10 task attempts against this service,
for two independent reasons, both measured on 2026-07-03:

1. The served Qwen2.5-Coder-32B-Instruct checkpoint does not generate the
   `<tool_call>` / `</tool_call>` marker tokens (vocabulary ids 151657 and 151658)
   that vLLM's hermes parser matches. At temperature 0, asked to reproduce a
   well-formed tool call byte for byte, the model's first emitted token is `{"` —
   the tags are omitted. The parser therefore never produces a `tool_calls`
   response and the tool JSON is returned as plain message text. The fix is an
   `AGENTS.md` rules file (below) that instructs the model to write the tags as
   ordinary characters; spelled out as text, the tags survive generation and the
   parser matches them after decoding.
2. MCP servers configured in your personal `~/.config/opencode/opencode.json` are
   advertised to the model as additional tools and inflate every prompt. In the
   verification, one personal MCP server raised the first request to 27,925 input
   tokens; vLLM rejected it (HTTP 400: input tokens plus the 8,192-token output
   budget exceed the 32,768 context) and opencode surfaced no error. The fix is a
   project-local `opencode.json` that disables personal MCP servers and restricts
   opencode to the local provider.

### 8.3 Procedure

Step 1 — start the session with tool calling enabled:

```bash
AGENT_CLIENT=1 bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh up
```

Step 2 — copy the proven configuration into the repository you will edit:

```bash
cd /path/to/your/repo
cp /project/rcc/mehta5/vllm/ai-session/opencode.example.json ./opencode.json
```

The example is the exact file used in the verification:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "rcc/qwen2.5_coder_32B",
  "small_model": "rcc/qwen2.5_coder_32B",
  "share": "disabled",
  "autoupdate": false,
  "enabled_providers": ["rcc"],
  "mcp": {
    "flytetest": {
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
        "baseURL": "http://localhost:8450/v1",
        "apiKey": "<SESSION_KEY>"
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

Adapt two things: replace `8450` in `baseURL` with the gateway port printed by
`up`, and replace `flytetest` with the name of each MCP server in your personal
`~/.config/opencode/opencode.json` (add one disabled entry per server; delete the
`mcp` block if you have none). A project-local `opencode.json` is merged over your
personal configuration, so nothing personal is modified. `small_model` must point
at the local provider: opencode otherwise sends session-title requests to an
externally hosted model. `enabled_providers` restricts opencode to the local
provider so no external API is selectable; `share: "disabled"` prevents session
upload.

Step 3 — create `AGENTS.md` in the repository root with exactly the following
content. Without this file the model does not emit parseable tool calls (0 of 10
task runs passed; see 8.2). The tag is deliberately spelled out in pieces: if the
literal tag string appeared in the file, it would tokenize to the marker token the
model cannot reproduce, and the instruction would be lost.

```markdown
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

Step 4 — sanity-check before spending tokens, then run:

```bash
opencode models   # must list exactly one model: rcc/qwen2.5_coder_32B
opencode          # interactive; or: opencode run "instruction"
```

If `opencode models` lists anything besides `rcc/qwen2.5_coder_32B`, the
project-local `opencode.json` is not being picked up; run opencode from the
directory containing it.

Cline (a VS Code extension) is the same class of tool. It has not been verified
against this service; reason 1 in 8.2 is a property of the served model, so Cline
is expected to need an equivalent rules file (its mechanism is `.clinerules`, not
`AGENTS.md`). If tool calls fail repeatedly with either tool, use aider, which
performs the same edits without function calling.

## 9. Model selection

The default is `qwen2.5_coder_32B` (Qwen2.5-Coder-32B-Instruct). Override the model,
tensor-parallel degree, and GPU constraint with environment variables on `up`.

Throughput figures below are aggregate, measured by the billing benchmark at
concurrency 64 over prefill-heavy, decode-heavy, and balanced request mixes; they are
the basis for the per-token charge, not single-stream latency.

| Model key | Parameters | Config | Prefill (tok/s) | Decode (tok/s) | Reservation floor |
|---|---|---|---:|---:|---:|
| `qwen2.5_coder_32B` (default) | 32B | TP=2, 2×A100-80GB | 4773 | 1679 | 2.0 SU/h |
| `qwen2.5_72B` | 72B | TP=4, 4×A100-80GB | 2901 | 1123 | 4.0 SU/h |
| `qwen2.5_72B` | 72B | TP=2, 2×H200 | 7594 | 2329 | 6.0 SU/h |

Overrides:

```bash
# general 72B model on A100:
MODEL=qwen2.5_72B TP=4 CONSTRAINT=A100 \
  bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh up

# coder model on the H200 throughput tier:
MODEL=qwen2.5_coder_32B TP=2 CONSTRAINT=H200 \
  bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh up
```

The coder model is the default because it is specialized for code, uses half the GPUs
of the 72B (TP=2 versus TP=4), and has a lower measured per-token cost than the 72B at
every benchmarked tier. The 72B is preferable for mixed code-and-prose work or when
the larger general model is specifically wanted.

`CONSTRAINT=A100` selects the 80 GB A100 nodes (uppercase). The 32B at TP=2 requires
two 80 GB cards; the 40 GB A100 nodes (lowercase `a100`) are not sufficient.

## 10. Context window and prompt sizing

Coding sessions serve a 32768-token context (set by `MAX_MODEL_LEN`, default 32768).
This is the native context length of the Qwen2.5 models; no rope/YaRN scaling is
applied. Chat sessions started by other helpers default to 8192.

The aider metadata file declares the split as 28000 input tokens and 4096 output
tokens, leaving headroom so that prompt plus generated tokens cannot exceed 32768.
If you add more file content than fits in 28000 tokens, aider will report the prompt
as too long; remove files with `/drop` or reset history with `/clear`.

Single-stream generation latency for the 32B at TP=2 is approximately 66 ms per
output token (measured median time-per-output-token), i.e. on the order of 15 tokens
per second as perceived by one interactive user. Aggregate throughput across
concurrent requests is higher (the table in Section 9).

## 11. Billing

A session reserves the GPU node for its wall-clock lifetime. The charge is the larger
of a token term and a reservation floor:

```
token_su = w_gpu(tier) * N_gpus * (T_in / prefill_tps + T_out / decode_tps) / 3600
floor_su = w_gpu(tier) * N_gpus * reserved_wall_hours
billed   = max(token_su, floor_su)
```

`w_gpu` is the GPU-tier weight (A100 = 1.0, H200 = 3.0); `N_gpus` is the number of
GPUs reserved; `prefill_tps` and `decode_tps` are the measured rates in Section 9.
For interactive coding the request volume is low and the reservation floor dominates,
so the practical cost is `w_gpu * N_gpus * hours`: 2.0 SU per hour for the default
32B at TP=2 on A100. The 32K context does not change this materially, because the
floor is independent of token counts.

`down` prints the itemized charge for the run, and the same summary is written to
`logs/usage/` under your state directory. `status` and `down` do not cost SU; only
the running GPU session does.

## 12. Data residency

The model executes on an RCC compute node. The gateway executes on an RCC login node.
The client reaches the gateway over `localhost` or an SSH-forwarded port. No prompt,
file content, or completion is transmitted to any service outside RCC. This is the
operative difference from hosted coding assistants and is the reason the service is
appropriate for unpublished or otherwise restricted source code.

## 13. Troubleshooting

| Symptom | Cause and resolution |
|---|---|
| `Unknown context window size` | The metadata file was not passed. Add `--model-metadata-file /project/rcc/mehta5/vllm/ai-session/aider_model_metadata.json`. The command printed by `up` already includes it. |
| Prompt reported as too long | Added file content exceeds the 28000-token input budget. Remove files with `/drop` or clear history with `/clear`. |
| aider rejects a diff edit | The model produced a malformed diff. Retry, or restart with `EDIT_FORMAT=whole bash .../run_coding_agent.sh up`. |
| Tool-call parse errors (opencode, Cline) | Expected for a locally served model. Confirm the session was started with `AGENT_CLIENT=1`; if errors persist, use aider. |
| `model 'qwen2.5_coder_32B' is not fully staged` | The model weights are not completely on disk. Wait for staging to finish, or start with `MODEL=qwen2.5_72B`. |
| Port already in use at `up` | Another session (yours or another user's) holds the default port. Choose another: `GW_PORT=8490 bash .../run_coding_agent.sh up`. |
| Client cannot connect from laptop | The SSH tunnel is not open, or points at the wrong login node. Use the tunnel command printed by `up`, which names the correct node. |
| Connection refused after working for a while | The SSH session that hosted the gateway closed. Restart with `up`, and run inside `tmux` to prevent recurrence. |

## 14. Design notes

The service serves an OpenAI-compatible endpoint via vLLM. aider is the default
client because its edit mechanism (chat completions plus text diffs) does not depend
on function calling, which is the least reliable part of serving a local model
through vLLM. Continue and opencode are documented for users who prefer an in-editor
workflow or an autonomous agent, and connect to the same endpoint. The endpoint, not
the choice of client, is the fixed interface; any OpenAI-compatible tool can use it.

For comparison, the Stanford Sherlock guide for this class of workflow uses Ollama
as the server and centers on native-tool-calling agents and the Zed editor. The
client list there reflects a serving stack in which tool calling is dependable. The
configuration mechanism is otherwise the same: a base URL, an API key, and a model
name supplied to each tool.
