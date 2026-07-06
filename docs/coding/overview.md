# Coding Sessions

A coding session serves a code-specialized language model on an RCC GPU node and
connects it to a coding tool that reads and edits source code in your git
repository. No prompt, file content, or completion leaves the cluster. The cost is
GPU reservation time, charged in Service Units (SU); 1 SU is 1 A100-GPU-hour (see
[Billing and Service Units](../billing.md)).

The helper script is `/project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh`, the
coding counterpart to the browser-chat stack on
[Getting Started](../getting-started.md). The difference in shape: a coding tool is
an interactive program you drive by hand, not a background server, so `up` starts
the session and the gateway and then prints the ready-to-run client command, which
you run yourself in the repository you want to edit. The default client is aider;
[Continue](continue.md) and [opencode and Cline](opencode.md) connect to the same
endpoint. The stack is the same three processes described on
[AI Sessions on RCC](../index.md): a vLLM server on a GPU compute node, a gateway (a
reverse proxy on the login node that gives every session one stable
OpenAI-compatible URL and records token counts for billing), and your client.

## Quick Start

| Step | Description | Command | Run on |
|---|---|---|---|
| 1 | Start the session and gateway; wait for the READY block and the printed client command | `bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh up` | Login node |
| 2 | Run the printed client command inside the git repository you want to edit | printed by `up` (aider by default) | Login node or Local machine |
| 3 | Check what is running at any time; costs nothing | `bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh status` | Login node |
| 4 | Stop the session, free the GPUs, print the SU charge | `bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh down` | Login node |

## Step 1: Start the session and gateway

Run this **on the login node**, inside `tmux` or `screen`, so that an SSH
disconnect does not terminate the gateway:

```bash
bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh up
```

`up` submits the Slurm job, waits until the model is loaded (typically several
minutes for the default 32B model), starts the gateway, and prints a block
containing the gateway port, the exact aider command, and the SSH tunnel command
for laptop access. Leave this terminal running.

!!! warning "A running session consumes SU whether or not you send requests"
    `up` starts a billed GPU reservation. Stop it as soon as you finish:
    `bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh down`

Verification — the command ends with a READY banner of this form:

```
==> [2/2] starting gateway on 127.0.0.1:<GW_PORT>
    gateway healthy (pid <PID>)  log: .../run/gateway.log

================ READY -- code with the local qwen2.5_coder_32B (ctx 32768) ================
```

followed by the aider command to copy, the connection parameters for other
clients, and the tunnel command. If `up` exits with an error instead, see
[Troubleshooting](../troubleshooting.md).

## Step 2: Run the coding tool

Open a second terminal, change into the git repository you want to edit, and run
the command printed in Step 1. aider requires the working directory to be inside a
git repository; run `git init` first if necessary. Per-client setup and usage:

- [aider](aider.md) — the default; terminal REPL, edits files as text diffs.
- [Continue (VS Code and JetBrains)](continue.md) — in-editor chat and edit/apply.
- [opencode and Cline (Tool-Calling Agents)](opencode.md) — autonomous agents;
  require the session to be started with `AGENT_CLIENT=1` (see the environment
  variable table below).

Before running any autonomous agent, read
[Agent responsibilities and risks](agents.md): an agent acts with your full
cluster permissions, can read anything you can read (including shared project
directories), and its actions and SU are your responsibility.

To run the tool on your laptop instead of the login node, first open the SSH
tunnel (next section).

Verification — **on the login node**, confirm the gateway answers before starting
the client (exits 0 when healthy):

```bash
curl -sf "http://127.0.0.1:<GW_PORT>/__gateway/health"
```

- Replace `<GW_PORT>` with the gateway port printed by `up`.

## Step 3: Stop the session

Run this **on the login node** the moment you stop working:

```bash
bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh down
```

`down` meters the session, cancels the Slurm job (releasing the GPUs), stops the
gateway, and prints the itemized SU charge for the run as its last output. The
same summary is written to `logs/usage/` under your state directory. `down` and
`status` themselves cost nothing; only the running GPU session does.

Verification — `status` afterwards reports no upstream and no listener:

```bash
bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh status
```

Expect `(none)` under both `-- gateway upstream --` and `-- listener on
:<GW_PORT> --`, and no session job in the `squeue` listing it prints.

## Connection parameters

Every client uses the same four values. The gateway port `GW_PORT` is derived from
your numeric user ID (`8400 + UID % 90`, so it differs per user) and is printed by
`up` and `status`.

| Parameter | Value |
|---|---|
| Base URL | `http://localhost:<GW_PORT>/v1` |
| API key | the session access key that `up` printed (see below) |
| Model name | `qwen2.5_coder_32B` (default) or `qwen2.5_72B`; must equal the model you started |
| Context window | 32768 tokens for coding sessions (8192 for the default chat sessions) |

The model name is the identifier vLLM serves under. Clients that route through
litellm (aider, Continue) prefix it with `openai/` to select the
OpenAI-compatible protocol, for example `openai/qwen2.5_coder_32B`.

### The session access key

`up` mints a random access key for the session and the gateway requires it on every
request. `up` prints it in the READY block and saves it, readable only by you, at
`<state-dir>/logs/gateway/session_key` (mode 600); `connect` and `status` (first six
characters only) also show it. Use it as the API key in every client below.

Because the gateway binds to `127.0.0.1` and accepts only requests carrying the key,
no one else on the shared login node can use your session by accident, and you can
deliberately share it with your lab: give a labmate the key, have them open their own
tunnel to your `GW_PORT`, and set the key as the API key in their client. All of
their usage bills to you, the starter — one key per session, no per-person split. A
request without the key is refused with HTTP 401. `down` deletes the key file, so the
key stops working when the session ends and the next `up` mints a fresh one. Only a
gateway you start by hand with no `AISESSION_GATEWAY_KEY` is keyless, in which case
any non-empty string works.

## Remote access from your laptop

The gateway listens on `127.0.0.1:<GW_PORT>` on the login node where you ran
`up`, so a client on your laptop needs a forwarded port. Run this **on your local
machine** and leave it running:

```bash
ssh -N -L <GW_PORT>:localhost:<GW_PORT> <cnetid>@<login-node>.rcc.uchicago.edu
```

- Replace `<GW_PORT>` with the gateway port printed by `up` (both occurrences).
- Replace `<cnetid>` with your CNetID.
- Replace `<login-node>` with the login node named in the output of `up`; the
  tunnel must target that node, not an arbitrary one.

`up` prints a ready-made tunnel command with the node filled in; that form routes
through a jump host (`-J <cnetid>@midway3.rcc.uchicago.edu`) and is equivalent.
The client on your laptop then uses `http://localhost:<GW_PORT>/v1` as its base
URL. When the client runs on a login node, no tunnel is needed; use the same
`localhost` URL directly. For what an SSH tunnel is and how to debug one, see
[Getting Started](../getting-started.md).

## Choosing a model and tier

The default is `qwen2.5_coder_32B` (Qwen2.5-Coder-32B-Instruct). Override the
model, tensor-parallel degree (TP, the number of GPUs the model's weights are
split across — all of them bill), and GPU constraint with environment variables on
`up`.

| Model key | Parameters | Configuration | Prefill (tok/s) | Decode (tok/s) | Reservation floor |
|---|---|---|---:|---:|---:|
| `qwen2.5_coder_32B` (default) | 32B | TP=2, 2×A100-80GB | 4773 | 1679 | 2.0 SU/h |
| `qwen2.5_72B` | 72B | TP=4, 4×A100-80GB | 2901 | 1123 | 4.0 SU/h |
| `qwen2.5_72B` | 72B | TP=2, 2×H200 | 7594 | 2329 | 6.0 SU/h |

Throughput figures are aggregate, measured by the billing benchmark at concurrency
64 over prefill-heavy, decode-heavy, and balanced request mixes (vLLM 0.10.2,
bfloat16; 32B on midway3-0377, 2026-06-10; 72B on A100 on midway3-0377 and on H200
on midway3-0605, both 2026-06-02). They are the basis of the per-token charge, not
the latency one user perceives (next section). The reservation floor is the
minimum charge for a session — GPU-tier weight times number of GPUs times
reserved wall-clock hours; a session bills the larger of the metered token work
and this floor. The canonical rate table and the full formula are on
[Billing and Service Units](../billing.md).

Override examples, **on the login node**:

```bash
# general 72B model on A100:
MODEL=qwen2.5_72B TP=4 CONSTRAINT=A100 \
  bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh up

# coder model on the H200 throughput tier:
MODEL=qwen2.5_coder_32B TP=2 CONSTRAINT=H200 \
  bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh up
```

!!! warning "Each of these starts a billed GPU reservation"
    The floors above apply from the moment the job starts. Stop with
    `bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh down`.

The coder model is the default because it is specialized for code, uses half the
GPUs of the 72B (TP=2 versus TP=4), and has a lower measured per-token cost than
the 72B at every benchmarked tier. The 72B is preferable for mixed
code-and-prose work or when the larger general model is specifically wanted.

!!! warning "CONSTRAINT=A100 (uppercase) selects the 80 GB nodes"
    The 32B at TP=2 requires two 80 GB cards; the 40 GB A100 nodes, selected by
    lowercase `a100`, are not sufficient.

All wrapper environment variables, with defaults:

| Env var | Default | Purpose |
|---|---|---|
| `AISESSION_STATE_DIR` | `/project/rcc/mehta5/ai-session-state/<user>` | per-user writable root; billing logs land here |
| `GW_PORT` | `8400 + UID % 90` | per-user gateway port (override on clash) |
| `MODEL` / `TP` / `CONSTRAINT` | `qwen2.5_coder_32B` / `2` / `A100` | model and serving tier |
| `TIME` | `02:00:00` | session walltime (`HH:MM:SS`); caps how long the GPU is held, and the maximum floor charge |
| `MAX_MODEL_LEN` | `32768` | served context length |
| `EDIT_FORMAT` | `diff` | aider edit format (`whole` for full-file rewrites) |
| `AGENT_CLIENT` | `0` | `1` enables vLLM tool calling for opencode/Cline |

## Context window and prompt sizing

Coding sessions serve a 32768-token context (set by `MAX_MODEL_LEN`, default
32768). This is the native context length of the Qwen2.5 models; no rope/YaRN
scaling is applied. Chat sessions started by the other helpers default to 8192.

The aider metadata file declares the split as 28000 input tokens and 4096 output
tokens, leaving headroom so that prompt plus generated tokens cannot exceed
32768. If your added files exceed the input budget, aider reports the prompt as
too long; see [Troubleshooting](../troubleshooting.md).

Single-stream generation latency for the 32B at TP=2 is approximately 66 ms per
output token (measured median time-per-output-token, same benchmark as above),
i.e. on the order of 15 tokens per second as perceived by one interactive user.
The aggregate throughput figures in the table above are higher because they sum
across 64 concurrent requests.

## Build your own agent

The listed clients are not the only way to use a session. Because the gateway
exposes the OpenAI chat-completions protocol, you can write your own agent in any
framework that targets an OpenAI-compatible endpoint (PydanticAI, LangGraph,
smolagents, or the OpenAI Agents SDK), pointing it at
`http://localhost:<GW_PORT>/v1` with the session access key as the API key and a
served model name (`qwen2.5_72B` or `qwen3_4b` for tool calling). A runnable,
self-contained PydanticAI example is at
`/project/rcc/mehta5/vllm/examples/agent_pydantic.py`; the
[Build your own agent](agents.md#build-your-own-agent) section walks through it.

Use `qwen2.5_72B` or `qwen3_4b`, not the coder model, for any tool-calling
agent: vLLM's `hermes` parser does not populate `tool_calls` for
Qwen2.5-Coder-32B (vLLM #29192), so a custom agent pointed at the coder model
runs but never calls your tools. Before running an autonomous agent, read
[Agent responsibilities and risks](agents.md): it acts with your full cluster
permissions and its actions are your responsibility.
