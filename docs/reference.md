# Command Reference

This page is intended for advanced users who want to drive the ai-session service
directly: the `ai_session.py` command-line interface, the environment variables the
wrapper scripts accept, and scripted access to the API. For everyday use you do not
need any of this — the two wrapper scripts cover the common cases end to end:

- Browser chat: `run_browser_demo.sh` — see [Getting Started: Browser Chat](getting-started.md).
- Coding: `run_coding_agent.sh` — see [Coding Sessions](coding/overview.md).

All commands on this page run **on the login node**. A session is a Slurm GPU job
serving one model through vLLM; the gateway is a reverse proxy on the login node at a
fixed port that forwards requests to whichever compute node the current session runs
on. Charges are counted in Service Units (SU), where 1 SU = 1 A100-GPU-hour; the
formula and rates are on [Billing and Service Units](billing.md).

Command summary:

| Task | Command | Run on |
|---|---|---|
| Start a session | `.../ai_session.py start --model <key> --wait` | Login node |
| Show session state and endpoint | `.../ai_session.py status` | Login node |
| Print client setup for the current session | `.../ai_session.py connect` | Login node |
| Meter, log, and cancel a session | `.../ai_session.py end` | Login node |
| Bring the browser stack up or down | `bash .../run_browser_demo.sh {up,down,status}` | Login node |
| Bring the coding stack up or down | `bash .../run_coding_agent.sh {up,down,status}` | Login node |

`...` is `/project/rcc/mehta5/vllm/ai-session` throughout.

## Where session state lives

The install is multi-tenant: code, model weights, and the serving environment are
shared and read-only, while everything a session writes goes to a per-user directory.

- `AISESSION_STATE_DIR` (default `/project/rcc/mehta5/ai-session-state/<user>`) is the
  per-user writable root. It holds session files (`logs/sessions/<user>_<jobid>.json`),
  the gateway pointer (`logs/gateway/upstream.json`), per-request gateway usage capture
  (`logs/gateway/usage-YYYYMMDD.jsonl`), billing output (`logs/usage/`), and the Slurm
  job logs (`logs/vllm/<model_key>-<jobid>.out` and `.err`).
- The environment (`/project/rcc/mehta5/conda-envs/vllm-probe`), the model weights
  under `/project/rcc/mehta5/vllm/models/`, and the code are shared read-only (group
  `rcc-staff`); you run the one managed install rather than copying it.
- Default ports are derived from your UID so two users on one login node do not
  collide: the gateway listens on `GW_PORT = 8400 + UID % 90` and the browser UI on
  `OWUI_PORT = 3000 + UID % 90`. Print yours with
  `echo $((8400 + $(id -u) % 90)) $((3000 + $(id -u) % 90))`.
- Running `ai_session.py` or `gateway.py` with `AISESSION_STATE_DIR` unset falls back
  to the original single-tenant layout, with state next to the code. The wrapper
  scripts always set the per-user directory; set it yourself when calling the CLI
  directly if you want the same layout the wrappers use.

## The ai_session.py CLI

Every subcommand runs **on the login node** using the shared `vllm-probe` interpreter.
No environment activation is needed if you call the interpreter by full path, as the
wrappers do:

```bash
PY=/project/rcc/mehta5/conda-envs/vllm-probe/bin/python
AIS=/project/rcc/mehta5/vllm/ai-session/ai_session.py
```

The Slurm partition and account default to `test` and `rcc-staff` and are CLI
parameters (`--partition`, `--account`). The `--model` argument takes a registry key:

| Key | Model | Served in Phase 1 | License |
|---|---|---|---|
| `qwen2.5_72B` | Qwen2.5-72B-Instruct | Yes | Qwen (Tongyi) community license |
| `qwen2.5_coder_32B` | Qwen2.5-Coder-32B-Instruct | Yes | Apache-2.0 |
| `qwen3_4b` | Qwen3-4B | Yes | Apache-2.0 |
| `llama3.1_70B` | Meta-Llama-3.1-70B-Instruct | No (requires `--force`) | Llama 3.1 Community License + Acceptable Use Policy |
| `qwen2.5_0.5B` | Qwen2.5-0.5B-Instruct | No (requires `--force`) | Apache-2.0 |

The license terms and the obligations that apply when you serve these models to
other people are set out on [Model licenses](licenses.md). Serving Llama 3.1 through
`--force` additionally requires a one-time recorded acknowledgment (see the
`--force` row below and [Model licenses](licenses.md#serving-llama-31-the-acknowledgment-gate)).

Qwen3 sessions serve with the `qwen3` reasoning parser, so the model's chain of
thought is returned in a separate `reasoning_content` field and the answer stays in
`content` — the raw `<think>…</think>` block is not mixed into the reply. Qwen2.5 and
Llama models do not think and are served without it.

### start

Submits the session as a Slurm job, records a session file, and (with `--wait`)
blocks until the model is loaded, then publishes the backend to the gateway.

```bash
$PY $AIS start --model <key> [--tp N] [--constraint TIER] [--gres gpu:N]
    [--partition NAME] [--account NAME] [--time HH:MM:SS]
    [--max-model-len N] [--gpu-mem-util F] [--enforce-eager]
    [--agent-client] [--wait] [--ready-timeout SECONDS] [--force]
```

| Flag | Default | Purpose |
|---|---|---|
| `--model` | required | model registry key (table above) |
| `--tp` | `4` | tensor-parallel size: the number of GPUs the model's weights are split across |
| `--constraint` | `A100` | GPU tier constraint (`A100`/`H200`/`H100`/`L40S`/`A40`) |
| `--gres` | `gpu:<tp>` | Slurm GPU request |
| `--account` | `rcc-staff` | Slurm account |
| `--partition` | `test` | Slurm partition |
| `--time` | `02:00:00` | Slurm time limit; the job ends when it expires |
| `--max-model-len` | `8192` | served context length in tokens (coding sessions use 32768) |
| `--gpu-mem-util` | `0.90` | vLLM GPU memory utilization fraction |
| `--enforce-eager` | off | disable torch.compile (keep in sync with the benchmark for that tier) |
| `--agent-client` | off | enable server-side tool calling for agent clients such as [opencode and Cline](coding/opencode.md); not needed for Open WebUI or aider |
| `--wait` | off | block until the endpoint is ready |
| `--ready-timeout` | `1800` | seconds to wait for readiness with `--wait` |
| `--force` | off | serve a registered model outside the Phase-1 served set; serving `llama3.1_70B` this way also requires the license acknowledgment `ACCEPT_LLAMA_LICENSE=1` (see below) |

Worked example — the coding configuration with tool calling enabled, run **on the
login node**:

```bash
$PY $AIS start --model qwen2.5_coder_32B --tp 2 --constraint A100 \
    --max-model-len 32768 --agent-client --wait
```

!!! warning "A running session consumes SU whether or not you send requests"
    `start` submits a GPU job billed at least the reservation floor (GPU-time held)
    until you stop it. When done, run `$PY $AIS end`.

Before it submits the job, `start` prints a pre-flight cost estimate — the
reservation-floor rate for the requested tier and GPU count, and the projected
maximum over the `--time` walltime — so you see the cost before any GPU is
committed. For the command above (`A100`, `--tp 2`, default `--time 02:00:00`):

```
[start] Estimated cost: 2 SU/h at tier a100 (N=2 GPU); projected max 4 SU over the 2 h walltime.
```

If the requested tier has no weight in the billing policy, the line instead reads
`Estimated cost: tier '<tier>' resolves at launch; floor is w_gpu*N per hour ...`.

Expected output once ready (placeholders for the values Slurm assigns):

```
[start] Estimated cost: 2 SU/h at tier a100 (N=2 GPU); projected max 4 SU over the 2 h walltime.
[start] submitted qwen2.5_coder_32B jobid=<jobid> port=<port>
[start] session file: <state-dir>/logs/sessions/<user>_<jobid>.json
[start] waiting for job <jobid> to start + load model (timeout 1800s)...
[start] READY. Direct endpoint: http://<node>:<port>/v1
[start]   model=qwen2.5_coder_32B  (direct /v1 needs the backend key -- use the gateway URL)
[start] gateway updated -> clients on the gateway URL now reach this session.
[start] run `ai_session.py connect` for client setup (Open WebUI / aider / tunnel).
```

Asking for a registered but unserved model without `--force` fails with this exact
message:

```
'llama3.1_70B' is not in PHASE1_SERVED ['qwen2.5_72B', 'qwen2.5_coder_32B', 'qwen3_4b']; pass --force to serve it anyway.
```

`llama3.1_70B` carries a further gate even with `--force`, because serving it to
others imposes obligations under the Llama 3.1 Community License and its Acceptable
Use Policy (see [Model licenses](licenses.md#serving-llama-31-the-acknowledgment-gate)).
The first `--force` start refuses until you record acceptance by setting
`ACCEPT_LLAMA_LICENSE=1`:

```bash
ACCEPT_LLAMA_LICENSE=1 $PY $AIS start --model llama3.1_70B --tp 4 --constraint A100 --force --wait
```

That writes a one-time per-user acceptance record under
`<state-dir>/logs/licenses/<user>_llama3.1_70B.accepted`; later starts reuse it and
need no environment variable. Without the acknowledgment and with no record on file,
`start` refuses before submitting any job, naming the on-disk license path and the
variable to set. The Apache-2.0 models (`qwen2.5_coder_32B`, `qwen3_4b`,
`qwen2.5_0.5B`) are not gated.

Verify with `$PY $AIS status`.

### status

Reports the session state and endpoint. With no `--jobid` it uses the most recent
session file.

```bash
$PY $AIS status [--jobid N]
```

| Flag | Default | Purpose |
|---|---|---|
| `--jobid` | latest session | select a specific session |

Worked example, run **on the login node**:

```bash
$PY $AIS status
```

Expected output for a running session:

```
jobid=<jobid> model=qwen2.5_coder_32B state=RUNNING node=<node> port=<port>
endpoint: http://<node>:<port>/v1  (model=qwen2.5_coder_32B)
/metrics: prompt_tokens=<n> generation_tokens=<n> success_requests=<n>
```

While the model is still loading, the last line reads
`/metrics: not reachable yet (model may still be loading)`. `status` costs nothing to
run; the session bills the same whether or not you query it.

### connect

Prints the client setup for the current session: the stable gateway URL, the SSH
tunnel command for a laptop, and ready-to-paste configuration for Open WebUI, aider,
Continue, and opencode.

```bash
$PY $AIS connect [--jobid N] [--gateway-host HOST] [--gateway-port PORT] [--gateway-key KEY]
```

| Flag | Default | Purpose |
|---|---|---|
| `--jobid` | latest session | select a specific session |
| `--gateway-host` | the host `connect` runs on | host running `gateway.py` |
| `--gateway-port` | `8080` | gateway port; pass your `<GW_PORT>` when using the wrapper-started gateway |
| `--gateway-key` | the session's saved key | overrides the key `connect` reads from `<state-dir>/logs/gateway/session_key`; must match the gateway's `AISESSION_GATEWAY_KEY` |

Worked example, run **on the login node** where the wrapper started your gateway:

```bash
$PY $AIS connect --gateway-port $((8400 + $(id -u) % 90))
```

The output includes the base URL, model name, and the session access key for each
client (Open WebUI, aider, Continue, opencode), showing exactly which field the key
goes in. When a wrapper started the session it printed and saved that key; `connect`
reads it from `<state-dir>/logs/gateway/session_key`. Pass `--gateway-key` to
override, or if there is no key file `connect` reports the session as keyless. Verify
by checking that the printed model matches the session you started.

### end

Meters the session, writes the usage log, and cancels the Slurm job. Run it as soon
as you stop working; a running session bills the reservation floor whether busy or
idle.

```bash
$PY $AIS end [--jobid N] [--usage-jsonl FILE] [--n-gpus N] [--policy FILE] [--rate-table FILE] [--no-cancel]
```

| Flag | Default | Purpose |
|---|---|---|
| `--jobid` | latest session | select a specific session |
| `--usage-jsonl` | none | client-written per-request usage JSONL, treated as the authoritative token source |
| `--n-gpus` | resolved from Slurm | override the billed GPU count N |
| `--policy` | `billing/billing_policy.yaml` | billing policy file |
| `--rate-table` | `billing/rate_table.json` | throughput rate table |
| `--no-cancel` | off | meter and log but leave the job running (debugging) |

Worked example, run **on the login node**:

```bash
$PY $AIS end
```

It prints an `=== ai-session billing summary ===` block (model, tier, GPUs billed,
tokens by source, floor SU, billed SU) and the path of the summary JSON under
`<state-dir>/logs/usage/`. Verify the job is gone:

```bash
squeue -u $USER
```

The session's job should no longer be listed.

## Wrapper environment variables

Both wrappers read their configuration from environment variables, so a one-off
override is a prefix on the command line, for example
`MODEL=qwen2.5_72B TP=4 CONSTRAINT=A100 bash /project/rcc/mehta5/vllm/ai-session/run_browser_demo.sh up`.

!!! warning "Wrapper up starts a GPU session and spends SU until you run down"
    Stop the stack you started with
    `bash /project/rcc/mehta5/vllm/ai-session/run_browser_demo.sh down` or
    `bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh down`.

| Variable | run_browser_demo.sh default | run_coding_agent.sh default | Purpose |
|---|---|---|---|
| `MODEL` | `qwen3_4b` | `qwen2.5_coder_32B` | model registry key to serve |
| `TP` | `1` | `2` | tensor-parallel size (GPUs per session) |
| `CONSTRAINT` | `A100` | `A100` | GPU tier; uppercase `A100` selects the 80 GB nodes |
| `TIME` | `02:00:00` | `02:00:00` | session walltime `HH:MM:SS`; caps how long the GPU is held and thus the maximum floor charge (`down` still ends it sooner) |
| `MAX_MODEL_LEN` | not read (sessions serve the 8192 CLI default) | `32768` | served context length in tokens |
| `GW_PORT` | `8400 + UID % 90` | `8400 + UID % 90` | gateway port on the login node |
| `OWUI_PORT` | `3000 + UID % 90` | not read | Open WebUI port (browser stack only) |
| `EDIT_FORMAT` | not read | `diff` | aider edit format; `whole` rewrites entire files |
| `AGENT_CLIENT` | not read | `0` | `1` starts the session with `--agent-client` (server-side tool calling) |
| `READY_TIMEOUT` | `900` | `900` | seconds to wait for the session to become ready |
| `AISESSION_STATE_DIR` | `/project/rcc/mehta5/ai-session-state/<user>` | same | per-user writable state root |

Both wrappers refuse to start if something already listens on the chosen port and
print the override to use; see [Troubleshooting](troubleshooting.md). Run one stack
at a time per user: both default to the same `GW_PORT`.

At `up`, each wrapper mints a random per-session access key
(`openssl rand -hex 16`), saves it at `<state-dir>/logs/gateway/session_key` (mode
600, readable only by you), exports it as `AISESSION_GATEWAY_KEY`, and starts the
gateway bound to `127.0.0.1` with that key required. The key appears in the READY
block; share it with your lab so they can use the session over their own tunnel, and
all of their usage bills to you, the starter. `status` shows only its first six
characters; `down` deletes the key file so the key stops working when the session
ends.

## Scripted access with curl and Python

Any OpenAI-compatible client works against the gateway. The base URL is
`http://localhost:<GW_PORT>/v1` on the login node running the gateway; from your
laptop, tunnel the port first:

```bash
ssh -N -L <GW_PORT>:localhost:<GW_PORT> <cnetid>@<login-node>.rcc.uchicago.edu
```

- Replace `<GW_PORT>` with your gateway port (`echo $((8400 + $(id -u) % 90))`, or
  whatever `GW_PORT` you set).
- Replace `<cnetid>` with your CNetID.
- Replace `<login-node>` with the login node where the gateway is running (the
  wrappers print it; `hostname -s` on that node shows it).

The API key is the session access key that `up` minted and printed (also saved at
`<state-dir>/logs/gateway/session_key`, and shown by `connect`). A gateway started
by a wrapper always requires it; a request without it is refused with HTTP 401. Only
a gateway you start by hand with no `AISESSION_GATEWAY_KEY` is keyless, and then any
non-empty string works. Substitute your session key for `<SESSION_KEY>` below. List
the served model — run this **on the login node** (or on your laptop through the
tunnel):

```bash
curl -s http://localhost:<GW_PORT>/v1/models -H "Authorization: Bearer <SESSION_KEY>"
```

Expected output (trimmed):

```
{"object": "list", "data": [{"id": "qwen2.5_coder_32B", "object": "model", ...}]}
```

A minimal chat completion with the `openai` Python package (install it in your own
environment, not in `vllm-probe`):

```python title="chat_example.py"
from openai import OpenAI

client = OpenAI(base_url="http://localhost:<GW_PORT>/v1", api_key="<SESSION_KEY>")
resp = client.chat.completions.create(
    model="qwen2.5_coder_32B",
    messages=[{"role": "user", "content": "Write a one-line docstring for a matrix transpose function."}],
)
print(resp.choices[0].message.content)
print(resp.usage)
```

- Replace `<GW_PORT>` with your gateway port, and `model` with the key of the session
  you started.

The gateway records per-request token usage automatically: every chat/completions
response's `usage` object is appended to
`<state-dir>/logs/gateway/usage-YYYYMMDD.jsonl`, and for streaming requests the
gateway injects `stream_options.include_usage` into the request so the final stream
chunk carries usage. `end` consumes this log as the billing source, so scripted
clients need no billing instrumentation. Server-side tool calling for agent
frameworks requires a session started with `--agent-client`; opencode support was
verified against the live service on 2026-07-03; see the
[coding agents guide](coding/opencode.md) for caveats.

??? question "What does the gateway do with paths other than /v1?"
    The gateway proxies `/v1`, `/metrics`, `/health`, `/version`, `/ping`,
    `/tokenize`, `/detokenize`, and `/pooling` to the current backend; other paths
    return 404, and the bare `/` returns a JSON hint. Its own health check is
    `GET /__gateway/health`, which reports gateway liveness and whether a backend
    is published (`{"gateway":"ok","backend_active":true|false}`) but not the
    backend's node:port -- that endpoint needs no key, so the address is withheld.
    When no session is active, proxied requests return 503 with
    `"type": "no_backend"`; see
    [Troubleshooting](troubleshooting.md).

## How the bill is computed at end

For auditing a charge, this is what `end` does, in order. The formula, the `w_gpu`
tier weights, and the measured rates live on [Billing and Service Units](billing.md).

1. Token counts are taken from the first available source, in priority order: an
   explicit `--usage-jsonl` file you supply; otherwise the gateway usage log filtered
   to the session's time window; otherwise the session delta of vLLM's own `/metrics`
   counters.
2. When both a per-request source and `/metrics` are available, `end` reconciles
   them and flags a mismatch over 2% in the summary's cross-check field.
3. The GPU tier is resolved from the node's Slurm features (`scontrol show node`),
   falling back to the launch `--constraint`. The billed GPU count N is the job's
   actual allocation (`scontrol show job`), falling back to the requested gres;
   `--n-gpus` overrides it, and `end` prints a note if the job held more GPUs than
   the tensor-parallel size. Reserved wall-hours come from `sacct` Elapsed, falling
   back to the recorded start-to-now interval.
4. The charge is the larger of the token-metered work and the reservation floor
   (tier weight times N times reserved wall-hours). The summary JSON is written to
   `<state-dir>/logs/usage/<user>_<jobid>_<ts>_summary.json`, plus a per-request
   `.jsonl` when a per-request source was used.

A session with no matching rate-table record bills the floor only and marks the token
term unrated; the floor dominates interactive use, so this is a usable fallback
rather than an error. The same floor-only fallback applies when the running vLLM
version no longer matches the version the matched record was measured under: `end`
reads the engine's `/version` and, on a mismatch, suspends the token term (basis
`UNRATED (running vLLM … != rate_table record …)`) until the tier is re-benchmarked.
This is fail-open — an unreadable version keeps the session rated and never breaks
`end`.

## For administrators

The serving environment `/project/rcc/mehta5/conda-envs/vllm-probe` is managed
centrally by RCC staff; users must not install into or otherwise modify it. The
benchmark harness that populates the rate table, rate-table maintenance, and the
gateway internals are documented in the repository operator guide at
`/project/rcc/mehta5/vllm/ai-session/README.md`. The billing tier weights (`w_gpu`)
live in `/project/rcc/mehta5/vllm/billing/billing_policy.yaml` and are editable by
RCC staff only; the measured throughput records live in
`/project/rcc/mehta5/vllm/billing/rate_table.json` and are regenerated by the
benchmark, not edited by hand.
