# AI Sessions on RCC

ai-session is a local large-language-model service on the university's research
computing cluster (RCC beagle3). Instead of sending data to a commercial provider,
faculty and students run open Qwen models on the GPUs the university already has.
You start a session, which is a job under Slurm (the cluster scheduler) serving an
OpenAI-compatible API on a GPU compute node; you reach it through one stable
gateway URL from a browser chat window or a coding tool; and when you stop it, the
usage is charged back in internal Service Units (SU) — a fair-usage accounting
unit, not dollars. The proven ways in are browser chat with Open WebUI
([Getting Started](getting-started.md)) and command-line or in-editor coding tools
([Coding Sessions](coding/overview.md)).

## How it works

Three processes are involved:

```
client                    gateway (login node)           vLLM server (compute node)
your laptop/login    ->   http://localhost:<port>/v1  -> http://<node>:<port>/v1
(browser/aider/...)       stable URL, reverse proxy      ephemeral, GPU-backed, SU-billed
```

- The **vLLM server** runs as a Slurm job on a GPU compute node and serves an
  OpenAI-compatible HTTP API; its node and port change every session, and the
  compute node has no inbound network route from outside the cluster.
- The **gateway** is a reverse proxy on a login node at a fixed port; it exists
  because the backend moves — it forwards to whatever vLLM server the current
  session is using, so the client sees one URL that does not change between
  sessions, and it records per-request token usage, which is the authoritative
  source for [billing](billing.md).
- The **client** — a browser chat window or a coding tool — is configured once
  against the gateway URL, over `localhost` on the login node or an SSH-forwarded
  port from your laptop.

The launch scripts and the per-client configuration live on the pages listed
under [Choose your path](#choose-your-path) below.

## Data residency

The model executes on an RCC compute node. The gateway executes on an RCC login
node. The client reaches the gateway over `localhost` or an SSH-forwarded port.
Along this serving path — client to gateway to vLLM — no prompt, file content, or
completion is transmitted to any service outside RCC. This is the operative
difference from hosted assistants and is the reason the service is appropriate for
unpublished or otherwise restricted code and data.

!!! note "Coding-tool telemetry is a separate concern you control in your client"
    The statement above covers the serving path — the model traffic itself. The
    coding tool you run as the client (aider, Continue, opencode) is separate
    software that may have its own usage telemetry, which phones home independently
    of the model traffic and is outside this service's control. Disable it in your
    own client: the aider commands documented here pass `--analytics-disable`, the
    documented Continue configuration sets `allowAnonymousTelemetry: false`, and the
    Open WebUI instance the browser launcher starts already has telemetry disabled
    (`ANONYMIZED_TELEMETRY=False`, `DO_NOT_TRACK`, `SCARF_NO_ANALYTICS`). Check your
    client's own settings for anything not covered here.

## Available models

Model weights are staged under `/project/rcc/mehta5/vllm/models/`. The model key
is the identifier you pass to the launch scripts and the name the API serves
under. "TP" is the tensor-parallel size: the model's weights are split across
that many GPUs, which serve each request together.

| Model key | Weights | Use | Context (tokens) | Default configuration | License |
|---|---|---|---:|---|---|
| `qwen2.5_coder_32B` | Qwen2.5-Coder-32B-Instruct | Coding (default for coding sessions) | 32768 | TP=2, 2 x A100-80GB | Apache-2.0 |
| `qwen2.5_72B` | Qwen2.5-72B-Instruct | General chat | 8192 | TP=4, 4 x A100-80GB | Qwen (Tongyi) community license |
| `qwen3_4b` | Qwen3-4B | Small and fast; single GPU | 8192 | TP=1, 1 x A100 | Apache-2.0 |

A Qwen2.5-0.5B-Instruct checkpoint (Apache-2.0) is also staged for operator smoke
tests, and a Meta-Llama-3.1-70B-Instruct checkpoint (Llama 3.1 Community License plus
an Acceptable Use Policy) for cross-checks; neither is offered for user sessions.
Guidance on choosing between the served models is on the
[coding overview](coding/overview.md) page. The license obligations that apply when
you serve these models to other people — attribution for the Qwen 72B model, the
acknowledgment gate for Llama — are set out on the [model licenses](licenses.md) page.

!!! note "Compute nodes have no internet access"
    Only models pre-staged under `/project/rcc/mehta5/vllm/models/` can be
    served; a session cannot download weights. New models are staged by the
    operators on request.

## What it costs, in one line

One SU equals one A100-GPU-hour, and the default coding session (Qwen2.5-Coder-32B
on 2 x A100) costs 2.0 SU per hour held. A session is billed the greater of its
metered token work and a reservation floor of `w_gpu x N_gpus x hours` — the
weighted cost of the GPUs held for the session's wall-clock lifetime. The GPU
weights, the full rate table with its benchmark provenance, and worked examples
are on the [billing page](billing.md).

!!! warning "A running session consumes SU whether or not you send requests"
    The floor is charged for the session's whole wall-clock lifetime. Stop your
    session as soon as you finish working; each launch page gives the matching
    stop command.

!!! note "ai-session SUs are not RCC allocation Service Units"
    The RCC user guide states that jobs on beagle3 consume no RCC service
    units. The SUs described here are the ai-session service's own fair-usage
    accounting, defined in its [billing policy](billing.md); they are not
    deducted from any RCC allocation.

## Choose your path

| You want | Start here |
|---|---|
| Chat with a model in the browser (Open WebUI) | [Getting Started: Browser Chat](getting-started.md) |
| Use a coding tool (aider, Continue, opencode) against your own repository | [Coding Sessions](coding/overview.md) |

The [command reference](reference.md) lists every command in one place, and
[troubleshooting](troubleshooting.md) collects the known failure symptoms and
their resolutions.

## Getting help

Two distinct support channels apply, depending on the question:

- Questions about the service itself — sessions, models, connection problems
  beyond the [troubleshooting page](troubleshooting.md), and billing — go to the
  ai-session operators: the RCC staff who maintain `/project/rcc/mehta5`.
- Questions about the cluster — accounts, SSH access, Slurm, partitions, and
  RCC allocations — go to the RCC help desk through the standard RCC support
  channel.
