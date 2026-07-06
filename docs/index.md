# AI Sessions on RCC

ai-session is a local large-language-model service on the university's research
computing cluster (RCC beagle3). Instead of sending data to a commercial provider,
faculty and students run open Qwen models on the GPUs the university already has.
You start a session, which serves a model on cluster GPUs behind an
OpenAI-compatible API; you reach it through one stable gateway URL from a browser
chat window or a coding tool; and when you stop it, the usage is charged back in
internal Service Units (SU) — a fair-usage accounting unit, not dollars. The proven
ways in are browser chat with Open WebUI ([Getting Started](getting-started.md))
and command-line or in-editor coding tools ([Coding Sessions](coding/overview.md)).

## The first five minutes

From a login node:

```bash
module use /project/rcc/mehta5/modulefiles   # goes away once RCC installs the module centrally
module load ai-session

ai-session chat      # chat in your browser, or:
ai-session code      # a coding session for aider / Continue / opencode

ai-session status    # ready, still loading, or stopped?
ai-session stop      # done -- free the GPUs and print the charge
```

Starting a session takes a few minutes while the model loads; the start command
waits and then prints everything you need — the web address, the session access
key, and (for laptops) the SSH command that makes the address reachable from your
machine. The per-client details are on the pages under
[Choose your path](#choose-your-path).

## How it works

Three pieces are involved:

```
client                     gateway (login node)           model server (GPU node)
your laptop/login     ->   http://localhost:<port>/v1  -> moves every session
(browser/aider/...)        stable URL, reverse proxy      GPU-backed, SU-billed
```

- The **model server** runs on a GPU node inside the cluster and serves an
  OpenAI-compatible HTTP API. Its address changes every session, and it is not
  reachable from outside the cluster.
- The **gateway** is a reverse proxy on a login node at a fixed port; it exists
  because the backend moves — it forwards to whatever model server the current
  session is using, so the client sees one URL that does not change between
  sessions, and it records per-request token usage, which is the authoritative
  source for [billing](billing.md).
- The **client** — a browser chat window or a coding tool — is configured once
  against the gateway URL, over `localhost` on the login node or an SSH-forwarded
  port from your laptop.

## Data residency

The model executes on an RCC GPU node. The gateway executes on an RCC login
node. The client reaches the gateway over `localhost` or an SSH-forwarded port.
Along this serving path — client to gateway to model server — no prompt, file
content, or completion is transmitted to any service outside RCC. This is the
operative difference from hosted assistants and is the reason the service is
appropriate for unpublished or otherwise restricted code and data.

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

Each preset picks a model and the right number and type of GPUs for it; you never
configure GPUs yourself. The model key is the name the API serves under and the
value you pass to `ai-session <preset> --model` when you want something other than
the preset's default.

| Preset | Model key | Weights | Use | Context (tokens) | GPUs it runs on | License |
|---|---|---|---|---:|---|---|
| `code` | `qwen2.5_coder_32B` | Qwen2.5-Coder-32B-Instruct | Coding | 32768 | 2 x A100-80GB | Apache-2.0 |
| `chat` | `qwen2.5_72B` | Qwen2.5-72B-Instruct | General chat | 8192 | 4 x A100-80GB | Qwen (Tongyi) community license |
| `fast` | `qwen3_4b` | Qwen3-4B | Small and fast; lowest cost | 8192 | 1 x A100 | Apache-2.0 |

A Qwen2.5-0.5B-Instruct checkpoint (Apache-2.0) is also staged for operator smoke
tests, and a Meta-Llama-3.1-70B-Instruct checkpoint (Llama 3.1 Community License plus
an Acceptable Use Policy) for cross-checks; neither is offered for user sessions.
Guidance on choosing between the served models is on the
[coding overview](coding/overview.md) page. The license obligations that apply when
you serve these models to other people — attribution for the Qwen 72B model, the
acknowledgment gate for Llama — are set out on the [model licenses](licenses.md) page.

!!! note "GPU nodes have no internet access"
    Only pre-staged models can be served; a session cannot download weights. New
    models are staged by the operators on request.

## What it costs, in one line

One SU equals one A100-GPU-hour, and the default coding session (Qwen2.5-Coder-32B
on 2 x A100) costs 2.0 SU per hour held. A session is billed the greater of its
metered token work and a reservation floor — the weighted cost of the GPUs held
for the session's wall-clock lifetime, whether or not you are actively using them.
The GPU weights, the full rate table with its benchmark provenance, and worked
examples are on the [billing page](billing.md).

!!! warning "A running session consumes SU whether or not you send requests"
    The floor is charged for the session's whole wall-clock lifetime. Stop your
    session with `ai-session stop` as soon as you finish working.

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
  ai-session operators: the RCC staff who maintain the service.
- Questions about the cluster — accounts, SSH access, and RCC allocations — go
  to the RCC help desk through the standard RCC support channel.
