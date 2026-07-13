# AI Sessions on RCC

ai-session is a local large-language-model service on the University of Chicago Research
Computing Cluster (RCC). Instead of sending data to a commercial provider,
faculty and students run open LLM models on the GPUs the university already has.
Compared with a cloud AI service: your prompts, code, and data never leave the
university's systems, which matters for unpublished results; usage is accounted
internally in Service Units (SU); and the served models are open-weights
checkpoints, so the exact model behind a result can be named and served again. 
You start a session, which serves a model on cluster GPUs, which is reached via ssh-tunnel. For chatting with the LLM you have two ways:
browser chat with Open WebUI ([Getting Started](getting-started.md)) or command-line/in-editor coding tools ([Coding Sessions](coding/overview.md)).

## Starting a session

From a login node:

```bash
module load ai-session

ai-session chat      # chat in your browser, or:
ai-session code      # a coding session for aider / Continue / opencode (see ([Coding Sessions](coding/overview.md)))

ai-session status    # ready, still loading, or stopped? # this will take a few minutes as the model loads
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
(browser/aider/...)        stable URL, relay              GPU-backed, SU-billed
```

- The **model server** runs on a GPU node inside the cluster and answers
  requests in the standard OpenAI API format that most AI tools can talk to.
  Its address changes every session, and it is not reachable from outside the
  cluster.
- The **gateway** is a small always-on relay on a login node at a fixed port;
  it exists because the model server moves — it forwards to whichever server
  the current session is using, so the client sees one web address that never
  changes between sessions, and it records per-request token usage for [billing](billing.md).
- The **client** — a browser chat window or a coding tool — is configured once
  against that address, over `localhost` on the login node or an SSH-forwarded
  port from your laptop.

## Data location

The LLM executes on a RCC GPU node. The gateway executes on an RCC login
node. The client reaches it over `localhost` or an SSH-forwarded port.
Along this serving path — client to gateway to model server — no prompt, file
content, or completion is transmitted to any service outside RCC. This is the main
difference from hosted assistants and is the reason the service is
appropriate for unpublished or otherwise restricted code and data. The one
exception is opt-in: starting browser chat with `AISESSION_TOOLS=1` adds web and
reference tools whose query terms do leave RCC; see
[Getting Started](getting-started.md#web-search-and-reference-tools-opt-in).

!!! note "Coding-tool monitor with external servers is a separate concern you control in your client"
    The statement above covers the serving path — the model traffic itself. The
    coding tool you run as the client (aider, Continue, opencode) is separate
    software that may have its own usage telemetry, which independently could send
    model traffic and is outside this service's control. Disable it in your
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

A good practice is to start small and scale up: prototype your prompts,
scripts, or agent setup against `ai-session fast` — the small model loads
quickest, waits least for free GPUs, and has the lowest floor cost (1.0 SU per
hour) — and move to the coder or 72B model once the workflow works. The larger
models are more capable, and switching requires no change to the client
configuration.

A Qwen3-32B checkpoint (Apache-2.0) is also available with `--model qwen3_32B`: a
thinking model whose chain of thought is returned separately from the answer,
served on two A100s. A Meta-Llama-3.1-70B-Instruct checkpoint (Llama 3.1 Community
License plus an Acceptable Use Policy) is also available to any user with
`--model llama3.1_70B`, once you record a one-time license acknowledgment (the
Llama 3.1 Community License permits this use, with conditions; see
[Model licenses](licenses.md)). A Qwen2.5-0.5B-Instruct
checkpoint (Apache-2.0) is staged for smoke tests only and is not offered for user
sessions.

Larger models are staged but not yet servable. Qwen3.5-122B-A10B (FP8) is
registered and its weights are on disk; it becomes available once it passes
validation on the cluster's H200 nodes. GLM-5.2 (FP8) is likewise registered
with its weights on disk, but it needs multi-node H200 serving that is not yet
built; GLM-5.1 comes later. The H200 hardware itself is already on the cluster;
what is pending is the serving work, not the machines.

Guidance on choosing between the served models is on the
[coding overview](coding/overview.md) page, and a rough capability frame of
reference against closed "frontier" models is in the
[Command Reference](reference.md#rough-capability-frame-of-reference). The license obligations that apply when
you serve these models to other people — attribution for the Qwen 72B model, the
acknowledgment gate for Llama — are set out on the [model licenses](licenses.md) page.

!!! note "GPU nodes have no internet access"
    Only pre-staged models can be served; a session cannot download weights. New
    models are staged by the users on request.

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
    The SUs described here are the ai-session service's own fair-usage
    accounting, defined in its [billing policy](billing.md); they are not
    deducted from any RCC allocation.

## Choose your path

| You want | Start here |
|---|---|
| Chat with a model in the browser (Open WebUI) | [Getting Started: Browser Chat](getting-started.md) |
| Use a coding tool (aider, Continue, opencode) against your own repository | [Coding Sessions](coding/overview.md) |
| Serve a model you fine-tuned yourself | [Your Own Fine-Tuned Model](lora.md) |

The [command reference](reference.md) lists every command in one place, and
[troubleshooting](troubleshooting.md) collects the known failure symptoms and
their resolutions.

## Getting help

Open a ticket through the standard RCC support channel. Mention `ai-session` if
the question is about the service itself — sessions, models, connection problems
beyond the [troubleshooting page](troubleshooting.md), or billing — so it reaches
the staff who maintain it. Questions about the cluster in general — accounts,
SSH access, RCC allocations — follow the normal help-desk path.
