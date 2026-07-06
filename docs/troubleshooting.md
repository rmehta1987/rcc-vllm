# Troubleshooting

This page collects the known failure modes of the ai-session service and their
fixes. Each section below is named after the error message or symptom you will
actually see, so searching this site for the error text lands here. Launch and
configuration instructions live on the client pages
([aider](coding/aider.md), [Continue](coding/continue.md),
[opencode and Cline](coding/opencode.md), [browser chat](getting-started.md));
this page assumes you followed one of them and something did not work.

## First checks

Before reading any symptom section, run this four-step diagnostic sequence. None
of these commands costs Service Units (SU): only the running GPU session bills,
and these commands only inspect it. Run all four **on the login node** where you
ran `up`.

| Step | Description | Command | Run on |
|---|---|---|---|
| 1 | Report the state of your whole stack (session, gateway, Slurm job) | `bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh status` | Login node |
| 2 | Confirm the gateway process is alive and knows its backend | `curl -sf http://127.0.0.1:<GW_PORT>/__gateway/health` | Login node |
| 3 | Confirm the vLLM backend on the compute node answers through the gateway | `curl -s http://localhost:<GW_PORT>/v1/models` | Login node |
| 4 | Confirm the session's Slurm job is running | `squeue -u $USER` | Login node |

- Replace `<GW_PORT>` with the gateway port printed by `up` and by `status`.
  The per-user default is `8400 + UID % 90`; print yours with
  `echo $((8400 + $(id -u) % 90))`.
- For a browser-chat stack, substitute
  `bash /project/rcc/mehta5/vllm/ai-session/run_browser_demo.sh status` in step 1;
  steps 2 to 4 are identical.

Step 1, `status`, prints the gateway upstream file, the listener on your gateway
port, the saved process IDs, and your Slurm queue. A healthy coding stack shows a
listener on `:<GW_PORT>` and one running job; a dead stack prints `(none)` under
the listener and upstream headings.

Step 2 checks the gateway, the login-node reverse proxy that gives clients one
stable URL while the compute-node backend changes every session. Expected output:

```
{"gateway":"ok","backend_active":true}
```

This endpoint is reachable without the access key, so it reports only liveness and
whether a backend is published -- never the backend's node:port (that address must
not leak to other users on the login node).

If this curl fails, the gateway process is gone: see
[Connection refused after working for a while](#connection-refused-after-working-for-a-while).
If it succeeds but `backend_active` is `false`, the gateway is up but no session has
published a backend; requests will return this error verbatim:

```
{"error": {"message": "no active ai-session backend; run `ai_session.py start`", "type": "no_backend"}}
```

In that case start a session from the owning page's recipe
([coding](coding/overview.md) or [browser chat](getting-started.md)).

Step 3 exercises the full path to the GPU. Expected output is an OpenAI-style
model list naming the model you started:

```
{"object":"list","data":[{"id":"qwen2.5_coder_32B", ...}]}
```

Step 4 shows the session's Slurm job. `R` in the `ST` column means running; `PD`
means pending — see
[`up` seems stuck or the model takes minutes to appear](#up-seems-stuck-or-the-model-takes-minutes-to-appear).

## `Unknown context window size`

**Symptom.** aider (via litellm) warns:

```
Unknown context window size
```

**Cause.** litellm was not given the metadata file that declares the served
model's context window (32768 tokens) and zero per-token cost, so it cannot size
prompts.

**Check.** Inspect the aider command you ran: it must include
`--model-metadata-file`.

**Fix.** Add the flag:

```bash
--model-metadata-file /project/rcc/mehta5/vllm/ai-session/aider_model_metadata.json
```

The aider command printed by `up` already includes it; the reliable fix is to
copy that printed command verbatim rather than retyping it. The full command and
an explanation of each flag are on the [aider page](coding/aider.md).

## Prompt reported as too long

**Symptom.** aider reports the prompt as too long, or refuses to send a request.

**Cause.** The metadata file splits the 32768-token context into a 28000-token
input budget and 4096 output tokens, so that prompt plus generated tokens can
never exceed the served context. Added file content has exceeded the 28000-token
input budget.

**Check.** Inside aider, run `/tokens` to see current context usage.

**Fix.** Remove files you are not editing with `/drop <path>`, or reset the
conversation history with `/clear` (added files remain). aider maintains a
repository map, so the model retains structural awareness of files you drop;
`/add` only the files you intend to edit.

## aider rejects a diff edit

**Symptom.** aider reports that it could not apply an edit the model proposed.

**Cause.** The model produced a malformed unified diff. This happens
occasionally; the default `--edit-format diff` is token-efficient but depends on
the model emitting a well-formed diff for that particular file.

**Fix.** First simply retry the request. If the same file fails repeatedly,
switch aider to whole-file rewrites: the `EDIT_FORMAT` variable only selects the
`--edit-format` flag in the aider command that `up` prints, so you do not need to
restart the GPU session. Exit aider and re-run the printed aider command with
`--edit-format whole` in place of `--edit-format diff`. To make whole-file edits
the default for a future session, start it with:

```bash
EDIT_FORMAT=whole bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh up
```

!!! warning "A running session consumes SU whether or not you send requests"
    `up` starts a GPU session; stop it with
    `bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh down` as soon as you finish.

## Tool calls fail silently in opencode or Cline

**Symptom.** The agent reports that the model responded, but no file changed and
no command ran; the tool-call JSON appears as plain text in the model's reply.
No error is raised on either end. This is the measured failure mode (verified
2026-07-03, opencode 1.14.41, Slurm job 51391003): the vLLM server log contained
zero tool-call parser exceptions while every out-of-the-box task run failed.

**Cause.** The served Qwen2.5-Coder-32B-Instruct checkpoint does not emit the
`<tool_call>` marker tokens that vLLM's `hermes` parser matches, so the tool
JSON streams back as ordinary assistant text that the agent ignores.

**Fix.** Create the `AGENTS.md` workaround file in the root of the repository
you are editing; its exact content and the reasoning behind it are on the
[opencode page](coding/opencode.md) and in section 8 of
`/project/rcc/mehta5/vllm/ai-session/CODING_AGENTS.md`.

**Check.** Also confirm the session was started with tool calling enabled — it
is off by default. Tool calling is on only if you ran:

```bash
AGENT_CLIENT=1 bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh up
```

If the session was started without `AGENT_CLIENT=1`, stop it with `down` and
start it again with the line above.

!!! warning "A running session consumes SU whether or not you send requests"
    Stop it with `bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh down` when finished.

If tool calls still misbehave with `AGENT_CLIENT=1` set and `AGENTS.md` in
place, use [aider](coding/aider.md), which performs the same edits through chat
completions and text diffs without function calling, against the same endpoint.

## `model '...' is not fully staged`

**Symptom.** `up` exits immediately with:

```
ERROR: model 'qwen2.5_coder_32B' is not fully staged at: <path>
       (missing config.json/*.safetensors, or a download is still in flight).
```

**Cause.** The model weights are not completely on disk under
`/project/rcc/mehta5/vllm/models/`. The wrapper refuses to submit a GPU job for a
model whose directory lacks `config.json` or `*.safetensors` files, or still
contains `*.incomplete` shards from an in-flight download.

**Check.** Look for leftover incomplete shards in the model directory named in
the error:

```bash
ls /project/rcc/mehta5/vllm/models/<model-dir>/*.incomplete 2>/dev/null
```

- Replace `<model-dir>` with the directory printed in the error message.

**Fix.** Wait for staging to finish and run `up` again, or start a model that is
already staged, for example:

```bash
MODEL=qwen2.5_72B TP=4 bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh up
```

!!! warning "The 72B reserves four GPUs and bills a higher floor than the 32B default"
    See the rate table on [Billing and Service Units](billing.md); stop with
    `bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh down` when finished.

## Port already in use at `up`

**Symptom.** `up` exits with:

```
Something is already listening on :<GW_PORT> (maybe a browser demo or another user).
```

**Cause.** The gateway port defaults to `8400 + UID % 90`, derived from your user
ID so two users on one login node normally do not collide. The check trips when
you already have a stack up on this node (a browser demo and a coding stack share
the same default port; run one at a time), or when another user overrode their
port onto yours.

**Check.** See what is listening and whether it is yours:

```bash
ss -ltnp | grep :<GW_PORT>
```

**Fix.** If the listener is your own leftover stack, tear it down first:

```bash
bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh down
```

Otherwise pick a free port explicitly:

```bash
GW_PORT=8490 bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh up
```

!!! warning "A running session consumes SU whether or not you send requests"
    Stop it with `GW_PORT=8490 bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh down` when finished.

!!! note "Set the same GW_PORT on every action"
    `status` and `down` read `GW_PORT` too. If you overrode it at `up`, prefix
    `status` and `down` with the same `GW_PORT=8490` or they will inspect the
    wrong port.

## The client on your laptop cannot connect

**Symptom.** A client on your laptop (aider, Continue, a browser) cannot reach
`http://localhost:<GW_PORT>` although the [first checks](#first-checks) pass on
the login node.

**Cause.** The gateway listens on `127.0.0.1` on the specific login node where
you ran `up`; it is not reachable from outside that node. Your laptop reaches it
only through an SSH tunnel, and the tunnel must target that same login node. The
usual causes are a tunnel that is not running, or a tunnel opened to a different
login node than the one hosting the gateway.

??? question "What is an SSH tunnel?"
    An SSH tunnel (`ssh -L`) forwards a port on your laptop to a port on a
    remote machine over the SSH connection. After
    `ssh -N -L 8412:localhost:8412 ...`, a client on your laptop that connects
    to `localhost:8412` is transparently connected to port 8412 on the login
    node. `-N` means the connection carries only the forward, no remote shell.

**Check.** With the tunnel supposedly open, run **on your laptop**:

```bash
curl -sf http://localhost:<GW_PORT>/__gateway/health
```

**Fix.** Re-run the tunnel command printed in the `READY` block by `up` — it
names the correct login node. Its form is:

```bash
ssh -N -L <GW_PORT>:localhost:<GW_PORT> <cnetid>@<login-node>.rcc.uchicago.edu
```

- Replace `<GW_PORT>` with your gateway port (both occurrences).
- Replace `<cnetid>` with your CNetID.
- Replace `<login-node>` with the login node named in the `READY` block printed
  by `up` — not an arbitrary login node.

Leave the tunnel running for as long as you work, then point the laptop client
at `http://localhost:<GW_PORT>/v1`.

## Connection refused after working for a while

**Symptom.** The client worked, then requests start failing with connection
refused; `curl -sf http://127.0.0.1:<GW_PORT>/__gateway/health` on the login node
fails.

**Cause.** The gateway runs as a background process on the login node, started
from the terminal where you ran `up`. When the SSH session hosting it closed
(laptop sleep, network drop), the gateway died with it. The GPU session may
still be running — and still billing.

**Check.** `squeue -u $USER` **on the login node**. A job still listed means the
session survived the gateway.

**Fix.** Tear down cleanly, then start again, this time inside `tmux` or
`screen` so an SSH drop cannot kill the gateway. `down` is safe to run even when
parts of the stack are already gone: it meters and cancels any remaining job,
stops any remaining gateway, and prints the SU charge.

```bash
bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh down
bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh up
```

!!! warning "A running session consumes SU whether or not you send requests"
    Stop it with `bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh down` as soon as you stop working.

## `up` seems stuck or the model takes minutes to appear

**Symptom.** `up` prints that it is starting the vLLM session, then appears to
hang.

**Cause.** This is usually normal. `up` blocks until the model has loaded and
answered a probe; loading the 32B coding model typically takes several minutes
after the job starts. `up` waits up to 900 seconds by default (override with the
`READY_TIMEOUT` environment variable, in seconds). The other common case is that
the job has not started at all because the partition is busy.

**Check.** In a second terminal **on the login node**:

```bash
squeue -u $USER
```

`R` in the `ST` column means the job is running and the model is loading: wait.
`PD` means the job is pending in the queue because the partition is busy: the
wait is unbounded, but it costs nothing — the billing floor is computed from the
job's elapsed run time, so SU accrue only once the job starts (see
[Billing and Service Units](billing.md)). For more detail, follow the launch log:

```bash
tail -f /project/rcc/mehta5/ai-session-state/$USER/run/start.log
```

**Fix.** Wait, or cancel with `Ctrl-C` followed by
`bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh down` and try a
different tier later. If the load is slow but progressing and 900 seconds will
not suffice, restart with a longer `READY_TIMEOUT`.

## Getting help

For anything not covered here, use the support channels listed on the
[front page](index.md). When reporting a problem, include the Slurm job ID from
`squeue -u $USER` (or from the `down` receipt). For a billing question, also
include the path of the usage summary for the session, which `down` writes under
your state directory as
`/project/rcc/mehta5/ai-session-state/<user>/logs/usage/<user>_<jobid>_<ts>_summary.json`.
