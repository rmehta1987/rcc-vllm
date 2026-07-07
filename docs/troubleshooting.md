# Troubleshooting

This page collects the known failure modes of the ai-session service and their
fixes. Each section below is named after the error message or symptom you will
actually see, so searching this site for the error text lands here. Launch and
configuration instructions live on the client pages
([aider](coding/aider.md), [Continue](coding/continue.md),
[opencode and Cline](coding/opencode.md), [browser chat](getting-started.md));
this page assumes you followed one of them and something did not work.

## First checks

Before reading any symptom section, run this three-step diagnostic sequence. None
of these commands costs Service Units (SU): only the running GPU session bills,
and these commands only inspect it. Run all three **on the login node** where you
started the session.

| Step | Description | Command |
|---|---|---|
| 1 | Is the session ready, still loading, or stopped? | `ai-session status` |
| 2 | Confirm the gateway process is alive and knows its backend | `curl -sf http://127.0.0.1:<GW_PORT>/__gateway/health` |
| 3 | Confirm the model server answers through the session URL | `eval "$(ai-session env)" && curl -s "$AISESSION_BASE_URL/models" -H "Authorization: Bearer $AISESSION_API_KEY"` |

- Replace `<GW_PORT>` with your port. The per-user default is
  `8400 + UID % 90`; print yours with `echo $((8400 + $(id -u) % 90))`.

Step 1 answers the usual question directly: `READY` (with the model and URL),
`STARTING` (the model is still loading — wait and re-check), or none running. It
also shows whether an access key is set and how long a running session has been
up.

Step 2 checks the gateway — the small always-on connection point the service
runs on the login node. It gives clients one stable web address while the GPU
session behind it changes. Expected output:

```
{"gateway":"ok","backend_active":true}
```

This endpoint is reachable without the access key, so it reports only liveness and
whether a backend is published — never the backend's internal address (which must
not leak to other users on the login node).

If this curl fails, the gateway process is gone: see
[Connection refused after working for a while](#connection-refused-after-working-for-a-while).
If it succeeds but `backend_active` is `false`, the gateway is up but no session
has published a backend; requests will return an error of type `no_backend`. In
that case start a session from the owning page's recipe
([coding](coding/overview.md) or [browser chat](getting-started.md)).

Step 3 exercises the full path to the GPU. Expected output is an OpenAI-style
model list naming the model you started:

```
{"object":"list","data":[{"id":"qwen2.5_coder_32B", ...}]}
```

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

**Fix.** Copy the aider command printed by `ai-session code` verbatim rather than
retyping it — it already includes the flag. The full command and an explanation
of each flag are on the [aider page](coding/aider.md).

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
switch aider to whole-file rewrites — this is a client-side flag, so you do not
need to restart the GPU session. Exit aider and re-run the printed aider command
with `--edit-format whole` in place of `--edit-format diff`. To make whole-file
edits the default for a future session, start it with:

```bash
EDIT_FORMAT=whole ai-session code
```

## Tool calls fail silently in opencode or Cline

**Symptom.** The agent reports that the model responded, but no file changed and
no command ran; the tool-call JSON appears as plain text in the model's reply.
No error is raised on either end. This is the measured failure mode (verified
2026-07-03, opencode 1.14.41): the server log contained zero tool-call parser
exceptions while every out-of-the-box task run failed.

**Cause.** The served Qwen2.5-Coder-32B-Instruct checkpoint does not emit the
`<tool_call>` marker tokens that the server's tool-call parser matches, so the
tool JSON streams back as ordinary assistant text that the agent ignores.

**Fix.** Create the `AGENTS.md` workaround file in the root of the repository
you are editing; its exact content and the reasoning behind it are on the
[opencode page](coding/opencode.md).

**Check.** Also confirm the session was started with tool calling enabled — it
is off by default. Tool calling is on only if you started with:

```bash
ai-session code --agent
```

If the session was started without `--agent`, stop it with `ai-session stop` and
start it again with the line above.

If tool calls still misbehave with `--agent` set and `AGENTS.md` in place, use
[aider](coding/aider.md), which performs the same edits through chat completions
and text diffs without function calling, against the same endpoint.

## `model '...' is not fully staged`

**Symptom.** The start command exits immediately with:

```
ERROR: model 'qwen2.5_coder_32B' is not fully staged at: <path>
       (missing config.json/*.safetensors, or a download is still in flight).
```

**Cause.** The model weights are not completely on disk in the service's model
store. The service refuses to reserve GPUs for a model whose directory lacks
`config.json` or `*.safetensors` files, or still contains `*.incomplete` shards
from an in-flight download.

**Fix.** Wait for staging to finish (the operators stage new models) and start
again, or start a model that is already staged, for example:

```bash
ai-session code --model qwen2.5_72B
```

!!! warning "The 72B reserves four GPUs and bills a higher floor than the 32B default"
    See the rate table on [Billing and Service Units](billing.md); stop with
    `ai-session stop` when finished.

## Port already in use at start

**Symptom.** The start command exits with:

```
Something is already listening on :<GW_PORT> (maybe a browser demo or another user).
```

**Cause.** The port defaults to `8400 + UID % 90`, derived from your user
ID so two users on one login node normally do not collide. The check trips when
you already have a session up on this node (chat and coding sessions share the
same default port; run one at a time), or when another user overrode their port
onto yours.

**Check.** See what is listening and whether it is yours:

```bash
ss -ltnp | grep :<GW_PORT>
```

**Fix.** If the listener is your own leftover session, stop it first:

```bash
ai-session stop
```

Otherwise pick a free port explicitly:

```bash
GW_PORT=8490 ai-session code
```

!!! note "Set the same GW_PORT on every command"
    `status`, `connect`, and `stop` read `GW_PORT` too. If you overrode it at
    start, prefix them with the same `GW_PORT=8490` or they will inspect the
    wrong port.

## The client on your laptop cannot connect

**Symptom.** A client on your laptop (aider, Continue, a browser) cannot reach
`http://localhost:<GW_PORT>` although the [first checks](#first-checks) pass on
the login node.

**Cause.** The session's web address is served on `127.0.0.1` on the specific
login node where you started the session; it is not reachable from outside that
node. Your laptop reaches it only through an SSH tunnel, and the tunnel must
target that same login node. The usual causes are a tunnel that is not running,
or a tunnel opened to a different login node than the one hosting your session.

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

**Fix.** Re-run the tunnel command printed in the `READY` block at start — it
names the correct login node. Its form is:

```bash
ssh -N -L <GW_PORT>:localhost:<GW_PORT> <cnetid>@<login-node>.rcc.uchicago.edu
```

- Replace `<GW_PORT>` with your port (both occurrences).
- Replace `<cnetid>` with your CNetID.
- Replace `<login-node>` with the login node named in the `READY` block — not an
  arbitrary login node.

Leave the tunnel running for as long as you work, then point the laptop client
at `http://localhost:<GW_PORT>/v1`.

## Connection refused after working for a while

**Symptom.** The client worked, then requests start failing with connection
refused; `curl -sf http://127.0.0.1:<GW_PORT>/__gateway/health` on the login node
fails.

**Cause.** The gateway runs as a background process on the login node, started
from the terminal where you started the session. When the SSH connection hosting it
closed (laptop sleep, network drop), it died with that connection. The GPU session
may still be running — and still billing.

**Check.** `ai-session status` **on the login node**. A server still listed
under `server:` means the GPU session survived the gateway.

**Fix.** Tear down cleanly, then start again, this time inside `tmux` or
`screen` so an SSH drop cannot kill the gateway. `ai-session stop` is safe to run
even when parts of the stack are already gone: it meters and ends any remaining
session, stops any remaining gateway, and prints the SU charge.

```bash
ai-session stop
ai-session code
```

## Start seems stuck or the model takes minutes to appear

**Symptom.** The start command prints that it is starting the session, then
appears to hang.

**Cause.** This is usually normal. The command blocks until the model has loaded
and answered a probe; loading the 32B coding model typically takes several
minutes after the GPUs are assigned. It waits up to 900 seconds by default
(override with the `READY_TIMEOUT` environment variable, in seconds). The other
common case is that the session has not been assigned GPUs yet because the
cluster is busy.

**Check.** In a second terminal **on the login node**:

```bash
ai-session status
```

`STARTING` means the model is loading: wait. If the session does not appear at
all yet, it is still waiting for free GPUs: the wait is unbounded, but it costs
nothing — the billing floor is computed from the time the GPUs are actually
held, so SU accrue only once the session starts (see
[Billing and Service Units](billing.md)). For more detail, follow the launch log
under your state directory (`run/start.log`).

**Fix.** Wait, or cancel with `Ctrl-C` followed by `ai-session stop` and try
again later. If the load is slow but progressing and 900 seconds will not
suffice, restart with a longer `READY_TIMEOUT`, for example
`READY_TIMEOUT=1800 ai-session code`.

## Getting help

For anything not covered here, use the support channels listed on the
[front page](index.md). When reporting a problem, include the session id shown
by `ai-session status` (or on the receipt printed at stop). For a billing
question, also include the receipt path printed by `ai-session receipt`.
