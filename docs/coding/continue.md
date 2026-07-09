# Continue (VS Code and JetBrains)

Continue is an editor extension that adds chat and model-driven editing inside VS
Code and the JetBrains IDEs. You install it yourself from your editor's marketplace;
nothing is installed on the cluster. It connects to the same gateway as every other
client — the small always-on connection point the service runs on the login node,
which gives each session one stable URL (see [Coding Sessions](overview.md)) —
so it needs no client software on RCC at all, only a reachable port.

Two things must be true before Continue can work:

- A coding session is running. Starting and stopping sessions is covered on
  [Coding Sessions](overview.md); this page assumes one is up.
- If your editor runs on your laptop rather than on a login node, the SSH tunnel to
  the session's port is open. The tunnel procedure and background live in the SSH
  tunnel section of [Coding Sessions](overview.md).

!!! note "A running session consumes SU whether or not Continue sends requests"
    A session bills for its wall-clock GPU reservation, idle or not (see
    [Billing and Service Units](../billing.md)). Stop the session as described on
    [Coding Sessions](overview.md) as soon as you stop working.

| Step | Description | Command | Run on |
|---|---|---|---|
| 1 | Confirm a coding session is running (start one per [Coding Sessions](overview.md)) | `ai-session status` | Login node |
| 2 | Open the SSH tunnel to the session's port (laptop editors only) | `ssh -N -L <GW_PORT>:localhost:<GW_PORT> <cnetid>@<login-node>.rcc.uchicago.edu` | Local machine |
| 3 | Install the Continue extension | Editor marketplace; no shell command | Local machine |
| 4 | Add the model definition | Edit `~/.continue/config.yaml` (Step 4 below) | Local machine |

## Step 1: Confirm the session and gateway are running

Continue only connects to a session; it never starts one. Check what is running —
this costs nothing. Run this **on the login node**:

```bash
ai-session status
```

A healthy session reports:

```
session: READY  (model qwen2.5_coder_32B, URL http://localhost:<GW_PORT>/v1)
access key: set (abc123...; full value: ai-session connect)
```

The port in the URL is the `<GW_PORT>` used in every later step. If the report
says the session is still starting, wait and re-check; if it says none is
running, start one per [Coding Sessions](overview.md).

## Step 2: Open the SSH tunnel (laptop editors only)

The connection point listens on `127.0.0.1:<GW_PORT>` on the login node where the
session was started, so an editor on your laptop reaches it only through an SSH
tunnel. If your editor runs on a login node (for example under VS Code
Remote-SSH), skip this step and use the `localhost` URL directly.

Run this **on your local machine** and leave it running; it prints nothing and
stays in the foreground:

```bash
ssh -N -L <GW_PORT>:localhost:<GW_PORT> <cnetid>@<login-node>.rcc.uchicago.edu
```

- Replace `<GW_PORT>` with the port shown by `ai-session status`.
- Replace `<cnetid>` with your CNetID.
- Replace `<login-node>` with the login node where the session was started. The
  tunnel command printed at start already names the correct node.

Details and background are in the SSH tunnel section of
[Coding Sessions](overview.md).

Verify the tunnel **on your local machine** by querying the gateway's health
endpoint:

```bash
curl -sf http://localhost:<GW_PORT>/__gateway/health
```

Expected output is one line of JSON reporting whether the connection point is up
and whether a backend is published (`backend_active` is `false` when no session
is active). It
does not expose the backend's node:port, which is kept off this keyless endpoint:

```
{"gateway":"ok","backend_active":true}
```

No output means the tunnel or the gateway is down; see
[Troubleshooting](../troubleshooting.md).

## Step 3: Install the Continue extension

Install Continue from the marketplace inside your editor: the Extensions view in VS
Code, or Settings, then Plugins, in a JetBrains IDE. This runs entirely on your own
machine and needs no cluster access. After installation the Continue panel appears
in the editor's side bar; opening it confirms the install.

## Step 4: Point Continue at the gateway

The session speaks the standard OpenAI API format that most AI tools can talk to,
so Continue's `openai` provider is used with the model name the session serves
under. Add a model definition **on the machine where the editor runs**:

```yaml title="~/.continue/config.yaml"
allowAnonymousTelemetry: false
models:
  - name: Qwen2.5-Coder-32B (RCC)
    provider: openai
    model: qwen2.5_coder_32B
    apiBase: http://localhost:<GW_PORT>/v1
    apiKey: <SESSION_KEY>
    roles: [chat, edit, apply]
```

Older Continue versions read a JSON file instead:

```json title="~/.continue/config.json"
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

- Continue reads a static config file, so the two placeholders are filled in with
  literal values: `ai-session connect` prints this exact block with `<GW_PORT>`
  and `<SESSION_KEY>` already substituted, ready to paste.
- `allowAnonymousTelemetry: false` turns off Continue's own usage telemetry. This is
  a client-side setting independent of the model traffic, which never leaves RCC; the
  [Data location note on the home page](../index.md#data-location) explains the
  distinction.
- `apiBase` must include the `/v1` suffix.
- `<SESSION_KEY>` is the session access key minted at start. Every request must
  carry it; a request without it is refused with HTTP 401. See
  [Coding Sessions](overview.md#the-session-access-key) for sharing it with your lab.
- `model` must equal the model the session was started with; `qwen2.5_coder_32B` is
  the default. If you started the session with a different model, use that key here
  (see model selection on [Coding Sessions](overview.md)).

Verify: in Continue's model selector choose "Qwen2.5-Coder-32B (RCC)" and send a
short chat message. A streamed reply confirms the whole path — editor, tunnel,
connection point, session. No reply means a connection problem; see
[Troubleshooting](../troubleshooting.md).

## Working in the editor

Use the chat and edit/apply features; the `roles: [chat, edit, apply]` line in the
configuration enables exactly these. Chat answers questions about code you attach
to the conversation; edit/apply proposes a change to the selected region and
applies it on your confirmation. Both are request-response interactions, which the
session serves well.

!!! tip "Leave tab-autocomplete disabled"
    Autocomplete fires on keystrokes and requires low latency. Single-stream
    generation for the 32B model on its two GPUs is approximately 66 ms per output
    token (median time-per-output-token; billing benchmark of 2026-06-10,
    midway3-0377.rcc.local, model-server version 0.10.2), which is not suitable for
    completion-as-you-type. If you want autocomplete, run a separate small-model
    session with `qwen3_4b` for that purpose only (see model selection on
    [Coding Sessions](overview.md)); it reserves one A100 and bills a floor of
    1.0 SU per hour (benchmarked 2026-06-02, midway3-0294.rcc.local, model-server
    version 0.10.2).

## Connection failures

When Continue cannot reach the model — connection refused, timeouts, or an empty
model list — the cause is almost always the tunnel or the session, not the
extension. The diagnosis table is on [Troubleshooting](../troubleshooting.md);
check the tunnel (Step 2 above) and the session (Step 1 above) first.
