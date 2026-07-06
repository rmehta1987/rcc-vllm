# Getting Started: Browser Chat

This page takes you from a login-node shell to chatting with an on-cluster model in
your browser, and it is the complete guide to the browser (Open WebUI) client. Three
processes are involved: a **vLLM server** (a Slurm job on a GPU compute node that
serves the model over an OpenAI-compatible API), a **gateway** (a reverse proxy on
the login node at a fixed per-user port, so the URL never changes between sessions
and every request's token counts are recorded for billing), and **Open WebUI** (the
chat interface, also on the login node). One script,
`/project/rcc/mehta5/vllm/ai-session/run_browser_demo.sh`, starts and stops all
three.

Only the vLLM server costs anything. Usage is charged in Service Units (SU), the
cluster's fair-usage accounting unit; 1 SU = 1 A100-GPU-hour. A session is billed
the larger of its metered token work and a reservation floor of
`w_gpu × N_GPUs × wall-clock hours` for the GPUs it holds, so it accrues charge for
as long as it is up, idle or not. See [Billing and Service Units](billing.md) for
the formula, the GPU-tier weights, and the measured rates.

For coding tools (aider, Continue, opencode) against the same service, see
[Coding Sessions](coding/overview.md) instead; this page covers browser chat only.

## Quick start

| Step | Description | Command | Run on |
|---|---|---|---|
| [1](#step-1-start-the-stack) | Start the session, gateway, and UI | `bash /project/rcc/mehta5/vllm/ai-session/run_browser_demo.sh up` | Login node |
| [2](#step-2-open-the-ssh-tunnel) | Forward the UI port to your machine | `ssh -N -L <OWUI_PORT>:localhost:<OWUI_PORT> -J <user>@midway3.rcc.uchicago.edu <user>@<login-node>` | Local machine |
| [3](#step-3-chat-in-the-browser) | Chat | Browse `http://localhost:<OWUI_PORT>` | Local machine |
| [4](#step-4-check-status) | Check what is running (no charge) | `bash /project/rcc/mehta5/vllm/ai-session/run_browser_demo.sh status` | Login node |
| [5](#step-5-stop-and-read-the-charge) | Stop everything and print the SU receipt | `bash /project/rcc/mehta5/vllm/ai-session/run_browser_demo.sh down` | Login node |

## Prerequisites

- An account in the `rcc-staff` Slurm account with read access to the project tree
  `/project/rcc/mehta5`. The shared Python environment, model weights, and scripts
  are read-only to the group; there is nothing to install or copy.
- Run the login-node commands inside `tmux` or `screen`, so that an SSH disconnect
  does not kill the gateway and UI processes that `up` leaves running.

Your writable state (chat history, session files, billing logs) is kept separate
from the shared install, under a per-user directory:
`/project/rcc/mehta5/ai-session-state/<user>` by default (override with the
`AISESSION_STATE_DIR` environment variable). Ports are likewise derived from your
numeric user ID, so two people on the same login node do not collide.

## Step 1: start the stack

Run this **on the login node**, inside `tmux` or `screen`:

```bash
bash /project/rcc/mehta5/vllm/ai-session/run_browser_demo.sh up
```

!!! warning "A running session consumes SU whether or not you send requests"
    From the moment `up` succeeds, the session accrues at least the reservation
    floor — 1.0 SU per hour for the default model (one A100, `w_gpu` = 1.0) — until
    you run `bash /project/rcc/mehta5/vllm/ai-session/run_browser_demo.sh down`.
    Always run `down` when you finish.

`up` does four things, in order:

1. Submits the Slurm GPU job for the vLLM server and blocks until the model
   actually answers, up to `READY_TIMEOUT` (default 900 seconds).
2. Starts the gateway on `127.0.0.1:<GW_PORT>` and waits for its health endpoint
   to return HTTP 200.
3. Starts Open WebUI on `127.0.0.1:<OWUI_PORT>` (its Python imports are heavy;
   expect roughly 30 to 60 seconds).
4. Prints the SSH tunnel command for the login node it ran on, the URL to browse,
   and the matching `down` command.

Defaults, all overridable as environment variables on the `up` command line:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `qwen3_4b` | Model key to serve (Qwen3-4B) |
| `TP` | `1` | Tensor-parallel degree: the number of GPUs the model is split across |
| `CONSTRAINT` | `A100` | Slurm feature constraint selecting the GPU type |
| `TIME` | `02:00:00` | Session walltime `HH:MM:SS`: the job ends after this even if you forget `down`, which caps the maximum floor charge. Printed as `walltime=…` at `up`. |
| `GW_PORT` | `8400 + UID % 90` | Gateway port on the login node (per-user) |
| `OWUI_PORT` | `3000 + UID % 90` | Open WebUI port on the login node (per-user) |
| `AISESSION_STATE_DIR` | `/project/rcc/mehta5/ai-session-state/<user>` | Writable per-user state root |
| `READY_TIMEOUT` | `900` | Seconds to wait for the model to become ready |

To serve the larger general model instead of the 4B default:

```bash
MODEL=qwen2.5_72B TP=4 CONSTRAINT=A100 \
  bash /project/rcc/mehta5/vllm/ai-session/run_browser_demo.sh up
```

!!! note "The 72B at TP=4 holds four A100s"
    Its reservation floor is 4.0 SU per hour, four times the default session.

Verification: `up` ends with this block (your port, user, login node, and model
filled in). If it does not, see [Troubleshooting](troubleshooting.md).

```
================ READY -- chat in your browser ================

  SESSION ACCESS KEY:  <64-hex-character key>

  The gateway now REQUIRES this key. The Open WebUI started here already uses it,
  so YOUR browser tab works out of the box. To let your lab use THIS session,
  share this key: each member points their own client at the gateway (their own
  SSH tunnel to :<GW_PORT>) and sets this as the OpenAI API key. ALL of their
  usage bills to YOU (<user>), the starter. Without the key the gateway refuses
  every request (401). Saved (mode 600, only you can read) at:
      <state-dir>/logs/gateway/session_key

On your LAPTOP, open the tunnel to THIS login node (<login-node>):

  ssh -N -L <OWUI_PORT>:localhost:<OWUI_PORT> -J <user>@midway3.rcc.uchicago.edu <user>@<login-node>

then browse:   http://localhost:<OWUI_PORT>      (pick model 'qwen3_4b')

The SU clock is running. When done (frees the GPU, stops billing):

  bash /project/rcc/mehta5/vllm/ai-session/run_browser_demo.sh down
==============================================================
```

## Step 2: open the SSH tunnel

The UI listens on `127.0.0.1` of the login node, so your browser cannot reach it
directly. Run the tunnel command that `up` printed **on your local machine** (it is
already filled in there; the general form is below):

```bash
ssh -N -L <OWUI_PORT>:localhost:<OWUI_PORT> -J <user>@midway3.rcc.uchicago.edu <user>@<login-node>
```

Replace:

- `<OWUI_PORT>` with the UI port printed by `up` (default `3000 + UID % 90`).
- `<user>` with your CNetID.
- `<login-node>` with the short hostname of the login node where you ran `up`;
  `up` prints it in the READY block. The `-J` jump through
  `midway3.rcc.uchicago.edu` lands you on that specific node.

The command prints nothing and does not return; `-N` means "forward ports only, run
no remote command". Leave this terminal open for the whole session. If your SSH
setup already reaches the specific login node directly, the jump host is
unnecessary: `ssh -N -L <OWUI_PORT>:localhost:<OWUI_PORT>
<user>@<login-node>.rcc.uchicago.edu`.

??? question "What is an SSH tunnel?"
    `ssh -L <port>:localhost:<port> ...` makes your local machine listen on
    `localhost:<port>` and relay every connection through the encrypted SSH
    session to `localhost:<port>` on the remote host. Your browser talks to a
    local port; SSH carries the traffic to the login node where Open WebUI is
    actually listening. Nothing is exposed on any public interface at either end.

!!! note "Working on a login node? Skip the tunnel"
    A browser running on the login node itself (for example under a remote
    desktop) reaches `http://localhost:<OWUI_PORT>` directly.

Verification, **on your local machine**, in a second terminal:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:<OWUI_PORT>
```

Expected output:

```
200
```

## Step 3: chat in the browser

Browse `http://localhost:<OWUI_PORT>` **on your local machine**. No account or
login is required: the UI runs with authentication disabled and is reachable only
through your own tunnel. The model you started (`qwen3_4b` by default) appears in
the model picker at the top of the chat pane; select it and chat.

Chat sessions serve an 8192-token context window (the coding stack serves 32768;
see [Coding Sessions](coding/overview.md)). Your chat history, uploads, and UI
settings persist between sessions in a private per-user database under your home
directory, at `$HOME/.ai-session/openwebui-data` (created mode 700, owner-only).
Storing it in your home directory keeps it readable only by you; other cluster
users cannot read your chats. Because the gateway URL is stable, the same history
is there the next time you bring the stack up.

Verification: send a message and watch the reply stream in. Every request passes
through the gateway, which records its token counts for the bill.

## The session access key

Each `up` mints a random access key for the session and requires it on every
gateway request. `up` prints the key in the READY block and saves it, readable
only by you, at `<state-dir>/logs/gateway/session_key` (mode 600). The Open WebUI
that `up` started already carries the key, so your own browser tab works without
any extra step.

The key is what lets you share one session with your lab. Because several people
may be on the same login node, the gateway binds to `127.0.0.1` and accepts only
requests that carry the key, so no one else on the node can use your session by
accident. To let a labmate use it, give them the key and have them:

1. open their own SSH tunnel to your gateway port (`ssh -N -L
   <GW_PORT>:localhost:<GW_PORT> -J <them>@midway3.rcc.uchicago.edu
   <them>@<login-node>`), then
2. set the key as the OpenAI API key in whatever client they use (in Open WebUI,
   Settings > Connections > API Key; for aider or a script, `OPENAI_API_KEY`).

All of their usage bills to you, the starter — there is one key per session and no
per-person split. A request without the key is refused with HTTP 401. `down`
deletes the key file, so the key stops working the moment you end the session;
the next `up` mints a fresh one. `status` shows only the first six characters, to
confirm a key is set without printing it.

## Step 4: check status

Run this **on the login node** at any time; it reads state and costs nothing:

```bash
bash /project/rcc/mehta5/vllm/ai-session/run_browser_demo.sh status
```

It prints your state directory, the gateway's current backend pointer
(`upstream.json`), whether a session access key is set (the first six characters
only), the listeners on your two ports, the saved gateway and UI process IDs, and
your Slurm queue (`squeue`).

To verify the gateway directly, poll its health endpoint **on the login node**:

```bash
curl -sf http://127.0.0.1:<GW_PORT>/__gateway/health
```

Replace `<GW_PORT>` with your gateway port (default `8400 + UID % 90`; `status`
prints it in its header line). A healthy gateway returns HTTP 200 and a JSON body
of the form `{"gateway": "ok", "backend_active": true}` (`false` when no session
has published a backend). This endpoint is reachable without the access key, so it
deliberately reports only liveness -- not the backend's node:port -- to keep the
compute-node address from leaking to other users on the login node.

## Step 5: stop and read the charge

Run this **on the login node** as soon as you finish:

```bash
bash /project/rcc/mehta5/vllm/ai-session/run_browser_demo.sh down
```

`down` meters the session (collects the token counts and computes the SU charge),
cancels the Slurm job (releasing the GPU and stopping the clock), stops the gateway
and Open WebUI, and prints the SU receipt last, so it cannot scroll away:

```
  SU CHARGE -- this session
    BILLED : <billed> SU      basis=<token|floor>
    model  : <model> / <tier> / TP=<tp>   (N=<n> GPU, w_gpu=<w>)
    usage  : reserved <hours> h   tokens in=<in> out=<out> (<requests> requests)
    job    : <jobid>
    receipt: <path to summary JSON>
```

The same summary is written to
`logs/usage/<user>_<jobid>_<ts>_summary.json` under your state directory
(default `/project/rcc/mehta5/ai-session-state/<user>`). How to read the receipt
fields, and the full rate table behind them, are on
[Billing and Service Units](billing.md).

Verification: run `status` again; the listener and Slurm-job sections should both
report nothing running, and the gateway upstream pointer should be cleared.

## Where each piece runs

| Piece | Runs on | Charged |
|---|---|---|
| vLLM server (the model) | Compute node (via the session job) | Yes — SU, floor-billed while up |
| Gateway | Login node | No |
| Open WebUI | Login node | No |
| `status` / `down` commands | Login node | No |

If a step fails — the READY block never appears, a port is already in use, the
browser cannot connect, or the model is missing from the picker — see
[Troubleshooting](troubleshooting.md).
