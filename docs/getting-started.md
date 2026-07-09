# Getting Started: Browser Chat

This page takes you from a login-node shell to chatting with an on-cluster model in
your browser, and it is the complete guide to the browser (Open WebUI) client. Three
pieces are involved: a **model server** (running on a GPU node inside the cluster,
speaking the standard OpenAI API format that most AI tools can talk to), a
**gateway** (a small always-on relay the service runs on the login node
at a fixed per-user port, so the session URL never changes between sessions and every
request's token counts are recorded for billing), and **Open WebUI** (the chat
interface, also on the login node). One command, `ai-session chat`, starts all
three; `ai-session stop` stops them.

Only the model server costs anything. Usage is charged in Service Units (SU), the
service's fair-usage accounting unit; 1 SU = 1 A100-GPU-hour. A session is billed
the larger of its metered token work and a reservation floor — the cost of the
GPUs it holds for its whole wall-clock lifetime — so it accrues charge for as long
as it is up, idle or not. See [Billing and Service Units](billing.md) for the
formula, the GPU weights, and the measured rates.

For coding tools (aider, Continue, opencode) against the same service, see
[Coding Sessions](coding/overview.md) instead; this page covers browser chat only.

## Quick start

| Step | Description | Command | Run on |
|---|---|---|---|
| [0](#step-0-load-the-module) | Put `ai-session` on your PATH | `module load ai-session` | Login node |
| [1](#step-1-start-the-session) | Start the session and the chat UI (first run: add `--account <acct> --partition <part>`) | `ai-session chat` | Login node |
| [2](#step-2-open-the-ssh-tunnel) | Forward the UI port to your machine | `ssh -N -f -L <UI_PORT>:localhost:<UI_PORT> <user>@<login-node>.rcc.uchicago.edu` | Local machine |
| [3](#step-3-chat-in-the-browser) | Chat | Browse `http://localhost:<UI_PORT>` | Local machine |
| [4](#step-4-check-status) | Check what is running (no charge) | `ai-session status` | Login node |
| [5](#step-5-stop-and-read-the-charge) | Stop everything and print the SU receipt | `ai-session stop` | Login node |

## Prerequisites

- **An RCC account.** No special group membership is required.  Get an [RCC Account here](https://rcc.uchicago.edu/accounts-allocations/request-account) .
- A **Slurm account and GPU partition** to run the GPU job under. These are unique
  to you and your PI: the first time you start a
  session you pass them with `--account` and `--partition`, and they are then
  remembered (see [Step 1](#step-1-start-the-session)). 
- Run the login-node commands inside `tmux` or `screen`, so that an SSH disconnect
  does not kill the relay and UI processes that the start command leaves running.
  
Your writable state (chat history, session files, billing logs) is kept separate
under a per-user directory. Ports are likewise derived from your numeric user ID.

## Step 0: load the module

Once per shell, **on the login node**:

```bash
module load ai-session
```

This puts the `ai-session` command on your PATH.

## Step 1: start the session

The first time you start any session, name your Slurm account and GPU partition
**on the login node**, inside `tmux` or `screen`:

```bash
ai-session chat --account <your-account> --partition <your-partition>
```

The two values are saved to `~/.ai-session/config`, so from then on
`ai-session chat` (or `code`, or `fast`) needs no flags; pass `--account` or
`--partition` again only to change them. Without a saved account and partition,
the command stops before reserving any GPU and tells you what to set — nothing is
billed.

!!! warning "A running session consumes SU whether or not you send requests"
    From the moment the session is up, it accrues at least the reservation floor —
    4.0 SU per hour for the `chat` preset (Qwen2.5 72B on four A100s) — until you
    run `ai-session stop`. Always stop the session when you finish. For casual use,
    `ai-session fast` serves a small model on one GPU at 1.0 SU per hour.

`ai-session chat` does four things, in order:

1. Starts the model server on cluster GPUs and blocks until the model actually
   answers. Loading takes a few minutes; the command prints its progress.
2. Starts the gateway on `127.0.0.1:<GW_PORT>` and waits for its health check
   address to return HTTP 200.
3. Starts Open WebUI on `127.0.0.1:<UI_PORT>` (its Python imports are heavy;
   expect roughly 30 to 60 seconds).
4. Prints the SSH tunnel command for the login node it ran on, the URL to browse,
   and the session access key.

Options and their defaults:

| Option | Default | Meaning |
|---|---|---|
| `--account NAME` | none — required once | Your Slurm account. Required on the first run, then remembered in `~/.ai-session/config`. |
| `--partition NAME` | none — required once | The GPU partition to run in. Required on the first run, then remembered. |
| `--time HH:MM:SS` | `02:00:00` | Session time limit. The session ends after this even if you forget `ai-session stop`, which caps the maximum floor charge. |
| `--model KEY` | preset's model | Serve a different registered model (e.g. `qwen3_32B`, a thinking model, or `llama3.1_70B`); the right GPU configuration is chosen for you. See [Command Reference](reference.md#models). |
| `--lora NAME=PATH` | none | Also serve your own fine-tuned adapter; see [Your Own Fine-Tuned Model](lora.md). Repeatable. |

The presets:

| Command | Model | GPUs held | Floor cost |
|---|---|---|---|
| `ai-session chat` | Qwen2.5 72B Instruct | 4 x A100-80GB | 4.0 SU/hour |
| `ai-session fast` | Qwen3 4B | 1 x A100 | 1.0 SU/hour |

Verification: the command ends with a READY block (your port, user, login node,
and model filled in). If it does not, see [Troubleshooting](troubleshooting.md).

```
================ READY -- chat in your browser ================

  SESSION ACCESS KEY:  <64-hex-character key>

  The gateway now REQUIRES this key. The Open WebUI started here already uses it,
  so YOUR browser tab works out of the box. ...

On your LAPTOP, open the tunnel to THIS login node (<login-node>) -- one login, -f backgrounds it:

  ssh -N -f -L <UI_PORT>:localhost:<UI_PORT> <user>@<login-node>.rcc.uchicago.edu

then browse:   http://localhost:<UI_PORT>
==============================================================
```

## Step 2: open the SSH tunnel

The UI listens on `127.0.0.1` of the login node, so your browser cannot reach it
directly. Run the tunnel command that the READY block printed **on your local
machine** (it is already filled in there; the general form is below):

```bash
ssh -N -f -L <UI_PORT>:localhost:<UI_PORT> <user>@<login-node>.rcc.uchicago.edu
```

Replace:

- `<UI_PORT>` with the UI port printed in the READY block.
- `<user>` with your CNetID.
- `<login-node>` with the login node where you started the session (e.g.
  `midway3-login4`); the READY block names it. The RCC login nodes are directly
  reachable — they are the same hosts the `midway3.rcc.uchicago.edu` round-robin
  points at — so this is a single connection with **one login prompt**.

`-N` means "forward ports only, run no remote command"; `-f` backgrounds the tunnel
once it connects, so you can close the terminal (stop it later by killing the ssh
process, e.g. `pkill -f "ssh -N -f -L <UI_PORT>"`).

Connect to the **specific** node the READY block names, not the
`midway3.rcc.uchicago.edu` alias — the alias may land you on a different login node
than the one running your UI. Only if your network cannot reach that node directly,
jump through the alias as a fallback (this authenticates **twice**):
`ssh -N -f -L <UI_PORT>:localhost:<UI_PORT> -J <user>@midway3.rcc.uchicago.edu <user>@<login-node>`.

??? question "What is an SSH tunnel?"
    `ssh -L <port>:localhost:<port> ...` makes your local machine listen on
    `localhost:<port>` and relay every connection through the encrypted SSH
    session to `localhost:<port>` on the remote host. Your browser talks to a
    local port; SSH carries the traffic to the login node where Open WebUI is
    actually listening. Nothing is exposed on any public interface at either end.

!!! note "Working on a login node? Skip the tunnel"
    A browser running on the login node itself (for example under a remote
    desktop) reaches `http://localhost:<UI_PORT>` directly.

Verification, **on your local machine**, in a second terminal:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:<UI_PORT>
```

Expected output:

```
200
```

## Step 3: chat in the browser

Browse `http://localhost:<UI_PORT>` **on your local machine**. No account or
login is required: the UI runs with authentication disabled and is reachable only
through your own tunnel. The model you started appears in the model picker at the
top of the chat pane; select it and chat.

Chat sessions serve an 8192-token context window (coding sessions serve 32768;
see [Coding Sessions](coding/overview.md)). Your chat history, uploads, and UI
settings persist between sessions in a private per-user database under your home
directory, at `$HOME/.ai-session/openwebui-data` (created mode 700, owner-only).
Storing it in your home directory keeps it readable only by you; other cluster
users cannot read your chats. Because the session URL is stable, the same history
is there the next time you start a session.

Verification: send a message and watch the reply stream in. Every request passes
through the relay, which records its token counts for the bill.

## The session access key

Each session start mints a random access key, which is required on every
request. The READY block prints the key and saves it, readable only by you, at
`<state-dir>/logs/gateway/session_key` (mode 600). The Open WebUI that was started
for you already carries the key, so your own browser tab works without any extra
step.

The key is what lets you share one session with your lab. Because several people
may be on the same login node, the gateway binds to `127.0.0.1` and accepts only
requests that carry the key, so no one else on the node can use your session by
accident. To let a labmate use it, give them the key and have them:

1. open their own SSH tunnel to your session's port (`ssh -N -f -L
   <GW_PORT>:localhost:<GW_PORT> <them>@<login-node>.rcc.uchicago.edu`), then
2. set the key as the OpenAI API key in whatever client they use (in Open WebUI,
   Settings > Connections > API Key; for aider or a script, `OPENAI_API_KEY`).

All of their usage bills to you, the starter — there is one key per session and no
per-person split. A request without the key is refused with HTTP 401.
`ai-session stop` deletes the key, so it stops working the moment you end the
session; the next start mints a fresh one. `ai-session status` shows only the
first six characters, to confirm a key is set without printing it.

## Step 4: check status

Run this **on the login node** at any time; it reads state and costs nothing:

```bash
ai-session status
```

It answers the question that matters — is the session ready, still loading, or
stopped — and shows whether an access key is set (first six characters only) and
how long the running session has been up. While the model is loading, it says so
and suggests re-checking; raw connection errors in your client during this window
mean the same thing.

To verify the gateway directly, poll its health check address **on the login node**:

```bash
curl -sf http://127.0.0.1:<GW_PORT>/__gateway/health
```

Replace `<GW_PORT>` with your port (`ai-session status` prints it when one
is running; `ai-session connect` always prints it). A healthy reply is
HTTP 200 with a JSON body of the form
`{"gateway": "ok", "backend_active": true}` (`false` when no session has published
a backend). This address is reachable without the access key, so it deliberately
reports only liveness — never the model server's internal address.

## Step 5: stop and read the charge

Run this **on the login node** as soon as you finish:

```bash
ai-session stop
```

`ai-session stop` meters the session (collects the token counts and computes the
SU charge), releases the GPUs and stops the clock, stops the relay and Open
WebUI, and prints the SU receipt last, so it cannot scroll away:

```
  SU CHARGE -- this session
    BILLED : <billed> SU      basis=<token|floor>
    model  : <model> on <n> x <GPU type> GPU   (weight <w> SU per GPU-hour)
    usage  : held <hours> h   tokens in=<in> out=<out> (<requests> requests)
    session: <session id>
    receipt: <path to summary JSON>
```

The same summary is written to a receipt file under your state directory. How to
read the receipt fields, and the full rate table behind them, are on
[Billing and Service Units](billing.md).

Verification: run `ai-session status` again; it should report no session running
and no access key set.

## Where each piece runs

| Piece | Runs on | Charged |
|---|---|---|
| Model server | GPU node | Yes — SU, floor-billed while up |
| Gateway | Login node | No |
| Open WebUI | Login node | No |
| `ai-session status` / `stop` | Login node | No |

If a step fails — the READY block never appears, a port is already in use, the
browser cannot connect, or the model is missing from the picker — see
[Troubleshooting](troubleshooting.md).
