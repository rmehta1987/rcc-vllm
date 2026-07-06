# ai-session — operator guide

Single-user-per-session language-model serving on UChicago RCC. Local Qwen models are
served by vLLM over an OpenAI-compatible API and charged in Service Units (SU). This
document covers operation and runtime. The published charging policy is in
[`BILLING_POLICY.md`](BILLING_POLICY.md); the end-user coding guide is in
[`CODING_AGENTS.md`](CODING_AGENTS.md).

## Runtime environment

All processes run in the conda/mamba environment
`/project/rcc/mehta5/conda-envs/vllm-probe` (vllm 0.10.2, torch 2.8.0+cu128,
transformers < 5). It runs against this cluster's NVIDIA driver 535 through CUDA
minor-version compatibility. The earlier `vllm-v0.7.3.sif` Apptainer image is retired.
The environment is managed centrally by RCC staff; users do not build or modify it.

Constraints carried over from the sibling Decrypto and MARSHAL projects:

- Do not `pip install -r requirements.txt` into this environment; the pinned
  requirements pull driver-incompatible torch 2.9 / vllm 0.13. Keep `transformers < 5`
  and `tokenizers < 0.22`.
- Activate with mamba (`module load python/miniforge-25.3.0; mamba activate <path>`).
  `source activate` falls through to the system anaconda 3.8.
- Per-job `TMPDIR` and `TORCHINDUCTOR_CACHE_DIR` are placed on project storage by the
  launcher and sbatch wrappers; do not point them at `/scratch` paths that a cancelled
  job may have removed.

## Layout

```
billing/
  su_formula.py        SU math (su_for_request, reservation floor, session max) + loaders
  billing_policy.yaml  w_gpu table, currency, floor and edge-case flags (RCC-staff editable)
  rate_table.json      benchmarked prefill_tps/decode_tps per (model, tier, TP) (generated)
  test_su_formula.py   unit tests, with the worked examples as fixtures
benchmark/
  bench_billing.py     launch vllm serve in production config, run vllm bench serve over
                       three profiles, write rate_table.json
  bench_billing.sbatch per-tier wrapper (override constraint/gres/TP/model/MAX_MODEL_LEN)
ai-session/
  launch_ai_session.sh launch one vllm server as a Slurm job; prints a JSON line (jobid/port/...)
  server.py            model registry, squeue discovery, node->tier and job->GPU-count resolvers
  ping_servers.py      health poll (matches 'Reply time' for readiness)
  metering.py          /metrics scrape, SU computation, usage logging, gateway-usage reader
  gateway.py           fixed-URL reverse proxy (usage capture, /status, per-client rate limit + body cap)
  ai_session.py        CLI: start / status / connect / end (one-session-per-user guard on start)
  idle_reaper.py       login-node watchdog: end THIS user's idle session to release its GPUs
  systemd/             systemd --user unit for supervising gateway.py (Restart=on-failure)
  run_browser_demo.sh  one-shot {up|down|status} for the browser-chat stack (multi-user)
  run_openwebui.sh     launch Open WebUI pointed at the gateway
  run_coding_agent.sh  one-shot {up|down|status} for the coding stack (session + gateway)
  aider_model_metadata.json  litellm context/cost metadata for the served model (declares 32K context)
  CODING_AGENTS.md     end-user coding guide (aider, Continue, opencode)
  print_su_receipt.py  render the SU-charge summary from a billing summary JSON (stdlib only)
```

## Prerequisites

- Model weights under `/project/rcc/mehta5/vllm/models/`. Currently staged:
  Qwen2.5-Coder-32B-Instruct (coding default), Qwen2.5-72B-Instruct, Qwen3-4B,
  Meta-Llama-3.1-70B-Instruct, Qwen2.5-0.5B-Instruct.
- Run the CLI from the `vllm-probe` environment:
  `module load python/miniforge-25.3.0 && mamba activate /project/rcc/mehta5/conda-envs/vllm-probe`.

## 1. Billing benchmark (populates `rate_table.json`)

`rate_table.json` holds one record per `(model, tier, TP)`. A session with no matching
record bills the reservation floor only and marks the token term unrated; the floor
dominates interactive use, so this is a usable fallback rather than an error. The table
currently holds five records:

| Model | Tier | TP | Prefill (tok/s) | Decode (tok/s) |
|---|---|---:|---:|---:|
| qwen2.5_72B | a100 | 4 | 2901 | 1123 |
| qwen2.5_72B | h100 | 4 | 3787 | 1810 |
| qwen2.5_72B | h200 | 2 | 7594 | 2329 |
| qwen2.5_coder_32B | a100 | 2 | 4773 | 1679 |
| qwen3_4b | a100 | 1 | 20063 | 4129 |

To add or refresh a record, submit the benchmark for that tier:

```bash
# A100 reference tier, Qwen2.5-72B TP=4 (sbatch defaults):
sbatch benchmark/bench_billing.sbatch

# Qwen2.5-Coder-32B, A100-80GB TP=2, measured at the 32K coding serve length:
sbatch --constraint=A100 --gres=gpu:2 \
       --export=ALL,MODEL_KEY=qwen2.5_coder_32B,MODEL_PATH=/project/rcc/mehta5/vllm/models/Qwen2.5-Coder-32B-Instruct,TP=2,MAX_MODEL_LEN=32768 \
       benchmark/bench_billing.sbatch

# H200 TP=2:
sbatch --constraint=H200 --gres=gpu:2 \
       --export=ALL,MODEL_KEY=qwen2.5_72B,MODEL_PATH=/project/rcc/mehta5/vllm/models/Qwen2.5-72B-Instruct,TP=2 \
       benchmark/bench_billing.sbatch

# Single-GPU anchor:
sbatch --gres=gpu:1 \
       --export=ALL,MODEL_KEY=qwen3_4b,MODEL_PATH=/project/rcc/mehta5/vllm/models/Qwen3-4B,TP=1 \
       benchmark/bench_billing.sbatch
```

The benchmark uses identical sweep parameters on every tier (prefill-heavy,
decode-heavy, balanced), sets `--tensor-parallel-size`, and serves with the same flags
production uses. The serve flags are kept in sync between
`bench_billing.py:build_serve_flags` and `launch_ai_session.sh`; a record is only valid
for the configuration it was measured under, so the two must not diverge. Each record
stores `prefill_tps`, `decode_tps`, `alpha_empirical`, ttft/tpot percentiles, and full
provenance (vllm version, dtype, serve flags, max_model_len, GPU name, node, a /metrics
cross-check, and a timestamp).

Validation: every record must have `prefill_tps > decode_tps`. `su_per_1k` is not
ordered across tiers — a faster GPU can have a lower per-token rate — so do not treat
its ordering as a check; the reservation floor is the cross-tier invariant.

### Upgrading vLLM (re-benchmark runbook)

Each `rate_table.json` record stores the vLLM version it was measured under
(`provenance.vllm_version`). Its `prefill_tps`/`decode_tps` only describe that engine
version — a vLLM upgrade can shift throughput enough to make a stored rate wrong, and
likewise if the production serve flags in `launch_ai_session.sh` change. Two mechanisms
keep billing honest across an upgrade:

- Version guard (automatic, no action needed). At `end`, metering reads the running
  engine version from the backend's `/version` endpoint and compares it to the matched
  record's `vllm_version`. On a mismatch it drops the token term and bills the
  reservation floor only, tagging the summary `basis` as `UNRATED` with both versions.
  This is fail-open: if the version cannot be read (a transient scrape miss) the session
  is still rated, and a mismatch never raises inside `end`. The floor dominates
  interactive use, so during the window between an upgrade and a re-benchmark users are
  neither over- nor under-charged — they simply pay the floor with the token term
  suspended.

- Re-benchmark (manual). After upgrading the serving environment, re-measure every live
  `(model, tier, TP)` record so the token term is rated again.

Checklist for an upgrade:

1. Record the new version:
   `/project/rcc/mehta5/conda-envs/vllm-probe/bin/python -c "import vllm; print(vllm.__version__)"`.
2. Confirm `benchmark/bench_billing.py:build_serve_flags` still matches the `vllm serve`
   flags in `launch_ai_session.sh`. If either changed, reconcile them first — a record
   is only valid for the exact flags it was measured under.
3. Re-run `benchmark/bench_billing.sbatch` once per `(model, tier, TP)` in the table
   above (the section-1 examples give the exact submissions). Each run overwrites its
   record with fresh `prefill_tps`/`decode_tps` and stamps the new `vllm_version`.
4. Verify the table: the same record count, each carrying the new `vllm_version` and
   `prefill_tps > decode_tps`.
5. Notify users. Until step 3 completes, sessions on the new engine bill floor-only, and
   receipts read `UNRATED (running vLLM … != rate_table record …)`. Tell users on the
   service that this is expected during a re-benchmark — the floor still applies, only
   the token term is paused — so no one mistakes it for an error.

## 2. Gateway

The vLLM endpoint changes node and port every session and sits on a compute node with
no inbound route from outside the cluster. `gateway.py` is an async reverse proxy on a
login node at a fixed port. It forwards to whichever backend the current session uses,
read from `logs/gateway/upstream.json` (written by `ai_session start`, cleared by
`end`), so no gateway restart is needed between sessions. Start it once and leave it
running:

```bash
# in the vllm-probe environment, on a login node:
python ai-session/gateway.py --host 127.0.0.1 --port 8080
# optional auth: export AISESSION_GATEWAY_KEY=<secret>  (clients send it as the API key)
```

The gateway binds to `127.0.0.1` by default so that on a shared login node no other
user can reach it; clients on a laptop tunnel the port. When `AISESSION_GATEWAY_KEY`
is set, every request must carry that value as the API key
(`Authorization: Bearer <key>`) or the gateway returns HTTP 401.

The gateway records per-request token usage (injecting `stream_options.include_usage`
for streaming requests) into `logs/gateway/usage-*.jsonl`. `ai_session end` consumes
this as the authoritative billing source, so users do not log usage themselves.

The gateway's CPU cost is low (async I/O). For an always-on production front door it
should run as a persistent service rather than a login-node process. The
`run_browser_demo.sh` and `run_coding_agent.sh` wrappers start and stop their own
gateway on a per-user port and do not require this manual step; each `up` also mints a
random per-session access key and starts the gateway with it required (see the
per-session access key note below).

### Liveness: the `/status` route

The gateway answers two key-free routes that clients and monitors can poll without the
session API key:

- `GET /__gateway/health` — a bare liveness probe: `{"gateway":"ok","backend_active":<bool>}`.
  The wrappers use it to know the gateway process is up.
- `GET /status` — a structured live/loading/gone answer, so a client learns the backend
  state without interpreting a raw 502/503:

  ```json
  {"gateway":"ok","backend_active":true,"backend_state":"ready","model_key":"qwen2.5_72B"}
  ```

  `backend_state` is `no_backend` (no session published), `loading` (published but the
  backend's `/health` is not answering yet, i.e. the model is still loading), or `ready`
  (backend `/health` returns success). Neither route reveals the backend `node:port`,
  job id, or backend key — only booleans, the coarse state, and the served model key —
  so a co-tenant who can reach only the gateway cannot discover the backend address.

### Rate limiting and request-size cap

`gateway.py` applies a lightweight per-client token-bucket rate limit and a maximum
request-body size in its proxy path, to blunt a single runaway client on the shared
login-node gateway. Both are env-configurable; the defaults do not affect ordinary
interactive chat or coding traffic:

| Env var | Default | Effect |
|---|---|---|
| `AISESSION_RATE_RPS` | `30` | sustained requests/second per client; burst is twice this. `<=0` disables the limiter. |
| `AISESSION_MAX_BODY_MB` | `16` | maximum request body in MiB. `<=0` disables the cap. |

A client is keyed by its Bearer key when it sends one, otherwise by its peer host. Over
the rate the gateway returns HTTP 429 with an OpenAI-style error body
(`type: rate_limit_exceeded`) and a `Retry-After` header; over the body size it returns
HTTP 413 (`code: payload_too_large`). The health and `/status` routes are not rate
limited. Set the values before launching the gateway, e.g.
`AISESSION_RATE_RPS=10 AISESSION_MAX_BODY_MB=8 python ai-session/gateway.py --port 8080`.

### Supervising the gateway with `systemd --user`

For a longer-lived front door than `nohup`, run the gateway as a user systemd service so
a crash is restarted automatically. The unit file is `ai-session/systemd/ai-session-gateway.service`
(`Restart=on-failure`). It reads `~/.config/ai-session/gateway.env` for `PYBIN`,
`GATEWAY_PY`, `GW_PORT`, `AISESSION_STATE_DIR`, and the per-session `AISESSION_GATEWAY_KEY`
(keep that file mode 600 — it holds the key). Manual install:

```bash
mkdir -p ~/.config/systemd/user
cp ai-session/systemd/ai-session-gateway.service ~/.config/systemd/user/
# write ~/.config/ai-session/gateway.env (see the header comment in the unit file), then:
systemctl --user daemon-reload
systemctl --user start ai-session-gateway.service
systemctl --user status ai-session-gateway.service
journalctl --user -u ai-session-gateway.service -f
```

The wrappers automate this: run either wrapper with `GATEWAY_SUPERVISE=1` and, where a
user-systemd session is available, they write the env file, install and start the unit,
and stop it on `down` instead of using `nohup`. On a login node without a working user
systemd they print a note and fall back to `nohup`, so the wrappers never break there.
To survive an SSH disconnect the user must have lingering enabled
(`loginctl enable-linger $USER`); otherwise run the wrapper inside `tmux` as before.

## 3. Serve a session

```bash
# start; blocks until ready, then publishes the backend to the gateway
python ai-session/ai_session.py start --model qwen2.5_72B --tp 4 --constraint A100 --wait

# print client setup for the current session (gateway URL, SSH tunnel, per-client config)
python ai-session/ai_session.py connect

# report state and endpoint
python ai-session/ai_session.py status

# end: meter, write usage log, scancel
python ai-session/ai_session.py end
```

`--partition`, `--account`, `--constraint`, `--gres`, `--tp`, and `--max-model-len` are
CLI parameters; partition and account default to `test` and `rcc-staff`. A per-PI
deployment changes these defaults.

`start` refuses to launch a second session while you already have an active ai-session
job (any of your `squeue --me` jobs whose name is `<model_key>:<port>`). Because every
reservation is floor-billed for its whole life, an accidental second `start` — a re-run,
or a coding wrapper launched on top of a browser demo — would silently double the
charge. End the running one first (`ai_session.py end`, or the wrapper `down`), or pass
`--allow-multiple` to run more than one on purpose. This guard only ever inspects your
own jobs, so it never sees another user's work.

## 4. Clients

Any OpenAI-compatible client connects to the gateway URL, which does not change between
sessions. `ai_session.py connect` prints the configuration for the current session. The
end-user coding instructions are in [`CODING_AGENTS.md`](CODING_AGENTS.md).

#### Per-session access key

Each wrapper `up` mints a random per-session access key (`openssl rand -hex 16`),
writes it to `logs/gateway/session_key` under the state dir (mode 600, readable only
by the starter), exports it as `AISESSION_GATEWAY_KEY`, and starts the gateway bound to
`127.0.0.1` with that key required. Every client below uses this key as its API key; a
request without it is refused with HTTP 401. Because the gateway is loopback-only and
key-gated, no other user on the shared login node can reach the session by accident,
and the starter can deliberately share the key with their lab: a labmate opens their
own tunnel to the starter's `GW_PORT` and sets the key as the API key. There is one key
per session and all usage bills to the starter — no per-person split and no per-user
keys. `status` shows only the first six characters; `down` deletes the key file, so the
key stops working when the session ends and the next `up` mints a fresh one. Open WebUI
started by `run_browser_demo.sh` receives the key automatically, so the starter's own
browser tab needs no extra step.

| Client | Use | `--agent-client` | Base URL |
|---|---|:---:|---|
| Open WebUI | general chat, documents, RAG; multi-user | no | gateway `/v1` |
| aider | coding (text-edit diffs); default coding client | no | gateway `/v1` |
| Continue | coding in VS Code / JetBrains | no | gateway `/v1` |
| opencode, Cline | autonomous tool-calling agents | yes (`AGENT_CLIENT=1`) | gateway `/v1` |
| curl, Python | scripting and tests | no | gateway `/v1` |

The coding default is `qwen2.5_coder_32B` (Qwen2.5-Coder-32B-Instruct) served at a
32768-token context, against the 8192 default for chat sessions. It is code-specialized
and uses two GPUs (TP=2) against the 72B's four.

Client tools each run in their own environment, separate from `vllm-probe`:

- Open WebUI: `/project/rcc/mehta5/openwebui-env` (python 3.11, `open-webui` 0.9.6).
  Launched against the gateway by `run_openwebui.sh`, which keeps the chat database
  (`DATA_DIR`) in the user's private home directory (`$HOME/.ai-session/openwebui-data`,
  mode 700), keeps `HF_HOME` in shared project storage, and disables the Ollama
  backend. Verified end-to-end on
  Qwen3-4B: model listing, normal and streaming chat through the gateway, and gateway
  metering of usage.
- aider: `/project/rcc/mehta5/aider-env` (python 3.11, `aider-chat`). It uses
  chat-completions and text-edit diffs and does not require `--agent-client`.

### Browser chat (`run_browser_demo.sh`)

`run_browser_demo.sh` brings the browser-chat stack (session, gateway, Open WebUI) up
or down in one command and can be run by any rcc-staff member:

```bash
bash ai-session/run_browser_demo.sh up       # start; consumes SU while up (floor-billed)
bash ai-session/run_browser_demo.sh status   # report state; no SU
bash ai-session/run_browser_demo.sh down      # meter, scancel, stop gateway and UI; no SU
# larger model:
MODEL=qwen2.5_72B TP=4 CONSTRAINT=A100 bash ai-session/run_browser_demo.sh up
```

`up` prints the laptop SSH-tunnel command for the login node it ran on; you then browse
`http://localhost:<port>`.

### Coding (`run_coding_agent.sh`)

`run_coding_agent.sh` is the coding counterpart to `run_browser_demo.sh`. Because aider
is interactive rather than a background server, `up` starts the session and gateway and
prints the aider command, which you run yourself in the git repository to edit:

```bash
bash ai-session/run_coding_agent.sh up        # start session + gateway; consumes SU
# run the printed aider command in your repository
bash ai-session/run_coding_agent.sh status    # report state; no SU
bash ai-session/run_coding_agent.sh down      # meter, scancel, stop gateway; prints SU receipt
```

The defaults are Qwen2.5-Coder-32B-Instruct on A100-80GB at TP=2. Override the model or
tier with environment variables:

```bash
MODEL=qwen2.5_72B TP=4 CONSTRAINT=A100 bash ai-session/run_coding_agent.sh up        # general 72B
MODEL=qwen2.5_coder_32B TP=2 CONSTRAINT=H200 bash ai-session/run_coding_agent.sh up  # H200 tier
```

Coding sessions serve a 32768-token context (`MAX_MODEL_LEN`), the native context
length of the Qwen2.5 models. Context length is independent of `--agent-client`, which
controls only tool calling. aider is invoked with `--model-metadata-file
aider_model_metadata.json` (declares the 32K window to litellm, split 28000 input /
4096 output so a request cannot exceed `max-model-len`), `--weak-model` set to the same
local model (so aider's auxiliary requests stay on-cluster), and `--edit-format diff`.
The reservation floor dominates interactive coding, so the context length has a
negligible effect on the charge.

| Env var | Default | Purpose |
|---|---|---|
| `AISESSION_STATE_DIR` | `/project/rcc/mehta5/ai-session-state/<user>` | per-user writable root; billing logs land here |
| `GW_PORT` | `8400 + UID % 90` | per-user gateway port (override on clash) |
| `MODEL` / `TP` / `CONSTRAINT` | `qwen2.5_coder_32B` / `2` / `A100` | model and serving tier |
| `MAX_MODEL_LEN` | `32768` | served context length |
| `EDIT_FORMAT` | `diff` | aider edit format (`whole` for full-file rewrites) |
| `AGENT_CLIENT` | `0` | `1` enables vLLM tool calling for opencode/Cline |

### Tool-calling agents

opencode and Cline drive the model through native function calling, which
`--agent-client` enables (`--enable-auto-tool-choice --tool-call-parser hermes` for
Qwen). The coding wrapper exposes this as `AGENT_CLIENT=1 bash
ai-session/run_coding_agent.sh up`. Directly:

```bash
python ai-session/ai_session.py start --model qwen2.5_coder_32B --tp 2 --constraint A100 \
    --max-model-len 32768 --agent-client --wait
```

opencode is supported, verified 2026-07-03 (opencode 1.14.41, job 51391003). It needs
two project-local files in the repo being edited: `opencode.json` (copy
`ai-session/opencode.example.json`, which also disables personal MCP servers) and the
`AGENTS.md` rules file from CODING_AGENTS.md section 8. Without `AGENTS.md` the failure
is silent — the served model never emits the `<tool_call>` marker tokens, the tool JSON
comes back as plain text, and no edit happens (zero parser exceptions server-side).
aider does not use tool calling and is the default coding client for that reason. An
exclusively reserved node is floor-billed regardless of request volume, so run `end`
(or `down`) as soon as you stop.

### Manual client commands

Against a gateway on `localhost:8080`. Substitute `<SESSION_KEY>` with the session
access key `up` printed (also saved at `logs/gateway/session_key` and shown by
`ai_session.py connect`); the gateway requires it. A gateway you start by hand with no
`AISESSION_GATEWAY_KEY` is keyless and any non-empty string works.

```bash
# Open WebUI (its own environment on a login node, not vllm-probe):
python -m venv ~/openwebui-env && source ~/openwebui-env/bin/activate && pip install open-webui
OPENAI_API_BASE_URL=http://localhost:8080/v1 OPENAI_API_KEY=<SESSION_KEY> open-webui serve --port 3000

# aider (its own environment; metadata file declares the 32K context):
OPENAI_API_BASE=http://localhost:8080/v1 OPENAI_API_KEY=<SESSION_KEY> \
  /project/rcc/mehta5/aider-env/bin/aider --model openai/qwen2.5_coder_32B \
    --weak-model openai/qwen2.5_coder_32B \
    --model-metadata-file ai-session/aider_model_metadata.json --edit-format diff \
    --analytics-disable
```

### Multi-tenant model

The environment, model weights, and code are shared and read-only (group `rcc-staff`,
setgid); colleagues run the one managed install rather than copying it. Each user's
writable state (session files, usage and billing logs, gateway pointer, Slurm logs) is
isolated under a per-user directory, and ports are derived from the UID so that two users
on one login node do not collide. The Open WebUI chat database is kept separately in the
user's private home directory (`$HOME/.ai-session/openwebui-data`, mode 700) so that no
other user can read it.

| Env var | Default | Purpose |
|---|---|---|
| `AISESSION_STATE_DIR` | `/project/rcc/mehta5/ai-session-state/<user>` | per-user writable root for that user's own session, usage, and billing summaries |
| `AISESSION_BILLING_DIR` | `/project/rcc/mehta5/ai-session-billing` | central staff-only ledger (mode `2770`, group `rcc-staff`); `end` and the sacct sweep both record each charge here |
| `GW_PORT` / `OWUI_PORT` | `8400 + UID % 90` / `3000 + UID % 90` | per-user gateway and UI ports (override on clash) |

Central accounting is a real pipeline, not just group-readable files: on
`ai_session end` the final charge is written to the `AISESSION_BILLING_DIR`
ledger as `<user>_<jobid>_end.json` (schema `ai-session-billing/1`, mode
`0640`), and the staff sacct sweep independently writes the authoritative
reservation floor for the same jobs (`source=sweep`). See
`BILLING_POLICY.md` -> "Central accounting". The write is best-effort and never
blocks `end`.

`AISESSION_STATE_DIR` is read by `gateway.py`, `ai_session.py`, and
`launch_ai_session.sh`, each defaulting to `<script_dir>/logs` when unset (the original
single-tenant layout). Running `ai_session.py` or `gateway.py` directly is therefore
unchanged; the env var (or the wrappers) selects the isolated multi-user layout.

When fixing permissions on the shared tree, apply `chmod 2775` last: `chgrp` clears the
setgid bit.

## 5. Billing token source

`end` selects the token source in priority order:

1. an explicit `--usage-jsonl` file,
2. the gateway usage log for the session's time window,
3. the `/metrics` session delta from vLLM's own counters.

When both per-request usage and `/metrics` are present, `end` reconciles them and flags
a mismatch over 2%. If the gateway usage is unavailable, billing proceeds from
`/metrics` alone.

## 6. Billing computation

`end` resolves the GPU tier (`scontrol show node` features, falling back to the launch
`--constraint`), the reserved GPU count N (`scontrol show job`, falling back to
requested gres), and the reserved wall hours (`sacct Elapsed`, falling back to recorded
start-to-now), then applies `billing/su_formula.py`:

```
token_su  = w_gpu(tier) * N * (T_in / prefill_tps + T_out / decode_tps) / 3600
floor_su  = w_gpu(tier) * N * reserved_wall_hours
billed_su = max(token_su, floor_su)        # with the floor disabled, token_su
```

Output is written to `logs/usage/<user>_<jobid>_<ts>_summary.json`, plus a per-request
`.jsonl` when a usage log was supplied. `BILLING_POLICY.md` gives the rationale and
worked examples.

## 7. Staff: billing sweep

`billing_sweep.py` reconstructs each session's authoritative reservation-floor charge
from Slurm accounting (`sacct`), which users cannot edit, and writes it to the central
ledger independent of whether the user ran `end`. A session holds its GPUs exclusively,
so the charge is `max(token_su, floor_su)` and the floor dominates interactive use; this
sweep captures the floor even for sessions the user forgot to close. It is the
authoritative record for the floor. The per-session `end` record (`source="end"`) carries
the token detail; the sweep record (`source="sweep"`) sets `token_su=null` and
`basis="floor"`. Both may coexist for one job and are kept, distinguished by `source`.

The script reuses the real billing formula (`billing/su_formula.py` for the `w_gpu`
weight and the floor) and `server.py`'s `MODEL_REGISTRY` and node-to-tier resolver, so it
must run under the `vllm-probe` python (that formula needs PyYAML). A job is an
ai-session job when its Slurm `JobName` is `<model_key>:<port>` with `model_key` in
`MODEL_REGISTRY`; benchmark (`bench_billing`) and unrelated jobs are excluded. Only
terminal states (COMPLETED, CANCELLED, TIMEOUT, FAILED, NODE_FAIL) are swept. The GPU
tier comes from the node's hardware features (`scontrol show node`, which persists after
the job ends), and the reserved GPU count `N` from the job's `AllocTRES` (`gres/gpu=<n>`),
which survives in accounting after the controller has purged the live job.

```bash
# preview last 7 days for all users, writing nothing:
/project/rcc/mehta5/conda-envs/vllm-probe/bin/python ai-session/billing_sweep.py --dry-run

# write the floor records for June onward, all users:
/project/rcc/mehta5/conda-envs/vllm-probe/bin/python ai-session/billing_sweep.py --since 2026-06-01
```

Flags: `--since <ISO or sacct time>` (default last 7 days), `--user <name>` (default all
users), `--billing-dir <path>` (default `$AISESSION_BILLING_DIR`, else
`/project/rcc/mehta5/ai-session-billing`), `--dry-run` (compute and print only),
`--force` (overwrite an existing record). The sweep is idempotent: it writes
`<user>_<jobid>_sweep.json` (mode 0640) once and skips it on later runs unless `--force`
is given, so it is safe to re-run over an overlapping window. A job whose tier or GPU
count cannot be resolved is reported `UNRESOLVED` and skipped; malformed accounting rows
are skipped with a warning; neither aborts the sweep. Each run prints a per-job line and
a totals summary (written / skipped / unresolved / malformed and the summed floor SU).

Suggested cadence: run it nightly from cron on a login node with a window that overlaps
the previous run (e.g. `--since` two days back), which the idempotent skip makes safe:

```cron
# 03:15 nightly: sweep the last two days into the central ledger
15 3 * * *  /project/rcc/mehta5/conda-envs/vllm-probe/bin/python /project/rcc/mehta5/vllm/ai-session/billing_sweep.py --since $(date -d '2 days ago' +\%Y-\%m-\%d) >> /project/rcc/mehta5/ai-session-billing/sweep.log 2>&1
```

## 8. Tests

```bash
# pure-logic formula tests; the vllm-probe env has no pytest, so use the system one:
/software/python-anaconda-2020.11-el8-x86_64/bin/python -m pytest billing/ -q
```

## 9. Open items

These do not block the code but should be settled before publishing rates:

1. Final `w_gpu` values. The defaults are peer-grounded estimates (Hopper set to Delta's
   3.0 / 2.0; V100 and RTX6000 excluded). RCC staff confirm them in `billing_policy.yaml`.
2. Whether the `test` partition allows sub-node GPU allocation or holds nodes whole.
   Submit `--gres=gpu:2` on a 4-GPU node and check whether the other two GPUs can be
   allocated to another job. This determines N (reserved gres versus whole node) in every
   charge. `ai_session end` detects `n_alloc > TP` and notes a whole-node reservation;
   `--n-gpus` forces the billed count.
3. End-to-end metering check: run `start --wait` then `end --usage-jsonl ...` with
   Qwen3-4B and confirm the `/metrics` token delta matches the summed per-request usage
   within 2% and that `squeue` shows the job gone.
4. Optional dollar cost-recovery target, if SU must map to currency.

## 10. Idle-session reaper (operator)

A session holds its GPUs exclusively and is floor-billed for the whole reservation
whether it is busy or idle, so a user who forgets to run `end`/`down` keeps paying.
`idle_reaper.py` is a login-node watchdog that ends the invoking user's own idle
session automatically. Each pass it finds that user's running ai-session job
(`squeue --me`, job-name `<model_key>:<port>`), reads the backend's request and token
counters with the same `/metrics` scrape the metering code uses, and if none of those
counters has advanced for `--idle-min` minutes (default 30) it warns and runs the normal
`end` path (final metering, central-ledger record, `scancel`) to release the GPUs. It
only ever inspects and ends the caller's own jobs — it can never touch another user's
session. It must run under the `vllm-probe` python (the `end` it invokes needs PyYAML)
and reads `AISESSION_STATE_DIR` like the other tools.

```bash
PY=/project/rcc/mehta5/conda-envs/vllm-probe/bin/python
# preview only, writes nothing, reaps nothing (safe to run any time):
$PY ai-session/idle_reaper.py --once --dry-run --idle-min 30

# one real pass (cron drives the cadence):
$PY ai-session/idle_reaper.py --once --idle-min 30

# long-running loop (run in tmux or under a systemd --user timer/service):
$PY ai-session/idle_reaper.py --idle-min 30 --poll-sec 300
```

Flags: `--idle-min <min>` (idle threshold, default 30), `--poll-sec <s>` (seconds
between passes in loop mode, default 300), `--once` (one pass then exit — the
cron-friendly mode), `--dry-run` (report only; writes no state and reaps nothing), and
`--email <addr>` (best-effort notification before reaping, via `mail`/`sendmail`;
defaults to `$AISESSION_REAPER_EMAIL`). Idle is judged across passes using a small
per-job state file under `$AISESSION_STATE_DIR/run/idle_reaper/`; any counter increase
resets the idle clock, and an in-flight generation (which advances the token counters)
is never mistaken for idleness. If `/metrics` is unreachable the reaper declines to reap
that pass, because it cannot confirm the session is idle.

Suggested cadence — a per-user cron entry that runs one pass every five minutes (the
`--idle-min` window, not the cron interval, decides when a session is reaped):

```cron
# every 5 min: end MY session if it has been idle 30 min
*/5 * * * *  AISESSION_STATE_DIR=/project/rcc/mehta5/ai-session-state/$USER /project/rcc/mehta5/conda-envs/vllm-probe/bin/python /project/rcc/mehta5/vllm/ai-session/idle_reaper.py --once --idle-min 30 >> $HOME/.ai-session/idle_reaper.log 2>&1
```

Or as a `systemd --user` timer paired with a `Type=oneshot` service that runs the same
`--once` command; `OnUnitActiveSec=5min` gives the same cadence and restarts with the
user session. Either way the reaper's `scancel` requires that user's own Slurm
credentials, so run it as the session owner, not from a shared service account.
