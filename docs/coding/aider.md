# aider

aider is the default coding client for AI sessions. It drives the model through the
chat-completions API and applies edits as text — unified diffs or whole-file
replacements — so it does not depend on the model's function-calling support, which
is the least reliable part of serving a local model through vLLM. Tool-calling
agents such as opencode and Cline are covered on
[opencode and Cline](opencode.md); the shared session lifecycle (`up`, `status`,
`down`) is on [Coding Sessions](overview.md).

aider is already installed at `/project/rcc/mehta5/aider-env/bin/aider`; you do not
install anything. It requires the working directory to be inside a git repository —
run `git init` first if necessary.

## Quick start

| Step | Description | Command | Run on |
|---|---|---|---|
| 1 | Start a coding session (procedure on [Coding Sessions](overview.md)) | `bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh up` | Login node |
| 2 | Change into the git repository you want to edit | `cd /path/to/your/repo` | Login node |
| 3 | Run the aider command printed by `up` | See Step 1 below | Login node |
| 4 | Stop the session when finished | `bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh down` | Login node |

!!! warning "A running session consumes SU whether or not you send requests"
    The default coding session (`qwen2.5_coder_32B` on 2 A100 GPUs) bills a floor of
    2.0 SU per hour of wall-clock time; stop it with
    `bash /project/rcc/mehta5/vllm/ai-session/run_coding_agent.sh down` as soon as
    you stop working. Rates and the billing formula are on
    [Billing and Service Units](../billing.md).

## Step 1: Run aider

Run aider **on the login node** where you ran `up`, in a second terminal (leave the
`up` terminal running). It can also run on your laptop through an SSH tunnel to the
gateway port; the tunnel procedure is on [Coding Sessions](overview.md).

Change into your repository and run the command exactly as `up` printed it:

```bash
cd /path/to/your/repo
OPENAI_API_BASE=http://localhost:<GW_PORT>/v1 \
OPENAI_API_KEY=<SESSION_KEY> \
/project/rcc/mehta5/aider-env/bin/aider \
  --model openai/qwen2.5_coder_32B \
  --weak-model openai/qwen2.5_coder_32B \
  --model-metadata-file /project/rcc/mehta5/vllm/ai-session/aider_model_metadata.json \
  --edit-format diff --analytics-disable
```

- Replace `<GW_PORT>` with the gateway port printed by `up` (also shown by
  `status`).
- Replace `<SESSION_KEY>` with the session access key `up` printed (also saved at
  `<state-dir>/logs/gateway/session_key` and shown by `connect`). The gateway
  requires it; without it every request is refused with HTTP 401. See
  [Coding Sessions](overview.md#the-session-access-key) for sharing it with your lab.
- Replace `/path/to/your/repo` with the git repository you want to edit.

The two `OPENAI_API_*` variables are read by litellm, the client library aider uses
to issue OpenAI-compatible requests. The API base must include the `/v1` suffix.

| Flag | Purpose |
|---|---|
| `--model openai/<key>` | Selects the served model. The `openai/` prefix selects the OpenAI-compatible protocol in litellm. |
| `--weak-model openai/<key>` | Routes aider's auxiliary requests (commit messages, history summarization) to the same local model rather than to `api.openai.com`. |
| `--model-metadata-file <path>` | Declares the model's context window (32768 tokens) and zero token cost to litellm. Without it, litellm cannot size prompts and prints `Unknown context window size`. |
| `--edit-format diff` | Requests unified-diff edits instead of full-file rewrites. Set `EDIT_FORMAT=whole` on `up` to switch to whole-file edits if diffs are rejected for a given file. |
| `--analytics-disable` | Permanently disables aider's own usage telemetry. This is a client-side concern separate from the model traffic, which never leaves RCC; the [data residency note on the home page](../index.md#data-residency) covers the distinction. |

The metadata file splits the 32768-token window as 28000 input tokens and 4096
output tokens, so prompt plus generated tokens cannot exceed the window. It carries
entries for both `qwen2.5_coder_32B` and `qwen2.5_72B`; if you started the session
with `MODEL=qwen2.5_72B`, substitute `openai/qwen2.5_72B` in both `--model` and
`--weak-model`.

Verification: aider starts its interactive prompt without printing
`Unknown context window size`. At the prompt, type `/tokens`; it reports current
context token usage against the window.

## In-session commands

| Command | Effect |
|---|---|
| `/add <path>` | Add a file to the editable context. |
| `/drop <path>` | Remove a file from the context. |
| `/ask <question>` | Ask a question without editing files. |
| `/tokens` | Report current context token usage. |
| `/clear` | Clear conversation history; added files remain. |
| `/run <cmd>` | Run a shell command and optionally add its output to the context. |

aider maintains a repository map, so the model has structural awareness of files you
have not explicitly added; add files with `/add` only when you intend to edit them.

## Non-interactive use

To make a single edit and exit, for example from a batch script, add `--yes-always`
(answers aider's confirmation prompts), `--no-auto-commit` (leaves the change
uncommitted for review), and `--message`:

```bash
OPENAI_API_BASE=http://localhost:<GW_PORT>/v1 \
OPENAI_API_KEY=<SESSION_KEY> \
/project/rcc/mehta5/aider-env/bin/aider \
  --model openai/qwen2.5_coder_32B \
  --weak-model openai/qwen2.5_coder_32B \
  --model-metadata-file /project/rcc/mehta5/vllm/ai-session/aider_model_metadata.json \
  --edit-format diff --analytics-disable \
  --yes-always --no-auto-commit \
  --message "add type hints to the public functions in utils.py"
```

Verification: `git diff` in the repository shows the edit; nothing was committed.

Standard input can be piped in, for example to analyze a job log:

```bash
cat slurm-${SLURM_JOB_ID}.out | \
OPENAI_API_BASE=http://localhost:<GW_PORT>/v1 OPENAI_API_KEY=<SESSION_KEY> \
/project/rcc/mehta5/aider-env/bin/aider --model openai/qwen2.5_coder_32B \
  --weak-model openai/qwen2.5_coder_32B \
  --model-metadata-file /project/rcc/mehta5/vllm/ai-session/aider_model_metadata.json \
  --analytics-disable \
  --message "explain the traceback in this log and propose a fix"
```

- Replace `<GW_PORT>` with the gateway port printed by `up`, in both examples.

## Errors

aider-specific error messages — `Unknown context window size`, the prompt reported
as too long, and rejected diff edits — are listed with resolutions on
[Troubleshooting](../troubleshooting.md).
