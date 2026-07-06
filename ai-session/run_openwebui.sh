#!/bin/bash
# Launch Open WebUI (the ai-session general chat client) against the gateway.
#
# Open WebUI lives in its OWN venv (NOT vllm-probe):
#   /project/rcc/mehta5/openwebui-env  (python 3.11, `pip install open-webui`)
# It talks to the stable gateway URL, so it never needs reconfiguring between
# sessions -- point it once at the gateway and it follows whatever backend the
# current `ai_session start` published.
#
# Usage:  bash ai-session/run_openwebui.sh [PORT]   (default 3000, binds 127.0.0.1)
# Then from your laptop:  ssh -N -L 3000:localhost:3000 mehta5@<this-login-node>
#                         browse http://localhost:3000
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../ai-session (== gateway.py's default state dir)
OWUI_ENV=/project/rcc/mehta5/openwebui-env
GW_PORT=${GW_PORT:-8421}
PORT=${1:-3000}

# DATA_DIR holds this user's chat history/uploads/sqlite -> it is PRIVATE, so
# default it under the user's HOME (mode 700, owner-only). Storing it in the
# group-readable project tree let any cluster user read another person's chats;
# HOME is owner-only, so keep the chat DB there.
# HF_HOME is a content-addressed model cache -> safe to share in project space.
export DATA_DIR=${DATA_DIR:-$HOME/.ai-session/openwebui-data}
export HF_HOME=${HF_HOME:-/project/rcc/mehta5/hf_cache}
mkdir -p "$HOME/.ai-session" "$DATA_DIR" "$HF_HOME"
# Lock the private home subtree to owner-only, so it stays unreadable to other
# users even if HOME were ever loosened.
chmod 700 "$HOME/.ai-session" "$DATA_DIR"

# Point the single OpenAI backend at the gateway; disable the Ollama backend.
# The gateway now requires a per-session access key. Prefer the one already in the
# environment (run_browser_demo.sh exports AISESSION_GATEWAY_KEY); else read the
# key file the wrapper wrote under the state dir; else fall back to the old literal
# so keyless dev (a gateway started with no key) still works.
KEYFILE="${AISESSION_STATE_DIR:-$HERE}/logs/gateway/session_key"
export ENABLE_OPENAI_API=True
export OPENAI_API_BASE_URL="http://localhost:${GW_PORT}/v1"
export OPENAI_API_KEY=${AISESSION_GATEWAY_KEY:-$( [ -f "$KEYFILE" ] && cat "$KEYFILE" )}
export OPENAI_API_KEY=${OPENAI_API_KEY:-ai-session}
export ENABLE_OLLAMA_API=False

# Demo posture: no login wall, no phone-home. (For multi-user, drop WEBUI_AUTH.)
export WEBUI_AUTH=False
export WEBUI_SECRET_KEY=${WEBUI_SECRET_KEY:-ai-session-demo-key}
export ANONYMIZED_TELEMETRY=False
export DO_NOT_TRACK=true
export SCARF_NO_ANALYTICS=true
export OFFLINE_MODE=${OFFLINE_MODE:-False}

echo "[openwebui] DATA_DIR=$DATA_DIR (private, home dir, mode 700)  backend=$OPENAI_API_BASE_URL  port=$PORT (127.0.0.1)"
exec "$OWUI_ENV/bin/open-webui" serve --host 127.0.0.1 --port "$PORT"
