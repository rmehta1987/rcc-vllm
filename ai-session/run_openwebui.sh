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
# Then from your laptop:  ssh -N -f -L 3000:localhost:3000 <you>@<this-login-node>.rcc.uchicago.edu
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
export DATA_DIR=${DATA_DIR:-$HOME/.ai-session/openwebui-data}
# HF_HOME is per-user WRITABLE: Open WebUI writes here (any embedding-model
# download, lock files, incidental cache). Keep it under the user's own state dir
# so a non-owner NEVER has to write into the shared install tree.
export HF_HOME=${HF_HOME:-${AISESSION_STATE_DIR:-$HOME/.ai-session/state}/hf_cache}
mkdir -p "$HOME/.ai-session" "$DATA_DIR" "$HF_HOME"
# Lock the private home subtree to owner-only, so it stays unreadable to other
# users even if HOME were ever loosened.
chmod 700 "$HOME/.ai-session" "$DATA_DIR"

# Embedding model (RAG) shared cache. On first run Open WebUI loads
# all-MiniLM-L6-v2; letting every user re-download it into their empty HF_HOME is
# slow and can overrun the UI bind-wait. If a world-READABLE shared cache holds it,
# read from there (reads only -- no write access to the shared cache required) and
# flip offline so the resolver takes the cache hit with no lock and no network.
# Reads resolve via HF_HUB_CACHE (shared); writes still go to HF_HOME (per-user),
# so this works for ANY user, rcc-staff or not. If the shared cache is absent or
# unreadable, fall through to a per-user online download. An explicit OFFLINE_MODE
# already in the environment is honored and never overridden. `find -readable`
# tests the CURRENT user's real access, so a non-owner who can't read it falls back.
SHARED_HF_CACHE=${AISESSION_SHARED_HF_CACHE:-/project/rcc/mehta5/hf_cache/hub}
EMBED_SNAP="$SHARED_HF_CACHE/models--sentence-transformers--all-MiniLM-L6-v2"
EMBED_OK=$(find "$EMBED_SNAP/snapshots" -maxdepth 2 -name model.safetensors -readable 2>/dev/null | head -1 || true)
if [ -z "${OFFLINE_MODE:-}" ] && [ -n "$EMBED_OK" ]; then
  export HF_HUB_CACHE="$SHARED_HF_CACHE"   # embedding model resolves here (shared, read-only)
  export HF_HUB_OFFLINE=1                  # cache hit: no lock, no download, no write to the shared cache
  export OFFLINE_MODE=True
  echo "[openwebui] embedding model: shared read-only cache $SHARED_HF_CACHE (offline, no download)"
else
  export OFFLINE_MODE=${OFFLINE_MODE:-False}   # per-user online download into HF_HOME on first run
  echo "[openwebui] embedding model: per-user cache $HF_HOME (online download on first run)"
fi

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
# OFFLINE_MODE is set above (shared-cache branch), not here.

# --- Optional web + reference tools (OPT-IN; default OFF preserves "nothing leaves RCC") ---
# Enable with AISESSION_TOOLS=1. Adds three capabilities the user turns on per chat:
#   1. Web search (RAG; UI-orchestrated, so it works with ANY served model, no tool-calling)
#   2. URL fetch (paste a link; Open WebUI loads and injects it)
#   3. Academic paper search -- arXiv/bioRxiv/medRxiv/PubMed/Semantic Scholar via the public
#      paper-search-mcp server, exposed to Open WebUI as an OpenAPI tool server behind mcpo.
# All three reach EXTERNAL services from the login node (only the GPU node is air-gapped), so
# the query terms leave RCC for those requests -- documented as a deliberate, opt-in tradeoff.
if [ "${AISESSION_TOOLS:-0}" = "1" ]; then
  export ENABLE_WEB_SEARCH=True
  export WEB_SEARCH_ENGINE=${WEB_SEARCH_ENGINE:-duckduckgo}   # keyless; override for searxng/tavily/brave/...
  export ENABLE_WEB_LOADER=True

  TOOLS_ENV=/project/rcc/mehta5/tools-env
  # Writable run dir for mcpo's pidfile+log. Standalone default matches the other
  # state paths above (the install dir is shared read-only, so never fall back there).
  RUN_DIR="${AISESSION_STATE_DIR:-$HOME/.ai-session/state}/run"
  mkdir -p "$RUN_DIR"
  MCPO_PORT=${MCPO_PORT:-$((PORT + 500))}
  if [ ! -x "$TOOLS_ENV/bin/mcpo" ]; then
    echo "[openwebui] AISESSION_TOOLS=1 but $TOOLS_ENV/bin/mcpo missing; web-search + URL-fetch on, paper-search OFF" >&2
  elif ss -ltn 2>/dev/null | grep -q ":$MCPO_PORT "; then
    # Never point the UI at a listener we did not start -- on a shared login node
    # it could be another user's process. Skip paper-search rather than guess.
    echo "[openwebui] :$MCPO_PORT is already in use (another user?); paper-search OFF." >&2
    echo "[openwebui] Pick a free port and restart: MCPO_PORT=<port> ... (web-search + URL-fetch stay on)" >&2
  else
    "$TOOLS_ENV/bin/mcpo" --host 127.0.0.1 --port "$MCPO_PORT" -- "$TOOLS_ENV/bin/paper-search-mcp" \
      > "$RUN_DIR/mcpo.log" 2>&1 &
    MCPO_PID=$!
    echo "$MCPO_PID" > "$RUN_DIR/mcpo.pid"   # so `ai-session stop` / run_browser_demo.sh down can reap it
    # Liveness check: only hand the UI a tool server that actually came up.
    i=0
    while ! ss -ltn 2>/dev/null | grep -q ":$MCPO_PORT "; do
      if ! kill -0 "$MCPO_PID" 2>/dev/null || [ "$i" -ge 30 ]; then break; fi
      sleep 1; i=$((i+1))
    done
    if ss -ltn 2>/dev/null | grep -q ":$MCPO_PORT "; then
      export TOOL_SERVER_CONNECTIONS='[{"url":"http://127.0.0.1:'"$MCPO_PORT"'","path":"openapi.json","auth_type":"none","config":{"enable":true},"info":{"id":"paper-search","name":"Academic paper search (arXiv/bioRxiv/PubMed/Semantic Scholar)"}}]'
      echo "[openwebui] tools ON: web-search=$WEB_SEARCH_ENGINE, URL-fetch, paper-search (mcpo 127.0.0.1:$MCPO_PORT, pid $MCPO_PID)" >&2
    else
      kill "$MCPO_PID" 2>/dev/null || true
      rm -f "$RUN_DIR/mcpo.pid"
      echo "[openwebui] mcpo never bound :$MCPO_PORT (see $RUN_DIR/mcpo.log); paper-search OFF." >&2
    fi
  fi
  echo "[openwebui] NOTE: web/reference tools reach EXTERNAL services -- query terms leave RCC for those requests." >&2
fi

echo "[openwebui] DATA_DIR=$DATA_DIR (private, home dir, mode 700)  backend=$OPENAI_API_BASE_URL  port=$PORT (127.0.0.1)"
exec "$OWUI_ENV/bin/open-webui" serve --host 127.0.0.1 --port "$PORT"
