#!/usr/bin/env python3
"""ai-session gateway: a stable URL in front of the ephemeral vLLM backend.

The vLLM endpoint changes node/port every session. Clients (Open WebUI, aider,
opencode, curl) hate that. This is a thin async reverse proxy that lives at ONE
fixed URL and forwards to whatever backend the *current* session is using.

  client  ->  http://<gateway-host>:8080/v1   ->  http://<compute-node>:<port>/v1
                       (fixed)                            (changes per session)

`ai_session start` writes the current backend to logs/gateway/upstream.json;
`ai_session end` clears it. The gateway re-reads that file (cheap, mtime-cached)
so it always points at the live session with no restart.

It also CAPTURES per-request token usage (the seam where Phase-2 per-request
metering/auth will live): every chat/completions response's `usage` is appended
to logs/gateway/usage-YYYYMMDD.jsonl, which `ai_session end` consumes as the
authoritative billing source -- so users don't have to log usage themselves.

Run (on a login node, in the vllm-probe env):
    python ai-session/gateway.py --host 127.0.0.1 --port 8080
Optional auth: set AISESSION_GATEWAY_KEY=... and clients send it as the API key.
"""

import argparse
import json
import os
import time

# NOTE: no `from __future__ import annotations` here -- FastAPI resolves the
# `request: Request` annotation on the proxy route, and with lazy (local)
# fastapi imports that name isn't in module globals, so string annotations would
# fail to resolve (422). Real annotations resolve via the closure at def-time.
#
# httpx / fastapi are imported lazily inside build_app() and the proxy
# helpers, so that importing this module for just the upstream-file helpers
# (write_upstream/clear_upstream/_Upstream — used by ai_session.py) needs no web
# stack. Only running the gateway server pulls in fastapi/httpx/uvicorn.

_HERE = os.path.dirname(os.path.abspath(__file__))
# Writable state dir: honor AISESSION_STATE_DIR (per-user, multi-tenant) so the
# gateway and `ai_session` agree on ONE upstream.json / usage log per user when
# several rcc-staff share this install. Unset -> next to the code (original).
# Keep this in lockstep with ai_session.py's _STATE.
_STATE = os.environ.get("AISESSION_STATE_DIR") or _HERE
GATEWAY_DIR = os.path.join(_STATE, "logs", "gateway")
UPSTREAM_FILE = os.path.join(GATEWAY_DIR, "upstream.json")

# Paths we proxy verbatim (everything under these prefixes).
_PROXY_PREFIXES = ("/v1", "/metrics", "/health", "/version", "/ping", "/tokenize", "/detokenize", "/pooling")
# Hop-by-hop / length headers we must not forward.
_DROP_REQ_HEADERS = {"host", "content-length", "connection", "keep-alive", "transfer-encoding"}
_DROP_RESP_HEADERS = {"content-length", "transfer-encoding", "content-encoding", "connection", "keep-alive"}


def _env_float(name: str, default: float) -> float:
    """Read a float env var; fall back to `default` on unset / blank / garbage."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class _TokenBucket:
    """A per-client token bucket: `rate` tokens/sec, burst up to `capacity`.

    Refill is lazy (computed from elapsed wall time on each `allow()`), so there
    is no background timer. Updated only from the single-threaded asyncio event
    loop, so the read-modify-write needs no lock.
    """

    __slots__ = ("rate", "capacity", "tokens", "last")

    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


# --------------------------------------------------------------------------- #
# upstream resolution (mtime-cached file read)
# --------------------------------------------------------------------------- #
class _Upstream:
    def __init__(self):
        self._mtime = None
        self._val = None

    def get(self):
        try:
            st = os.stat(UPSTREAM_FILE)
        except FileNotFoundError:
            self._mtime, self._val = None, None
            return None
        if st.st_mtime != self._mtime:
            self._mtime = st.st_mtime
            try:
                with open(UPSTREAM_FILE) as f:
                    data = json.load(f)
                self._val = data if data.get("active") and data.get("base_url") else None
            except (json.JSONDecodeError, OSError):
                self._val = None
        return self._val


def _atomic_write_private(path: str, data: dict) -> None:
    """Atomically write `data` as JSON to `path`, OWNER-ONLY (mode 0600).

    upstream.json carries the backend node:port AND the per-session backend API
    key. The per-user state dir is group rcc-staff and group-writable under the
    deploy umask (002), so with default (0664) perms a co-tenant could either READ
    the backend key or OVERWRITE this file to silently repoint the live gateway
    (a MITM on every lab member funneling through it). 0600 closes both: only the
    owner -- the gateway and ai_session, which run as that same user -- can read
    or replace it. O_CREAT's mode is masked by umask, and a stale tmp from a
    crashed run keeps its old perms, so chmod explicitly to pin 0600.
    """
    os.makedirs(GATEWAY_DIR, exist_ok=True)
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def write_upstream(base_url: str, **meta) -> None:
    """Helper used by ai_session start: publish the current backend (owner-only)."""
    data = {"active": True, "base_url": base_url.rstrip("/"), "updated": time.time(), **meta}
    _atomic_write_private(UPSTREAM_FILE, data)


def clear_upstream() -> None:
    """Helper used by ai_session end: mark no active backend."""
    _atomic_write_private(UPSTREAM_FILE, {"active": False, "updated": time.time()})


def usage_log_path(epoch: float = None) -> str:
    day = time.strftime("%Y%m%d", time.localtime(epoch))
    return os.path.join(GATEWAY_DIR, f"usage-{day}.jsonl")


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #
def build_app(require_key: str = None, client=None):
    import httpx
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    upstream = _Upstream()
    state = {"client": client}  # may be injected (tests) or created on startup

    def _client() -> "httpx.AsyncClient":
        if state["client"] is None:  # lazy fallback (e.g. when lifespan isn't run)
            state["client"] = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0))
        return state["client"]

    # --- item 16: per-client rate limit + max request-body size ------------- #
    # Both are env-configurable with defaults that do not affect ordinary
    # interactive chat/coding traffic; they exist to blunt a single runaway
    # client on the shared login-node gateway. AISESSION_RATE_RPS<=0 disables the
    # limiter; AISESSION_MAX_BODY_MB<=0 disables the body-size cap.
    rate_rps = _env_float("AISESSION_RATE_RPS", 30.0)
    burst = max(2.0 * rate_rps, 1.0)               # allow a couple seconds of headroom
    max_body_mb = _env_float("AISESSION_MAX_BODY_MB", 16.0)
    max_body_bytes = int(max_body_mb * 1024 * 1024) if max_body_mb > 0 else 0
    buckets = {}                                   # client-key -> _TokenBucket
    # A single GLOBAL token bucket, checked ALONGSIDE the per-client bucket, as a
    # fallback ceiling. It bounds aggregate load even when per-client keying is
    # weak or defeated (e.g. a keyless client rotating its Authorization header,
    # or -- in keyed mode -- every lab member sharing the one gateway key). Sized
    # well above the per-client rate so ordinary single-client traffic is never
    # limited by it.
    global_rate = rate_rps * 4.0
    global_bucket = (_TokenBucket(global_rate, max(2.0 * global_rate, 1.0))
                     if rate_rps > 0 else None)

    def _rate_ok(client_key: str) -> bool:
        if rate_rps <= 0:
            return True
        b = buckets.get(client_key)
        if b is None:
            if len(buckets) > 4096:                # bound memory: drop idle (refilled) buckets
                for k in [k for k, v in buckets.items() if v.tokens >= v.capacity]:
                    buckets.pop(k, None)
            b = buckets[client_key] = _TokenBucket(rate_rps, burst)
        return b.allow()

    def _global_ok() -> bool:
        return global_bucket.allow() if global_bucket is not None else True

    def _too_large():
        from fastapi.responses import JSONResponse as _JR
        return _JR(
            {"error": {"message": f"request body exceeds AISESSION_MAX_BODY_MB={max_body_mb:g}",
                       "type": "invalid_request_error", "code": "payload_too_large"}},
            status_code=413,
        )

    @asynccontextmanager
    async def lifespan(app):
        _client()                       # startup: ensure the shared AsyncClient exists
        try:
            yield
        finally:
            if state["client"]:         # shutdown: close it (skip an injected test client)
                await state["client"].aclose()

    app = FastAPI(title="ai-session gateway", lifespan=lifespan)

    @app.get("/__gateway/health")
    async def _gw_health():
        # This endpoint is reachable WITHOUT the API key (the wrappers poll it for
        # liveness). Report only liveness + whether a backend is published -- never
        # the backend node:port / jobid / backend_key. Leaking the address here
        # would let a co-tenant who can only reach the gateway (127.0.0.1 on the
        # shared login node) discover the backend and hit its /v1 directly.
        up = upstream.get()
        return {"gateway": "ok", "backend_active": up is not None}

    status_cache = {"t": 0.0, "val": None}         # last /status result, TTL-cached
    STATUS_TTL = 1.5                               # seconds

    @app.get("/status")
    async def _status():
        # Structured liveness for clients: instead of a raw 502/503 they get a
        # coarse live/loading/gone answer. Keyless, like /__gateway/health (the
        # wrappers and clients poll it without the API key). It reports only
        # booleans + a state string + the model key -- NEVER the backend
        # node:port. A co-tenant who can reach only the gateway (127.0.0.1 on the
        # shared login node) must not be able to discover the backend address.
        #   no_backend : no session published (gone)
        #   loading    : published but /health not yet answering (model loading)
        #   ready      : /health answers 2xx/3xx (serving)
        #
        # The result is cached for STATUS_TTL so an unauthenticated caller cannot
        # spam /status to fan out one backend /health probe (up to a 5s hung
        # connection each) per request: at most one probe per TTL window.
        now = time.monotonic()
        cached = status_cache["val"]
        if cached is not None and (now - status_cache["t"]) < STATUS_TTL:
            return cached
        up = upstream.get()
        if not up:
            result = {"gateway": "ok", "backend_active": False, "backend_state": "no_backend"}
            status_cache["t"], status_cache["val"] = now, result
            return result
        backend_state = "loading"
        try:
            r = await _client().get(up["base_url"] + "/health", timeout=httpx.Timeout(5.0))
            if r.status_code < 400:
                backend_state = "ready"
        except httpx.HTTPError:
            backend_state = "loading"
        result = {"gateway": "ok", "backend_active": True,
                  "backend_state": backend_state, "model_key": up.get("model_key")}
        status_cache["t"], status_cache["val"] = now, result
        return result

    @app.api_route("/{full_path:path}",
                   methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    async def _proxy(full_path: str, request: Request):
        path = "/" + full_path
        if not any(path == p or path.startswith(p + "/") or path == p for p in _PROXY_PREFIXES):
            if path == "/":
                return JSONResponse({"service": "ai-session gateway",
                                     "hint": "point an OpenAI client at this URL + /v1"})
            return JSONResponse({"error": f"path {path} not proxied"}, status_code=404)

        # Client identity for auth AND rate-limiting: the Bearer key if present,
        # else the peer host. Extracted before the auth branch so a keyless
        # gateway still rate-limits per client.
        sent = (request.headers.get("authorization", "") or "").removeprefix("Bearer ").strip()

        # optional auth
        if require_key and sent != require_key:
            return JSONResponse({"error": "invalid or missing API key"}, status_code=401)

        # item 16: per-client token-bucket rate limit (429 with an OpenAI-style
        # body). Checked before the upstream lookup so it applies even while a
        # backend is (re)loading.
        #
        # Rate-limit keying: only trust the Bearer key when a key is REQUIRED (in
        # keyed mode `sent == require_key` has just been validated). In keyless
        # (dev) mode the Authorization header is attacker-controlled -- rotating
        # it would mint a fresh bucket per request and defeat the limiter -- so we
        # key on the peer host only and ignore the header. The global bucket then
        # backstops the shared-key case in keyed mode.
        peer_host = request.client.host if request.client else "unknown"
        client_key = (sent if require_key else "") or peer_host
        if not _rate_ok(client_key) or not _global_ok():
            return JSONResponse(
                {"error": {"message": f"rate limit exceeded ({rate_rps:g} req/s per client); retry shortly",
                           "type": "rate_limit_exceeded", "code": "rate_limit_exceeded"}},
                status_code=429, headers={"Retry-After": "1"},
            )

        # item 16: reject oversized bodies early via the Content-Length header
        # (the actual bytes are re-checked after reading, below).
        if max_body_bytes:
            clen = request.headers.get("content-length")
            if clen and clen.isdigit() and int(clen) > max_body_bytes:
                return _too_large()

        up = upstream.get()
        if not up:
            return JSONResponse(
                {"error": {"message": "no active ai-session backend; run `ai_session.py start`",
                           "type": "no_backend"}},
                status_code=503,
            )

        target = up["base_url"] + path
        if request.url.query:
            target += "?" + request.url.query
        # Read the body with a running byte counter and abort as soon as the cap
        # is exceeded, so a chunked / no-Content-Length upload can NOT be fully
        # buffered into memory before rejection (the Content-Length prefilter
        # above only stops clients that send a truthful Content-Length).
        if max_body_bytes:
            chunks, total = [], 0
            async for chunk in request.stream():
                total += len(chunk)
                if total > max_body_bytes:
                    return _too_large()
                chunks.append(chunk)
            body = b"".join(chunks)
        else:
            body = await request.body()
        fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in _DROP_REQ_HEADERS}

        # The paid vLLM backend has its OWN api key (VLLM_API_KEY, minted per
        # session by launch_ai_session.sh and published in upstream.json). Client
        # auth is enforced HERE via require_key; we swap in the backend key before
        # forwarding so that a co-tenant who scrapes the backend node:port (from
        # squeue or elsewhere) still cannot reach /v1 without going through this
        # gateway. Absent (older/keyless backends) -> forward the client's header.
        backend_key = up.get("backend_key")
        if backend_key:
            fwd_headers = {k: v for k, v in fwd_headers.items() if k.lower() != "authorization"}
            fwd_headers["Authorization"] = "Bearer " + backend_key

        # Is this a streaming generation request? If so, make sure usage is emitted.
        is_gen = path.endswith("/chat/completions") or path.endswith("/completions")
        streaming = False
        model_name = None
        if is_gen and request.method == "POST" and body:
            try:
                payload = json.loads(body)
                model_name = payload.get("model")
                if payload.get("stream"):
                    streaming = True
                    opts = payload.get("stream_options") or {}
                    opts["include_usage"] = True          # so the final SSE chunk carries usage
                    payload["stream_options"] = opts
                    body = json.dumps(payload).encode()
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        client: httpx.AsyncClient = _client()
        if streaming:
            return await _stream_proxy(client, request.method, target, body, fwd_headers,
                                       path, model_name, up)
        return await _buffered_proxy(client, request.method, target, body, fwd_headers,
                                     path, model_name, up, is_gen)

    return app


async def _buffered_proxy(client, method, target, body, headers, path, model, up, is_gen):
    import httpx
    from fastapi.responses import JSONResponse, Response
    try:
        resp = await client.request(method, target, content=body, headers=headers)
    except httpx.HTTPError as e:
        return JSONResponse({"error": {"message": f"backend unreachable: {e}", "type": "bad_gateway"}},
                            status_code=502)
    if is_gen and resp.status_code < 400:
        try:
            usage = resp.json().get("usage")
            if usage:
                _log_usage(path, model, usage, resp.status_code, stream=False, backend=up)
        except (json.JSONDecodeError, ValueError):
            pass
    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in _DROP_RESP_HEADERS}
    return Response(content=resp.content, status_code=resp.status_code,
                    headers=out_headers, media_type=resp.headers.get("content-type"))


async def _stream_proxy(client, method, target, body, headers, path, model, up):
    import httpx
    from fastapi.responses import StreamingResponse
    media_type = "text/event-stream"

    async def gen():
        captured = {"usage": None}
        buf = b""
        try:
            async with client.stream(method, target, content=body, headers=headers) as resp:
                nonlocal media_type
                media_type = resp.headers.get("content-type", media_type)
                async for chunk in resp.aiter_bytes():
                    yield chunk                      # pass through immediately (no buffering of the stream)
                    buf += chunk
                    # opportunistically scan completed SSE events for a usage object
                    while b"\n\n" in buf:
                        event, buf = buf.split(b"\n\n", 1)
                        _scan_event_for_usage(event, captured)
        except httpx.HTTPError:
            err = json.dumps({"error": {"message": "backend stream error", "type": "bad_gateway"}})
            yield f"data: {err}\n\n".encode()
            return
        finally:
            if captured["usage"]:
                _log_usage(path, model, captured["usage"], 200, stream=True, backend=up)

    return StreamingResponse(gen(), media_type=media_type)


def _scan_event_for_usage(event_bytes: bytes, captured: dict) -> None:
    for line in event_bytes.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[len(b"data:"):].strip()
        if data == b"[DONE]" or not data:
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("usage"):
            captured["usage"] = obj["usage"]          # keep the latest non-null usage


def _log_usage(path, model, usage, status, stream, backend) -> None:
    line = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "path": path,
        "model": model,
        "stream": stream,
        "status": status,
        "success": status < 400,
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        },
        "backend_jobid": (backend or {}).get("jobid"),
    }
    try:
        os.makedirs(GATEWAY_DIR, exist_ok=True)
        with open(usage_log_path(), "a") as f:
            f.write(json.dumps(line) + "\n")
    except OSError:
        pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    import uvicorn
    key = os.environ.get("AISESSION_GATEWAY_KEY") or None
    if key:
        print("[gateway] API-key auth ENABLED (clients must send AISESSION_GATEWAY_KEY as the API key)")
    else:
        print("[gateway] no API-key auth (set AISESSION_GATEWAY_KEY to require one)")
    print(f"[gateway] listening on http://{args.host}:{args.port}  ->  backend from {UPSTREAM_FILE}")
    uvicorn.run(build_app(require_key=key), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
