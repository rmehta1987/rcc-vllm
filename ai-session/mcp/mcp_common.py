"""Minimal MCP (Model Context Protocol) server over stdio, stdlib only.

The official ``mcp`` Python package is not installed in the vllm-probe env and no
installs are allowed on this cluster, so this module hand-implements the small
subset of the protocol the two servers in this directory need. It speaks
newline-delimited JSON-RPC 2.0 on stdin/stdout (the MCP stdio transport): one JSON
object per line, UTF-8, no embedded newlines.

Methods handled:
  initialize                 -> capabilities + serverInfo (echoes client protocol)
  notifications/initialized  -> acknowledged, no reply (it is a notification)
  ping                       -> {}
  tools/list                 -> the server's fixed tool set
  tools/call                 -> dispatch to a Python handler
  shutdown / exit / notifications/cancelled -> tolerated, no error

A tool is a :class:`Tool` whose handler takes the arguments dict and returns a
string (rendered as MCP text content). Handlers signal failures by raising:
  * :class:`InvalidParams`  -> JSON-RPC error -32602 (argument rejected, e.g. a
    value that fails a regex/whitelist check -- the request is refused, not run);
  * :class:`ToolError`      -> a tools/call result with ``isError: true`` (the
    tool ran but could not answer, e.g. a job you do not own).

There is deliberately no generic "run this command" tool and no shell: every
server here exposes a fixed, typed set of read-only tools.
"""

import json
import sys

PROTOCOL_VERSION = "2024-11-05"


class ToolError(Exception):
    """Tool executed but could not answer; reported as isError content."""


class InvalidParams(Exception):
    """Argument failed validation; reported as JSON-RPC -32602 (rejected)."""


class Tool:
    def __init__(self, name, description, input_schema, handler):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler

    def spec(self):
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


def _write(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _result(msg_id, result):
    _write({"jsonrpc": "2.0", "id": msg_id, "result": result})


def _error(msg_id, code, message):
    _write({"jsonrpc": "2.0", "id": msg_id,
            "error": {"code": code, "message": message}})


def _text(text, is_error=False):
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def run_server(server_name, server_version, tools):
    """Serve ``tools`` (a list of :class:`Tool`) on stdin/stdout until EOF."""
    by_name = {t.name: t for t in tools}

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _error(None, -32700, "Parse error")
            continue
        if not isinstance(msg, dict):
            _error(None, -32600, "Invalid Request")
            continue

        method = msg.get("method")
        msg_id = msg.get("id")
        is_notification = "id" not in msg

        if method == "initialize":
            client_pv = None
            params = msg.get("params") or {}
            if isinstance(params, dict):
                client_pv = params.get("protocolVersion")
            _result(msg_id, {
                "protocolVersion": client_pv or PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": server_name, "version": server_version},
            })
        elif method == "notifications/initialized":
            pass  # notification: acknowledged, no reply
        elif method == "ping":
            if not is_notification:
                _result(msg_id, {})
        elif method == "tools/list":
            _result(msg_id, {"tools": [t.spec() for t in tools]})
        elif method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name") if isinstance(params, dict) else None
            args = params.get("arguments") if isinstance(params, dict) else None
            if not isinstance(args, dict):
                args = {}
            tool = by_name.get(name)
            if tool is None:
                _error(msg_id, -32602, "Unknown tool: %r" % (name,))
                continue
            try:
                text = tool.handler(args)
                _result(msg_id, _text(text, is_error=False))
            except InvalidParams as exc:
                # Argument rejected by validation -> hard JSON-RPC error.
                _error(msg_id, -32602, "Invalid arguments: %s" % (exc,))
            except ToolError as exc:
                _result(msg_id, _text("Error: %s" % (exc,), is_error=True))
            except Exception as exc:  # never crash the server on one bad call
                _result(msg_id, _text("Internal error: %s" % (exc,),
                                      is_error=True))
        elif method in ("shutdown",):
            if not is_notification:
                _result(msg_id, {})
        elif method in ("exit", "notifications/cancelled"):
            pass
        else:
            if not is_notification:
                _error(msg_id, -32601, "Method not found: %r" % (method,))
