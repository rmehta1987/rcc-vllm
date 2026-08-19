#!/usr/bin/env python
"""Tool-call emission probe for one (model, tool-call-parser) pair.

Answers the question the MTP smoke could not: does Qwen3.8-27B actually EMIT tool
calls, and does the configured vLLM parser turn them into OpenAI `tool_calls`?

Two failure modes look identical to a client, so we capture BOTH the parsed
`tool_calls` and the raw leftover `content`:
  - model never emits the tokens          -> tool_calls empty AND no <tool_call> in content
  - model emits, parser does not match it -> tool_calls empty BUT <tool_call> visible in content
Qwen2.5-Coder-32B is the first case (vLLM #29192); that is why the AGENTS.md prompt
workaround exists. If a parser here works, that workaround can be dropped for this model.
"""
import json, re, sys, time, urllib.error, urllib.request

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read the contents of a file from disk.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "Absolute path"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List files in a directory.",
        "parameters": {"type": "object",
                       "properties": {"directory": {"type": "string", "description": "Absolute dir path"}},
                       "required": ["directory"]}}},
]

CASES = [
    ("single_tool", "Read the file /etc/hostname and tell me what it contains.", True),
    ("multi_tool", "List the Python files in /tmp, then read the first one you find.", True),
    ("no_tool_needed", "What is 2+2? Answer with just the number.", False),
]

TOOLCALL_MARKERS = ("<tool_call>", "</tool_call>", '"name":', "function")


def _post(url, payload, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def _get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read().decode()


def wait_ready(root, budget_s=1800):
    t0 = time.time()
    while time.time() - t0 < budget_s:
        try:
            if _get(f"{root}/health")[0] == 200:
                return True, time.time() - t0
        except Exception:
            pass
        time.sleep(5)
    return False, time.time() - t0


def run_case(base, served, prompt, expect_call):
    payload = {"model": served,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.0, "max_tokens": 2048,
               "reasoning_effort": "low",
               "tools": TOOLS, "tool_choice": "auto"}
    t0 = time.time()
    try:
        status, body = _post(f"{base}/chat/completions", payload)
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode()[:500]}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    calls = msg.get("tool_calls") or []
    parsed = [{"name": (c.get("function") or {}).get("name"),
               "arguments": (c.get("function") or {}).get("arguments")} for c in calls]
    return {
        "ok": True, "http_status": status, "elapsed_s": round(time.time() - t0, 2),
        "finish_reason": ch.get("finish_reason"),
        "expected_tool_call": expect_call,
        "n_tool_calls": len(calls),
        "parsed_calls": parsed,
        "content_chars": len(content),
        "content_head": content[:400],
        "raw_shows_toolcall_markup": any(m in content for m in TOOLCALL_MARKERS),
        "raw_has_tool_call_tag": "<tool_call>" in content,
        "reasoning_chars": len(msg.get("reasoning") or msg.get("reasoning_content") or ""),
    }


def main():
    base, served, out_path, parser_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    root = base.rsplit("/v1", 1)[0]
    res = {"parser": parser_name, "served_name": served}
    ready, load_s = wait_ready(root)
    res["ready"], res["time_to_ready_s"] = ready, round(load_s, 1)
    if not ready:
        res["error"] = "server never became ready"
        json.dump(res, open(out_path, "w"), indent=2)
        print(f"PARSER {parser_name}: NOT READY")
        return 1

    res["cases"] = {}
    for name, prompt, expect in CASES:
        r = run_case(base, served, prompt, expect)
        res["cases"][name] = r
        print(f"  [{parser_name}/{name}] ok={r.get('ok')} n_calls={r.get('n_tool_calls')} "
              f"raw_tag={r.get('raw_has_tool_call_tag')} err={str(r.get('error'))[:150]}")

    want = [c for n, c in res["cases"].items() if c.get("expected_tool_call")]
    res["verdict"] = (
        "PARSER_WORKS" if all(c.get("ok") and c.get("n_tool_calls", 0) > 0 for c in want)
        else "MODEL_EMITS_PARSER_MISMATCH" if any(c.get("raw_has_tool_call_tag") for c in want)
        else "NO_TOOL_CALLS_AT_ALL" if all(c.get("ok") for c in want)
        else "ERROR")
    json.dump(res, open(out_path, "w"), indent=2)
    print(f"PARSER {parser_name}: VERDICT {res['verdict']} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
