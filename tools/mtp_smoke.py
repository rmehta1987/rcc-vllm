#!/usr/bin/env python
"""MTP + reasoning_effort smoke client (one arm of the A/B).

Three probes against an already-serving vLLM endpoint, then a /metrics scrape:
  A) xhigh validation -- does reasoning_effort=xhigh serve, and what does it cost?
  B) bounded throughput -- 3 short greedy requests at reasoning_effort=low, tok/s
  C) tool-call probe -- does the model actually EMIT tool calls? (Qwen2.5-Coder never did;
     that silently broke opencode, so this is the ops question that matters to us.)
Writes one JSON per arm. Scoring/comparison happens on the CPU side afterwards.
"""
import json, sys, time, urllib.error, urllib.request

TIMEOUT = 1200


def _post(url, payload, timeout=TIMEOUT):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def _get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read().decode()


def wait_ready(root, budget_s=1800):
    t0 = time.time()
    while time.time() - t0 < budget_s:
        try:
            if _get(f"{root}/health", timeout=10)[0] == 200:
                return True, time.time() - t0
        except Exception:
            pass
        time.sleep(5)
    return False, time.time() - t0


PROMPT = ("Write a Python function `merge_intervals(intervals)` that merges overlapping "
          "closed intervals given as a list of [start, end] pairs and returns the merged "
          "list sorted by start. Return only the code in a single fenced block.")

THROUGHPUT_PROMPTS = [
    "Write a Python function to reverse a linked list iteratively. Code only.",
    "Write a Python function that returns the n-th Fibonacci number using memoization. Code only.",
    "Write a Python function that checks whether a string is a valid palindrome ignoring case and non-alphanumerics. Code only.",
]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file from disk.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute path to the file"}},
            "required": ["path"],
        },
    },
}]


def scrape_spec_metrics(root):
    """Pull the spec-decode counters out of the Prometheus text exposition."""
    out = {}
    try:
        _, body = _get(f"{root}/metrics", timeout=30)
    except Exception as e:
        return {"error": f"metrics scrape failed: {e}"}
    for line in body.splitlines():
        if line.startswith("#") or "spec_decode" not in line:
            continue
        try:
            name, val = line.rsplit(" ", 1)
            out[name.strip()] = float(val)
        except ValueError:
            continue
    acc = sum(v for k, v in out.items() if "num_accepted_tokens_total" in k)
    draft = sum(v for k, v in out.items() if "num_draft_tokens_total" in k)
    drafts = sum(v for k, v in out.items() if "num_drafts" in k)
    out["_derived_acceptance_rate"] = round(acc / draft, 4) if draft else None
    out["_derived_accepted_total"] = acc
    out["_derived_draft_total"] = draft
    out["_derived_num_drafts"] = drafts
    return out


def one_request(base, served, prompt, effort, max_tokens, tools=None):
    payload = {
        "model": served,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "top_p": 1.0, "n": 1, "max_tokens": max_tokens,
        "reasoning_effort": effort,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    t0 = time.time()
    try:
        status, body = _post(f"{base}/chat/completions", payload)
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode()[:400]}",
                "effort": effort, "elapsed_s": round(time.time() - t0, 2)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "effort": effort, "elapsed_s": round(time.time() - t0, 2)}
    el = time.time() - t0
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    usage = body.get("usage") or {}
    comp = usage.get("completion_tokens")
    reasoning = msg.get("reasoning") or msg.get("reasoning_content")
    return {
        "ok": True, "http_status": status, "effort": effort,
        "elapsed_s": round(el, 2),
        "finish_reason": ch.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": comp,
        "completion_tok_per_s": round(comp / el, 2) if comp and el else None,
        "has_content": bool(msg.get("content")),
        "content_chars": len(msg.get("content") or ""),
        "content_head": (msg.get("content") or "")[:300],
        "reasoning_chars": len(reasoning or ""),
        "has_reasoning": bool(reasoning),
        "tool_calls": msg.get("tool_calls"),
        "n_tool_calls": len(msg.get("tool_calls") or []),
    }


def main():
    base, served, out_path, arm = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    root = base.rsplit("/v1", 1)[0]
    res = {"arm": arm, "served_name": served, "base_url": base}

    ready, load_s = wait_ready(root)
    res["ready"] = ready
    res["time_to_ready_s"] = round(load_s, 1)
    if not ready:
        res["error"] = "server never became ready"
        json.dump(res, open(out_path, "w"), indent=2)
        print(f"ARM {arm}: NOT READY after {load_s:.0f}s")
        return 1

    # Baseline metrics BEFORE any traffic, so the arm's counters are attributable.
    res["metrics_before"] = scrape_spec_metrics(root)

    print(f"ARM {arm}: probe A -- reasoning_effort=xhigh")
    res["probe_a_xhigh"] = one_request(base, served, PROMPT, "xhigh", 16384)
    print(f"  -> {json.dumps({k: v for k, v in res['probe_a_xhigh'].items() if k not in ('content_head','tool_calls')})}")

    print(f"ARM {arm}: probe B -- bounded throughput at reasoning_effort=low")
    res["probe_b_throughput"] = [one_request(base, served, p, "low", 2048)
                                 for p in THROUGHPUT_PROMPTS]
    oks = [r for r in res["probe_b_throughput"] if r.get("ok") and r.get("completion_tok_per_s")]
    if oks:
        tot_tok = sum(r["completion_tokens"] for r in oks)
        tot_s = sum(r["elapsed_s"] for r in oks)
        res["throughput_agg_tok_per_s"] = round(tot_tok / tot_s, 2) if tot_s else None
        res["throughput_n_ok"] = len(oks)

    print(f"ARM {arm}: probe C -- tool-call emission")
    res["probe_c_toolcall"] = one_request(
        base, served, "Read the file /etc/hostname and tell me what it contains.",
        "low", 2048, tools=TOOLS)

    res["metrics_after"] = scrape_spec_metrics(root)
    json.dump(res, open(out_path, "w"), indent=2)
    print(f"ARM {arm}: wrote {out_path}")
    print(f"ARM {arm}: agg_tok_per_s={res.get('throughput_agg_tok_per_s')} "
          f"acceptance={res['metrics_after'].get('_derived_acceptance_rate')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
