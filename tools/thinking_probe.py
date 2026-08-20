#!/usr/bin/env python
"""Thinking on/off probe for a served model.

Answers three things the frozen benchmark cannot, because it pins thinking OFF:
  1. Does the reasoning parser actually SEPARATE the chain of thought? A working parser puts
     the CoT in `reasoning` and the answer in `content`. A broken one leaves the CoT in
     `content`, which silently corrupts any downstream scoring (this is exactly why
     serve_cu129.sbatch sets a per-model --reasoning-parser).
  2. What does thinking COST? Under GPU-time billing, reasoning tokens are money. We measure
     completion tokens and wall time both ways on identical prompts.
  3. Does it HELP? Same prompts, greedy, so any difference in the answer is attributable.

Thinking is toggled per model family: Gemma 4 uses chat_template_kwargs.enable_thinking,
Qwen3.5-family uses the top-level reasoning_effort field. Pass MODE=gemma or MODE=qwen.
"""
import json, sys, time, urllib.error, urllib.request

PROMPTS = [
    ("easy", "Write a Python function that returns the n-th Fibonacci number using memoization. Code only."),
    ("hard", "Given a list of integers and an integer k, write a Python function "
             "`max_subarray_sum_k(nums, k)` returning the maximum sum of any contiguous "
             "subarray whose length is at most k, in O(n) time. Explain nothing; code only."),
]


def _post(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def _get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read().decode()


def wait_ready(root, budget_s=2400):
    t0 = time.time()
    while time.time() - t0 < budget_s:
        try:
            if _get(f"{root}/health")[0] == 200:
                return True, time.time() - t0
        except Exception:
            pass
        time.sleep(5)
    return False, time.time() - t0


def run(base, served, prompt, thinking, mode, max_tokens):
    payload = {"model": served, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.0, "top_p": 1.0, "n": 1, "max_tokens": max_tokens}
    if mode == "gemma":
        payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    else:  # qwen3.5 family
        payload["reasoning_effort"] = "xhigh" if thinking else "low"
    t0 = time.time()
    try:
        status, body = _post(f"{base}/chat/completions", payload)
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode()[:300]}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    el = time.time() - t0
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    usage = body.get("usage") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    comp = usage.get("completion_tokens")
    return {
        "ok": True, "thinking": bool(thinking), "elapsed_s": round(el, 2),
        "finish_reason": ch.get("finish_reason"),
        "completion_tokens": comp,
        "tok_per_s": round(comp / el, 2) if comp and el else None,
        "content_chars": len(content), "reasoning_chars": len(reasoning),
        # A working parser yields reasoning in its own field. If reasoning is EMPTY while
        # thinking was requested, the CoT is probably sitting inside content instead.
        "parser_separated_cot": bool(reasoning) if thinking else None,
        "content_head": content[:220],
        "reasoning_head": reasoning[:220],
    }


def main():
    base, served, out_path, mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 8192
    root = base.rsplit("/v1", 1)[0]
    res = {"served_name": served, "mode": mode, "max_tokens": max_tokens, "cases": {}}
    ready, load_s = wait_ready(root)
    res["ready"], res["time_to_ready_s"] = ready, round(load_s, 1)
    if not ready:
        json.dump(res, open(out_path, "w"), indent=2); print("NOT READY"); return 1

    for label, prompt in PROMPTS:
        for thinking in (False, True):
            k = f"{label}_thinking_{'on' if thinking else 'off'}"
            r = run(base, served, prompt, thinking, mode, max_tokens)
            res["cases"][k] = r
            print(f"  [{k}] ok={r.get('ok')} tok={r.get('completion_tokens')} "
                  f"s={r.get('elapsed_s')} reasoning_chars={r.get('reasoning_chars')} "
                  f"finish={r.get('finish_reason')} err={str(r.get('error'))[:110]}", flush=True)

    # cost ratio: thinking-on completion tokens vs thinking-off, per prompt
    ratios = {}
    for label, _ in PROMPTS:
        off = res["cases"].get(f"{label}_thinking_off", {})
        on = res["cases"].get(f"{label}_thinking_on", {})
        if off.get("completion_tokens") and on.get("completion_tokens"):
            ratios[label] = {
                "token_ratio_on_over_off": round(on["completion_tokens"] / off["completion_tokens"], 2),
                "time_ratio_on_over_off": round(on["elapsed_s"] / off["elapsed_s"], 2) if off.get("elapsed_s") else None,
            }
    res["cost_ratios"] = ratios
    json.dump(res, open(out_path, "w"), indent=2)
    print(f"WROTE {out_path}  cost_ratios={json.dumps(ratios)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
