#!/usr/bin/env python3
"""Drive a mix of normal + streaming requests through the ai-session gateway,
so the gateway's usage-capture path is exercised end-to-end (STEP 3 e2e).

Run AFTER `ai_session start` has published the backend to the gateway:
    /project/rcc/mehta5/conda-envs/vllm-probe/bin/python ai-session/e2e_send_requests.py \
        --base http://127.0.0.1:8421 --model qwen2.5_72B

Prints each request's reported usage and a local tally; the gateway logs the
same usage to logs/gateway/usage-YYYYMMDD.jsonl, which `ai_session end` bills from.
"""
import argparse
import json
import sys

import httpx


def chat_nonstream(client, base, model, messages, max_tokens, key):
    r = client.post(
        f"{base}/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": messages, "max_tokens": max_tokens,
              "temperature": 0.7, "seed": 7},
        timeout=httpx.Timeout(None, connect=10.0),
    )
    r.raise_for_status()
    u = r.json().get("usage", {})
    return u


def chat_stream(client, base, model, messages, max_tokens, key):
    """Streaming chat; the gateway injects stream_options.include_usage so the
    final SSE chunk carries usage. We parse it here too for a local tally."""
    usage = {}
    with client.stream(
        "POST",
        f"{base}/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": messages, "max_tokens": max_tokens,
              "temperature": 0.7, "seed": 7, "stream": True},
        timeout=httpx.Timeout(None, connect=10.0),
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]" or not data:
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("usage"):
                usage = obj["usage"]
    return usage


def legacy_completion(client, base, model, prompt, max_tokens, key):
    r = client.post(
        f"{base}/v1/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
              "temperature": 0.5, "seed": 7},
        timeout=httpx.Timeout(None, connect=10.0),
    )
    r.raise_for_status()
    return r.json().get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8421")
    ap.add_argument("--model", default="qwen2.5_72B")
    ap.add_argument("--key", default="ai-session")
    args = ap.parse_args()

    big_doc = ("The quick brown fox jumps over the lazy dog. " * 200)  # ~long-ish input

    plan = [
        ("nonstream", "Say hello in one short sentence.", 32),
        ("nonstream", "List three primary colors, comma-separated.", 32),
        ("nonstream", "Write a two-sentence summary of what a GPU does.", 96),
        ("stream",    "Count slowly from one to ten, words only.", 80),
        ("stream",    "Explain tensor parallelism to a beginner in 3 sentences.", 160),
        ("stream",    "Write a short haiku about supercomputers.", 64),
        ("legacy",    "Complete this: The capital of France is", 16),
        ("longdoc",   "Summarize the following text in one sentence:\n\n" + big_doc, 64),
    ]

    client = httpx.Client()
    tot_in = tot_out = 0
    n_ok = 0
    print(f"sending {len(plan)} requests through {args.base} (model={args.model})\n")
    for i, (kind, prompt, mt) in enumerate(plan, 1):
        try:
            if kind == "stream":
                u = chat_stream(client, args.base, args.model,
                                [{"role": "user", "content": prompt}], mt, args.key)
            elif kind == "legacy":
                u = legacy_completion(client, args.base, args.model, prompt, mt, args.key)
            else:  # nonstream / longdoc
                u = chat_nonstream(client, args.base, args.model,
                                   [{"role": "user", "content": prompt}], mt, args.key)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] {kind:9s} ERROR: {e}")
            continue
        pin = int(u.get("prompt_tokens", 0))
        pout = int(u.get("completion_tokens", 0))
        tot_in += pin
        tot_out += pout
        n_ok += 1
        print(f"  [{i}] {kind:9s} in={pin:5d} out={pout:4d}")
    client.close()

    print(f"\nlocal tally: {n_ok}/{len(plan)} ok  in={tot_in}  out={tot_out}")
    return 0 if n_ok == len(plan) else 1


if __name__ == "__main__":
    sys.exit(main())
