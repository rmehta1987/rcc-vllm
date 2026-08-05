#!/usr/bin/env python
"""Stage-2 GPU generation client: send the frozen 60-problem LCB subset to a co-located vLLM
server under the frozen DECODE, collect completions, write a completions JSON. NO scoring here
(scoring is CPU work on caslake -- session_start.md §4 routing); this only reserves the GPU to
GENERATE, exactly like the Stage-1 smoke client but for all 60 problems.

Invoked by tools/serve_cu129.sbatch in BENCH=1 mode, inside the vllm-serve-cu129 env on the GPU
node (127.0.0.1 co-located, no cross-node HTTP). Reads the SAME prompts/decode as the scorer via
tools/stage2_lcb (uniform by construction). Baseline and every candidate run this SAME client.

  client <base_url> <served_name> <result_path>

Stdlib + huggingface_hub (via stage2_lcb). A hard deadline (NEST_DEADLINE_EPOCH, set by the sbatch
from the SBATCH box) guarantees a completions file is written before Slurm's wall-clock kill.
"""
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2_lcb as lcb  # noqa: E402

READY_TIMEOUT_S = 1800      # fallback ready budget if no NEST_DEADLINE_EPOCH (large model load)
GEN_TIMEOUT_S = 600         # per-request read timeout for one greedy completion
CONCURRENCY = int(os.environ.get("STAGE2_CONCURRENCY", "8"))
TEARDOWN_S = 180            # stop generating this long before the box end, leaving time to write


def _http_get(url, timeout=10):
    with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=timeout) as r:
        return r.status, r.read()


def _post_chat(base, served_name, messages, timeout):
    payload = {"model": served_name, "messages": messages, **lcb.DECODE}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def _deadline():
    v = os.environ.get("NEST_DEADLINE_EPOCH")
    try:
        return float(v) if v else None
    except ValueError:
        return None


def _serve_pid():
    v = os.environ.get("NEST_SERVE_PID")
    try:
        return int(v) if v else None
    except ValueError:
        return None


def _proc_alive(pid):
    if pid is None:
        return True
    try:
        with open(f"/proc/{pid}/stat") as fh:
            after = fh.read().rsplit(") ", 1)[-1].split()
        return bool(after) and after[0] not in ("Z", "X", "x")
    except FileNotFoundError:
        return False
    except OSError:
        return True


def _wait_ready(base, deadline, serve_pid):
    while time.time() < deadline:
        if not _proc_alive(serve_pid):
            return "dead"
        try:
            st, _ = _http_get(base + "/health", timeout=10)
            if st == 200:
                return "ready"
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(10)
    return "timeout"


def _gen_one(base, served_name, problem, req_timeout):
    """Return a completion record for one problem. Any request failure -> empty content (which the
    scorer counts as unsolved); the error is recorded verbatim so it is diagnosable, never opaque."""
    messages = lcb.build_messages(problem)
    try:
        st, data = _post_chat(base, served_name, messages, req_timeout)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:400]
        except Exception:  # noqa: BLE001
            pass
        return {"content": None, "finish_reason": None, "http_status": e.code,
                "error": f"HTTP {e.code}: {body or e.reason}"}
    except Exception as e:  # noqa: BLE001
        return {"content": None, "finish_reason": None, "http_status": None,
                "error": f"{type(e).__name__}: {e}"}
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")   # read `content`, never `reasoning`
    return {"content": content, "finish_reason": choice.get("finish_reason"),
            "http_status": st, "error": None}


def _write(result_path, obj):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
        with open(result_path, "w") as fh:
            json.dump(obj, fh, indent=2)
    except OSError as e:
        print(f"could not write completions to {result_path}: {e}", file=sys.stderr)


def client(base_url, served_name, result_path):
    base = base_url.rstrip("/")
    hard = _deadline()
    serve_pid = _serve_pid()
    problems = lcb.load_subset()   # asserts SUBSET_SHA256 (prereg.md §2) before any generation
    print(f"loaded {len(problems)} problems; concurrency={CONCURRENCY}; served={served_name}")

    ready_deadline = time.time() + READY_TIMEOUT_S
    if hard is not None:
        ready_deadline = min(ready_deadline, hard - TEARDOWN_S)
    status = _wait_ready(base, ready_deadline, serve_pid)
    if status != "ready":
        reason = {"dead": "server process exited during load",
                  "timeout": "server not ready before deadline"}[status]
        _write(result_path, {"stage": "2-gen", "served_name": served_name, "pass": False,
                             "reason": reason, "completions": {}})
        print(f"RESULT stage2-gen {served_name} : FAIL - {reason}")
        return 1

    gen_deadline = (hard - TEARDOWN_S) if hard is not None else (time.time() + 3600)
    completions = {}
    n = len(problems)
    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        fut = {}
        for p in problems:
            req_timeout = GEN_TIMEOUT_S
            fut[ex.submit(_gen_one, base, served_name, p,
                          min(GEN_TIMEOUT_S, max(5.0, gen_deadline - time.time())))] = p["question_id"]
        for f in cf.as_completed(fut):
            qid = fut[f]
            try:
                completions[qid] = f.result()
            except Exception as e:  # noqa: BLE001
                completions[qid] = {"content": None, "finish_reason": None,
                                    "http_status": None, "error": f"future error: {e}"}
            done = len(completions)
            if done % 10 == 0 or done == n:
                print(f"  generated {done}/{n} ({int(gen_deadline - time.time())}s to deadline)")
            if time.time() >= gen_deadline:
                print("hit generation deadline; recording remaining as not-generated")
                break
    for p in problems:
        completions.setdefault(p["question_id"],
                               {"content": None, "finish_reason": None, "http_status": None,
                                "error": "not generated before deadline"})

    n_ok = sum(1 for c in completions.values() if c.get("content"))
    _write(result_path, {
        "stage": "2-gen", "served_name": served_name, "pass": True,
        "n": n, "n_with_content": n_ok, "concurrency": CONCURRENCY,
        "subset_sha256": lcb.SUBSET_SHA256, "decode": lcb.DECODE,
        "completions": completions,
    })
    print(f"RESULT stage2-gen {served_name} : wrote {n} completions ({n_ok} non-empty) -> {result_path}")
    return 0


def writefail(result_path, served_name, reason):
    """Record a gen FAIL (used by the BENCH preflight-fail path so the scorer / next driver reads a
    NO-GO instead of an absent file). Always exits 0 -- the sbatch leaves the queue cleanly."""
    _write(result_path, {"stage": "2-gen", "served_name": served_name, "pass": False,
                         "reason": reason, "completions": {}})
    print(f"RESULT stage2-gen {served_name} : FAIL - {reason}")
    return 0


def main(argv):
    if len(argv) == 5 and argv[1] == "client":
        return client(argv[2], argv[3], argv[4])
    if len(argv) == 5 and argv[1] == "writefail":
        return writefail(argv[2], argv[3], argv[4])
    print("usage: stage2_bench.py client <base_url> <served_name> <result_path>", file=sys.stderr)
    print("       stage2_bench.py writefail <result_path> <served_name> <reason>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
