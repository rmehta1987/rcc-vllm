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
GEN_TIMEOUT_S = 600         # per-request read-timeout CEILING for one greedy completion. Generous
                            # enough for a full max_tokens=8192 answer on the slow dense baseline
                            # (~14 tok/s worst case) so a legit long completion is not truncated and
                            # scored unsolved (adversary R2). It does NOT threaten the box: each
                            # request's ACTUAL timeout is re-bounded at its start by the time left to
                            # gen_deadline (below), and os._exit skips any lingering worker at the end.
# Default concurrency 1: vLLM greedy under continuous batching is NOT bitwise-deterministic (batch
# composition changes FP reduction order, occasionally flipping a token), and that variance can differ
# between the MoE candidate and the dense baseline -- an uncontrolled asymmetry near the +3.0 margin
# (adversary Strike 7). Serial generation removes cross-request batch-composition variance and is
# applied identically to both models; it still fits the 02:00:00 box for a 60-problem run. Override
# with STAGE2_CONCURRENCY only for a deliberately faster, less-reproducible run.
CONCURRENCY = int(os.environ.get("STAGE2_CONCURRENCY", "1"))
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


def _gen_one(base, served_name, problem, gen_deadline):
    """Return a completion record for one problem. Any request failure -> empty content, recorded with
    a verbatim error (which the scorer counts as infra-incomplete, not a real unsolved). The read
    timeout is computed HERE, at the request's actual start, as the smaller of GEN_TIMEOUT_S and the
    time left to gen_deadline -- so a tail request started near the deadline is auto-bounded and the
    serial (concurrency-1) loop cannot block past the box waiting on it (adversary R2 / Fence-3)."""
    req_timeout = max(5.0, min(GEN_TIMEOUT_S, gen_deadline - time.time()))
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


def _is_real_answer(rec):
    """A prior completion worth KEEPING on resume: a content-bearing answer, OR a legitimate HTTP-200
    empty answer (the model genuinely returned no code -- a real unsolved, error is None). NOT an infra
    failure (deadline-skip / read-timeout / HTTP error / request exception), which must be regenerated.
    Mirrors the scorer's n_infra rule EXACTLY (stage2_score.py: `gen_error and not has_content`) so a
    problem is regenerated iff the scorer would have voided the gate on it."""
    if not isinstance(rec, dict):
        return False
    if rec.get("content"):
        return True
    return rec.get("error") is None and rec.get("http_status") == 200


def _plan_resume(problems, prior, served_name):
    """Return (kept, todo): the prior real answers to KEEP and the problems to (re)generate. A prior
    file is trusted ONLY if it is the SAME served_name AND the SAME frozen subset AND the SAME frozen
    DECODE -- otherwise everything is regenerated (kept={}), so a stale/foreign file can never leak a
    completion into a run it does not belong to. Pure (no I/O); unit-tested before any GPU spend."""
    kept = {}
    if (isinstance(prior, dict)
            and prior.get("subset_sha256") == lcb.SUBSET_SHA256
            and prior.get("served_name") == served_name
            and prior.get("decode") == lcb.DECODE):
        kept = {q: r for q, r in (prior.get("completions") or {}).items() if _is_real_answer(r)}
    todo = [p for p in problems if p["question_id"] not in kept]
    return kept, todo


def client(base_url, served_name, result_path):
    base = base_url.rstrip("/")
    hard = _deadline()
    serve_pid = _serve_pid()
    problems = lcb.load_subset()   # asserts SUBSET_SHA256 (prereg.md §2) before any generation
    n = len(problems)

    # RESUME across successive same-decode boxes -- the concurrency-1 completion path for a large/slow
    # model that cannot finish all 60 in ONE 02:00:00 box. A re-submit of the SAME job (same
    # served_name -> same default RESULT_PATH) loads the prior file, KEEPS every real answer
    # (content-bearing, or a legit HTTP-200 empty), and regenerates ONLY the infra-incomplete problems
    # (deadline-skip, read-timeout, HTTP error). Every kept AND new completion is still produced
    # serially at batch size 1, so NO batch-composition variance is introduced (adversary Strike 7 --
    # the reason CONCURRENCY defaults to 1) and each problem's greedy completion is identical to a
    # single-box run; the baseline's identical concurrency-1 decode is untouched, so the head-to-head
    # is not gamed. adjudicate still requires n_infra==0 before Gate 2, so an incomplete resume merely
    # reruns -- it can NEVER pass an incomplete head-to-head. First run (no prior file) -> kept={} ->
    # a no-op (full 60 generated as before).
    prior = {}
    if os.path.exists(result_path):
        try:
            prior = json.load(open(result_path))
        except (json.JSONDecodeError, OSError, ValueError):
            prior = {}
    kept, todo = _plan_resume(problems, prior, served_name)
    completions = dict(kept)
    print(f"loaded {n} problems; concurrency={CONCURRENCY}; served={served_name}; "
          f"resume_kept={len(kept)}; to_generate={len(todo)}")

    if todo:
        ready_deadline = time.time() + READY_TIMEOUT_S
        if hard is not None:
            ready_deadline = min(ready_deadline, hard - TEARDOWN_S)
        status = _wait_ready(base, ready_deadline, serve_pid)
        if status != "ready":
            reason = {"dead": "server process exited during load",
                      "timeout": "server not ready before deadline"}[status]
            # NEVER clobber prior real answers on a transient serve failure: on a resume that already
            # holds kept answers, leave the existing (more-complete) file intact for the next resume
            # instead of overwriting it with an empty FAIL record (that would throw away real progress).
            if kept:
                print(f"RESULT stage2-gen {served_name} : server not ready ({reason}); left "
                      f"{len(kept)} prior answers intact for next resume")
                return 1
            _write(result_path, {"stage": "2-gen", "served_name": served_name, "pass": False,
                                 "reason": reason, "completions": {}})
            print(f"RESULT stage2-gen {served_name} : FAIL - {reason}")
            return 1

        gen_deadline = (hard - TEARDOWN_S) if hard is not None else (time.time() + 3600)
        ex = cf.ThreadPoolExecutor(max_workers=CONCURRENCY)
        try:
            fut = {ex.submit(_gen_one, base, served_name, p, gen_deadline): p["question_id"]
                   for p in todo}
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
        finally:
            # Do NOT block on a normal `with` exit (shutdown(wait=True)): it would drain running AND
            # still-queued futures, so on a slow/wedged server client() could return only AFTER the box
            # kill -> no verdict written (config-reviewer Fence-3 High). Cancel queued work and return.
            ex.shutdown(wait=False, cancel_futures=True)
    else:
        print("RESUME: all 60 already have real answers; writing complete file without re-serving")

    # Distinguish infra-incompleteness (never generated before the deadline) from a real empty answer
    # (the model returned no code) -- the former compromises the head-to-head (adversary Strike 6),
    # the latter is a legitimate unsolved. adjudicate refuses a run with any not-generated problem.
    not_generated = []
    for p in problems:
        if p["question_id"] not in completions:
            completions[p["question_id"]] = {"content": None, "finish_reason": None,
                                             "http_status": None, "error": "not generated before deadline"}
        if completions[p["question_id"]].get("error") == "not generated before deadline":
            not_generated.append(p["question_id"])

    n_ok = sum(1 for c in completions.values() if c.get("content"))
    _write(result_path, {
        "stage": "2-gen", "served_name": served_name, "pass": True,
        "n": n, "n_with_content": n_ok, "n_not_generated": len(not_generated),
        "not_generated": not_generated, "concurrency": CONCURRENCY, "resumed_kept": len(kept),
        "subset_sha256": lcb.SUBSET_SHA256,
        "subset_content_fingerprint": lcb.subset_content_fingerprint(problems),
        "decode": lcb.DECODE, "completions": completions,
    })
    print(f"RESULT stage2-gen {served_name} : wrote {n} completions ({n_ok} non-empty, "
          f"{len(not_generated)} not-generated) -> {result_path}")
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
        rc = client(argv[2], argv[3], argv[4])
        # The verdict is already on disk. A single request may still be in-flight in a worker thread
        # (cancel_futures cancels only QUEUED work); the normal interpreter-exit join would block up
        # to GEN_TIMEOUT_S, keeping the sbatch -- and thus the Slurm job -- RUNNING and delaying the
        # orchestrator's next iteration. os._exit returns immediately, killing that thread with the
        # process; the sbatch then tears down the serve and the job leaves the queue promptly.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    if len(argv) == 5 and argv[1] == "writefail":
        return writefail(argv[2], argv[3], argv[4])
    print("usage: stage2_bench.py client <base_url> <served_name> <result_path>", file=sys.stderr)
    print("       stage2_bench.py writefail <result_path> <served_name> <reason>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
