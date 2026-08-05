#!/usr/bin/env python
"""Stage-2 shared LiveCodeBench harness: frozen-subset load, prompt build, pass@1 score.

Consumed by BOTH the GPU generation client (tools/stage2_bench.py, in the vllm-serve-cu129
env on a test-partition GPU node) and the CPU scorer (tools/stage2_score.py, on caslake).
Every constant here is transcribed from the FROZEN pre-registration prompts/60_model_refresh/
prereg.md (§2 subset, §3 decode, §4 prompt+scoring); the gauntlet reference-checker verifies
they match. Baseline and every candidate run this SAME module, so subset/prompt/decode/scoring
are uniform by construction (session_start.md §1 fence: an env/decode/harness mismatch would
silently game Gate 2).

Subcommands (this file is ALSO the sandbox child so model code never runs in the scorer proc):
  _runtests <spec.json>   execute one completion against its test cases; print a JSON verdict.
  selftest                materialize the subset, assert SHA256, score known good/bad solutions.

Deps: stdlib + huggingface_hub (offline cache resolution only). No torch/vllm import here, so it
loads on a CPU caslake node and inside the serve env alike.
"""
import ast
import base64
import json
import os
import pickle
import re
import signal
import subprocess
import sys
import tempfile
import zlib

# --- FROZEN constants (prereg.md §1/§2/§3) --------------------------------------- #
LCB_DATASET = "livecodebench/code_generation_lite"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LCB_SHARD = "test6.jsonl"
SUBSET_N = 60
SUBSET_SHA256 = "b3c2b75377c305f495de834ec699bc07d70b6b507230fe16c8d73bf550b7021b"
# The frozen 60 question_ids (prereg.md §2), canonical sorted order == the source of truth.
FROZEN_IDS = [
    "3717", "3744", "3750", "3759", "3765", "3773", "3777", "3784", "3788", "3789",
    "3791", "3793", "3794", "3795", "3799", "3801", "3805", "3809", "3811", "3817",
    "3832", "abc397_a", "abc397_b", "abc397_c", "abc397_d", "abc397_e", "abc397_f",
    "abc397_g", "abc398_a", "abc398_b", "abc398_c", "abc398_d", "abc398_f", "abc398_g",
    "abc399_a", "abc399_b", "abc399_c", "abc399_d", "abc399_e", "abc399_f", "abc400_a",
    "abc400_b", "abc400_c", "abc400_d", "abc400_e", "abc400_g", "arc194_a", "arc194_b",
    "arc194_c", "arc194_d", "arc194_e", "arc195_a", "arc195_b", "arc195_c", "arc195_d",
    "arc195_e", "arc196_a", "arc196_b", "arc196_c", "arc196_d",
]

# prereg.md §3 DECODE -- ONE policy applied identically to baseline and every candidate.
DECODE = {
    "temperature": 0.0,
    "top_p": 1.0,
    "n": 1,
    "seed": 0,
    "max_tokens": 8192,
    "chat_template_kwargs": {"enable_thinking": False},
}

PER_TEST_TIMEOUT_S = 6      # prereg.md §4 per-test wall timeout
SYS_MSG = ("You are an expert Python programmer. You will be given a competitive programming "
           "problem. Provide a correct, self-contained Python solution.")

# A standard import prelude prepended to EVERY executed completion so a correct solution that
# omits `from typing import List` (etc.) is not false-failed on an infra error. Applied identically
# to baseline and candidate -> it cannot bias the head-to-head (it only removes noise). Mirrors the
# LiveCodeBench call-based runner convention.
PRELUDE = (
    "import sys, math, collections, itertools, heapq, bisect, functools, re, string, random, io\n"
    # NOT `from math import *`: that binds `pow = math.pow`, shadowing the BUILTIN pow. 3-arg
    # modular pow(a, b, m) (ubiquitous in these problems: modular inverse, Fermat) would then raise
    # TypeError and 2-arg pow would return a float -- silently marking CORRECT answers WRONG, and
    # not symmetrically between models. Import the common math NAMES explicitly, leaving builtin pow.
    "from math import (sqrt, gcd, lcm, factorial, comb, perm, isqrt, floor, ceil, log, log2, log10, "
    "exp, inf, nan, pi, e, tau, hypot, prod, degrees, radians)\n"
    "from collections import *\n"
    "from itertools import *\n"
    "from functools import *\n"
    "from bisect import *\n"
    "from heapq import *\n"
    "from typing import *\n"
    "sys.setrecursionlimit(1000000)\n"
)


# ------------------------------------------------------------- subset loading ----- #
def _resolve_shard():
    """Offline-resolve the pinned shard from the HF cache (prereg.md §1). No network: the
    caslake/GPU nodes have none, so HF_HUB_OFFLINE is forced on."""
    os.environ.setdefault("HF_HOME", "/project/rcc/mehta5/hf-cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    from huggingface_hub import hf_hub_download
    return hf_hub_download(LCB_DATASET, LCB_SHARD, repo_type="dataset", revision=LCB_REVISION)


def _decode_private(row):
    """private_test_cases is either plain JSON or base64(zlib(pickle(json_str))) (LCB format)."""
    raw = row.get("private_test_cases", "")
    try:
        return json.loads(raw)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))


def load_subset(assert_sha=True):
    """Materialize the frozen 60-problem subset (prereg.md §2 selection rule), asserting
    SUBSET_SHA256 before returning. A hash mismatch is a hard error (session_start.md §9 blocker;
    never a silent re-pin). Returns a list of normalized problem dicts in canonical sorted-id order.
    """
    import hashlib
    path = _resolve_shard()
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # Selection: sort by (contest_date, question_id) asc, take the 60 latest (prereg.md §2).
    rows_sorted = sorted(rows, key=lambda r: (r.get("contest_date", ""), str(r.get("question_id"))))
    sel = rows_sorted[-SUBSET_N:]
    sel_ids = sorted(str(r["question_id"]) for r in sel)
    sha = hashlib.sha256("\n".join(sel_ids).encode()).hexdigest()
    if assert_sha and sha != SUBSET_SHA256:
        raise SystemExit(
            f"SUBSET_SHA256 MISMATCH: materialized {sha} != frozen {SUBSET_SHA256} "
            f"(prereg.md §2). Refusing to score -- this is a §9 blocker, not a re-pin.")
    if assert_sha and sel_ids != sorted(FROZEN_IDS):
        raise SystemExit("subset id set != frozen FROZEN_IDS despite SHA match -- refusing.")
    probs = []
    for r in sorted(sel, key=lambda r: str(r["question_id"])):
        starter = (r.get("starter_code") or "")
        meta = r.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        pub = r["public_test_cases"]
        pub = json.loads(pub) if isinstance(pub, str) else pub
        probs.append({
            "question_id": str(r["question_id"]),
            "question_content": r.get("question_content") or "",
            "starter_code": starter,
            "platform": r.get("platform"),
            "difficulty": r.get("difficulty"),
            "func_name": (meta or {}).get("func_name"),
            "is_functional": bool(starter.strip()),
            "tests": pub + _decode_private(r),   # public + private, scored together (prereg.md §4)
        })
    return probs


# --------------------------------------------------------------- prompt build ----- #
def subset_content_fingerprint(problems):
    """A provenance hash over the FULL problem content (statement, starter, func_name, and every
    public+private test), not just the id set that SUBSET_SHA256 binds (prereg.md §2 hashes ids
    only). Recorded (non-gating) in the gen/score outputs so a later change to the cached shard's
    test CONTENT -- which the id-set SHA would not catch -- is detectable after the fact."""
    import hashlib
    h = hashlib.sha256()
    for p in problems:   # already in canonical sorted-id order from load_subset
        for part in (p["question_id"], p["question_content"], p["starter_code"],
                     str(p.get("func_name")), json.dumps(p["tests"], sort_keys=True, default=str)):
            h.update((part or "").encode())
            h.update(b"\0")
    return h.hexdigest()


def build_messages(problem):
    """The uniform prompt (prereg.md §4): starter-code (functional) vs stdin template."""
    qc = problem["question_content"]
    if problem["is_functional"]:
        user = (
            "### Question:\n" + qc + "\n\n"
            "### Starter code:\n```python\n" + problem["starter_code"] + "\n```\n\n"
            "### Instructions:\n"
            "Complete the solution. Respond with a single self-contained Python code block that "
            "defines the required class/function. Do not include explanations."
        )
    else:
        user = (
            "### Question:\n" + qc + "\n\n"
            "### Instructions:\n"
            "Write a complete Python program that reads from standard input and writes the answer "
            "to standard output. Respond with a single self-contained Python code block. Do not "
            "include explanations."
        )
    return [{"role": "system", "content": SYS_MSG}, {"role": "user", "content": user}]


def extract_code(content):
    """Last fenced code block (```lang\\n...```); if none, the raw content (prereg.md §4)."""
    if not content:
        return ""
    blocks = re.findall(r"```(?:[A-Za-z0-9_+-]+)?[ \t]*\r?\n(.*?)```", content, re.DOTALL)
    return blocks[-1] if blocks else content


# ------------------------------------------------------------------ scoring ------- #
def _parse_literal(s):
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        return ast.literal_eval(s)


def _canon(x):
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, dict):
        return {k: _canon(v) for k, v in x.items()}
    return x


def _eq(g, e):
    # bool must match bool exactly (avoid True==1 leaking a wrong int through a bool-answer)
    if isinstance(e, bool) or isinstance(g, bool):
        return g is e or g == e and isinstance(g, bool) == isinstance(e, bool)
    if isinstance(e, float) or isinstance(g, float):
        try:
            return abs(float(g) - float(e)) <= 1e-6 * max(1.0, abs(float(e)))
        except Exception:
            return False
    if isinstance(e, list) and isinstance(g, list):
        return len(g) == len(e) and all(_eq(a, b) for a, b in zip(g, e))
    if isinstance(e, dict) and isinstance(g, dict):
        return set(g) == set(e) and all(_eq(g[k], e[k]) for k in e)
    return g == e


def deep_equal(got, exp):
    return _eq(_canon(got), _canon(exp))


def _norm_stdout(s):
    lines = (s or "").replace("\r\n", "\n").split("\n")
    lines = [ln.rstrip() for ln in lines]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


class _Timeout(Exception):
    pass


def _with_alarm(seconds, fn):
    def _handler(_s, _f):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _run_functional(code, func_name, tests):
    """exec the completion, then for each test parse args (one literal per input line), call the
    target method under a per-test alarm, deep-compare the return value. Short-circuits on the first
    failing test (a problem is SOLVED iff ALL tests pass -- prereg.md §4)."""
    ns = {"__name__": "__solution__"}
    exec(PRELUDE + "\n" + code, ns)  # our own benchmark harness; model code isolated in this child
    sol_cls = ns.get("Solution")
    if sol_cls is None:
        return False, "no class Solution defined"
    if not func_name:
        return False, "no func_name in metadata"
    for i, t in enumerate(tests):
        args = [_parse_literal(ln) for ln in t["input"].split("\n") if ln.strip() != ""]
        try:
            expected = _parse_literal(t["output"])
        except Exception as e:  # noqa: BLE001
            return False, f"cannot parse expected output for test {i}: {e}"
        try:
            got = _with_alarm(PER_TEST_TIMEOUT_S,
                              lambda: getattr(sol_cls(), func_name)(*[_canon(a) for a in args]))
        except _Timeout:
            return False, f"test {i}: timeout (> {PER_TEST_TIMEOUT_S}s)"
        except Exception as e:  # noqa: BLE001
            return False, f"test {i}: {type(e).__name__}: {e}"
        if not deep_equal(got, expected):
            return False, f"test {i}: wrong answer (got {repr(got)[:120]} want {repr(expected)[:120]})"
    return True, f"all {len(tests)} tests pass"


def _run_stdin(code, tests):
    """Run the program as a REAL subprocess per test with the test input piped to stdin, comparing
    normalized stdout to expected (prereg.md §4). A real process (not an in-proc StringIO) is more
    faithful: competitive-programming fast-IO templates use `sys.stdin.buffer.readline`, which a
    StringIO stdin lacks; the subprocess also gives a hard per-test timeout and full isolation.
    Short-circuits on the first failing test (SOLVED iff ALL pass)."""
    src = PRELUDE + "\n" + code
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        solpath = fh.name
    try:
        for i, t in enumerate(tests):
            try:
                p = subprocess.run([sys.executable, solpath], input=t["input"],
                                   capture_output=True, text=True, timeout=PER_TEST_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                return False, f"test {i}: timeout (> {PER_TEST_TIMEOUT_S}s)"
            except Exception as e:  # noqa: BLE001
                return False, f"test {i}: {type(e).__name__}: {e}"
            if _norm_stdout(p.stdout) != _norm_stdout(t["output"]):
                return False, (f"test {i}: wrong output (got {repr(_norm_stdout(p.stdout))[:120]} "
                               f"want {repr(_norm_stdout(t['output']))[:120]})")
    finally:
        try:
            os.unlink(solpath)
        except OSError:
            pass
    return True, f"all {len(tests)} tests pass"


def run_tests(problem, code):
    """Run all tests for one problem IN THIS PROCESS. Meant to be called only inside the _runtests
    child (subprocess-isolated so a runaway completion cannot wedge the scorer)."""
    if not (code or "").strip():
        return False, "empty completion"
    if problem["is_functional"]:
        return _run_functional(code, problem.get("func_name"), problem["tests"])
    return _run_stdin(code, problem["tests"])


def score_one(problem, code, python_exe=None):
    """Score one completion by spawning THIS file's _runtests child (model code never executes in
    the scorer process). Returns {solved, detail, n_tests}. A problem-level timeout / crashed child
    counts as unsolved (uniform rule, prereg.md §4)."""
    python_exe = python_exe or sys.executable
    n_tests = len(problem["tests"])
    if not (code or "").strip():
        return {"solved": False, "detail": "empty completion", "n_tests": n_tests}
    spec = {"problem": {k: problem[k] for k in
                        ("question_id", "is_functional", "func_name", "tests")}, "code": code}
    # Outer wall bound: per-test budget x #tests + slack, capped so one problem cannot dominate the
    # caslake box. Correct solutions finish in ms/test; only a wedged child approaches this.
    outer = min(400, PER_TEST_TIMEOUT_S * n_tests + 30)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(spec, fh)
        specpath = fh.name
    try:
        p = subprocess.run([python_exe, os.path.abspath(__file__), "_runtests", specpath],
                           capture_output=True, text=True, timeout=outer)
        line = (p.stdout or "").strip().splitlines()
        verdict = json.loads(line[-1]) if line else None
        if verdict is None:
            return {"solved": False, "n_tests": n_tests,
                    "detail": f"child produced no verdict (rc={p.returncode}): {(p.stderr or '')[-200:]}"}
        return {"solved": bool(verdict["solved"]), "detail": verdict["detail"], "n_tests": n_tests}
    except subprocess.TimeoutExpired:
        return {"solved": False, "detail": f"problem-level timeout (> {outer}s)", "n_tests": n_tests}
    finally:
        try:
            os.unlink(specpath)
        except OSError:
            pass


# --------------------------------------------------------------- subcommands ------ #
def _cmd_runtests(specpath):
    spec = json.load(open(specpath))
    solved, detail = run_tests(spec["problem"], spec["code"])
    print(json.dumps({"solved": solved, "detail": detail}))
    return 0


def _cmd_selftest():
    """CPU self-test: assert the subset SHA, then score a known-CORRECT and known-WRONG solution on
    one functional (3799) and one stdin (abc397_a) problem, so a harness regression is caught BEFORE
    any GPU spend. Exit 0 iff all four verdicts are as expected."""
    probs = {p["question_id"]: p for p in load_subset()}
    print("subset OK: 60 problems, SHA256 asserted")
    good = {
        "3799": "class Solution:\n    def totalNumbers(self, digits):\n        seen=set()\n"
                "        for p in permutations(digits,3):\n"
                "            if p[0]==0 or p[2]%2: continue\n"
                "            seen.add(p[0]*100+p[1]*10+p[2])\n        return len(seen)\n",
        "abc397_a": "x=float(input())\nprint(1 if x>=38.0 else 2 if x>=37.5 else 3)\n",
    }
    bad = {
        "3799": "class Solution:\n    def totalNumbers(self, digits):\n        return 0\n",
        "abc397_a": "x=float(input())\nprint(99)\n",
    }
    ok = True
    for qid in ("3799", "abc397_a"):
        gr = score_one(probs[qid], good[qid])
        br = score_one(probs[qid], bad[qid])
        print(f"{qid}: good->{gr['solved']} ({gr['detail']}) | bad->{br['solved']} ({br['detail']})")
        ok = ok and gr["solved"] is True and br["solved"] is False

    # Regression guard for the `from math import *` shadow bug (adversary Strike 1): a completion
    # using 3-arg modular pow MUST score solved in BOTH the functional and stdin paths. pow(2,10,1000)
    # = 24. If the PRELUDE ever re-shadows builtin pow with math.pow, these flip to unsolved.
    pow_func = {"question_id": "_powtest_func", "is_functional": True, "func_name": "modexp",
                "tests": [{"input": "2\n10\n1000", "output": "24"},
                          {"input": "3\n5\n7", "output": "5"}]}   # 3^5=243, 243%7=5
    pow_stdin = {"question_id": "_powtest_stdin", "is_functional": False, "func_name": None,
                 "tests": [{"input": "2 10 1000", "output": "24"}, {"input": "3 5 7", "output": "5"}]}
    pf = score_one(pow_func, "class Solution:\n    def modexp(self, a, b, m):\n        return pow(a, b, m)\n")
    ps = score_one(pow_stdin, "a,b,m=map(int,input().split())\nprint(pow(a,b,m))\n")
    print(f"pow-regression: functional->{pf['solved']} ({pf['detail']}) | stdin->{ps['solved']} ({ps['detail']})")
    ok = ok and pf["solved"] is True and ps["solved"] is True

    print("subset content fingerprint:", subset_content_fingerprint(list(probs.values()))[:16], "...")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv):
    if len(argv) == 3 and argv[1] == "_runtests":
        return _cmd_runtests(argv[2])
    if len(argv) == 2 and argv[1] == "selftest":
        return _cmd_selftest()
    print("usage: stage2_lcb.py _runtests <spec.json> | selftest", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
