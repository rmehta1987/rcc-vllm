#!/usr/bin/env python
"""Stage-1 smoke: fail-cheap GPU pre-flight + co-located OpenAI-endpoint accept-check.

Invoked by tools/serve_cu129.sbatch in SMOKE=1 mode, INSIDE the vllm-serve-cu129 env
on the GPU compute node. Three subcommands:

  preflight <model_dir>
      Cheap checks BEFORE the multi-minute weight load, so a doomed reservation
      fails in seconds (session_start.md §4): CUDA is live, a real GPU compute op
      runs, every architecture in config.json is registered in this vLLM, and (for
      an fp8/fp4-quantized model) the quant storage dtype exists / the compute
      capability clears the kernel floor. Exit 0 iff all pass; nonzero => "do not
      bother loading the weights". NOTE: this proves the GPU is in the right ballpark;
      the DEFINITIVE quant-kernel test is the serve load that follows (the V4-Flash
      FP4-on-Hopper trap resolves there, session_start.md §2 Stage 1).

  client <base_url> <served_name> <result_path>
      Poll <base_url>/health until ready, send the frozen Gate-1 SMOKE_PROMPT
      (prereg.md §6) to /v1/chat/completions under the frozen DECODE policy
      (prereg.md §3 -- greedy t0/top_p1/n1/seed0/max8192, enable_thinking:false,
      read `content` NOT `reasoning`), run the deterministic two_sum accept-check,
      and write a JSON verdict to <result_path>. Exit 0 iff the accept-check passes.

  writefail <result_path> <served_name> <reason>
      Record a FAIL verdict (used by the preflight-fail path so the next driver
      session reads a NO-GO instead of an absent file). Always exits 0.

Stdlib only (urllib/json) -- adds no dependency to the serve env. The SMOKE_PROMPT,
DECODE, and accept-check below are transcribed from the frozen prereg.md (§3, §6);
the gauntlet reference-checker verifies they match. Baseline and every candidate run
this SAME harness, so decode is uniform by construction (session_start.md §1 fence).
"""
import json
import os
import re
import signal
import sys
import time
import urllib.error
import urllib.request

# --- frozen Gate-1 constants (prereg.md §6 SMOKE_PROMPT, §3 DECODE) -------------- #
SYS_MSG = "You are an expert Python programmer. Provide correct, self-contained code."
USER_MSG = (
    "Write a Python function with the exact signature "
    "`def two_sum(nums: list[int], target: int) -> list[int]:` that returns the "
    "indices of the two numbers in `nums` that add up to `target`. Exactly one "
    "solution exists and the same element may not be used twice. Respond with only "
    "a single Python code block containing the function."
)
# prereg.md §6 accept check: exec the returned code, then require ALL three.
ACCEPT_CASES = [
    (([2, 7, 11, 15], 9), [0, 1]),
    (([3, 2, 4], 6), [1, 2]),
    (([3, 3], 6), [0, 1]),
]
# prereg.md §3 DECODE -- identical for baseline and every candidate.
DECODE = {
    "temperature": 0.0,
    "top_p": 1.0,
    "n": 1,
    "seed": 0,
    "max_tokens": 8192,
    "chat_template_kwargs": {"enable_thinking": False},
}
READY_TIMEOUT_S = 1200   # fallback ready budget when no absolute SMOKE_DEADLINE_EPOCH is set
GEN_TIMEOUT_S = 180      # read timeout for one greedy completion (trivial task -> seconds)
ACCEPT_TIMEOUT_S = 20    # a pathological completion cannot hang the job


def _write(result_path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    with open(result_path, "w") as fh:
        json.dump(obj, fh, indent=2)


def _hard_deadline():
    """Absolute epoch deadline the SMOKE sbatch derives from the SBATCH box, so the client
    ALWAYS writes a verdict and returns before Slurm's wall-clock kill (which would skip the
    verdict + the graceful serve teardown). None when not launched by the sbatch."""
    v = os.environ.get("SMOKE_DEADLINE_EPOCH")
    try:
        return float(v) if v else None
    except ValueError:
        return None


def _serve_pid():
    v = os.environ.get("SMOKE_SERVE_PID")
    try:
        return int(v) if v else None
    except ValueError:
        return None


def _proc_alive(pid):
    """False only if the serve process is gone or a zombie; any other read outcome returns
    True so a transient /proc hiccup never false-fails a healthy serve. Lets the client bail
    within one poll of a serve that crashes on load instead of burning the whole ready box."""
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


# ------------------------------------------------------------------ preflight ---- #
def preflight(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    archs = cfg.get("architectures") or []
    import torch

    print("cuda_available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("PREFLIGHT FAIL: torch.cuda.is_available() is False")
        return 1
    cc = torch.cuda.get_device_capability()
    print("device        :", torch.cuda.get_device_name(0), "cc", cc)

    # A real GPU compute op -- catches a driver/runtime mismatch that is_available() misses.
    try:
        x = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
        _ = float((x @ x).item())
        torch.cuda.synchronize()
        print("gpu_compute   : bf16 matvec OK")
    except Exception as e:  # noqa: BLE001 -- report any GPU compute failure verbatim
        print(f"PREFLIGHT FAIL: gpu compute -> {type(e).__name__}: {e}")
        return 1

    ok = True
    from vllm import ModelRegistry

    supported = set(ModelRegistry.get_supported_archs())
    for a in archs:
        hit = a in supported
        print(f"arch          : {a} -> {'SUPPORTED' if hit else 'NOT SUPPORTED'}")
        ok = ok and hit

    # Quant probe: informational cc + a cheap dtype/floor check for the exact quant.
    q = cfg.get("quantization_config") or {}
    qs = json.dumps(q).lower()
    quant = q.get("quant_method") or q.get("quant_algo") or (
        "fp8" if "fp8" in qs else
        "fp4" if ("fp4" in qs or "mxfp4" in qs or "nvfp4" in qs) else "none"
    )
    print(f"quant         : {quant} (device cc {cc})")
    if "fp8" in qs:
        # FP8 e4m3 tensor-core GEMM is Ada(sm89)/Hopper(sm90)+; on this cluster the FP8
        # candidates serve on H200 (cc 9.0). A cc floor catches an FP8 model mis-routed to a
        # pre-Ada GPU (e.g. A100 cc 8.0) cheaply -- a storage-only dtype alloc, which is not
        # compute-capability-gated, would false-PASS it into the multi-minute weight load.
        if cc < (8, 9):
            print(f"PREFLIGHT FAIL: fp8 tensor cores need >= sm89 (Ada/Hopper), have {cc}")
            return 1
        try:
            _ = torch.zeros(16, device="cuda", dtype=torch.float8_e4m3fn)
            print(f"quant_probe   : float8_e4m3fn tensor OK on this GPU (cc {cc})")
        except Exception as e:  # noqa: BLE001
            print(f"PREFLIGHT FAIL: fp8 dtype -> {type(e).__name__}: {e}")
            return 1
    if "fp4" in qs or "mxfp4" in qs or "nvfp4" in qs:
        # FP4 tensor-core compute is Blackwell (sm100). On Hopper (sm90) vLLM 0.26.0
        # routes MXFP4 experts to a Marlin kernel (sm80+); the serve load is definitive.
        if cc < (8, 0):
            print(f"PREFLIGHT FAIL: fp4/mxfp4 needs >= sm80 for the Marlin path, have {cc}")
            return 1
        print("quant_probe   : fp4/mxfp4 present; relies on Marlin(sm90) -- serve load is definitive")

    print("PREFLIGHT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------- client ---- #
def _http_get(url, timeout=10):
    with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=timeout) as r:
        return r.status, r.read()


def _http_post_json(url, payload, timeout=GEN_TIMEOUT_S):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def _wait_ready(base, deadline, serve_pid):
    while time.time() < deadline:
        if not _proc_alive(serve_pid):
            return "dead"   # serve crashed during load -> fail now, do not burn the box
        try:
            st, _ = _http_get(base + "/health", timeout=10)
            if st == 200:
                return "ready"
        except (urllib.error.URLError, ConnectionError, OSError):
            pass  # server not up yet
        time.sleep(10)
    return "timeout"


def _extract_code(content):
    # Last fenced block (```lang\n...```); if none, the raw content (prereg.md §4/§6).
    blocks = re.findall(r"```(?:[A-Za-z0-9_+-]+)?[ \t]*\r?\n(.*?)```", content, re.DOTALL)
    return blocks[-1] if blocks else content


class _Timeout(Exception):
    pass


def _accept(content):
    code = _extract_code(content or "")

    def _alarm(_signum, _frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(ACCEPT_TIMEOUT_S)
    try:
        ns = {}
        exec(code, ns)  # our own benchmark; the two_sum task is trivial and time-boxed
        fn = ns.get("two_sum")
        if not callable(fn):
            return False, "no two_sum defined"
        for args, expected in ACCEPT_CASES:
            got = fn(*args)
            if sorted(got) != expected:
                return False, f"wrong answer for {args}: got {got}, want {expected}"
        return True, "ok"
    except _Timeout:
        return False, "accept-check timed out (pathological completion)"
    except Exception as e:  # noqa: BLE001 -- any exec/runtime error is a Gate-1 fail
        return False, f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def client(base_url, served_name, result_path):
    base = base_url.rstrip("/")
    hard = _hard_deadline()
    serve_pid = _serve_pid()
    ready_deadline = time.time() + READY_TIMEOUT_S
    if hard is not None:
        ready_deadline = min(ready_deadline, hard)

    status = _wait_ready(base, ready_deadline, serve_pid)
    if status != "ready":
        reason = {"dead": "server process exited during load",
                  "timeout": "server not ready before deadline"}[status]
        _write(result_path, {"stage": "1", "served_name": served_name, "pass": False, "reason": reason})
        print(f"RESULT stage1 {served_name} : FAIL - {reason}")
        return 1

    # Bound the single completion by whatever the box still allows, so the client always
    # returns (and writes a verdict) before the SBATCH wall-clock kill.
    gen_timeout = GEN_TIMEOUT_S
    if hard is not None:
        gen_timeout = min(GEN_TIMEOUT_S, max(5.0, hard - time.time()))

    payload = {"model": served_name,
               "messages": [{"role": "system", "content": SYS_MSG},
                            {"role": "user", "content": USER_MSG}],
               **DECODE}
    try:
        st, data = _http_post_json(base + "/v1/chat/completions", payload, timeout=gen_timeout)
    except Exception as e:  # noqa: BLE001
        _write(result_path, {"stage": "1", "served_name": served_name, "pass": False,
                             "reason": f"chat/completions error: {type(e).__name__}: {e}"})
        print(f"RESULT stage1 {served_name} : FAIL - request error {e}")
        return 1

    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")   # read `content`, never `reasoning`
    finish = choice.get("finish_reason")
    ok, reason = _accept(content) if content else (False, f"empty content (finish_reason={finish})")

    _write(result_path, {"stage": "1", "served_name": served_name, "pass": ok, "reason": reason,
                         "http_status": st, "finish_reason": finish,
                         "content_head": (content or "")[:400]})
    print(f"RESULT stage1 {served_name} : {'PASS' if ok else 'FAIL'} - {reason}")
    return 0 if ok else 1


def writefail(result_path, served_name, reason):
    try:
        _write(result_path, {"stage": "1", "served_name": served_name, "pass": False, "reason": reason})
    except OSError as e:
        print(f"writefail: could not write verdict to {result_path}: {e}", file=sys.stderr)
    print(f"RESULT stage1 {served_name} : FAIL - {reason}")
    return 0


# ----------------------------------------------------------------------- main ---- #
def main(argv):
    if len(argv) >= 3 and argv[1] == "preflight":
        return preflight(argv[2])
    if len(argv) == 5 and argv[1] == "client":
        return client(argv[2], argv[3], argv[4])
    if len(argv) == 5 and argv[1] == "writefail":
        return writefail(argv[2], argv[3], argv[4])
    print("usage: stage1_smoke.py preflight <model_dir>", file=sys.stderr)
    print("       stage1_smoke.py client <base_url> <served_name> <result_path>", file=sys.stderr)
    print("       stage1_smoke.py writefail <result_path> <served_name> <reason>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
