#!/usr/bin/env python
"""Stage-2 CPU scorer + Gate-2 adjudicator (caslake; no GPU -- session_start.md §4 routing).

Reads a completions JSON written by tools/stage2_bench.py, re-materializes the frozen subset
(asserting SUBSET_SHA256 via stage2_lcb.load_subset -> prereg.md §2), scores greedy pass@1 with the
frozen harness, and writes a score JSON. Baseline and every candidate are scored by this SAME code
on the SAME subset -> the head-to-head is controlled by construction.

Subcommands:
  score <gen.json> <out.json> [label]     score one model's completions -> pass@1
  adjudicate <cand.json> <base.json>       Gate-2: cand pass@1 - baseline pass@1 >= WIN_MARGIN?
  prereg-check                             anti-back-fit assertions (prereg.md §8) -- run pre-submit
"""
import concurrent.futures as cf
import datetime
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2_lcb as lcb  # noqa: E402

WIN_MARGIN_PTS = 3.0            # prereg.md §5 (absolute pass@1 percentage points)
BASELINE_KEY = "qwen2.5_coder_32B"
PREREG = "prompts/60_model_refresh/prereg.md"
FIRST_SUBMIT = "_scratch/first_scoring_submit.txt"
SCORE_WORKERS = int(os.environ.get("STAGE2_SCORE_WORKERS", "8"))


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------- score -------- #
def score(gen_path, out_path, label=None):
    gen = json.load(open(gen_path))
    if gen.get("subset_sha256") and gen["subset_sha256"] != lcb.SUBSET_SHA256:
        raise SystemExit(f"gen file subset_sha256 {gen['subset_sha256']} != frozen {lcb.SUBSET_SHA256}")
    completions = gen.get("completions") or {}
    problems = lcb.load_subset()                 # asserts SUBSET_SHA256 before any scoring
    served = gen.get("served_name")
    label = label or served
    # Integrity: the shard CONTENT the scorer sees must match what generation saw (adversary Strike 4;
    # the id-set SHA alone would not catch a test-content change). A mismatch is a §9 blocker.
    content_fp = lcb.subset_content_fingerprint(problems)
    if gen.get("subset_content_fingerprint") and gen["subset_content_fingerprint"] != content_fp:
        raise SystemExit(f"subset_content_fingerprint mismatch: gen {gen['subset_content_fingerprint']} "
                         f"!= score {content_fp} -- dataset content changed between gen and score.")
    missing = [p["question_id"] for p in problems if p["question_id"] not in completions]
    if missing:
        print(f"WARNING: {len(missing)} qids absent from completions (scored as unsolved): {missing[:5]}...")

    def _score(p):
        rec = completions.get(p["question_id"]) or {}
        code = lcb.extract_code(rec.get("content"))
        r = lcb.score_one(p, code)
        return p["question_id"], {
            "solved": r["solved"], "detail": r["detail"], "n_tests": r["n_tests"],
            "difficulty": p["difficulty"], "is_functional": p["is_functional"],
            "finish_reason": rec.get("finish_reason"), "gen_error": rec.get("error"),
        }

    per = {}
    with cf.ThreadPoolExecutor(max_workers=SCORE_WORKERS) as ex:
        for qid, rec in ex.map(_score, problems):
            per[qid] = rec
    solved = sum(1 for r in per.values() if r["solved"])
    n = len(problems)
    pass1 = solved / n
    by_diff = {}
    for r in per.values():
        d = r["difficulty"] or "unknown"
        by_diff.setdefault(d, [0, 0])
        by_diff[d][1] += 1
        by_diff[d][0] += 1 if r["solved"] else 0
    out = {
        "stage": "2-score", "label": label, "served_name": served,
        "n": n, "solved": solved, "pass_at_1": round(pass1, 6),
        "pass_at_1_pct": round(100 * pass1, 3),
        "by_difficulty": {d: {"solved": s, "n": t, "pct": round(100 * s / t, 1)} for d, (s, t) in by_diff.items()},
        "n_with_content": gen.get("n_with_content"),
        "n_not_generated": gen.get("n_not_generated", 0),
        "not_generated": gen.get("not_generated", []),
        "concurrency": gen.get("concurrency"),
        "subset_sha256": lcb.SUBSET_SHA256, "subset_content_fingerprint": content_fp,
        "decode": lcb.DECODE, "per_problem": per,
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"SCORE {label}: pass@1 = {solved}/{n} = {100*pass1:.2f}%  by_diff={out['by_difficulty']}")
    print(f"  wrote {out_path}")
    return 0


# ---------------------------------------------------------------- adjudicate ------ #
def adjudicate(cand_path, base_path):
    cand = json.load(open(cand_path))
    base = json.load(open(base_path))
    # Integrity: both sides must be scored on the SAME shard content (adversary Strike 4).
    cfp, bfp = cand.get("subset_content_fingerprint"), base.get("subset_content_fingerprint")
    if cfp and bfp and cfp != bfp:
        raise SystemExit(f"content-fingerprint mismatch cand {cfp} != base {bfp} -- not comparable.")
    # Completeness guard: a run with any NOT-GENERATED problem (infra/deadline, not a real empty
    # answer) makes the head-to-head silently unfair (adversary Strike 6). Such a run is INCOMPLETE
    # -> an infra-retry, NOT a decided gate. The gate is `win AND complete`.
    ng_c, ng_b = cand.get("n_not_generated", 0), base.get("n_not_generated", 0)
    complete = (ng_c == 0 and ng_b == 0)
    c, b = cand["pass_at_1_pct"], base["pass_at_1_pct"]
    delta = round(c - b, 3)
    win = delta >= WIN_MARGIN_PTS
    print(f"candidate {cand['label']}: {c:.2f}%  ({cand['solved']}/{cand['n']}); not_generated={ng_c}")
    print(f"baseline  {base['label']}: {b:.2f}%  ({base['solved']}/{base['n']}); not_generated={ng_b}")
    print(f"delta = {delta:+.2f} pts ; WIN_MARGIN = +{WIN_MARGIN_PTS} pts (prereg.md §5)")
    if not complete:
        print(f"RUN INCOMPLETE: {ng_c + ng_b} problem(s) never generated before the box -> INFRA-RETRY, "
              f"NOT a decided gate (adjudicate refuses to call Gate 2 on an incomplete run).")
    else:
        print(f"GATE 2: {'PASS (candidate wins the tier)' if win else 'FAIL (keep baseline -- pre-registered NO-GO)'}")
    manifest = {"stage": "2", "model": cand["label"], "metric": "pass_at_1_pct",
                "value": c, "baseline": b, "baseline_key": BASELINE_KEY,
                "delta_pts": delta, "win_margin_pts": WIN_MARGIN_PTS,
                "complete": complete, "n_not_generated_cand": ng_c, "n_not_generated_base": ng_b,
                "pass": bool(win and complete)}
    print("MANIFEST " + json.dumps(manifest))
    return 0


# --------------------------------------------------------------- prereg-check ----- #
def _git(root, *args):
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)


def _parse_iso(s):
    """Parse an ISO-8601 timestamp (with a numeric offset or 'Z') to an aware datetime. Comparing
    the RAW strings lexically is wrong when offsets differ (adversary Strike 3): '...T14:00-05:00'
    string-sorts before '...T18:00Z' yet is chronologically LATER. Parse to epoch and compare."""
    return datetime.datetime.fromisoformat(s.strip().replace("Z", "+00:00"))


def _verify_constants_vs_prereg(txt):
    """Mechanically assert the SCORING-DECISIVE constants in code EQUAL the frozen prereg.md, rather
    than trusting the module docstring's claim that a subagent checks it (adversary Strike 2). Any
    mismatch (or a prereg format the regexes cannot read) fails the gate."""
    checks = []

    def _add(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    m = re.search(r"SUBSET_SHA256 = ([0-9a-f]{64})", txt)
    _add("SUBSET_SHA256", m and m.group(1) == lcb.SUBSET_SHA256, m.group(1) if m else "not found")

    mj = re.search(r"```json\s*(\[.*?\])\s*```", txt, re.DOTALL)
    ids = json.loads(mj.group(1)) if mj else []
    _add("FROZEN_IDS (60, set-equal)", len(ids) == 60 and sorted(map(str, ids)) == sorted(lcb.FROZEN_IDS),
         f"{len(ids)} ids")

    def _num(pat):
        mm = re.search(pat, txt)
        return float(mm.group(1)) if mm else None
    _add("DECODE.temperature", _num(r"\|\s*temperature\s*\|\s*([0-9.]+)") == lcb.DECODE["temperature"])
    _add("DECODE.top_p", _num(r"\|\s*top_p\s*\|\s*([0-9.]+)") == lcb.DECODE["top_p"])
    _add("DECODE.n", _num(r"\|\s*n \(samples/problem\)\s*\|\s*([0-9]+)") == lcb.DECODE["n"])
    _add("DECODE.seed", _num(r"\|\s*seed\s*\|\s*([0-9]+)") == lcb.DECODE["seed"])
    _add("DECODE.max_tokens", _num(r"\|\s*max_tokens\s*\|\s*([0-9]+)") == lcb.DECODE["max_tokens"])
    et = bool(re.search(r"enable_thinking\"?\s*:\s*false", txt, re.I))
    _add("DECODE.enable_thinking:false", et and lcb.DECODE["chat_template_kwargs"]["enable_thinking"] is False)
    _add("WIN_MARGIN", _num(r"\+([0-9.]+) absolute percentage points") == WIN_MARGIN_PTS)
    _add("PER_TEST_TIMEOUT", _num(r"Per-test wall timeout ([0-9]+) s") == lcb.PER_TEST_TIMEOUT_S)
    return checks


def prereg_check():
    """Anti-back-fit assertions (prereg.md §8 / session_start.md §3). Run in the build allocation
    BEFORE submitting any Stage-2 generation job. Exit 0 iff ALL pass; any failure is a §9 blocker."""
    root = _repo_root()
    ok = True

    # (1) prereg.md exists with `^Status: PRE-RESULT` in the first 5 lines.
    pth = os.path.join(root, PREREG)
    head = open(pth).read().splitlines()[:5] if os.path.exists(pth) else []
    c1 = any(l.startswith("Status: PRE-RESULT") for l in head)
    print(f"(1) prereg.md PRE-RESULT header in first 5 lines: {'PASS' if c1 else 'FAIL'}")
    ok = ok and c1

    # (2) the commit that ADDED prereg.md precedes the first-scoring-submit timestamp -- compared as
    #     timezone-aware instants, NOT lexically (adversary Strike 3).
    add_dates = _git(root, "log", "--diff-filter=A", "--format=%cI", "--", PREREG).stdout.split()
    add_date = add_dates[-1].strip() if add_dates else ""
    fs = os.path.join(root, FIRST_SUBMIT)
    first_ts = open(fs).read().splitlines()[0].strip() if os.path.exists(fs) else ""
    try:
        c2 = bool(add_date) and bool(first_ts) and _parse_iso(add_date) < _parse_iso(first_ts)
    except ValueError as e:
        print(f"    timestamp parse error: {e}")
        c2 = False
    print(f"(2) prereg add-commit {add_date!r} precedes first-submit {first_ts!r} (tz-aware): {'PASS' if c2 else 'FAIL'}")
    ok = ok and c2

    # (3) prereg.md unmodified since its commit (working tree clean vs HEAD).
    c3 = _git(root, "diff", "--quiet", "HEAD", "--", PREREG).returncode == 0
    print(f"(3) prereg.md unmodified vs HEAD (git diff --quiet): {'PASS' if c3 else 'FAIL'}")
    ok = ok and c3

    # (4) the frozen subset re-materializes with the frozen SHA256.
    try:
        lcb.load_subset()
        c4 = True
    except SystemExit as e:
        print(e)
        c4 = False
    print(f"(4) subset re-materializes with frozen SUBSET_SHA256: {'PASS' if c4 else 'FAIL'}")
    ok = ok and c4

    # (5) the scoring-decisive CODE constants equal the frozen prereg.md (adversary Strike 2): the
    #     doc being frozen is worthless if the code the harness actually runs drifts from it.
    txt = open(pth).read() if os.path.exists(pth) else ""
    c5 = True
    for name, passed, detail in _verify_constants_vs_prereg(txt):
        if not passed:
            print(f"    (5) code!=prereg for {name}: {detail}")
        c5 = c5 and passed
    print(f"(5) code constants == frozen prereg.md (sha/ids/decode/margin/timeout): {'PASS' if c5 else 'FAIL'}")
    ok = ok and c5

    print("PREREG-CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv):
    if len(argv) >= 4 and argv[1] == "score":
        return score(argv[2], argv[3], argv[4] if len(argv) > 4 else None)
    if len(argv) == 4 and argv[1] == "adjudicate":
        return adjudicate(argv[2], argv[3])
    if len(argv) == 2 and argv[1] == "prereg-check":
        return prereg_check()
    print("usage: stage2_score.py score <gen.json> <out.json> [label]", file=sys.stderr)
    print("       stage2_score.py adjudicate <cand.json> <base.json>", file=sys.stderr)
    print("       stage2_score.py prereg-check", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
