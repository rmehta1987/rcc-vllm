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
import json
import os
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
        "subset_sha256": lcb.SUBSET_SHA256, "decode": lcb.DECODE, "per_problem": per,
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
    c, b = cand["pass_at_1_pct"], base["pass_at_1_pct"]
    delta = round(c - b, 3)
    win = delta >= WIN_MARGIN_PTS
    print(f"candidate {cand['label']}: {c:.2f}%  ({cand['solved']}/{cand['n']})")
    print(f"baseline  {base['label']}: {b:.2f}%  ({base['solved']}/{base['n']})")
    print(f"delta = {delta:+.2f} pts ; WIN_MARGIN = +{WIN_MARGIN_PTS} pts (prereg.md §5)")
    print(f"GATE 2: {'PASS (candidate wins the tier)' if win else 'FAIL (keep baseline -- pre-registered NO-GO)'}")
    manifest = {"stage": "2", "model": cand["label"], "metric": "pass_at_1_pct",
                "value": c, "baseline": b, "baseline_key": BASELINE_KEY,
                "delta_pts": delta, "win_margin_pts": WIN_MARGIN_PTS, "pass": win}
    print("MANIFEST " + json.dumps(manifest))
    return 0 if True else 1   # adjudication itself always succeeds; the gate value is `win`


# --------------------------------------------------------------- prereg-check ----- #
def _git(root, *args):
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)


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

    # (2) the commit that ADDED prereg.md precedes the first-scoring-submit timestamp.
    add_dates = _git(root, "log", "--diff-filter=A", "--format=%cI", "--", PREREG).stdout.split()
    add_date = add_dates[-1].strip() if add_dates else ""
    fs = os.path.join(root, FIRST_SUBMIT)
    first_ts = open(fs).read().splitlines()[0].strip() if os.path.exists(fs) else ""
    c2 = bool(add_date) and bool(first_ts) and add_date < first_ts
    print(f"(2) prereg add-commit {add_date!r} precedes first-submit {first_ts!r}: {'PASS' if c2 else 'FAIL'}")
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
