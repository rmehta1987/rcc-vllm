# Pre-registration of Stage-2 constants — coding-model refresh

Status: PRE-RESULT. No Stage-1 or Stage-2 gate has scored at the time this file is committed. This file is
written by the constants pilot (`session_start.md` §3; loop prompt Phase 0 orient) and
committed as its OWN commit before the first Stage-1/Stage-2 `mrefresh-nest` job is submitted. The gate formulas
and fail-branches live in `session_start.md`; this file fixes only the numeric/textual constants those gates
consume. Nothing here is edited after a gate scores (`git diff --quiet HEAD -- prompts/60_model_refresh/prereg.md`
is asserted before every scoring run).

Authored 2026-08-04 in the `build` allocation (the only allocation with internet). The LiveCodeBench data is
fetched and cached HERE, because Stage-2 scoring runs on `caslake`, which has no internet.

## 1. LCB_VERSION — dataset and exact revision

- Dataset: `livecodebench/code_generation_lite` (HuggingFace), the gzip-compressed "lite" code-generation set.
- Pinned revision (commit sha, stronger than the `release_v6` name — the repo carries no git tags):
  `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`.
- The problems live in six JSONL shards (`test.jsonl` .. `test6.jsonl`); `test6.jsonl` is the release_v6
  increment (the most recent contest window). Only `test6.jsonl` is needed for the pinned subset below.
- Cached (offline-resolvable) at:
  `/project/rcc/mehta5/hf-cache/hub/datasets--livecodebench--code_generation_lite/snapshots/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test6.jsonl`
  (134,303,240 bytes).
- Offline load recipe for the `caslake` scorer (no internet — resolves from the cache above):

  ```bash
  export HF_HOME=/project/rcc/mehta5/hf-cache HF_HUB_OFFLINE=1
  python -c 'from huggingface_hub import hf_hub_download; \
    print(hf_hub_download("livecodebench/code_generation_lite","test6.jsonl", \
    repo_type="dataset", revision="0fe84c3912ea0c4d4a78037083943e8f0c4dd505"))'
  ```

## 2. LCB_SUBSET — the pinned 60 problems

Selection rule (deterministic, fixed before any score): load every problem in `test6.jsonl` at the revision
above (175 problems, contest dates 2025-01-04 .. 2025-04-06), sort by `(contest_date, question_id)` ascending,
and take the 60 with the LATEST contest dates (the tail of that sort). This is the freshest available LCB window,
2025-03-08 .. 2025-04-06 — entirely AFTER the baseline `qwen2.5_coder_32B` (Qwen2.5-Coder-32B-Instruct,
late-2024) training cutoff, so the baseline is not contamination-advantaged on it (see §9).

- Count: 60. Composition (of the selected 60): platforms atcoder + leetcode; difficulty mix as materialized
  from the pinned data. The parent shard `test6.jsonl` is 43 easy / 52 medium / 80 hard across 175 problems.
- Integrity: `SUBSET_SHA256 = b3c2b75377c305f495de834ec699bc07d70b6b507230fe16c8d73bf550b7021b`, computed as
  `sha256("\n".join(sorted(question_ids)))`. The Stage-2 scorer re-materializes the subset from the cached shard
  and asserts this hash BEFORE the first scoring call; a mismatch is a §9 blocker (never a silent re-pin).
- The 60 `question_id`s (canonical order = sorted ascending; this is the frozen list, the source of truth):

  ```json
  ["3717","3744","3750","3759","3765","3773","3777","3784","3788","3789","3791","3793","3794","3795","3799","3801","3805","3809","3811","3817","3832","abc397_a","abc397_b","abc397_c","abc397_d","abc397_e","abc397_f","abc397_g","abc398_a","abc398_b","abc398_c","abc398_d","abc398_f","abc398_g","abc399_a","abc399_b","abc399_c","abc399_d","abc399_e","abc399_f","abc400_a","abc400_b","abc400_c","abc400_d","abc400_e","abc400_g","arc194_a","arc194_b","arc194_c","arc194_d","arc194_e","arc195_a","arc195_b","arc195_c","arc195_d","arc195_e","arc196_a","arc196_b","arc196_c","arc196_d"]
  ```

## 3. DECODE — one uniform policy for the baseline and every candidate

The fence (`session_start.md` §3) is that decode is IDENTICAL across baseline and candidate so an env/decode
mismatch cannot silently game Gate 2. This is ONE policy applied uniformly; models differ only in weights.

| Knob | Value |
| --- | --- |
| temperature | 0.0 (greedy — deterministic pass@1, no sampling-seed dependence) |
| top_p | 1.0 |
| n (samples/problem) | 1 |
| seed | 0 |
| max_tokens | 8192 |
| chat_template_kwargs | `{"enable_thinking": false}` requested uniformly |
| response field scored | `content` (never `reasoning`) |

Notes, applied identically to all models:
- `enable_thinking:false` is REQUESTED of every model. `qwen2.5_coder_32B` and `Qwen3-Coder-30B-A3B-Instruct` are
  non-thinking (the flag is a no-op). `Qwen3.5-122B` honors it (emits no `<think>`). If a model ignores the flag
  and reasons anyway (a possibility for `DeepSeek-V4-Flash`), the scorer still reads only the final `content`;
  the sampling knobs above are unchanged, so no per-model decode difference is introduced. Any such intrinsic
  reasoning is disclosed in `verdict.md`.
- `serve_cu129.sbatch` adds `--reasoning-parser qwen3` for `Qwen3*` model dirs, which routes any CoT to
  `reasoning` and leaves the answer in `content`; the scorer reads `content`, so the parser presence is neutral.
- max_tokens 8192 is generous for a code answer (correct solutions are typically < 1000 tokens) and bounds
  the time box; a completion that hits 8192 with `content=null`/`finish_reason=length` counts as unsolved
  (uniform rule, not a per-model exception).

## 4. Benchmark prompt template and scoring (pass@1)

Prompt (uniform across all models). System message:

> You are an expert Python programmer. You will be given a competitive programming problem. Provide a correct,
> self-contained Python solution.

User message, starter-code (LeetCode-style, `starter_code` non-empty):

> ### Question:\n{question_content}\n\n### Starter code:\n```python\n{starter_code}\n```\n\n### Instructions:\n
> Complete the solution. Respond with a single self-contained Python code block that defines the required
> class/function. Do not include explanations.

User message, stdin (AtCoder-style, empty `starter_code`):

> ### Question:\n{question_content}\n\n### Instructions:\nWrite a complete Python program that reads from
> standard input and writes the answer to standard output. Respond with a single self-contained Python code
> block. Do not include explanations.

Scoring (LCB-faithful, run on `caslake`, offline):
- Extract the LAST fenced code block (```python ... ``` or ``` ... ```); if none, use the raw `content`.
- starter-code problems: call-based harness — construct `Solution`, invoke the target method per test input,
  deep-compare the returned value to the expected output.
- stdin problems: run the program with the test input on stdin, compare stdout to expected with trailing-
  whitespace/newline normalization.
- Per-test wall timeout 6 s; a runtime error, timeout, or any wrong answer fails the problem.
- A problem is SOLVED iff its single greedy completion passes ALL provided public + private test cases.
- `pass@1 = solved / 60`.
- Deviation from the LCB leaderboard protocol (n=10, temperature 0.2, sampled pass@1) is deliberate: greedy n=1
  gives a deterministic, time-boxable point estimate for a same-decode head-to-head. Applied identically to all
  models, so the COMPARISON is controlled; the absolute number is a greedy pass@1, not the sampled leaderboard
  figure. Disclosed in `verdict.md`.

## 5. WIN_MARGIN — the Gate-2 decision rule

- A candidate WINS its tier iff its measured greedy pass@1 exceeds the baseline `qwen2.5_coder_32B` (measured on
  the SAME subset/decode/harness/env) by at least **+3.0 absolute percentage points**. On 60 problems, 3.0 pts
  = 1.8 problems, so the candidate must solve at least 2 more problems than the baseline.
- If no candidate at a tier clears +3.0, the pre-registered fail-branch applies: keep the baseline; record the
  measured shortfall as a valid NO-GO (`session_start.md` §2 Stage 2). Never lower this margin after seeing a
  number.
- Honest statistical caveat: on n=60 a 3-point gap is within binomial sampling noise (SE ≈ 6 pts near p=0.4).
  +3.0 is a SERVICE-DECISION threshold ("meaningfully better before we re-point production"), not a
  significance claim. An operator wanting significance can commission a larger re-run; that is out of loop scope.

## 6. SMOKE_PROMPT — the Gate-1 accept check (text-only)

Gate 1 proves the server reaches ready and returns a coherent, compilable, correct answer to a fixed text-only
code prompt within the smoke time box. One prompt, same decode policy as §3.

System message: `You are an expert Python programmer. Provide correct, self-contained code.`
User message:

> Write a Python function with the exact signature `def two_sum(nums: list[int], target: int) -> list[int]:`
> that returns the indices of the two numbers in `nums` that add up to `target`. Exactly one solution exists and
> the same element may not be used twice. Respond with only a single Python code block containing the function.

Accept check (deterministic): extract the last fenced code block, `exec` it in a fresh namespace, then require
ALL of:
- `sorted(two_sum([2,7,11,15], 9)) == [0,1]`
- `sorted(two_sum([3,2,4], 6)) == [1,2]`
- `sorted(two_sum([3,3], 6)) == [0,1]`

Any exec error, missing `two_sum`, or failed assertion is a Gate-1 accept FAIL. two_sum is a canonical,
unambiguous task, so a serving coding model failing it is a real serve/decode fault, not prompt ambiguity.

## 7. Time boxes

| Job | Wall clock |
| --- | --- |
| Stage-1 serve smoke (`mrefresh-nest-stage1`) | `00:30:00` |
| Stage-2 benchmark, per model (`mrefresh-nest-stage2`) | `02:00:00` |
| Stage-D download | in the `build` allocation, monitored background (no nested job) |
| Stage-2 scoring (`mrefresh-nest-score`, `caslake`, CPU) | right-sized `caslake` nested job |

## 8. Anti-back-fit protocol (enforced before every Stage-1/Stage-2 scoring run)

Assert, in order (`session_start.md` §3); any failure is a §9 blocker, never a post-hoc edit of this file:
1. `prompts/60_model_refresh/prereg.md` exists and a line matching `^Status: PRE-RESULT` appears in its header
   (within the first 5 lines) — checked with `grep`, not by assuming it is the first non-empty line (line 1 is the H1 title).
2. The commit that introduced this file PRECEDES the first Stage-1/Stage-2 `mrefresh-nest` submit timestamp
   recorded in `_scratch/first_scoring_submit.txt` (that timestamp is written only AFTER this commit lands;
   `--is-ancestor` of HEAD is NOT sufficient and is not relied upon).
3. `git diff --quiet HEAD -- prompts/60_model_refresh/prereg.md` (unmodified since its commit).
4. The Stage-2 scorer re-materializes the subset from the cached shard and asserts `SUBSET_SHA256` (§2).

## 9. Known limitations, disclosed up front

- Contamination direction: the 2025-03/04 window post-dates the baseline's cutoff (fair to the baseline) but
  may pre-date the 2026 candidates' training cutoffs, so a candidate's score could be contamination-aided. This
  is unavoidable (no public LCB window post-dates 2026 cutoffs) and is conservative for a "should we serve the
  newer model" decision; the same problems/decode are used for all models, and WIN_MARGIN guards a marginal edge.
- n=60 greedy is a point estimate, not the sampled LCB leaderboard number (§4, §5).
- Difficulty skews hard in this window (parent shard is 80/175 hard); pass rates may be low for all models. The
  window is the honest freshest set and is not re-picked to engineer difficulty.
