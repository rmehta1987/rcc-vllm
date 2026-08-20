# Frozen benchmark artifacts (moved out of gitignored `_scratch/`, 2026-08-19)

These are the only surviving records of measurements whose model weights have been
deleted from disk. `_scratch/` is gitignored; these files are not. Do not delete them.

## Why they matter

`_scratch/score_stage2.sbatch` reuses an existing baseline score file rather than
re-scoring (`if [ -s "$BASE_OUT" ]`). So a stored score JSON keeps a baseline usable
for adjudication even after its weights are gone. What a stored score file CANNOT do is
be regenerated, re-verified, or re-run under a new condition (different subset, decode,
or thinking mode) -- that needs the weights.

## Contents

| file | what it is |
| --- | --- |
| `stage2_score_bench-coder32b.json` | **Retiring baseline.** Qwen2.5-Coder-32B-Instruct, 26.67% (16/60). Weights DELETED 2026-08-19. This file is now the sole record. |
| `stage2_gen_bench-coder32b.json` | its raw 60 completions (job 53057094) |
| `stage2_score_bench-qwen38-27b.json` | **Incoming baseline.** Qwen3.8-27B, 50.00% (30/60), score job 53531932. |
| `stage2_gen_bench-qwen38-27b.json` | its raw 60 completions (job 53496745) |
| `stage2_score_bench-gemma4-31b.json` | **Gemma-4-31B-it, 66.67% (40/60)**, Gate-2 PASS, score job 53554777. |
| `stage2_gen_bench-gemma4-31b.json` | its raw 60 completions (gen 53546905 + tail resume 53554776) |
| `model_refresh_manifest.jsonl` | per-stage measured outcomes for the whole round |

## Frozen protocol these were measured under

LiveCodeBench `code_generation_lite@0fe84c39`, 60-problem subset
`subset_sha256=b3c2b753...7021b`, window 2025-03-08..2025-04-06, greedy
(temp 0, top_p 1, n 1, max_tokens 8192), `enable_thinking: false`, concurrency 1,
all models served on `vllm-serve-cu129` (vLLM 0.26.0). Win margin +3.0 pts.

## Re-baselining to Qwen3.8-27B -- READ BEFORE EDITING THE HARNESS

`tools/stage2_score.py:26` still reads `BASELINE_KEY = "qwen2.5_coder_32B"`, and
`prereg-check` verifies the harness's scoring-decisive constants against the FROZEN
`prompts/60_model_refresh/prereg.md`. Editing `BASELINE_KEY` in place makes
`prereg-check` fail -- by design; that gate exists to stop exactly this kind of
post-hoc edit. Promoting Qwen3.8-27B's 50.00% to the baseline requires a NEW
pre-registration for the next round, committed before the first scoring submit of
that round. Do not retrofit the existing one.

Note also that 50.00% was measured with thinking OFF, which is NOT this model's
default mode (`reasoning_effort: xhigh`). Any new prereg should state explicitly
which mode the baseline is measured in.
