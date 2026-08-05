# Stage 1 (GPU serve smoke) — candidate accept-checks

Recorded 2026-08-04. Each Stage-1 job loads one candidate on its tier's GPU under the
SERVE DISCIPLINE (`session_start.md §2`) and runs the frozen Gate-1 accept-check
(`prereg.md §6`): the server must reach ready inside the `00:30:00` box and return a
compilable, correct answer to the fixed `two_sum` text-only prompt. The serve env is
`vllm-serve-cu129` (`vllm==0.26.0+cu129`, torch 2.11.0+cu129, driver-535-compatible;
`cluster_provenance.md`), submitted via `tools/serve_cu129.sbatch` — never
`launch_ai_session.sh` (which forces the 0.10.2 `vllm-probe` env and a floor-billed
`<key>:<port>` job name). Every serve names its job `mrefresh-nest-stage1` and its
served-model-name `bench-*` (not a `MODEL_REGISTRY` key), so `billing_sweep.py::model_key_of`
never bills it and `server.py::_discover_servers_from_squeue` never surfaces it to
production clients.

Anti-back-fit (asserted before this scoring run, `prereg.md §8`): `prereg.md` carries
`Status: PRE-RESULT`; its introducing commit `2bb927c` (2026-08-04 11:17 CDT) and doc-fix
`177bc4e` (11:28 CDT) both precede the first Stage-1 submit timestamp
`2026-08-04T18:05:56Z` recorded in `_scratch/first_scoring_submit.txt`; and
`git diff --quiet HEAD -- prompts/60_model_refresh/prereg.md` is clean.

## Gate 1 — Qwen3-Coder-30B-A3B-Instruct (Tier A, BF16, TP=2): PASS

Job `53041215` (`--partition test --account rcc-staff --constraint "A100|a100" --gres=gpu:2`,
`--time=00:30:00`), node midway3-0294, served `bench-qwen3coder30b` from
`models/Qwen3-Coder-30B-A3B-Instruct` on port 8411. vLLM 0.26.0, `quantization=None`
(BF16), `tensor_parallel_size=2`, `max_model_len=16384`, `enforce_eager`,
`--reasoning-parser qwen3`.

| Measurement | Value |
| --- | --- |
| GPUs | 2× NVIDIA A100-PCIE-40GB (cc 8.0, driver 535.216.03) |
| Pre-flight | `cuda_available=True`; bf16 matvec OK; `Qwen3MoeForCausalLM` SUPPORTED; quant `none` (no FP8 cc-floor) |
| Weight load | 28.51 GiB / 23.72 s per worker |
| Engine warmup (profile + KV cache + warmup) | 15.74 s |
| Time to ready | serve start 18:28:55 → `Application startup complete` ≈18:30:21 (≈86 s) |
| Accept-check HTTP | 200, `finish_reason=stop` |
| `two_sum` accept | 3/3 cases pass (`[2,7,11,15],9→[0,1]`; `[3,2,4],6→[1,2]`; `[3,3],6→[0,1]`) |
| Verdict | `RESULT stage1 bench-qwen3coder30b : PASS - ok`; `accept-check rc=0` |

The accept-check (`tools/stage1_smoke.py::_accept`) extracts the last fenced code block,
`exec`s it in a fresh namespace, and requires all three `sorted(two_sum(...))` equalities
under the frozen DECODE (`prereg.md §3`: greedy t0 / top_p1 / n1 / seed0 / max_tokens 8192,
`enable_thinking:false`, response `content` read — never `reasoning`). The model returned
the canonical hash-map solution. Run log: `tools/.mrefresh-serve-53041215.log`.

Honest note on hardware: `--constraint "A100|a100"` landed a 40 GB PCIE A100 pair, not the
80 GB SXM A100 of the Tier-A production target. The 30.5B BF16 weights load 28.51 GiB per
GPU and serve at 16384 context on 40 GB — a STRICTER memory test than the 80 GB target, so
a clean serve here bounds the 80 GB case. Stage 2 requests the same constraint; Stage 3
pins the production node type.

Infra history: the first Tier-A attempt (job `53030556`) failed the accept-check with HTTP
400 "maximum context length" — the frozen `max_tokens=8192` left no prompt room in an
8192-token context window. Fixed in `1e68c20` (`serve_cu129.sbatch` `MAX_MODEL_LEN=16384`,
scoring-neutral: `max_tokens` unchanged, greedy output identical). This was an infra-retry
(2/5, `_scratch/stage1_tiera_infra_retries`), not a Gate-1 FAIL.

## Tier B — pending

`Qwen3.5-122B-A10B-FP8` (H200, FP8, TP measured) and `deepseek-ai/DeepSeek-V4-Flash` (H200,
FP8 dense + MXFP4 experts) have not yet run Stage 1. Both cleared Gate 0
(`stage0_preflight.md`); V4-Flash carries the FP4-on-Hopper quant pre-flight that fails
cheap before the full load if MXFP4/FlashMLA are Blackwell-only on this cluster
(`session_start.md §2` Stage 1). Recorded here when they land.

## Compute provenance (this stage)

| Job | Partition / constraint | Time box | GPUs | Proves |
| --- | --- | --- | --- | --- |
| 53030556 | test / A100 | 00:30:00 | 2× A100 | infra-fail: HTTP 400 ctx-overflow (fixed `1e68c20`); no gate FAIL, no floor-bill |
| 53041215 | test / A100 | 00:30:00 | 2× A100-PCIE-40GB | Gate 1 Tier A PASS: serves + `two_sum` 3/3 |

Both jobs served `bench-qwen3coder30b` (served-name `bench-*`, not a registry key) on
`vllm-serve-cu129`; neither is billable by `billing_sweep.py::model_key_of` (job name
`mrefresh-nest-stage1`, not `<registry-key>:<port>`), and neither disturbs production. No
`mrefresh-nest*` job remains in `squeue`.
