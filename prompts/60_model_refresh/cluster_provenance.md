# Cluster GPU-driver provenance — model-refresh loop

Recorded 2026-08-03 21:26 UTC from two `srun` probes on the `test` partition (raw `srun`, not
`ai_session.py` — no SU floor). This is the gating fact for which serve env the loop can use.

## Driver / hardware (H200)

| Node | GPU | Driver | Driver max CUDA | Compute cap | HBM |
|---|---|---|---|---|---|
| midway3-0605 | NVIDIA H200 | **535.216.03** | **12.2** | 9.0 (Hopper) | 143771 MiB (~141 GB) |

## Env compatibility (measured on midway3-0605, driver 535.216.03)

| Env | vLLM | torch | CUDA | `torch.cuda.is_available()` | GPU matmul |
|---|---|---|---|---|---|
| `vllm-probe` (production) | 0.10.2 | 2.8.0+cu128 | 12.8 | **True** | OK |
| `vllm-qwen35` (loop, as built) | 0.26.0 | 2.11.0+cu130 | 13.0 | **False** — "driver too old (found 12020)" | — |

## Implication (load-bearing for the loop)

The `vllm-qwen35` env (vLLM 0.26.0 → torch 2.11/**cu130**/CUDA 13.0) **cannot serve on the current
cluster driver.** CUDA 13 is a *major* version bump; driver 535.216.03 caps at CUDA 12.2. CUDA-12.x
torch works on this driver via minor-version compatibility (cu128 proven above), but CUDA 13 does not
(needs driver ~R580+). The loop's Stage-1 pre-flight (`torch.cuda.is_available()` before the full load)
would therefore correctly NO-GO **every** GPU serve on `vllm-qwen35`.

**Required before any Stage-1/Stage-2 GPU serve (OPERATOR-DECISION-PENDING):** either
1. **Rebuild the serve env on a CUDA-12.x vLLM** — a version that BOTH registers `qwen3_5_moe` +
   `glm_moe_dsa` (≥ 0.17.0, see [[project_h200_staging]]) AND ships a `cu128`/`cu129` torch (not
   `cu130`). vLLM ~0.17–0.23 predates the torch-2.11/cu130 bump; pin the newest such version and
   re-run `tools/arch_dryrun.py` to confirm the archs still register. `vllm-qwen35` (0.26.0) stays only
   as the offline arch-resolution reference. OR
2. **Request an RCC driver upgrade to ≥ R580** on the H200/A100 test nodes so CUDA 13 (and vLLM 0.26.0)
   runs — a cluster-admin action, not self-serviceable.

NOTE: only the H200 node was probed; the A100 test nodes are assumed to carry the same 535.x image
(cluster-uniform) but this should be confirmed with the same probe before a Tier-A serve.

## Why a plain CUDA-12.x pip rebuild is NOT possible here (measured 2026-08-03)

The whole cluster is **el8 / glibc 2.28** (verified on the build node midway3-0200 AND the H200 node
midway3-0603 — both `ldd (GNU libc) 2.28`). vLLM's wheel platform tags vs glibc:

| vLLM | torch / CUDA (default) | wheel tag | runs on glibc 2.28? |
|---|---|---|---|
| 0.17.0–0.19.0 | 2.10.0 / **cu128 (12.8)** ✓ driver-OK | `manylinux_2_31` | **NO** (needs glibc ≥2.31) |
| 0.20.0 | 2.11.0 / cu130 | `manylinux_2_35` | no |
| 0.26.0 | 2.11.0 / **cu130 (13.0)** ✗ driver-fails | `manylinux_2_28` | yes (but CUDA-13) |

So the arch-capable **CUDA-12.x** versions ship only **glibc-2.31** wheels that cannot run on this el8
cluster; a normal `pip install` of 0.17–0.19 on the el8 build node rejects the wheel and falls back to a
source build, which needs `nvcc` (absent from base PATH) → the 2026-08-03 build job 52974060 FAILED here.
The only el8-runnable recent wheel (0.26.0) is CUDA-13 and fails on driver 535. **The two remaining paths:**
1. **Source-build vLLM 0.19.0 against CUDA 12.8** — feasible (modules `cuda/12.8` + `gcc/12.2.0` exist;
   pre-install `torch==2.10.0` cu128), but heavy (~30–60 min compile, failure-prone).
2. **RCC driver upgrade to ≥ R580** — then the already-built `vllm-qwen35` (0.26.0, el8 `manylinux_2_28`
   wheel) runs as-is with zero rebuild, and every future (CUDA-13) vLLM works too. Cluster-admin action.

## Provenance ledger (for verdict.md)

| Job ID | Partition | Constraint | Node | What it proves |
|---|---|---|---|---|
| 52973706 | test | H200 | midway3-0605 | driver 535.216.03 / CUDA 12.2; cu130 fails to init |
| 52973737 | test | H200 | midway3-0605 | cu128 (production torch) inits + matmuls; cu130 does not |
| 52974060 | build | — | midway3-0200 | vLLM 0.19.0 pip install FAILED: manylinux_2_31 wheel rejected on glibc 2.28 → source build → no nvcc |
| (srun) | test | H200 | midway3-0603 | H200 node is glibc 2.28 (manylinux_2_31 wheels can't run cluster-wide) |

Reproduce:
```
srun --partition=test --account=rcc-staff --constraint=H200 --gres=gpu:1 --time=00:05:00 \
  bash -lc 'nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader; \
    /project/rcc/mehta5/conda-envs/vllm-probe/bin/python -c "import torch;print(torch.__version__,torch.cuda.is_available())"'
```
