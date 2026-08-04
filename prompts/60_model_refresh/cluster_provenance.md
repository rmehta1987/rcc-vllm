# Cluster GPU-driver provenance — model-refresh loop

Recorded 2026-08-03 21:26 UTC from two `srun` probes on the `test` partition (raw `srun`, not
`ai_session.py` — no SU floor). This is the gating fact for which serve env the loop can use.

## RESOLUTION (2026-08-03 22:40 UTC): prebuilt cu129 wheel serves — NO source build, NO driver upgrade

The driver blocker is resolved. vLLM publishes per-CUDA wheels at `https://wheels.vllm.ai/<ver>/<cuda>`;
`vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl` clears every wall: `manylinux_2_28` runs on
el8/glibc-2.28; `+cu129` (CUDA 12.9) is a CUDA-12.x minor version so it runs on driver 535 (only cu130/
CUDA-13 was the wall); built *against* cu129 so its `_C` loads `libcudart.so.12` (avoids the mismatch in
vllm #43435); vLLM 0.26.0 registers both new archs with its native transformers 5.14.1. **Serve env
`/project/rcc/mehta5/conda-envs/vllm-serve-cu129`** = vLLM 0.26.0+cu129 + torch 2.11.0+cu129 (CUDA 12.9)
+ transformers 5.14.1. Install recipe: `sbatch_build_vllm_cu129.sh` (build partition, ~7 min, no compile).

**PROVEN end-to-end on H200 driver 535 (jobs below):** (1) torch cu129 CUDA init + bf16 matmul; (2) full
vLLM serve of Qwen3-4B (TP=1) — FlashAttention v3 + FlashInfer sampling, real generation; (3) **Qwen3.5-
122B-A10B-FP8 (`qwen3_5_moe`, FP8/E4M3) at TP=2 on 2×H200 — loaded 122 GB in 114 s and generated code.**

Three env fixes any launcher of this env MUST reproduce (a bare `$ENV/bin/python` or `mamba activate`
alone is NOT enough):
1. `export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH` — GPU compute nodes put a system lib path ahead of
   the env's RUNPATH, so el8's `/lib64/libstdc++.so.6` (CXXABI≤1.3.11) gets picked over the env's 6.0.35
   (has CXXABI_1.3.15, required by ICU→sqlite3). Prepend the env lib so it wins (put it AFTER any module
   loads, since gcc/12.2.0's libstdc++ is also too old).
2. `module load cuda/12.8 gcc/12.2.0` — FlashInfer JIT-compiles its sampling kernels (sm_90a) on first use
   with nvcc + a host compiler; without a real CUDA_HOME torch defaults it to the anaconda base (no nvcc)
   and the JIT fails. Compiled kernels cache to `~/.cache/flashinfer/0.6.14/90a/` (home → shared, warms once).
3. `export VLLM_USE_DEEP_GEMM=0` and `export VLLM_MOE_USE_DEEP_GEMM=0` — DeepGEMM (JIT FP8 GEMM for Hopper)
   resolves CUDA DRIVER-API symbols by version via `cuGetProcAddress`; driver 535 (CUDA 12.2) lacks the
   newer-version symbol → "Failed to load CUDA driver API" assert during MoE profiling. Disabling forces
   the prebuilt CUTLASS FP8 MoE kernels (`_moe_C`, in the wheel). This is the ONE place the old driver
   genuinely bites; it costs some FP8-MoE throughput (recoverable only by a driver upgrade to ≥R580), not
   correctness. Serve-config note: Qwen3.5 has "thinking" on by default → use `--reasoning-parser`.

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
| 52984532 | build | — | midway3-0200 | prebuilt `vllm==0.26.0+cu129` + torch 2.11.0+cu129 installed (7 min, no compile); archs register; arch_dryrun PASS both models |
| 52984938 | test | H200 | midway3-0605 | Stage-1 PASS: torch cu129 matmul + full vLLM serve of Qwen3-4B (FlashAttn v3 + FlashInfer) on driver 535 (needed libstdc++ path + cuda/gcc modules) |
| 52985604 | test | H200 | 2× | Stage-2 PASS: Qwen3.5-122B-A10B-FP8 (qwen3_5_moe, FP8) TP=2 loaded in 114 s and generated code, with DeepGEMM disabled (CUTLASS FP8 fallback) |

Reproduce:
```
srun --partition=test --account=rcc-staff --constraint=H200 --gres=gpu:1 --time=00:05:00 \
  bash -lc 'nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader; \
    /project/rcc/mehta5/conda-envs/vllm-probe/bin/python -c "import torch;print(torch.__version__,torch.cuda.is_available())"'
```
