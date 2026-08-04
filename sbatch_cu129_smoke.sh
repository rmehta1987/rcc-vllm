#!/bin/bash
#SBATCH --job-name=cu129-smoke
#SBATCH --partition=test
#SBATCH --account=rcc-staff
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/project/rcc/mehta5/vllm/tools/.cu129-smoke-%j.log

# Stage-1 GPU smoke test for the prebuilt cu129 serve env. Isolates ONE variable:
# do vLLM 0.26.0+cu129's compiled kernels dlopen and EXECUTE on H200 driver 535.216.03
# (max CUDA 12.2)? torch cu128 is already proven on this driver; cu129 vLLM kernels are
# the new unknown. A tiny model (Qwen3-4B, BF16) is enough -- size is irrelevant to the
# cu129/driver question; we only need vLLM to load its CUDA ops and generate. enforce_eager
# skips CUDA-graph/torch.compile so this is the cleanest possible kernel-load probe.
# Raw sbatch, generic job name -> NOT floor-billed, NOT seen by production discovery.
# FP8 kernels (needed by Qwen3.5-122B) are validated separately in the TP=2 122B load.

set -u
ENV=/project/rcc/mehta5/conda-envs/vllm-serve-cu129
PY="$ENV/bin/python"
export HF_HOME=/project/rcc/mehta5/hf-cache
export VLLM_LOGGING_LEVEL=INFO

# Toolkit for FlashInfer's runtime JIT: vLLM's bundled flashinfer compiles sampling kernels
# (sm_90a) on first use with nvcc + a host compiler and caches them under ~/.cache/flashinfer
# (home -> shared across nodes, so this warms once). Without a real CUDA_HOME, torch defaults it
# to the anaconda base (no nvcc) and the JIT fails. cuda/12.8 nvcc compiling for a cu129 runtime
# is fine (CUDA 12.x forward/backward compat); gcc/12.2.0 is a supported host compiler.
module load cuda/12.8 gcc/12.2.0
: "${CUDA_HOME:=$(dirname "$(dirname "$(command -v nvcc)")")}"
export CUDA_HOME

# libstdc++ ordering: the env ships 6.0.35 (has CXXABI_1.3.15, needed by ICU); el8's /lib64 and
# even gcc/12.2.0 are too old. Prepend the env's lib LAST so it wins over the module paths. Any
# launcher that runs this env directly (ai_session / the loop) must reproduce these two exports.
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"

echo "=== toolchain: nvcc=$(command -v nvcc) CUDA_HOME=$CUDA_HOME ==="
nvcc --version 2>&1 | tail -1

echo "=== $(date '+%F %T') node $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader

echo "=== step 1: torch cu129 CUDA init + bf16 matmul on driver 535 ==="
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| avail", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul sum =", (x @ x).sum().item())
print("TORCH_CU129_OK")
PY
echo "torch_rc=$?"

echo "=== step 2: vLLM cu129 kernels dlopen + end-to-end generate (Qwen3-4B, TP=1, eager) ==="
"$PY" - <<'PY'
from vllm import LLM, SamplingParams
llm = LLM(model="/project/rcc/mehta5/vllm/models/Qwen3-4B",
          tensor_parallel_size=1, gpu_memory_utilization=0.40,
          max_model_len=2048, enforce_eager=True, trust_remote_code=True)
out = llm.generate(["Write a Python function that returns the nth Fibonacci number:"],
                   SamplingParams(max_tokens=48, temperature=0))
print("GEN:", repr(out[0].outputs[0].text[:300]))
print("VLLM_CU129_SMOKE_PASS")
PY
echo "vllm_rc=$?"
echo "=== $(date '+%F %T') smoke done ==="
