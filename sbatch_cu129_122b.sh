#!/bin/bash
#SBATCH --job-name=cu129-122b-smoke
#SBATCH --partition=test
#SBATCH --account=rcc-staff
#SBATCH --constraint=H200
#SBATCH --gres=gpu:2
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=/project/rcc/mehta5/vllm/tools/.cu129-122b-%j.log

# Stage-2 GPU smoke: the REAL target. Qwen3.5-122B-A10B-FP8 (arch Qwen3_5MoeForConditionalGeneration)
# at TP=2 on 2x H200. Exercises everything the 4B BF16 smoke did NOT:
#   - the novel qwen3_5_moe (VLM-MoE) arch loading real weights,
#   - FP8 (E4M3) tensor-core kernels on Hopper sm_90a,
#   - tensor-parallel across 2 GPUs (122 GB FP8 weights -> ~61 GB/GPU, fits 2x141 GB).
# Raw sbatch, generic job name -> NOT floor-billed, NOT seen by production discovery. Self-terminates.

set -u
ENV=/project/rcc/mehta5/conda-envs/vllm-serve-cu129
PY="$ENV/bin/python"
MODEL=/project/rcc/mehta5/vllm/models/Qwen3.5-122B-A10B-FP8
export HF_HOME=/project/rcc/mehta5/hf-cache
export VLLM_LOGGING_LEVEL=INFO

# DeepGEMM (vLLM's JIT FP8 GEMM for Hopper) resolves CUDA DRIVER-API symbols by version via
# cuGetProcAddress; driver 535 (CUDA 12.2) doesn't provide the newer-version symbol it requests
# -> "Failed to load CUDA driver API" assert in the MoE profiling pass. Disable both DeepGEMM
# paths so vLLM uses the prebuilt CUTLASS FP8 MoE kernels (_moe_C, shipped in the wheel, no
# driver-API JIT). This is a performance-optimization fallback, not a correctness compromise.
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0

# Same env fixes proven in Stage-1: real CUDA toolkit for FlashInfer JIT, and the env's libstdc++
# ahead of the module/system ones (ICU needs CXXABI_1.3.15). FlashInfer sm_90a cache is already
# warm from Stage-1, so no first-run compile is expected here.
module load cuda/12.8 gcc/12.2.0
: "${CUDA_HOME:=$(dirname "$(dirname "$(command -v nvcc)")")}"
export CUDA_HOME
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"

echo "=== $(date '+%F %T') node $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

echo "=== load Qwen3.5-122B FP8, TP=2, and generate ==="
# Real file (not stdin heredoc): TP=2 uses spawn, which re-imports the main module by path.
"$PY" /project/rcc/mehta5/vllm/tools/cu129_122b_test.py "$MODEL" 2
echo "vllm_rc=$?"
echo "=== $(date '+%F %T') 122b smoke done ==="
