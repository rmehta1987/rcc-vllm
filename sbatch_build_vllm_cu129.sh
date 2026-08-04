#!/bin/bash
#SBATCH --job-name=build-vllm-cu129
#SBATCH --partition=build
#SBATCH --account=rcc-staff
#SBATCH --qos=build
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/project/rcc/mehta5/vllm/tools/.build-vllm-cu129-%j.log

# The SERVE env for the model-refresh loop, built from PREBUILT wheels -- no source compile.
#
# vLLM publishes per-CUDA wheels at https://wheels.vllm.ai/<version>/<cuda>. For 0.26.0 the CUDA-12.x
# option is cu129 (CUDA 12.9): vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl. This clears every
# wall at once:
#   - manylinux_2_28  -> runs on this el8 / glibc-2.28 cluster (the glibc wall that blocked 0.17-0.19)
#   - +cu129 (CUDA 12.9) -> runs on H200 driver 535.216.03 (max CUDA 12.2) via CUDA-12.x minor-version
#       compatibility, same as the production cu128 torch. Only cu130 (CUDA 13, a MAJOR bump) fails.
#   - built AGAINST cu129 -> its _C.so loads libcudart.so.12, not .so.13 (avoids the mismatch import
#       error seen when a PyPI cu130 vLLM is paired with a cu12.x torch, vllm issue #43435).
#   - vLLM 0.26.0 -> registers Qwen3_5MoeForConditionalGeneration + GlmMoeDsaForCausalLM with its native
#       transformers 5.x (arch resolution already proven offline by tools/arch_dryrun.py).
#
# Production env (conda-envs/vllm-probe, 0.10.2) is left untouched so its version-pinned rate_table
# stays valid. GPU load is verified separately by an srun H200 smoke test (cu129 init + real model load).

set -u
ENV=/project/rcc/mehta5/conda-envs/vllm-serve-cu129
REPO=/project/rcc/mehta5/vllm
TORCH_SPEC="torch==2.11.0+cu129"
TORCH_INDEX="https://download.pytorch.org/whl/cu129"
VLLM_SPEC="vllm==0.26.0+cu129"
VLLM_INDEX="https://wheels.vllm.ai/0.26.0/cu129"

export PIP_CACHE_DIR=/project/rcc/mehta5/.pip-cache
export HF_HOME=/project/rcc/mehta5/hf-cache
mkdir -p "$PIP_CACHE_DIR"

echo "=== $(date '+%F %T') host $(hostname) : prebuilt cu129 wheel install (NO compile) ==="
module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
[ -x "$ENV/bin/python" ] || mamba create -p "$ENV" python=3.12 -y || { echo "env create FAILED"; exit 1; }
PY="$ENV/bin/python"; PIP="$ENV/bin/pip"
"$PIP" install --upgrade pip

echo "=== $(date '+%F %T') 1/2 $TORCH_SPEC (CUDA 12.9, el8 manylinux_2_28) ==="
"$PIP" install "$TORCH_SPEC" --index-url "$TORCH_INDEX" || { echo "torch install FAILED"; exit 1; }

echo "=== $(date '+%F %T') 2/2 $VLLM_SPEC (prebuilt, arch-capable) ==="
rc=1
for attempt in 1 2 3; do
  "$PIP" install "$VLLM_SPEC" --extra-index-url "$VLLM_INDEX" --extra-index-url "$TORCH_INDEX"
  rc=$?; [ "$rc" -eq 0 ] && break
  echo "attempt $attempt failed rc=$rc; retry in 30s"; sleep 30
done
[ "$rc" -eq 0 ] || { echo "vllm install FAILED rc=$rc"; exit 1; }

echo "=== $(date '+%F %T') guarantee torch stayed cu129 (vLLM dep resolution can tug it) ==="
"$PIP" install --no-deps --force-reinstall "$TORCH_SPEC" --index-url "$TORCH_INDEX"

echo "=== $(date '+%F %T') versions + arch registration ==="
"$PY" - <<'PY'
import vllm, transformers, torch
print("vllm        :", vllm.__version__)
print("transformers:", transformers.__version__)
print("torch       :", torch.__version__, "| torch.version.cuda:", torch.version.cuda)
cu = torch.version.cuda or ""
print("torch CUDA-12.x?:", "YES" if cu.startswith("12.") else f"NO ({cu}) -- WOULD FAIL ON DRIVER 535")
from vllm import ModelRegistry
a = set(ModelRegistry.get_supported_archs())
for k in ["Qwen3_5MoeForConditionalGeneration", "GlmMoeDsaForCausalLM", "Qwen3MoeForCausalLM", "DeepseekV4ForCausalLM"]:
    print(f"  arch {k}: {'SUPPORTED' if k in a else 'absent'}")
PY

echo "=== $(date '+%F %T') OFFLINE arch dry-run: Qwen3.5-122B ==="
"$PY" "$REPO/tools/arch_dryrun.py" "$REPO/models/Qwen3.5-122B-A10B-FP8"; q35=$?
echo "=== $(date '+%F %T') OFFLINE arch dry-run: GLM-5.2 ==="
"$PY" "$REPO/tools/arch_dryrun.py" "$REPO/models/GLM-5.2-FP8"; glm=$?
echo "=== $(date '+%F %T') DONE install_rc=$rc q35=$q35 glm=$glm ==="
