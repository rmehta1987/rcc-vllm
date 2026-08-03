#!/bin/bash
#SBATCH --job-name=build-vllm-serve
#SBATCH --partition=build
#SBATCH --account=rcc-staff
#SBATCH --qos=build
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/project/rcc/mehta5/vllm/tools/.build-vllm-serve-%j.log

# The SERVE env for the model-refresh loop: a CUDA-12.x vLLM that (a) registers the new archs
# (qwen3_5_moe / glm_moe_dsa need vLLM >= 0.17.0) AND (b) actually runs on the cluster's H200 driver
# 535.216.03 (max CUDA 12.2). vLLM 0.19.0 pins torch==2.10.0, whose DEFAULT PyPI build is CUDA 12.8
# (nvidia-cuda-runtime-cu12 12.8.90) -- proven to init CUDA on driver 535 (cluster_provenance.md), unlike
# vLLM 0.26.0 (torch 2.11/cu130/CUDA 13, which fails). vLLM 0.20+ moved to torch 2.11 -> CUDA 13, so 0.19.0
# is the NEWEST arch-capable version still on the CUDA-12.x line. Plain install (no index forcing): torch's
# default cu12.8 is exactly what we want, and vLLM 0.19.0's kernels are compiled against it.

set -u
ENV=/project/rcc/mehta5/conda-envs/vllm-serve-cu128
REPO=/project/rcc/mehta5/vllm
VLLM_VERSION="${VLLM_VERSION:-0.19.0}"

export PIP_CACHE_DIR=/project/rcc/mehta5/.pip-cache
export HF_HOME=/project/rcc/mehta5/hf-cache
mkdir -p "$PIP_CACHE_DIR"

echo "=== $(date '+%F %T') creating env $ENV (python 3.12) ==="
module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
[ -x "$ENV/bin/python" ] || mamba create -p "$ENV" python=3.12 -y || { echo "env create FAILED"; exit 1; }
PY="$ENV/bin/python"; PIP="$ENV/bin/pip"

echo "=== $(date '+%F %T') pip installing vllm==${VLLM_VERSION} (default index -> torch 2.10.0 cu128) ==="
"$PIP" install --upgrade pip setuptools wheel
rc=1
for attempt in 1 2 3; do
  "$PIP" install "vllm==${VLLM_VERSION}"; rc=$?
  [ "$rc" -eq 0 ] && break
  echo "=== attempt $attempt failed (rc=$rc); retry in 30s ==="; sleep 30
done
[ "$rc" -eq 0 ] || { echo "pip install vllm FAILED (rc=$rc)"; exit 1; }

echo "=== $(date '+%F %T') installed versions + CUDA runtime major ==="
"$PY" - <<'PY'
import vllm, transformers, torch, importlib.metadata as m
print("vllm        :", vllm.__version__)
print("transformers:", transformers.__version__)
print("torch       :", torch.__version__, "| torch.version.cuda:", torch.version.cuda)
try:
    print("cuda-runtime:", m.version("nvidia-cuda-runtime-cu12"), "(cu12 = good for driver 535)")
except Exception:
    try: print("cuda-runtime cu13:", m.version("nvidia-cuda-runtime-cu13"), "(cu13 = WRONG, would fail on 535)")
    except Exception as e: print("cuda-runtime: unknown", e)
from vllm import ModelRegistry
a=set(ModelRegistry.get_supported_archs())
for k in ["Qwen3_5MoeForConditionalGeneration","GlmMoeDsaForCausalLM","Qwen3MoeForCausalLM","DeepseekV4ForCausalLM"]:
    print(f"  arch {k}: {'SUPPORTED' if k in a else 'absent'}")
PY

echo
echo "=== $(date '+%F %T') OFFLINE arch dry-run (CPU): Qwen3.5-122B ==="
"$PY" "$REPO/tools/arch_dryrun.py" "$REPO/models/Qwen3.5-122B-A10B-FP8"; q35=$?
echo "=== $(date '+%F %T') DONE. qwen3.5_rc=$q35 ==="
