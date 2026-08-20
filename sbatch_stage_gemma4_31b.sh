#!/bin/bash
#SBATCH --job-name=hf-stage-gemma4-31b
#SBATCH --partition=build
#SBATCH --account=rcc-staff
#SBATCH --qos=build
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/project/rcc/mehta5/vllm/models/.stage-gemma4-31b-%j.log

# Stage google/gemma-4-31B-it -- candidate to evaluate against the incumbent
# qwen3.8_27B on the frozen LiveCodeBench subset.
#
# Why this variant: 30.7B dense BF16 (~62 GB), the closest Gemma 4 size to the
# 27B incumbent's 51.8 GB, so the comparison is roughly footprint-matched. The
# INSTRUCT (-it) variant, not the base checkpoint. Apache-2.0 and UNGATED --
# unlike Llama 3.1 (retired today partly because its weights are licence-gated
# and therefore not freely re-stageable), this needs no token and no acceptance.
#
# vLLM 0.26.0 support confirmed in-env: Gemma4ForCausalLM,
# Gemma4ForConditionalGeneration, and Gemma4MTP (it ships an MTP draft head, like
# Qwen3.8-27B) are all registered.
#
# Build partition, CPU only, no GPU, no billing (billing_sweep only floor-bills
# <registry-key>:<port> job names -- this is neither). Reuses the proven stager
# (xet disabled -- it hangs on this cluster; retry loop; exact tensor-byte verify
# against the safetensors index == Gate D).

set -u
cd /project/rcc/mehta5/vllm

echo "=== $(date '+%F %T') staging google/gemma-4-31B-it -> models/Gemma-4-31B-it ==="
tools/stage_model_bg.sh google/gemma-4-31B-it /project/rcc/mehta5/vllm/models/Gemma-4-31B-it
rc=$?
echo "=== $(date '+%F %T') stage finished rc=$rc ==="
exit $rc
