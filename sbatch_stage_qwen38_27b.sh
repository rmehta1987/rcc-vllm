#!/bin/bash
#SBATCH --job-name=hf-stage-qwen38-27b
#SBATCH --partition=build
#SBATCH --account=rcc-staff
#SBATCH --qos=build
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/project/rcc/mehta5/vllm/models/.stage-qwen38-27b-%j.log

# Stage Qwen/Qwen3.8-27B (dense, hybrid Gated-DeltaNet/Gated-Attention arch --
# same family as Qwen3-Next, which already SERVES on driver 535 via the
# vllm-serve-cu129 env; see cluster_provenance.md). Native safetensors, vLLM-
# supported per the model card.
#
# UPDATED 2026-08-19: this model is now the PRODUCTION CODING INCUMBENT
# (MODEL_REGISTRY key qwen3.8_27B, in PHASE1_SERVED, served TP=2 on H100 via the
# vllm-serve-cu129 env). This file is therefore the re-staging recipe for a model
# the service depends on -- keep it. Re-run it verbatim to restore the weights.
#
# Build partition, CPU only, no GPU, no billing (billing_sweep only floor-bills
# <registry-key>:<port> job names -- this is neither). Reuses the proven
# stager (xet disabled -- it hangs on this cluster; retry loop; exact
# tensor-byte verify against the safetensors index == Gate D from the
# 2026-08-05/06 candidate staging run).

set -u
cd /project/rcc/mehta5/vllm

echo "=== $(date '+%F %T') staging Qwen/Qwen3.8-27B -> models/Qwen3.8-27B ==="
tools/stage_model_bg.sh Qwen/Qwen3.8-27B /project/rcc/mehta5/vllm/models/Qwen3.8-27B
rc=$?
echo "=== $(date '+%F %T') stage finished rc=$rc ==="
exit $rc
