#!/bin/bash
#SBATCH --job-name=hf-stage-gemma4-qat
#SBATCH --partition=build
#SBATCH --account=rcc-staff
#SBATCH --qos=build
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/project/rcc/mehta5/vllm/models/.stage-gemma4-qat-%j.log
# Stage google/gemma-4-31B-it-qat-w4a16-ct -- quantization-AWARE-trained 4-bit Gemma 4.
# 21.67 GiB vs 58.25 GiB BF16, so it fits TP=1 on ONE A40 (48GB) or A100 40GB, halving the
# GPU count and therefore the reservation floor again. compressed-tensors pack-quantized,
# num_bits 4, group_size 32, symmetric; vision tower + lm_head in the ignore list.
# Kernel path verified in our env: Marlin supports group_size 32 and min capability 75, so
# sm80/sm86/sm90 all work. Machete is skipped (needs g64/g128) -- no Hopper bonus.
# Do NOT pass --quantization; vLLM auto-detects from the checkpoint.
# QAT rather than post-training quant, so quality should be near-lossless -- but Google
# publishes NO quantized-vs-BF16 benchmark, so our frozen gauntlet is the only evidence.
set -u
cd /project/rcc/mehta5/vllm
echo "=== $(date '+%F %T') staging google/gemma-4-31B-it-qat-w4a16-ct ==="
tools/stage_model_bg.sh google/gemma-4-31B-it-qat-w4a16-ct /project/rcc/mehta5/vllm/models/Gemma-4-31B-it-qat-w4a16
rc=$?
echo "=== $(date '+%F %T') stage finished rc=$rc ==="
exit $rc
