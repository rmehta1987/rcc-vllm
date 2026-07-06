#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH --partition=test
#SBATCH --account=rcc-staff
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=vllm-%j.log

set -e

module load apptainer/1.4.1

SIF=/project/rcc/mehta5/vllm/vllm-v0.7.3.sif
MODEL_DIR=/project/rcc/mehta5/vllm/models
PORT=8000
HOST=$(hostname -f)

echo "================================================"
echo "vLLM starting on ${HOST}:${PORT}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "================================================"
echo "Connect with:"
echo "  api_base: http://${HOST}:${PORT}/v1"
echo "================================================"

apptainer run --nv \
    --bind ${MODEL_DIR}:/models:ro \
    ${SIF} \
    --model /models/Qwen2.5-0.5B-Instruct \
    --served-model-name Qwen/Qwen2.5-0.5B-Instruct \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.5 \
    --host 0.0.0.0 \
    --port ${PORT}

