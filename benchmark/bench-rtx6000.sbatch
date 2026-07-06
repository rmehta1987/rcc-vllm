#!/bin/bash
#SBATCH --job-name=bench-rtx6000
#SBATCH --partition=test
#SBATCH --account=rcc-staff
#SBATCH --constraint=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/project/rcc/mehta5/vllm/benchmark/bench-rtx6000-%j.log

export GPU_TAG=rtx6000
# RTX 6000 (Turing, CC 7.5) does not support bfloat16
export EXTRA_VLLM_ARGS="--dtype=half"
source /project/rcc/mehta5/vllm/benchmark/bench-template.sh
