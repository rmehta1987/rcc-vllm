#!/bin/bash
#SBATCH --job-name=bench-l40s
#SBATCH --partition=test
#SBATCH --account=rcc-staff
#SBATCH --constraint=L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/project/rcc/mehta5/vllm/benchmark/bench-l40s-%j.log

export GPU_TAG=l40s
source /project/rcc/mehta5/vllm/benchmark/bench-template.sh
