#!/bin/bash
#SBATCH --job-name=bench-a40
#SBATCH --partition=test
#SBATCH --account=rcc-staff
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/project/rcc/mehta5/vllm/benchmark/bench-a40-%j.log

export GPU_TAG=a40
source /project/rcc/mehta5/vllm/benchmark/bench-template.sh
