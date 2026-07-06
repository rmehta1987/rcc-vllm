#!/bin/bash
#SBATCH --job-name=vllm-pull
#SBATCH --partition=build
#SBATCH --account=rcc-staff
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=pull-%j.log

set -e

module load apptainer/1.4.1

export APPTAINER_TMPDIR=/project/rcc/mehta5/some_temp
export APPTAINER_CACHEDIR=/project/rcc/mehta5/some_temp/cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR

cd /project/rcc/mehta5/vllm

apptainer pull vllm-v0.7.3.sif docker://vllm/vllm-openai:v0.7.3

echo "=== done ==="
ls -lh vllm-v0.7.3.sif
