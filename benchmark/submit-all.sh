#!/bin/bash
# Submit one benchmark job per GPU type. Each one is independent.
set -e
cd "$(dirname "$0")"

for gpu in h200 h100 a100 a40 l40s rtx6000; do
    echo "Submitting bench-${gpu}.sbatch..."
    sbatch "bench-${gpu}.sbatch"
done

echo
echo "Watch with: squeue -u \$USER"
echo "Results land in: $(pwd)/results/<gpu>-raw.txt"
