#!/bin/bash
#SBATCH --job-name=qwen-smoke
#SBATCH --partition=test
#SBATCH --account=rcc-staff
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=smoke-qwen-%j.log

set -e

module load apptainer/1.4.1
#module load cuda/12.8

SIF=/project/rcc/mehta5/vllm/vllm-v0.7.3.sif
MODEL_DIR=/project/rcc/mehta5/vllm/models

PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
HOST=$(hostname -f)

echo "Starting vLLM on ${HOST}:${PORT}"

apptainer run --nv \
    --bind ${MODEL_DIR}:/models:ro \
    ${SIF} \
    --model /models/Qwen2.5-0.5B-Instruct \
    --served-model-name Qwen/Qwen2.5-0.5B-Instruct \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.5 \
    --host 0.0.0.0 \
    --port ${PORT} &

VLLM_PID=$!

echo "Waiting for vLLM to start..."
for i in {1..60}; do
    if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "vLLM ready after $((i*5))s"
        break
    fi
    sleep 5
done

echo "=== /v1/models ==="
curl -s "http://localhost:${PORT}/v1/models" | python3 -m json.tool

echo "=== chat completion ==="
curl -s "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 50
    }' | python3 -m json.tool

echo "=== /v1/models ==="
curl -s "http://localhost:${PORT}/v1/models" | python3 -m json.tool

echo ""
echo "================================================"
echo "vLLM is running on ${HOST}:${PORT}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "To connect from login node:"
echo "  ssh -N -L 8000:${HOST}:${PORT} ${HOST} &"
echo "  export OPENAI_API_BASE=http://localhost:8000/v1"
echo "  export OPENAI_API_KEY=dummy"
echo "================================================"
echo ""
echo "Will run until --time limit or scancel ${SLURM_JOB_ID}"

# Keep job alive until time limit or scancel
wait $VLLM_PID
