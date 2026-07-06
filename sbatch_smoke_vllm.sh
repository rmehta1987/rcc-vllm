#!/bin/bash
#SBATCH --job-name=vllm-smoke
#SBATCH --partition=test
#SBATCH --account=rcc-staff
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=smoke-%j.log

set -e

module load apptainer/1.4.1
SIF=/project/rcc/mehta5/vllm/vllm-v0.7.3.sif
MODEL_CACHE=/project/rcc/mehta5/vllm/models

mkdir -p $MODEL_CACHE

PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
HOST=$(hostname -f)

echo "Starting vLLM on ${HOST}:${PORT}"

apptainer run --nv \
    --bind ${MODEL_CACHE}:/root/.cache/huggingface \
    ${SIF} \
    --model Qwen/Qwen2.5-0.5B-Instruct \
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

echo "=== done ==="
kill $VLLM_PID
