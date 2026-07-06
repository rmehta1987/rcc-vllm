#!/bin/bash
# Sourced by each bench-<gpu>.sbatch — runs vLLM, benchmarks, writes a result line

set -e
module load apptainer/1.4.1

SIF=/project/rcc/mehta5/vllm/vllm-v0.7.3.sif
MODEL_DIR=/project/rcc/mehta5/vllm/models
MODEL_PATH=/models/Qwen2.5-0.5B-Instruct
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
RESULTS_DIR=/project/rcc/mehta5/vllm/benchmark/results
mkdir -p $RESULTS_DIR

PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
HOST=$(hostname -f)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ',' | tr ' ' '_')

echo "=== Benchmarking on ${GPU_NAME} (${GPU_TAG}) ==="

# Start vLLM in background
apptainer run --nv \
    --bind ${MODEL_DIR}:/models:ro \
    ${SIF} \
    --model ${MODEL_PATH} \
    --served-model-name ${MODEL_NAME} \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --disable-log-requests \
    --host 0.0.0.0 \
    --port ${PORT} \
    ${EXTRA_VLLM_ARGS:-} > vllm-server-${SLURM_JOB_ID}.log 2>&1 &

VLLM_PID=$!

# Wait for server
echo "Waiting for vLLM..."
for i in {1..60}; do
    if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "Ready after $((i*5))s"
        break
    fi
    sleep 5
done

# Run benchmark inside the container (it has benchmark_serving.py)
BENCH_OUT="${RESULTS_DIR}/${GPU_TAG}-raw.txt"

apptainer exec --nv ${SIF} python3 -c "
import asyncio, time, json, sys
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url='http://localhost:${PORT}/v1', api_key='dummy')

PROMPT = 'Write a detailed explanation of how a CPU works, covering the fetch-decode-execute cycle, pipelining, and cache hierarchy.'
MAX_TOK = 256
CONCURRENCY = 16
N_REQUESTS = 500
WARMUP = 16

async def one_request():
    r = await client.chat.completions.create(
        model='${MODEL_NAME}',
        messages=[{'role': 'user', 'content': PROMPT}],
        max_tokens=MAX_TOK,
    )
    return r.usage.prompt_tokens, r.usage.completion_tokens

async def main():
    # Warmup
    print('Warmup...', file=sys.stderr)
    await asyncio.gather(*[one_request() for _ in range(WARMUP)])

    print('Benchmark...', file=sys.stderr)
    sem = asyncio.Semaphore(CONCURRENCY)
    async def bounded():
        async with sem:
            return await one_request()

    start = time.time()
    results = await asyncio.gather(*[bounded() for _ in range(N_REQUESTS)])
    elapsed = time.time() - start

    total_in = sum(r[0] for r in results)
    total_out = sum(r[1] for r in results)
    total = total_in + total_out

    print(json.dumps({
        'gpu_tag': '${GPU_TAG}',
        'gpu_name': '${GPU_NAME}',
        'model': '${MODEL_NAME}',
        'concurrency': CONCURRENCY,
        'n_requests': N_REQUESTS,
        'elapsed_s': round(elapsed, 2),
        'input_tokens': total_in,
        'output_tokens': total_out,
        'total_tokens': total,
        'total_tokens_per_sec': round(total/elapsed, 1),
        'output_tokens_per_sec': round(total_out/elapsed, 1),
        'requests_per_sec': round(N_REQUESTS/elapsed, 2),
    }))

asyncio.run(main())
" > "$BENCH_OUT" 2>&1 || echo "Benchmark failed"

cat "$BENCH_OUT"

# Clean shutdown
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo "=== Done: ${GPU_TAG} ==="
