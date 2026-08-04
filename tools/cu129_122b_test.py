"""Stage-2 load+generate test for Qwen3.5-122B-A10B-FP8 on the cu129 serve env.

Must be a real file (not piped via stdin) and guarded by __main__: TP>=2 uses the
`spawn` multiprocessing start method (CUDA is already initialized in the parent),
and spawn re-imports the main module by re-executing its path -- a stdin heredoc
has no path, so workers die with FileNotFoundError('<stdin>'). Run as:
    python tools/cu129_122b_test.py [MODEL_DIR] [TP]
"""
import sys
import time

from vllm import LLM, SamplingParams


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else \
        "/project/rcc/mehta5/vllm/models/Qwen3.5-122B-A10B-FP8"
    tp = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    t0 = time.time()
    llm = LLM(model=model, tensor_parallel_size=tp, gpu_memory_utilization=0.90,
              max_model_len=4096, enforce_eager=True, trust_remote_code=True)
    print(f"LOAD_OK in {time.time() - t0:.1f}s (tp={tp})", flush=True)

    prompts = [
        "Write a Python function `merge_intervals(intervals)` that merges "
        "overlapping intervals. Return only code.",
        "Implement binary search in Rust as "
        "`fn bsearch(a: &[i32], t: i32) -> Option<usize>`.",
    ]
    out = llm.generate(prompts, SamplingParams(max_tokens=192, temperature=0))
    for i, o in enumerate(out):
        print(f"--- gen[{i}] ---", flush=True)
        print(o.outputs[0].text[:600], flush=True)
    print("VLLM_CU129_122B_PASS", flush=True)


if __name__ == "__main__":
    main()
