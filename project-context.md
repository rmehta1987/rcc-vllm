# HPC LLM Service Project Context

## Goal
Deploy local Qwen LLM on RCC cluster (beagle3) for faculty/students via vLLM + Slurm.

## What's done so far
- vLLM v0.7.3 SIF at /project/rcc/mehta5/vllm/vllm-v0.7.3.sif
- Qwen2.5-0.5B-Instruct downloaded to /project/rcc/mehta5/vllm/models/
- Smoke test passed on beagle3-0013 with test partition
- Account: rcc-staff
- GPU types available: h200, h100, a100, a40, l40, rtx6000

## Architecture decisions
- Phase 1: PI dedicated nodes only, single user per session
- Billing: token-based (not wall-clock)
- SU formula: SU = (tokens / throughput / 3600) × n_gpus × SU_rate (flat 1.0)
- vLLM container via Apptainer
- Compute nodes have NO internet

## Next steps
- Benchmark all GPU types with small model
- Scale up to Qwen2.5-7B or 14B
- Build ai-session wrapper
