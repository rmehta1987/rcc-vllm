# rcc-vllm — ai-session

Local large-language-model serving on the UChicago RCC cluster. Slurm-launched
vLLM sessions serve open Qwen and Llama models over an OpenAI-compatible API,
behind a stable per-user gateway, with usage charged in Service Units (SU).

## Documentation

The user guide is built with MkDocs and published to GitHub Pages:

<https://rmehta1987.github.io/rcc-vllm/>

To preview it locally on a login node:

    /project/rcc/mehta5/mkdocs-env/bin/mkdocs serve

The source pages are under `docs/`.

## Layout

- `ai-session/` — session launcher, gateway, metering, CLI, clients, MCP servers
- `billing/` — SU formula, policy, benchmarked rate table, and tests
- `benchmark/` — billing-grade throughput benchmark
- `docs/` — MkDocs user guide (the published site)
- `examples/` — an agent example that runs against the gateway
- `IMPLEMENTATION_ROADMAP.md` — ranked backlog and current status

Model weights, container images, virtualenvs, caches, logs, and session keys are
intentionally excluded from git (see `.gitignore`).

## Operating the service

See `ai-session/README.md` (operator guide) and `ai-session/BILLING_POLICY.md`
(charging policy).
