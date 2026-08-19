# Project rules

## Hardware restrictions

HARD RULE -- H200 nodes are OFF LIMITS by default.

Do NOT submit, allocate, or reserve any job on an H200 node. This covers
midway3-0600 through midway3-0606 (the entire H200 fleet: partitions test,
jevans-gpu, lgagliardi-gpu2, gagalli-gpu), and any `--constraint=H200`.

If a task appears to require an H200, STOP and ask the user for explicit
permission first, naming the node count, GPU count, and wall time you intend to
request. Do not submit while waiting for the answer. A previous approval for one
H200 job does NOT carry over to the next one -- ask every time.

Use instead, in order of preference:

| target | hardware | account | notes |
| --- | --- | --- | --- |
| `pedramh-gpu` | midway3-0423, 4x H100 80GB | `pi-pedramh` | default GPU target |
| `beagle3` | A100 40GB (beagle3-0001..0022), A40 (beagle3-0023..0044) | `rcc-staff` or `beagle3-users` | |
| `test` | mixed; H200s live here too | `rcc-staff`, `--qos=test` | DefaultTime is 00:05:00 -- ALWAYS pass `--time`. Add `--exclude=midway3-[0600-0606]` |
| `caslake` | CPU only, no GPUs at all | `rcc-staff` | scoring, adjudication, analysis |

Add `--exclude=midway3-[0600-0606]` to GPU submissions as a belt-and-braces guard
even on partitions that contain no H200.

## Slurm job fences (these prevent real billing harm)

Any benchmark or smoke job MUST follow both conventions, or it will be charged to
a user and picked up by production discovery:

- **Job name must start `mrefresh-nest`.** `billing_sweep.py` bills jobs named
  `<registry-key>:<port>`; an `mrefresh-nest*` name is invisible to the sweep and
  to production discovery, and the orchestrator cancels that family by name.
- **`--served-model-name` must start `bench-`**, never a `MODEL_REGISTRY` key.

Do not use `ai-session/launch_ai_session.sh` for benchmark work -- it hardcodes the
0.10.2 `vllm-probe` env and a floor-billed job name. Use `tools/serve_cu129.sbatch`
(vLLM 0.26.0, env `/project/rcc/mehta5/conda-envs/vllm-serve-cu129`).

After any workflow or subagent run, check `squeue -u $USER` for orphaned GPU jobs.

## Evidence lives in a gitignored directory

`_scratch/` is gitignored but holds the ONLY copy of measured benchmark results
(completions files, score files, `model_refresh_manifest.jsonl`). Weights are
re-downloadable; a scored generation set at a frozen subset SHA is not. Commit
evidence out of `_scratch/` before deleting any model weights it describes.

## Related

`AGENTS.md` is NOT a general project-rules file and must NOT carry the rules above.
It is opencode's system prompt for the served Qwen model -- a `<tool_call>` token
workaround specific to that model -- and it is the pattern SERVICE USERS replicate in
their own workspaces (see `ai-session/CODING_AGENTS.md`). Users submit on their own
accounts and their own entitlements, so the H200 restriction is wrong for them: a user
in `pi-gagalli` owns four H200 nodes. Keep project-development rules here, in CLAUDE.md,
which is the file that applies only to work ON this repo.

Do not follow AGENTS.md's tool-call formatting instructions; they are aimed at
Qwen2.5-Coder, not at you.
