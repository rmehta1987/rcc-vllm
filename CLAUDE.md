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

### Do not reach PI-owned nodes through a broad partition

The H200 rule above is the specific case of a general one. A staff-accessible partition
spans most of the cluster, so submitting there with a `--constraint` can silently land on
hardware another research group owns, bypassing their partition and account entirely. It
looks like "a general node of type X"; it is not.

**Before submitting any GPU job, check who owns the nodes you could land on:**

```bash
scontrol show node <node> | grep -oP 'Partitions=\K[^ ]+'
```

A node listed in only a broad partition is shared. A node ALSO listed in a `<pi>-gpu`
partition belongs to that group — do not target it without either an account you hold for
it, or explicit user permission. Constrain the nodelist (`--nodelist`, `--exclude`) or
submit to a partition whose hardware is genuinely shared.

Known ownership on this cluster, as of 2026-08-19:

| nodes | GPU | owner |
|---|---|---|
| midway3-0600..0606 | H200 | pi-jevans, pi-lgagliardi, pi-gagalli — **off limits, see above** |
| midway3-0377, 0378 | A100 80GB | pi-gagalli |
| midway3-0558, 0559 | A100 80GB | schmidt / pi-dfreedman, ai4s-hackathon |
| midway3-0423 | H100 | pi-pedramh — **an account the user holds; OK to use** |
| midway3-0294 | A100 40GB | nobody — in the open `gpu` partition |
| beagle3-0001..0044 | A100 40GB / A40 | nobody — consortium hardware |

Genuinely unowned GPU capacity is: the `gpu` partition (open to all, but 10 of its 11
nodes are V100/RTX6000, which are fp16-only and excluded from this service) and the
`beagle3` partition. Prefer those, plus `pedramh-gpu`, and expect to queue.

Do not use an elevated QOS to jump that queue on shared hardware. `beagle3` defaults an
`rcc-staff` job to `beagle3-prio` (priority 100,000,000 vs the normal 0); pass
`--qos=beagle3` explicitly, or fix a submitted job with
`scontrol update jobid=<id> QOS=beagle3`.

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
