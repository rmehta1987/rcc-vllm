# Project rules

## Hardware restrictions

HARD RULE -- GPU work runs ONLY on the three targets below. This is an ALLOWLIST, not a
preference order. If the work does not fit one of them, STOP and ask the user.

| # | target | hardware | account / qos | why it is allowed |
| --- | --- | --- | --- | --- |
| 1 | `pedramh-gpu` | midway3-0423, 4x H100 80GB | `pi-pedramh` | an account the user actually holds |
| 2 | `beagle3` | A100 40GB (beagle3-0001..0022), A40 (beagle3-0023..0044) | `rcc-staff` or `beagle3-users`, **`--qos=beagle3`** | consortium hardware, owned by no single group |
| 3 | `gpu` | midway3-0294 (A100 40GB) only | `rcc-staff`, `--qos=gpu` | `AllowAccounts=ALL`, genuinely open to everyone |

CPU-only work (scoring, adjudication, analysis) goes to `caslake`, which has no GPU nodes.

### Everything else is off limits

**Do NOT use the `test` partition for GPU work.** It spans almost the whole cluster, so a
bare `--constraint` there silently lands on hardware another research group owns, bypassing
their partition and account entirely. It looks like "a general node of type X"; it is not.
This is how 103 historical jobs reached pi-gagalli and pi-lgagliardi hardware -- 85 of them
on H200 nodes -- without ever naming those partitions. `test` is still fine for CPU jobs.

**Do NOT use H200 at all.** midway3-0600..0606, and any `--constraint=H200`. If a task
appears to require one, STOP and ask, naming node count, GPU count, and wall time. Do not
submit while waiting. Approval for one H200 job does NOT carry to the next -- ask every time.

**Do NOT target another group's nodes even when they are idle.** Idle is not available.
midway3-0558/0559 (schmidt / pi-dfreedman) and midway3-0377/0378 (pi-gagalli) are A100 80GB
and often free; they are still theirs. Ask the user first, every time.

**Do NOT use an elevated QOS to jump the queue on shared hardware.** `beagle3` defaults an
`rcc-staff` job to `beagle3-prio` (priority 100,000,000 vs the normal 0). Pass `--qos=beagle3`
explicitly, or fix a submitted job with `scontrol update jobid=<id> QOS=beagle3`.

### Node ownership on this cluster, as of 2026-08-20

| nodes | GPU | owner | usable? |
|---|---|---|---|
| midway3-0423 | H100 80GB | pi-pedramh | **YES** -- an account the user holds |
| beagle3-0001..0022 | A100 40GB | nobody (consortium) | **YES** via `beagle3` |
| beagle3-0023..0044 | A40 48GB | nobody (consortium) | **YES** via `beagle3` |
| midway3-0294 | A100 40GB | nobody (open `gpu`) | **YES** via `gpu` |
| midway3-0600..0606 | H200 | pi-jevans, pi-lgagliardi, pi-gagalli | no -- ask every time |
| midway3-0377, 0378 | A100 80GB | pi-gagalli | no -- ask first |
| midway3-0558, 0559 | A100 80GB | schmidt / pi-dfreedman | no -- ask first |
| midway3-0293 | A100 40GB | pi-lgagliardi | no -- ask first |
| midway3-0372, 0385, 0426, 0432 | H100 | depablo / jevans / schmidt / sriesenfeld | no -- ask first |
| midway3-0277..0286 | V100, RTX6000 | nobody (open `gpu`) | no -- fp16-only, excluded (cc 7.5) |

Verify before submitting if unsure:

```bash
scontrol show node <node> | grep -oP 'Partitions=\K[^ ]+'
```

A node listed ONLY in a broad partition is shared. A node ALSO listed in a `<pi>-gpu`
partition belongs to that group.

Add `--exclude=midway3-[0600-0606]` to GPU submissions as a belt-and-braces guard even on
partitions that contain no H200.

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
