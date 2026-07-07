# LoRA fine-tuning: design note for a supported training path

Status: DESIGN ONLY (2026-07-07). The serving side is built: `ai-session
<preset> --lora NAME=/abs/path` registers a PEFT adapter with the session's vLLM
server, validated before submission (see the "LoRA adapters" section of
README.md). This note designs the other half — how a researcher produces that
adapter — for an RCC decision before any build.

## Problem

Researchers want models adapted to their domain (lab-specific code style,
instrument logs, protocol text). Full fine-tunes of 32B/72B models are out of
reach on a shared allocation (hundreds of GPU-hours, terabytes of optimizer
state), but LoRA adapters are not: rank 16-64 adapters on a 32B base train in
single-digit GPU-hours on one or two 80GB cards and weigh tens to hundreds of
megabytes. The serving side already accepts them; today the training side is
"figure it out yourself", which in practice means users will do it wrong
(wrong base snapshot, wrong chat template, wrong target modules) and then file
tickets when the adapter misbehaves.

## Options

### Option A — recipe, not service (recommended first step)

The service ships a maintained, tested training recipe the user runs on their
OWN allocation; nothing new runs under the service account.

- A `train_lora/` directory with: a pinned environment spec (separate from the
  untouchable vllm-probe env), one `train_lora.sbatch` the user submits from
  their own account/partition, and a `train_lora.py` built on PEFT + TRL's
  SFTTrainer.
- The recipe hard-codes the correct settings per served base model: the exact
  base checkpoint path under `/project/rcc/mehta5/vllm/models/` (same snapshot
  the service serves — this is the error the recipe exists to prevent), the
  model's chat template applied to the user's JSONL, target modules, and a
  default rank (16 or 32) within the serving cap (256).
- Input contract: a JSONL of `{"messages": [...]}` conversations on project
  storage; the recipe validates it before training.
- Output contract: an adapter directory that `--lora` accepts unchanged, plus a
  `training_manifest.json` (base path + hash, data path, hyperparameters, loss
  curve tail) so operators can reproduce/debug an adapter that behaves badly.
- Cost: billed to the user's own Slurm account by Slurm itself; no SU-billing
  work needed. A rough guide goes in the user docs (a 3-epoch rank-32 run on
  ~10k conversations, Coder-32B base, 2 x A100: low single-digit GPU-hours).

### Option B — managed training verb (`ai-session tune`)

The service runs training as a billed session type: user points at a JSONL,
the service submits the training job, meters it like a session (floor = GPU
holding cost; no token term), and drops the adapter + manifest into the user's
state directory.

- Pros: one-command UX; settings cannot be wrong; manifest always present.
- Cons: the service takes on job babysitting (OOM, bad data, divergence),
  data-quality support burden, and a new billing surface (a `tune` session
  kind in the ledger). Requires the Option-A recipe to exist anyway — the verb
  is a wrapper around it.

### Option C — do nothing (rejected)

Serving-without-training invites unsupported, half-right adapters and support
load with no artifact (no manifest) to debug from.

## Recommendation

Ship Option A now (it is small: an env spec, one sbatch, one script, one docs
page), collect real usage, and revisit Option B only if recipe uptake shows
demand. This mirrors the service's pattern elsewhere: the hard part (correct
serve and training configuration) is captured in maintained code, while the GPUs and
accountability stay with the user.

## Policy questions for RCC before building Option A

1. Which base models are supported for tuning? (Proposal: the served Qwen2.5
   models; Llama only after its license gate, which the recipe must re-check.)
2. Where do adapters live? (Proposal: the user's own project space; the service
   never stores user adapters centrally.)
3. Data policy: training data never leaves the cluster (same stance as
   inference privacy); the recipe must refuse paths under `/tmp` or scratch
   that other users can read? (Proposal: warn, do not refuse.)
4. Support boundary: operators debug the recipe, not the user's data or the
   adapter's quality.

## Build sketch for Option A (when approved)

1. New env `/project/rcc/mehta5/lora-train-env` (python 3.11 venv: torch,
   transformers, peft, trl, datasets; pinned). Never touch vllm-probe.
2. `ai-session/train_lora/{train_lora.py,train_lora.sbatch,README.md}` +
   validation of the JSONL and of the base-model snapshot hash.
3. A user docs page ("Fine-tune a model") written against the user's own
   allocation, ending at `ai-session code --lora ...`.
4. A smoke run on the 4B base (cheapest card, minutes) before recommending
   32B recipes. Requires explicit SU/allocation approval like any GPU work.
