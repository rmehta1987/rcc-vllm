# Fine-Tuning a Model on Your Own Data

This page shows how to produce a LoRA adapter — a small set of extra weights that
adjust one of the served text models toward a task or style of your own — and then
serve it. Once you have an adapter, [Using Your Own Fine-Tuned Model](lora.md)
covers loading it into a session; this page is the step before that, where the
adapter is made.

!!! note "What the service provides, and what you run yourself"
    The **serving** side is built and verified: `ai-session <preset> --lora
    NAME=PATH` loads a finished adapter (see [Using Your Own Fine-Tuned
    Model](lora.md)). The **training** side below is a worked recipe you run on
    your **own** GPU allocation — Slurm bills your account directly, and no
    Service Units are involved. The service does not yet ship a packaged,
    one-command training tool, so treat the script here as a starting point to
    adapt, not a tested black box. The two settings it exists to get right — the
    base checkpoint and the chat template — are called out below.

## Is a LoRA adapter the right tool?

The models served today are text language models, so this page trains text
adapters. A LoRA adapter is worth producing when one text-to-text mapping recurs
across many inputs — a fixed output schema, free text mapped to structured
fields, a house coding or writing style — and prompting alone will not hold the
pattern. The worked example below, turning thousands of pathology reports into
one fixed set of fields, is that case.

Several nearby problems are not. Answering questions from a corpus of papers or
lab protocols is retrieval over embeddings, not training. Reading a finding from
a whole-slide image or micrograph is a job for the vision-language model
`qwen3.5_122B` (once served), prompted directly rather than fine-tuned.
Recovering a simulator's parameters is simulation-based inference, which wants a
density estimator, not a language model at all.

The single-transformation case is the common trap. "I have a statistical-genetics
codebase that computes true effect sizes from noisy GWAS estimates and I want to
port it to PyTorch" is *one* translation with the code already in front of you,
not a durable pattern; open the repository in a [coding session](coding/overview.md)
and let the tool read your files. Fine-tune only when the pattern recurs across
many inputs, as in the worked example below.

## Worked example: extracting structured data from a document corpus

A common and broadly applicable case: you have thousands of unstructured
documents — pathology or radiology reports, clinical notes, interview
transcripts, curated literature — and you need each turned into the same set of
structured fields for analysis. The mapping from varied free text to one fixed
schema is a repeated pattern, which is what a LoRA learns well.

This is also a task the on-cluster service is specifically suited to: when the
text is protected (for example PHI under HIPAA) it cannot be sent to a hosted
model, so an adapter that runs entirely inside the cluster is the enabling
option, not merely a convenience. A base model with a good prompt may extract a
few reports acceptably; a fine-tune earns its keep at volume, where per-call
prompt length, throughput, and *consistency of the schema across every document*
are what matter.

The example below extracts a small pathology-report schema. The reports shown are
synthetic, not real patient data.

## What you need

1. **A dataset of examples** of the mapping you want, as a JSONL file with one
   chat conversation per line (`{"messages": [...]}`). Fifty to a few hundred
   hand-checked examples is a good range; quality matters far more than volume,
   because every assistant turn is a target the model learns to reproduce.
2. **Your own GPU allocation** (your PI's Slurm account and partition). Training
   does not run under the service; it runs as an ordinary batch job on your
   account.
3. **A training environment** — a Python virtual environment with `torch`,
   `transformers`, `peft`, `trl`, and `datasets`. Build it once on a login node,
   which has internet; nothing for training is preinstalled.

## Step 1: Build the dataset

Each line pairs an input document with the exact structured output you want back.
Note how the assistant turns normalize varied source phrasing into one schema —
"Nottingham grade 2", "moderately differentiated", and "poorly differentiated"
all map to an integer grade, and centimetres are converted to millimetres. That
normalization, applied consistently across a whole corpus, is the reason to train
rather than prompt.

```json title="train.jsonl (one object per line; shown pretty-printed)"
{"messages": [
  {"role": "user", "content": "Extract the fields from this pathology report as JSON: site, diagnosis, grade (1-3), tumor_size_mm, margins, nodes_positive, nodes_examined.\n\nSpecimen: left breast, lumpectomy. Invasive ductal carcinoma, Nottingham grade 2. Tumor size 1.8 cm. Margins negative, closest margin 3 mm. 2 of 3 sentinel lymph nodes positive."},
  {"role": "assistant", "content": "{\"site\": \"left breast\", \"diagnosis\": \"invasive ductal carcinoma\", \"grade\": 2, \"tumor_size_mm\": 18, \"margins\": \"negative\", \"nodes_positive\": 2, \"nodes_examined\": 3}"}
]}
{"messages": [
  {"role": "user", "content": "Extract the fields from this pathology report as JSON: site, diagnosis, grade (1-3), tumor_size_mm, margins, nodes_positive, nodes_examined.\n\nSpecimen: sigmoid colon, segmental resection. Moderately differentiated adenocarcinoma invading through the muscularis propria into pericolic fat. Largest dimension 4.2 cm. Resection margins uninvolved. 0 of 14 lymph nodes involved."},
  {"role": "assistant", "content": "{\"site\": \"sigmoid colon\", \"diagnosis\": \"adenocarcinoma\", \"grade\": 2, \"tumor_size_mm\": 42, \"margins\": \"negative\", \"nodes_positive\": 0, \"nodes_examined\": 14}"}
]}
{"messages": [
  {"role": "user", "content": "Extract the fields from this pathology report as JSON: site, diagnosis, grade (1-3), tumor_size_mm, margins, nodes_positive, nodes_examined.\n\nSpecimen: stomach, distal gastrectomy. Poorly differentiated adenocarcinoma, greatest dimension 5.5 cm. Margins negative. 6 of 21 regional lymph nodes positive for metastatic carcinoma."},
  {"role": "assistant", "content": "{\"site\": \"stomach\", \"diagnosis\": \"adenocarcinoma\", \"grade\": 3, \"tumor_size_mm\": 55, \"margins\": \"negative\", \"nodes_positive\": 6, \"nodes_examined\": 21}"}
]}
```

The file itself has one JSON object per physical line (no pretty-printing); the
blocks above are expanded only for reading. Put the file on project or scratch
storage — for example `/project/<pi>/pathology/train.jsonl`.

## Step 2: Build the training environment

Once, on a login node (which has internet):

```bash
python -m venv ~/lora-train-env
source ~/lora-train-env/bin/activate
pip install torch transformers peft trl datasets
```

Keep this separate from anything the service ships; it is yours to manage.

## Step 3: Write the training script

The two settings that must be exactly right are the **base checkpoint** — the
same one the service serves, loaded from the shared model directory, not a fresh
download that may be a different snapshot — and the **chat template**, applied
with the base model's own tokenizer so the fine-tune sees the prompt framing
inference will use. This example fine-tunes the small model (`Qwen3-4B`), which
is cheap to train and enough for a well-defined extraction schema; scale to the
72B for harder tasks by changing one path. Save this as `train_lora.py`:

```python title="train_lora.py"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 1. SAME checkpoint the service serves -- not a fresh Hugging Face download.
BASE = "/project/rcc/mehta5/vllm/models/Qwen3-4B"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto")

# 2. Format each example with the base model's OWN chat template.
ds = load_dataset("json", data_files="/project/<pi>/pathology/train.jsonl", split="train")
ds = ds.map(lambda r: {"text": tok.apply_chat_template(r["messages"], tokenize=False)})

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"])

trainer = SFTTrainer(
    model=model, train_dataset=ds, peft_config=lora,
    args=SFTConfig(
        output_dir="/project/<pi>/adapters/path-extract",
        num_train_epochs=3, per_device_train_batch_size=4,
        gradient_accumulation_steps=4, learning_rate=2e-4,
        bf16=True, dataset_text_field="text", max_seq_length=2048))
trainer.train()
trainer.save_model()   # writes adapter_config.json + the adapter weights
```

Rank 16 is well inside the serving cap of 256. To fine-tune the general 72B model
instead, point `BASE` at `/project/rcc/mehta5/vllm/models/Qwen2.5-72B-Instruct`
(it needs more GPUs and a longer run) and serve the matching preset later.

## Step 4: Submit the training job

Training runs on **your** allocation, so submit it with your own account and
partition. A minimal `train_lora.sbatch` for the 4B base:

```bash title="train_lora.sbatch"
#!/bin/bash
#SBATCH --job-name=lora-path-extract
#SBATCH --account=<your-account>
#SBATCH --partition=<your-gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

source ~/lora-train-env/bin/activate
python train_lora.py
```

```bash
sbatch train_lora.sbatch
```

A three-epoch, rank-16 run on a few hundred examples with the 4B base takes
well under an hour on one A100-80GB. Slurm bills your account for that time; the
ai-session service and its Service Units are not involved in training at all.

When the job finishes, `/project/<pi>/adapters/path-extract` contains
`adapter_config.json` and the adapter weights — exactly what the serving side
validates.

## Step 5: Serve your adapter

Start a session with the adapter loaded, giving it a short name you will type in
your client. The 4B adapter is served by the `fast` preset:

```bash
ai-session fast --lora pathextract=/project/<pi>/adapters/path-extract
```

Send documents with `model=pathextract` and the model returns the schema you
taught; request the base model (`qwen3_4b`) in the same session to compare
against stock behavior at no extra cost. To guarantee the output parses as JSON
rather than trusting the model to stay in format, validate each response and
retry on failure; a server-enforced JSON mode is a planned addition. The full
serving details — per-client usage, cost, and limits — are on
[Using Your Own Fine-Tuned Model](lora.md).

## Training the base model to act as an agent in your domain

The [LoRA route above](#is-a-lora-adapter-the-right-tool) teaches a frozen base
model a text-to-text pattern, and retrieval puts domain facts in front of a
general model at inference. Neither changes how the base model behaves as an
*agent* — how well it holds a multi-turn tool-calling loop and stays in a strict
output format. When that agentic behavior is what you need from an open model you
can serve entirely on the cluster, the base weights have to change, and the
recipe is heavier than a LoRA by orders of magnitude.

The biomedical agent in
[Biomni (*Science* 2026)](https://www.science.org/doi/10.1126/science.adz4351) is
the reference point, and it cuts two ways:

- Its headline system needs **no training at all**: a strong general model
  (the published work uses Claude, zero-shot) drives the agent, wrapped in
  retrieval over a large curated biomedical data lake and code execution. If your
  goal is an agent that is capable in your field, that agent-plus-retrieval
  route — see [Agent responsibilities and risks](coding/agents.md) — is cheaper,
  keeps improving as your data changes, and is the first thing to try.
- Its open-weights variant, **Biomni-R0**, *is* trained, from Qwen3-8B and
  Qwen3-32B, because the vanilla Qwen3 models struggle with multi-turn tool use
  and strict format compliance. The two-phase recipe below is that variant's
  (from the published *Science* version's supplement; it is not in the earlier
  bioRxiv preprint), for the case where you need an open model — no external API
  in the loop — to be the agent itself.

### The two-phase recipe

Biomni-R0's procedure, from the paper's supplement, is distillation followed by
reinforcement learning. The numbers are the paper's, sized to that task; treat
them as a starting point, not universal constants.

| Phase | What it does | Data | Settings (from the paper) |
|---|---|---|---|
| 1 — Rejection-sampled SFT | Full-parameter supervised fine-tuning on expert trajectories, to teach the tool loop and the output format | 834 trajectories kept by rejection sampling: for each task, generate 8 rollouts with a strong teacher and keep the highest-reward one | 4 epochs, batch size 16, learning rate 1e-5, cosine annealing |
| 2 — Reinforcement learning (GRPO) | Refine the policy on a group of rollouts per prompt, scored by a task ground-truth reward plus a formatting reward | 4447 samples | 1 epoch, batch size 32, 8 rollouts per sample, learning rate 1e-6, cosine |

The teacher used for rejection sampling in the paper is Claude-4-Sonnet, a
frontier API model. **That step sends your task prompts off the cluster**, which
gives up the data-locality reason for using this service. If your trajectories
contain protected or unpublished data, generate them with a strong *local* model
instead — the served `qwen2.5_72B`, or `qwen3.5_122B` once it is served — and
accept that a weaker teacher yields weaker distillation. If the data is public,
an external teacher is fine.

### Compute

Full-parameter training of a 32B model does not fit on one GPU: the weights,
gradients, and Adam optimizer state together need roughly 16 bytes per parameter
— about 520 GB for Qwen3-32B — before activations, so the model is sharded across
many GPUs. The paper's allocation:

| Model | GPUs | Rollout parallelism | Actor-update parallelism |
|---|---|---|---|
| Biomni-R0-8b | 8 × A100 | 2-way tensor parallel | FSDP + 4-way sequence parallel |
| Biomni-R0-32b | 16 × A100 | 4-way tensor parallel | FSDP + 4-way sequence parallel |

16 × A100-80GB is two full GPU nodes. This is a multi-node, multi-day job on
**your own allocation** — it does not run under ai-session and costs no Service
Units; Slurm bills your account for the GPUs directly, as with the LoRA training
above. If you do not have that budget, the LoRA and retrieval routes above are
the realistic alternatives; there is no cheap path to full 32B reinforcement
learning.

### Tools

- **Phase 1 (SFT)** is ordinary full-parameter supervised fine-tuning: the
  [Step 3 script](#step-3-write-the-training-script) above with the LoRA config
  removed (so every weight updates) and a sharding backend added — DeepSpeed
  ZeRO-3 or PyTorch FSDP — because the full model no longer fits on one GPU.
  Point `BASE` at `/project/rcc/mehta5/vllm/models/Qwen3-32B` and format each
  trajectory with that model's own chat template, exactly as in the LoRA example.
- **Phase 2 (RL)** uses GRPO (Group Relative Policy Optimization) with per-prompt
  rollout groups and separate rollout and actor-update parallelism — which
  frameworks like [verl](https://github.com/volcengine/verl) and
  [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) implement: they run the rollout
  generation through vLLM with tensor parallelism and the policy update through
  FSDP, which is exactly the split in the compute table. Configure one of these
  with the base checkpoint, your reward function, and the hyperparameters above
  rather than writing the RL loop yourself.

### Serving what you trained

A fully retrained model is a new base checkpoint, not an adapter, so it is **not**
loaded with `--lora` at session start. The service serves only pre-staged,
registered models, and GPU nodes have no internet, so to serve your trained model
through ai-session it must be staged like any other served model — copied into
the model store and registered — which RCC staff do on request (see the staging
note on the [home page](index.md#available-models)). Keep the tokenizer and chat
template with the checkpoint so it serves and bills like the model it was trained
from. Until it is staged, you can serve it yourself with your own vLLM on your own
allocation.

## Getting help

Training on your own data raises questions this page cannot answer in advance
(which base to choose, how much data you need, why an adapter underperforms). For
those, and for whether a packaged training command should exist, contact
RCC staff — see [the FAQ](faq.md#who-runs-this-and-how-do-i-get-help).
