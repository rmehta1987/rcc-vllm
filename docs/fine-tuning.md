# Fine-Tuning a Model on Your Own Data

This page shows how to produce a LoRA adapter — a small set of extra weights that
adjust one of the served models toward your own code, data, or writing style — and
then serve it. Once you have an adapter, [Using Your Own Fine-Tuned
Model](lora.md) covers loading it into a session; this page is the step before
that, where the adapter is made.

!!! note "What the service provides, and what you run yourself"
    The **serving** side is built and verified: `ai-session <preset> --lora
    NAME=PATH` loads a finished adapter (see [Using Your Own Fine-Tuned
    Model](lora.md)). The **training** side below is a worked recipe you run on
    your **own** GPU allocation — Slurm bills your account directly, and no
    Service Units are involved. The service does not yet ship a packaged,
    one-command training tool, so treat the script here as a starting point to
    adapt, not a tested black box. The two settings it exists to get right — the
    base checkpoint and the chat template — are called out below.

## First decide whether you need to fine-tune at all

Fine-tuning teaches a model *durable patterns* from many examples: a house
coding style, a domain's conventions, an API it should reach for by default. It
is the wrong tool for a **one-time task** you could do with the code already in
front of you.

A common example: "I have a statistical-genetics codebase that computes true
effect sizes from noisy GWAS estimates, and I want to port it to PyTorch." That
is a translation of one existing codebase, not a durable pattern. Fine-tuning
would be slow and worse than the direct route, which is to open the repository in
a coding tool and let it read your actual files:

```bash
cd /path/to/statgen-repo
module use /project/rcc/mehta5/modulefiles
module load ai-session
ai-session code            # start a session; then, in the repo, run the printed aider command
```

Then ask in plain language, for example: *"Port `shrinkage.py` from NumPy to
PyTorch. Keep the function signatures, make the estimators differentiable so
`betahat` can carry gradients, and move the array math to torch ops."* The tool
works from your exact code, which beats any fine-tune for a single conversion.
See [Coding Sessions](coding/overview.md).

Fine-tune instead when you will write **new** code of this kind repeatedly and
want the model to default to your idioms — your naming (`betahat`, `se`,
`tau2`), your convention that estimators stay differentiable and batched over
SNPs — without being told each time. The rest of this page works that example
through end to end.

## What you need

1. **A dataset of examples** of the behavior you want, as a JSONL file with one
   chat conversation per line (`{"messages": [...]}`). Fifty to a few hundred
   hand-checked examples is a good range; quality matters far more than volume,
   because every assistant turn is something the model will learn to imitate.
2. **Your own GPU allocation** (your PI's Slurm account and partition). Training
   does not run under the service; it runs as an ordinary batch job on your
   account.
3. **A training environment** — a Python virtual environment with `torch`,
   `transformers`, `peft`, `trl`, and `datasets`. Build it once on a login node,
   which has internet; nothing for training is preinstalled.

## Step 1: Build the dataset

Each line pairs a request with the code you would sign off on. For the
statistical-genetics example, the examples teach the model your PyTorch idioms
for shrinkage and empirical-Bayes estimators:

```json title="train.jsonl (one object per line; shown pretty-printed)"
{"messages": [
  {"role": "user", "content": "Port this James-Stein shrinkage estimator to PyTorch. Keep it differentiable and batched over SNPs.\n\ndef james_stein(betahat, se):\n    z = betahat / se\n    p = len(z)\n    shrink = max(1 - (p - 2) / np.sum(z**2), 0)\n    return shrink * betahat"},
  {"role": "assistant", "content": "import torch\n\ndef james_stein(betahat, se):\n    # betahat, se: (n_snps,) tensors; returns shrunk effect sizes, differentiable in betahat\n    z = betahat / se\n    p = betahat.numel()\n    shrink = torch.clamp(1.0 - (p - 2) / (z**2).sum(), min=0.0)\n    return shrink * betahat"}
]}
{"messages": [
  {"role": "user", "content": "Write the empirical-Bayes posterior mean for effect sizes under a N(0, tau2) prior, PyTorch, batched over SNPs."},
  {"role": "assistant", "content": "import torch\n\ndef eb_posterior_mean(betahat, se, tau2):\n    # normal-means model: betahat ~ N(beta, se^2), beta ~ N(0, tau2)\n    # per-SNP shrinkage weight; tau2 can be a leaf tensor fit by marginal likelihood\n    w = tau2 / (tau2 + se**2)\n    return w * betahat"}
]}
{"messages": [
  {"role": "user", "content": "Give me the negative marginal log-likelihood of tau2 for the normal-means model so I can fit it with torch.optim."},
  {"role": "assistant", "content": "import torch\n\ndef neg_marginal_loglik(tau2, betahat, se):\n    # marginal: betahat ~ N(0, tau2 + se^2); sum over SNPs, minimize w.r.t. tau2\n    var = tau2 + se**2\n    return 0.5 * (torch.log(2 * torch.pi * var) + betahat**2 / var).sum()"}
]}
```

The file itself has one JSON object per physical line (no pretty-printing); the
blocks above are expanded only for reading. Put the file on project or scratch
storage — for example `/project/<pi>/statgen/train.jsonl`.

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
inference will use. Save this as `train_lora.py`:

```python title="train_lora.py"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 1. SAME checkpoint the service serves -- not a fresh Hugging Face download.
BASE = "/project/rcc/mehta5/vllm/models/Qwen2.5-Coder-32B-Instruct"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto")

# 2. Format each example with the base model's OWN chat template.
ds = load_dataset("json", data_files="/project/<pi>/statgen/train.jsonl", split="train")
ds = ds.map(lambda r: {"text": tok.apply_chat_template(r["messages"], tokenize=False)})

lora = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"])

trainer = SFTTrainer(
    model=model, train_dataset=ds, peft_config=lora,
    args=SFTConfig(
        output_dir="/project/<pi>/adapters/statgen-torch",
        num_train_epochs=3, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=1e-4,
        bf16=True, dataset_text_field="text", max_seq_length=4096))
trainer.train()
trainer.save_model()   # writes adapter_config.json + the adapter weights
```

Rank 32 is well inside the serving cap of 256. To fine-tune the general model or
the small model instead, point `BASE` at
`/project/rcc/mehta5/vllm/models/Qwen2.5-72B-Instruct` or
`/project/rcc/mehta5/vllm/models/Qwen3-4B`, and serve the matching preset later.

## Step 4: Submit the training job

Training runs on **your** allocation, so submit it with your own account and
partition. A minimal `train_lora.sbatch`:

```bash title="train_lora.sbatch"
#!/bin/bash
#SBATCH --job-name=lora-statgen
#SBATCH --account=<your-account>
#SBATCH --partition=<your-gpu-partition>
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00

source ~/lora-train-env/bin/activate
python train_lora.py
```

```bash
sbatch train_lora.sbatch
```

A three-epoch, rank-32 run on a few hundred examples with the Coder-32B base
takes low single-digit GPU-hours on one or two A100-80GB cards. Slurm bills your
account for that time; the ai-session service and its Service Units are not
involved in training at all.

When the job finishes, `/project/<pi>/adapters/statgen-torch` contains
`adapter_config.json` and the adapter weights — exactly what the serving side
validates.

## Step 5: Serve your adapter

Start a session with the adapter loaded, giving it a short name you will type in
your client:

```bash
ai-session code --lora statgen=/project/<pi>/adapters/statgen-torch
```

Request `model=statgen` and the model writes in the style you taught; request the
base model (`qwen2.5_coder_32B`) in the same session to compare against stock
behavior at no extra cost. The full serving details — per-client usage, cost, and
limits — are on [Using Your Own Fine-Tuned Model](lora.md).

## Getting help

Training on your own data raises questions this page cannot answer in advance
(which base to choose, how much data you need, why an adapter underperforms). For
those, and for whether a packaged training command should exist, contact the
service operators — see [the FAQ](faq.md#who-runs-this-and-how-do-i-get-help).
