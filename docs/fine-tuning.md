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

The served models are **text** language models. A LoRA adapter on them is worth
producing when a task is a **repeated pattern over many text inputs** — a fixed
output format, a mapping from free text to a schema, a house writing or coding
style — that you cannot get reliably from prompting alone. It is the wrong tool
for several adjacent problems:

| Your goal | Better tool |
|---|---|
| A repeated text-to-text pattern over many inputs (this page) | LoRA adapter |
| Answering from a body of facts or documents | Retrieval (embeddings), not training |
| A single transformation with the material in hand | A coding session with the file in context |
| Classifying images, or inference over a simulator (SBI) | A vision or density model — not a text model, so not this service |

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

## Getting help

Training on your own data raises questions this page cannot answer in advance
(which base to choose, how much data you need, why an adapter underperforms). For
those, and for whether a packaged training command should exist, contact the
service operators — see [the FAQ](faq.md#who-runs-this-and-how-do-i-get-help).
