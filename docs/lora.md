# Using Your Own Fine-Tuned Model

If you have fine-tuned one of the served models with LoRA (the common
parameter-efficient method; tools such as Hugging Face PEFT, Axolotl,
LLaMA-Factory, and Unsloth all produce it), a session can serve your adapter
alongside the base model. You choose per request which one answers: requests
that name your adapter get the fine-tuned behavior, requests that name the base
model get the stock model. Nothing about the rest of the workflow changes —
same commands, same URL, same key, same billing.

The service does not run the fine-tuning itself yet; you train elsewhere (for
example on your own GPU allocation) and bring the resulting adapter directory.
For a worked example of producing that adapter — dataset format, training script,
and submission — see [Fine-Tuning a Model on Your Own Data](fine-tuning.md).

## What you need

1. **An adapter directory** containing `adapter_config.json` and the adapter
   weights (`adapter_model.safetensors` or `adapter_model.bin`). This is what a
   PEFT `save_pretrained` call writes. A full fine-tune (a directory of
   multi-gigabyte `model-*.safetensors` shards without `adapter_config.json`)
   is not an adapter and cannot be loaded this way.
2. The directory must be **on project or scratch storage** and readable by you
   from GPU nodes (not only from your laptop). An absolute path is required.
3. The adapter must have been **trained on the same base model the session
   serves**. An adapter trained on Qwen2.5-Coder-32B belongs in a `code`
   session; one trained on Qwen2.5-72B-Instruct belongs in a `chat` session.
   The start command warns if the adapter's recorded base does not match, but
   it cannot verify quality — a mismatched adapter usually produces poor
   output rather than an error.

## Start a session with your adapter

Add `--lora NAME=PATH` to any start command. `NAME` is the model name you will
type in your client; choose something short and memorable.

```bash
ai-session code --lora myft=/project/<pi>/adapters/my-coder-adapter
```

The option is repeatable, so several colleagues' adapters (or several
checkpoints of one training run) can be served in the same session:

```bash
ai-session code \
  --lora run3=/project/<pi>/adapters/run3 \
  --lora run4=/project/<pi>/adapters/run4
```

The adapter set is fixed for the life of the session; to add or swap one, stop
and start again. Every adapter is validated before anything is reserved, so a
typo in a path fails immediately and costs nothing.

## Use it

Wherever a client asks for a model name, give the adapter's `NAME` instead of
the base model's:

- **Browser chat:** the adapter appears in the model picker next to the base
  model; select it.
- **aider:** `aider --model openai/myft ...` (everything else as on the
  [aider page](coding/aider.md)).
- **A script:**

```bash
eval "$(ai-session env)"
curl -s "$AISESSION_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $AISESSION_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "myft", "messages": [{"role": "user", "content": "Hello"}]}'
```

Requests that keep the base model's name (for example `qwen2.5_coder_32B`)
are answered by the unmodified base model, in the same session, at no extra
cost — useful for A/B-comparing your fine-tune against stock behavior.

## Cost

A session with adapters costs the same as one without: the per-hour holding
cost of the GPUs it reserves, as described on
[Billing and Service Units](billing.md). Serving an adapter adds a small
per-request overhead but no additional charge.

## Limits

| Limit | Value |
|---|---|
| Adapter format | LoRA (PEFT-style directory with `adapter_config.json`) |
| LoRA rank (`r`) | up to 256 |
| Base model | must match the model the session serves |
| Changing adapters | fixed per session; restart to change |
| Path | absolute, on project/scratch storage, no spaces |

If your adapter exceeds these limits (a higher rank, a full fine-tune, a
different base model), contact RCC staff — see
[the FAQ](faq.md#who-runs-this-and-how-do-i-get-help).
