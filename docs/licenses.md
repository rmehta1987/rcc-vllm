# Model licenses

The model weights served by ai-session are governed by the license each model
publisher attached to them. When you run a private session for yourself, you are the
licensee using the model. When you share your session's access key with labmates so
they reach it over their own tunnels — the sharing model described on
[Coding Sessions](coding/overview.md#the-session-access-key) and in the
[command reference](reference.md#the-session-access-key) — you are making the model
available to other people, which is where the licenses differ. This page states, per
model, what the license is, where its authoritative text sits on disk, and what you
must do when you serve it to others. It is a practical summary, not legal advice; the
on-disk license file is the controlling text in every case.

## License at a glance

In the table, `<models>` is the service's model store, `$AISESSION_HOME/models`
after `module load ai-session`.

| Model key | License | On-disk license file | What serving to others requires |
|---|---|---|---|
| `qwen2.5_coder_32B` | Apache-2.0 | `<models>/Qwen2.5-Coder-32B-Instruct/LICENSE` | Nothing beyond keeping the license and any `NOTICE` with redistributed weights. |
| `qwen3_4b` | Apache-2.0 | `<models>/Qwen3-4B/LICENSE` | Same as above. |
| `qwen3_32B` | Apache-2.0 | `<models>/Qwen3-32B/LICENSE` | Same as above. |
| `qwen2.5_0.5B` | Apache-2.0 | `<models>/Qwen2.5-0.5B-Instruct/LICENSE` | Same as above. |
| `qwen2.5_72B` | Qwen (Tongyi) community license | `<models>/Qwen2.5-72B-Instruct/LICENSE` | Retain the Qwen attribution notice; observe the "Built with Qwen" and large-scale-use terms below. |
| `llama3.1_70B` | Llama 3.1 Community License + Acceptable Use Policy | `<models>/Meta-Llama-3.1-70B-Instruct/LICENSE` and `.../USE_POLICY.md` | Provide the license, display "Built with Llama", follow the Acceptable Use Policy, and record acceptance via the [acknowledgment gate](#serving-llama-31-the-acknowledgment-gate). |

Models on the roadmap (see [Command Reference](reference.md#models)) will be added
to this table with their licenses when they become servable: `qwen3.5_122B`
(Qwen3.5-122B-A10B) is Apache-2.0; the GLM-5.1 / GLM-5.2 license terms will be
recorded here before those models are offered.

## The Apache-2.0 models

`qwen2.5_coder_32B`, `qwen3_4b`, `qwen3_32B`, and `qwen2.5_0.5B` are released under
the Apache License, Version 2.0. Apache-2.0 is a permissive open-source license: it places no
restriction on serving the model to other users and imposes no in-product attribution
requirement. The only standing obligation is that if you redistribute the weights
themselves (copy them elsewhere, not merely serve inference from them), you keep the
`LICENSE` file and any accompanying `NOTICE` with the copy. Serving these models
through ai-session, whether to yourself or to your lab, needs nothing further.

## The Qwen 72B community license

`qwen2.5_72B` (Qwen2.5-72B-Instruct) is released under the Qwen LICENSE AGREEMENT, a
community license from Alibaba Cloud rather than an OSI-approved open-source license.
It permits research and commercial use but attaches obligations that matter once the
model is offered to others as a service:

- Attribution. Copies you distribute must retain, in a `Notice` text file, the exact
  line the license specifies: "Qwen is licensed under the Qwen LICENSE AGREEMENT,
  Copyright (c) Alibaba Cloud. All Rights Reserved."
- "Built with Qwen". If you use the model's outputs to train, fine-tune, or otherwise
  improve another AI model that you then distribute or make available, you must
  prominently display "Built with Qwen" or "Improved using Qwen" in that product's
  documentation.
- Changed files. If you modify the materials, the modified files must carry prominent
  notices stating that you changed them.
- Large-scale commercial use. If you use the materials commercially in a product or
  service with more than 100 million monthly active users, you must request a
  separate license from Alibaba Cloud. This threshold is far above any RCC lab
  setting and is noted only for completeness.
- Trademarks. No trademark license is granted beyond what is needed to satisfy the
  attribution requirement above.

When you host `qwen2.5_72B` for your lab through ai-session, the practical duty is the
attribution notice; the full and controlling terms are in the on-disk `LICENSE` above.

## The Llama 3.1 license

`llama3.1_70B` (Meta-Llama-3.1-70B-Instruct) is released under the Llama 3.1 Community
License Agreement, accompanied by a separate Acceptable Use Policy
(`USE_POLICY.md`). Like the Qwen license, it is a community license with conditions,
and it is the most restrictive of the models staged here. Serving it to others carries
these obligations:

- Provide the license. If you distribute or make the Llama materials (or a product
  using them) available to a third party, you must provide a copy of the Llama 3.1
  Community License Agreement with it.
- "Built with Llama". You must prominently display "Built with Llama" on a related
  website, interface, or documentation when you make the model available.
- Naming. If you use Llama to create, train, or improve another AI model that you
  distribute, its name must begin with "Llama".
- Attribution. Retain, in a `Notice` file, the line the license specifies: "Llama 3.1
  is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc.
  All Rights Reserved."
- Acceptable Use Policy. Your use, and the use by anyone you serve, must comply with
  the Acceptable Use Policy in `USE_POLICY.md` (which prohibits a specific list of
  harmful uses).
- Large-scale commercial use. If the product or service using Llama exceeded 700
  million monthly active users on the release date, you must request a license from
  Meta. As with Qwen, this is noted only for completeness.

Because these duties bind whoever offers the model, ai-session refuses to serve
`llama3.1_70B` until you have acknowledged them, as described next.

### Serving Llama 3.1: the acknowledgment gate

`llama3.1_70B` is served to any user — Llama 3.1 is free to run for research on
university hardware — but the first attempt to serve it is refused until you
record that you accept the license. This is a deliberate, non-interactive gate so
that scripts can satisfy it once and proceed: the serving command is refused until
it is run with `ACCEPT_LLAMA_LICENSE=1` set in the environment. The first time:

```bash
ACCEPT_LLAMA_LICENSE=1 ai-session chat --model llama3.1_70B
```

Setting `ACCEPT_LLAMA_LICENSE=1` writes a one-time acceptance record to your per-user
state directory at `<state-dir>/logs/licenses/<user>_llama3.1_70B.accepted`. The
record holds the timestamp, your username, the model key, and the on-disk license
path you accepted. Once it exists, later starts reuse it and need no environment
variable.

Without the acknowledgment, and with no record already on file, the start is
refused before any GPU is reserved, and the refusal prints the on-disk license
paths and the variable to set.

The Apache-2.0 models are permissive and are not gated; `qwen2.5_0.5B` is a
smoke-test checkpoint (not a user model) and needs no license acknowledgment.
`qwen2.5_72B` is served and is not license-gated, but its attribution obligation
above still applies when you host it for others.

## Reading the authoritative text

Every license summarized here is quoted in full in the on-disk file listed in the
table. To read one directly on a login node (after `module load ai-session`), for
example the Llama license and its Acceptable Use Policy:

```bash
less "$AISESSION_HOME/models/Meta-Llama-3.1-70B-Instruct/LICENSE"
less "$AISESSION_HOME/models/Meta-Llama-3.1-70B-Instruct/USE_POLICY.md"
```

If in doubt about an obligation for your specific use, read the on-disk license, which
controls, and raise questions about service policy with the ai-session operators.
