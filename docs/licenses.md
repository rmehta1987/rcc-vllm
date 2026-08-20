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
| `qwen3.8_27B` | Apache-2.0 | `<models>/Qwen3.8-27B/LICENSE` | Nothing beyond keeping the license and any `NOTICE` with redistributed weights. |
| `gemma4_31B` | Apache-2.0 | `<models>/Gemma-4-31B-it/LICENSE` | Same as above. Note Gemma 4 is Apache-2.0 and ungated, unlike earlier Gemma releases. |
| `qwen3_4b` | Apache-2.0 | `<models>/Qwen3-4B/LICENSE` | Same as above. |
| `qwen2.5_0.5B` | Apache-2.0 | `<models>/Qwen2.5-0.5B-Instruct/LICENSE` | Same as above. |
| `qwen2.5_72B` | Qwen (Tongyi) community license | `<models>/Qwen2.5-72B-Instruct/LICENSE` | Retain the Qwen attribution notice; observe the "Built with Qwen" and large-scale-use terms below. |

Models on the roadmap (see [Command Reference](reference.md#models)) will be added to
this table with their licenses when they become servable: `qwen3.5_122B`
(Qwen3.5-122B-A10B) is Apache-2.0.

Retired 2026-08-19 and removed from disk: `qwen2.5_coder_32B` and `qwen3_32B`
(both Apache-2.0), superseded by `qwen3.8_27B`; and the unserved GLM-5.2-FP8 (MIT)
and DeepSeek-V4-Flash checkpoints.

## The Apache-2.0 models

`qwen3.8_27B`, `qwen3_4b`, and `qwen2.5_0.5B` are released under
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

## The Llama 3.1 license (retired)

`llama3.1_70B` (Meta-Llama-3.1-70B-Instruct) was retired on 2026-08-19 and its weights
were removed. It was the only model here under a non-permissive licence, and the only one
requiring a per-user acknowledgment before it could be served. No obligations remain: the
model is not distributed, not served, and not staged.

Every model the service now offers is Apache-2.0 except `qwen2.5_72B`, which is under the
Qwen (Tongyi) community licence described above.

## Reading the authoritative text

Every license summarized here is quoted in full in the on-disk file listed in the
table. To read one directly on a login node (after `module load ai-session`), for
example the Qwen community licence:

```bash
less "$AISESSION_HOME/models/Qwen2.5-72B-Instruct/LICENSE"
```

If in doubt about an obligation for your specific use, read the on-disk license, which
controls, and raise questions about service policy with RCC staff.
