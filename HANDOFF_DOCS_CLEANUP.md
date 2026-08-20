# HANDOFF — documentation cleanup after the 2026-08 model refresh

**Green floor: `b27780b`.** All code, billing and hardware changes are committed and the
working tree is clean. This handoff is a **documentation-only** task. Do not change serving
behaviour, `MODEL_REGISTRY`, `PHASE1_SERVED`, `rate_table.json`, or any pin while doing it.

## Why this is needed

Over 2026-08-19/20 the fleet went from thirteen models (1.79 TB) to four served models
(395 GB). Seven models were deleted, the coding default changed twice, a second coding
option was added, every rate row was re-measured on a new vLLM, and the GPU targeting rule
was rewritten. The docs were updated *incrementally alongside each change*, which means they
are individually accurate but collectively carry sediment: retirement notices for models
nobody will ever look up, licence sections for weights that no longer exist, and a lot of
"this changed on 2026-08-19" scaffolding that was useful during the transition and is now
noise.

The goal is docs that read as if the current state had always been the state — while keeping
the small number of caveats a user genuinely needs.

## Verify the current state first

Do not trust this document's summary of what is served. Read it from the source:

```bash
cd /project/rcc/mehta5/vllm
python3 -c "import sys;sys.path.insert(0,'ai-session');import server;print(sorted(server.PHASE1_SERVED))"
python3 -c "import json;d=json.load(open('billing/rate_table.json'));[print(r['model_key'],r['gpu_tier'],r['tp'],r['decode_tps'],r['su_per_1k_out']) for r in d['records']]"
ls models/
```

As of `b27780b`: served = `qwen2.5_72B`, `qwen3.8_27B`, `gemma4_31B`, `qwen3_4b`.
Registered-not-served = `qwen3.5_122B` (FP8, Hopper-only), `qwen2.5_0.5B` (staff smoke).
Also on disk but NOT registered: `Gemma-4-31B-it-qat-w4a16` (22 GB, staged and Gate-1 PASS,
quality unmeasured — see "leave alone" below).

## Task 1 — remove retirement scaffolding

These models are gone from disk and from the registry. A user has no way to reach them and
no reason to read about them:

`qwen2.5_coder_32B`, `qwen3_32B`, `llama3.1_70B`, `glm5.2_753B`, plus the deleted
Qwen3.6-35B-A3B, Qwen3-Coder-Next, Qwen3-Coder-30B-A3B, DeepSeek-V4-Flash.

Current stale references:

| file | what to do |
|---|---|
| `docs/index.md`, `docs/coding/mcp.md`, `docs/faq.md`, `docs/troubleshooting.md`, `docs/coding/opencode.md`, `docs/coding/agents.md` | all mention "Coder-32B" in historical framing — cut to present tense |
| `docs/reference.md` | has a "Retired 2026-08-19 and deleted from disk" paragraph — delete it entirely |
| `docs/billing.md` | rate table keeps a `qwen2.5_coder_32B *(retired)*` row — delete the row; keep the *provenance* note that stale-version rows bill the floor, which is still true of the `qwen2.5_72B` h100 row |

**Judgement call, and it is yours to make:** the tool-calling sections in `opencode.md`,
`agents.md`, `mcp.md`, `troubleshooting.md` and `faq.md` currently explain at length that the
OLD model could not emit tool calls and the NEW ones can. Users arriving today do not need
the history — they need "tool calling works, here is how". Compress each to the present
tense. The one thing worth keeping is the **explicit instruction to delete an old
`AGENTS.md` workaround file** if the reader has one from the previous instructions, because
that file is actively counterproductive with the current models. Do not lose that.

## Task 2 — cut the licensing content down

`docs/licenses.md` is 92 lines for a fleet where **every served model is Apache-2.0 except
one**. Current sections:

```
## License at a glance
## The Apache-2.0 models
## The Qwen 72B community license
## The Llama 3.1 license (retired)     <- delete outright
## Reading the authoritative text
```

Target: a short page. A table of the four served models and their licences; a paragraph on
the Qwen (Tongyi) community licence for `qwen2.5_72B`, which is the only one with real
obligations (attribution, "Built with Qwen", large-scale-use terms); one line noting the rest
are Apache-2.0 with nothing required beyond keeping the licence file; and the `less
<models>/…/LICENSE` pointer. Delete the retired-Llama section — there are no obligations to
retain for weights we do not hold and do not serve.

Check the other pages for licence duplication and remove it from there: the model tables in
`docs/reference.md` and `docs/index.md` each carry a licence column, which is fine, but any
prose *explaining* licences outside `licenses.md` should just link to it.

While there: `ai-session/CODING_AGENTS.md` is 591 lines and is the user-facing coding-agent
guide. It predates both coding-model changes. Audit it, but it is a bigger job than the docs
site and can be a separate pass if time is short.

## Task 3 — the transition scaffolding

Search for and remove date-stamped "this changed" framing that has served its purpose:

```bash
grep -rn "2026-08-19\|2026-08-20\|changed on\|as of 2026-08\|no longer required\|Retired" docs/
```

Keep dates only where a user needs them to interpret a number — for instance "measured
2026-08-20 on vLLM 0.26.0" attached to a throughput figure is useful provenance. Remove them
where they are narrating a migration.

## Caveats that MUST survive

Do not tidy these away. Each is a live, user-affecting fact, and each was measured:

1. **`gemma4_31B` bills the reservation floor on any tier it lands on** where the rate row
   does not match, and **prefer A40 over A100** — A40 is faster *and* half the price for this
   model (349 vs 336 tok/s decode; 0.000797 vs 0.001652 SU per 1k out). The "prefer A40" note
   in `docs/coding/overview.md` is counterintuitive and must stay.
2. **Enabling Gemma's thinking costs 5–12× tokens and wall time** with no measured quality
   gain. Off by default; opt in per request. Under GPU-time billing that is a 5–12× cost
   multiplier.
3. **`qwen3.5_122B` requires H100/H200** because its weights are FP8 and Ampere has no FP8
   tensor cores — and on this cluster Hopper is PI-owned, so most users cannot start it. The
   "Which GPUs each model needs" section in `docs/reference.md` covers this; keep it, and keep
   the sentence telling users who lack that access that `qwen3.8_27B` is the better choice
   anyway.
4. **The two coding models differ in character, not just quality.** Qwen thinks by default and
   suits hard problems; Gemma is cheaper and faster by default and scored higher on our
   benchmark. The "Choosing between the two coding models" section in
   `docs/coding/overview.md` is the most useful page for a new user — keep it, tighten it.
5. **The benchmark comparison is not like-for-like.** Gemma's 66.67% vs Qwen's 50.00% was
   measured with thinking disabled, which is Gemma's native default but suppresses Qwen's
   `xhigh`. Wherever those two numbers appear together, that caveat appears too.

## Do NOT mention

- The `test` partition. It is `Hidden=YES` and staff-only; users should not learn of it.
  It appears in `CLAUDE.md` and some `#SBATCH` defaults, which is fine — those are staff
  files. It must not appear in `docs/`.
- `Gemma-4-31B-it-qat-w4a16`. Staged and proven to serve on one A40 at a 0.5 SU/h floor, but
  its **coding quality is entirely unmeasured** and Google publishes no quantized-vs-BF16
  comparison. It is not registered and not served. Documenting it would advertise something
  users cannot start.

## Definition of done

```bash
cd /project/rcc/mehta5/vllm
/project/rcc/mehta5/mkdocs-env/bin/mkdocs build --strict -d /tmp/docs-check   # must pass
grep -rn "qwen2.5_coder_32B\|qwen3_32B\|llama3.1_70B\|glm5.2_753B" docs/      # must be empty
grep -rniE "test partition|--partition=test" docs/                            # must be empty
```

Then re-read `docs/index.md` and `docs/coding/overview.md` end to end as a new user would.
If either still reads like a migration log, it is not done.

Commit as a docs-only change. Do not push — the user pushes to `origin` themselves, and
never to `upstream`.

## House style

Scientist-to-scientist prose. No checkmarks, no emoji, no marketing language. Numbered
steps, exact commands, tables, and measured numbers with their provenance. Prefer deleting a
paragraph to hedging it.
