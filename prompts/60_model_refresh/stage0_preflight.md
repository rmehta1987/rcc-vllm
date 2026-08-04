# Stage 0 (CPU load-safety) + candidate pre-flight triage

Recorded 2026-08-04. All work here is CPU-only, run in-allocation on the `build`
node (midway3-0200, job 53017932) in the `vllm-serve-cu129` env
(`vllm==0.26.0+cu129`, `torch 2.11.0+cu129`, `torch.version.cuda==12.9`). No GPU
was reserved; no SU floor billed. This is the anti-floor-bill front matter that
must clear before any candidate reaches an H200/A100 (`session_start.md §1`).

## Gate 0 — Qwen3.5-122B-A10B-FP8 (Tier B, already staged): PASS

`tools/arch_dryrun.py models/Qwen3.5-122B-A10B-FP8` returned exit 0:

- architectures `Qwen3_5MoeForConditionalGeneration` -> SUPPORTED in the vLLM
  0.26.0 registry;
- vLLM model class imports (`load_model_cls()` on the `_LazyRegisteredModel`);
- `transformers` 5.14.1 parses the config as `Qwen3_5MoeConfig`
  (`text_config=Qwen3_5MoeTextConfig` — it is a vision-language MoE);
- tokenizer loads (`Qwen2Tokenizer`, vocab ~248044).

Gate 0 proves the architecture loads offline; it does NOT prove the FP8 kernel
runs on the target GPU — that is the Stage-1 pre-flight. FP8-on-Hopper for this
exact model is already PROVEN on this env (job 52985604, `cluster_provenance.md`:
`qwen3_5_moe` FP8/E4M3 at TP=2 loaded 122 GB in 114 s and generated code).

## Candidate pre-flight — DeepSeek-V4-Flash: the frozen NO-GO premise is REFUTED

`session_start.md §2` and the loop prompt pre-register a **V4-Flash / FP4-on-Hopper
NO-GO**: "NVFP4 compute is Blackwell (sm_100+); these H200s are Hopper (sm_90). If
vLLM 0.26.0 cannot run V4-Flash's FP4 on sm_90, that is a clean Gate-1 NO-GO." The
premise is that V4-Flash's experts are NVFP4. That premise is **factually false**
for the released `deepseek-ai/DeepSeek-V4-Flash`, per the model's own `config.json`
(fetched config-only, no weights):

```
architectures        : ['DeepseekV4ForCausalLM']
model_type           : deepseek_v4
quantization_config  : {quant_method: fp8, fmt: e4m3, scale_fmt: ue8m0,
                        weight_block_size: [128,128], activation_scheme: dynamic}
```

This is DeepSeek **block-FP8** (E4M3 weights, UE8M0 block scales, 128x128 blocks) —
the SAME quant family already proven serving on this cluster (the 122B FP8 above,
and DeepSeek-V3-style FP8 broadly). It is **not** NVFP4. Corroborating measurements
in the `vllm-serve-cu129` env:

- `DeepseekV4ForCausalLM` IS in `ModelRegistry.get_supported_archs()`.
- vLLM 0.26.0 ships a dedicated `DeepseekV4FP8Config` quant method
  (`get_min_capability()==75`), plus generic `Fp8Config` (min_cap 75). FP8 block
  quant is native on Hopper (sm_90 >= 75). (Note also: even the NVFP4 methods —
  `ModelOptNvFp4Config`, `nvfp4_per_token` — report min_cap 75 in this build, i.e.
  NVFP4 is not Blackwell-only here either; but that is moot since the model is FP8.)
- Footprint from `model.safetensors.index.json`: `total_size` = 159 609 485 896 B
  (159.6 GB / 148.6 GiB) across 46 shards. That is ~1 byte/param -> ~160B params in
  FP8 (the plan's "~148 GB" size was right; its "NVFP4 / 284B" quant reasoning was
  wrong). 159.6 GB fits one 4xH200 node (4 x 141 = 564 GB) at TP=4 — no hand-written
  data-parallel path is needed (contra `session_start.md §2` "DP=4").

**Decision: V4-Flash is NOT a pre-flight NO-GO. It proceeds to Stage D.** This is an
evidence-based correction of a wrong factual premise (the model's declared quant),
not a moved gate: the frozen gate is conditional ("IF FP4 unsupported on Hopper"),
and its antecedent does not hold because the model is not FP4. The honest Tier-B
comparison therefore runs BOTH Qwen3.5-122B and V4-Flash, as originally intended.
Memory [[project_coding_model_landscape_2026]] currently records the inferred NVFP4
assumption and must be corrected at verdict.

## Stage D — downloads launched (build allocation, background, Xet-disabled)

Both un-staged candidates are downloading in-allocation via `tools/stage_model_bg.sh`
(mirrors `sbatch_stage_glm52.sh` minus the `#SBATCH` header; `HF_HUB_DISABLE_XET=1`,
retry loop, exact tensor-byte Gate-D verify). Neither is a nested `build` job (the
`build` QOS is one job per user; the orchestrator holds it — `session_start.md §0`).
Progress/pids persisted to `_scratch/stage_d_tiera.json` and
`_scratch/stage_d_tierb_v4.json`; Gate D (all shards present + non-zero, no
`.incomplete`, tensor bytes == index `total_size`) is verified per candidate when
its loop exits.

- Tier A: `Qwen/Qwen3-Coder-30B-A3B-Instruct` -> `models/Qwen3-Coder-30B-A3B-Instruct`
  (16 safetensors shards, ~61 GB).
- Tier B: `deepseek-ai/DeepSeek-V4-Flash` -> `models/DeepSeek-V4-Flash`
  (46 shards, 159.6 GB, FP8).

## Compute provenance (this stage)

No GPU job. CPU work only: `arch_dryrun.py` (122B) and two config-only HF fetches
(V4-Flash `config.json` + `model.safetensors.index.json`) in the build allocation.
