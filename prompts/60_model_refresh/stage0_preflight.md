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

## Candidate pre-flight — DeepSeek-V4-Flash: NOT a clean CPU NO-GO (proceeds)

CORRECTION of the first draft of this file (commit 2810f70), which claimed V4-Flash
is "FP8, not FP4." That was WRONG: it read only the top-level `quantization_config`
and missed the top-level `expert_dtype` field. The released
`deepseek-ai/DeepSeek-V4-Flash` is **mixed precision** (`config.json`, fetched
config-only, no weights):

```
architectures        : ['DeepseekV4ForCausalLM']    model_type: deepseek_v4
quantization_config  : {quant_method: fp8, fmt: e4m3, scale_fmt: ue8m0,
                        weight_block_size: [128,128]}   # dense/attention linear
expert_dtype         : "fp4"                            # the 256 routed experts
n_routed_experts=256, n_shared_experts=1, num_experts_per_tok=6, num_hidden_layers=43
```

So the memory [[project_coding_model_landscape_2026]] is RIGHT that the experts are
4-bit; it is imprecise only in calling them "NVFP4" (this checkpoint is **MXFP4**,
see below) and in "DP=4" (TP=4 fits, see footprint). The frozen plan's FP4-on-Hopper
NO-GO is therefore live and must be adjudicated on real vLLM-0.26.0 support, not on
the generic hardware fact that native FP4 tensor cores are Blackwell.

Adjudication (measured against the installed vLLM 0.26.0 source, `vllm-serve-cu129`):

1. **Sparse MLA attention runs on Hopper.** DeepSeek-V4 uses DeepSeek Sparse
   Attention (sparse MLA). vLLM ships two backends:
   `vllm/models/deepseek_v4/sparse_mla.py::DeepseekV4FlashMLABackend` (name
   `FLASHMLA_SPARSE_DSV4`) whose `supports_compute_capability` returns
   `capability.major in [9, 10]` — **9 is Hopper** — and
   `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` (name
   `FLASHINFER_MLA_SPARSE_DSV4`) which is Blackwell-only ("requires SM10x or
   SM12x"). On sm_90 the FlashMLA backend is the one selected; Hopper is supported.
2. **The FP4 experts are MXFP4, and MXFP4 has a Hopper backend.**
   `vllm/models/deepseek_v4/quant_config.py::DeepseekV4FP8Config.get_quant_method`
   routes `RoutedExperts` with `expert_dtype=="fp4"` (and no
   `moe_quant_algo=="NVFP4"`, which this config does not declare) to
   `Mxfp4MoEMethod` — its docstring: "MXFP4 experts with ue8m0 FP8 linear scales."
   `Mxfp4Config.get_min_capability()==80` (sm_80+), and
   `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py::select_deepseek_v4_mxfp4_moe_backend`
   walks a priority list and returns the first backend whose `is_supported_config`
   passes: on sm_90 the Blackwell backends (TRTLLM / FlashInfer-cutlass) report
   unsupported and it falls back to a Triton/Marlin MXFP4 backend. (MXFP4-on-Hopper
   is the established gpt-oss path; DeepGEMM is disabled by `tools/serve_cu129.sbatch`
   so `DEEPGEMM_MXFP4` is never selected.) `NotImplementedError` fires only if NO
   backend matches — not the case on sm_90.
3. **FP8 dense/attention linear** is native on Hopper (proven, the 122B above).
4. **Footprint** from `model.safetensors.index.json`: `total_size` =
   159 609 485 896 B (159.6 GB / 148.6 GiB) across 46 shards — fits one 4xH200 node
   (4 x 141 = 564 GB) at TP=4 (no hand-written data-parallel path needed).

**Decision: V4-Flash is NOT a clean CPU NO-GO. It proceeds to Stage D.** Every
kernel path (sparse MLA, MXFP4 experts, FP8 dense) has a declared Hopper code path
in vLLM 0.26.0, so the frozen NO-GO ("IF FP4 unsupported on Hopper") does not fire
at CPU pre-flight — the antecedent is not established. The definitive test is the
Stage-1 GPU serve, preceded by the cheap quant-support pre-flight of
`session_start.md §4` (compute-capability probe for the exact MXFP4-MoE +
FlashMLA-sparse kernels BEFORE the full load). If either kernel fails to
build/run on sm_90 there, THAT is the clean Gate-1 NO-GO — it fails cheap, before
the full load, and is NOT retried 5x (per the frozen plan). This preserves the
anti-floor-bill fence: the H200 reservation is entered only after CPU pre-flight
finds a plausible Hopper path, and the GPU pre-flight bails before the full load if
the kernel is in fact Blackwell-only.

## Stage D — downloads launched (build allocation, background, Xet-disabled)

Both un-staged candidates download in-allocation via `tools/stage_model_bg.sh`
(mirrors `sbatch_stage_glm52.sh` minus the `#SBATCH` header; `HF_HUB_DISABLE_XET=1`,
retry loop, exact tensor-byte Gate-D verify). Neither is a nested `build` job (the
`build` QOS is one job per user; the orchestrator holds it — `session_start.md §0`).
Pids/logs persisted to `_scratch/stage_d_tiera.json` and
`_scratch/stage_d_tierb_v4.json`; Gate D (all shards present + non-zero, no
`.incomplete`, tensor bytes == index `total_size`) is verified per candidate when
its loop exits.

- Tier A: `Qwen/Qwen3-Coder-30B-A3B-Instruct` -> `models/Qwen3-Coder-30B-A3B-Instruct`
  (16 safetensors shards, ~61 GB).
- Tier B: `deepseek-ai/DeepSeek-V4-Flash` -> `models/DeepSeek-V4-Flash`
  (46 shards, 159.6 GB; FP8 dense + MXFP4 experts).

## Compute provenance (this stage)

No GPU job. CPU work only: `arch_dryrun.py` (122B), two config-only HF fetches
(V4-Flash `config.json` + `model.safetensors.index.json`), and a read of the
installed vLLM 0.26.0 DeepSeek-V4 source, all in the build allocation.
