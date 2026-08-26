# Project rules — DGX Spark LLM service

A single-box vLLM service: three users, one GB10 GPU, models served over an OpenAI-compatible
API to a browser client and three coding clients. Nothing on the serving path leaves this
machine.

**Every hardware number below is `UNVERIFIED` — this file was written off-box.** Each carries
the command that settles it. Run them before relying on anything here, and replace the number
*and* the marker in the same edit.

---

## 1. Hardware reality

### 1.1 Memory budget — in gigabytes, and it comes first

Unified memory is not VRAM. `--gpu-memory-utilization` is a fraction of the pool the OS, the
desktop session, Open WebUI and the gateway are also using. On a discrete GPU an over-commit
kills the job; here it causes system-wide pressure and OOM-killed processes. **vLLM's ordinary
0.9 default is unsafe on this box.**

- **Placeholder budget: 100 GB serve / 20 GB OS floor / 8 GB margin.** `UNVERIFIED`, derived
  from 128 GB × 0.85 ≈ 108 then held back further. **Do not allocate against it** — it is a
  stand-in until the derivation below has been run.
- Derive the real numbers: `budget = free unified memory − OS floor − margin`, measured with
  **Open WebUI and the gateway already resident**, not on an idle box.
  ```bash
  systemctl --user start ai-gateway open-webui   # bring the resident services up first
  free -g                                        # then read the pool
  ```
- Prefer `--kv-cache-memory` (absolute) to `--gpu-memory-utilization` (fraction) wherever the
  pinned build supports it. If it does not, compute the fraction as `budget ÷ total pool` and
  record it — never fall back to a default.
- A preflight must fail before loading weights unless `MemAvailable` ≥ weights bytes +
  `--kv-cache-memory` + 5 GB, printing all three terms. Do not let a load OOM the box.

### 1.2 The platform

| assumed | verify with |
|---|---|
| GB10 Grace-Blackwell, one GPU, sm_121 | `nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv` |
| 128 GB LPDDR5X unified, **shared with the OS** | `free -g`; `nvidia-smi -q -d MEMORY` — same pool? |
| ~273 GB/s memory bandwidth (vendor spec) | no read-only probe exists; run a STREAM-class microbenchmark and record it |
| 20-core Arm, **aarch64** | `uname -m` |
| DGX OS, container runtime present, **CUDA 13.0** | `cat /etc/os-release`; `docker info`; `nvcc --version` |
| ~4 TB NVMe | `df -h` — the weights budget is whatever is actually free |

### 1.3 Decode is bandwidth-bound

```
tok/s_ceiling ≈ bandwidth / bytes_read_per_token
bytes_read_per_token ≈ active_parameter_bytes + KV_bytes_read
```

**Interactive speed is governed by bytes read per token, not parameter count.** Size any
checkpoint with this before downloading it.

Measured on this hardware — **a field report supplied by the box owner, 2026-08-26, not
measured by this project**; checkpoint `unsloth/Qwen3.8-27B-NVFP4`:

| | |
|---|---|
| decode at 0 / 32K / 100K context | 11.5 / 10.9 / 9.8 tok/s |
| aggregate, 10 concurrent streams | 84.3 tok/s (~8.4 per user) |
| prefill | ~4,000 tok/s |
| single-stream under Ollama (speculative decoding on by default) | 26.5 tok/s |

Three consequences:

1. **Realized decode was 98 % of the computed ceiling** on the one measured point (computed
   11.7 from 23.4 GB ÷ 273 GB/s; measured 11.5). Treat the ceiling as the estimate; do not
   discount it. For an MoE only active experts are read, so its ceiling is a **range** —
   bracket it between active-experts-only and active-plus-trunk, and measure it early.
2. **Context is nearly free at decode** — 15 % loss across 100K tokens. Plan for large
   contexts; they are this hardware's comparative advantage. **Prefill is what scales with
   context**, which is why §4.4 exists.
3. **Per-user decode does not divide by user count.** Ten streams gave 84.3 tok/s aggregate
   while per-user fell only 11.5 → ~8.4. Do not ration on a divide-by-N model.

### 1.4 What this rules out

- **Serving BF16 weights of any dense 27–31B model.** 55–63 GB read per token is a 4.4–4.9
  tok/s ceiling against 11.5 measured at 4-bit. Do not register a BF16 checkpoint as served.
- **Tensor parallelism.** One GPU. Any `--tensor-parallel-size` above 1 is a bug, and there is
  no second box — do not add the flag "for later".
- **A second concurrently-serving model** (§5.3), and **any LAN-bound listener** (§6).

---

## 2. Install and pinning

**Container only. Never build vLLM from source.** On aarch64 for sm_121 that means compiling
kernels for a compute capability with no prebuilt artifacts — the cold-compile failure in §5.4.

**Target CUDA 13.0**, per NVIDIA's DGX Spark vLLM playbook. Any image built against an older
CUDA is the wrong branch, however convenient its tag looks.

**The container is a field of the recipe, not a separate pin we maintain.** sparkrun (§4.7)
resolves it: its execution flow is `prepare_image()` (build or pull) → sync image and model to
hosts → launch → `pre_exec` → serve → health check. So the decision is *which recipe you fork*,
and the pin lives in that fork's `container:` field alongside the flags it was validated with.
Do not pair a recipe's flags with a different image and assume the combination holds.

Known images, none of them yet confirmed on this box:

| image | from | note |
|---|---|---|
| `ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38` | the NVFP4+DSpark reference recipe | **GB10-specific**, and the only one built for this hardware |
| `ghcr.io/spark-arena/dgx-vllm-eugr-nightly` | the official Spark recipe | community nightly |
| `vllm/vllm-openai:qwen38` | vLLM's own recipe for this model | a fork build in the 0.27 range |
| `vllm/vllm-openai:v0.27.1-aarch64-cu129-ubuntu2404` | upstream stable | published `linux/arm64`, but **cu129, not CUDA 13** |

Start from the GB10-specific one, since the reference recipe's ~75 tok/s target was measured
with it. `UNVERIFIED` for whichever you pick: that it carries sm_121 kernels. Confirm at first
load — the startup log must select a Marlin/W4A16 kernel, with no "no kernel image is
available" error and no PTX JIT-fallback warning. Pin the digest once pulled.

`Qwen3_5ForConditionalGeneration` is registered from vLLM 0.17.0 onward, so that is the
architecture floor. The reason to run something much newer is kernels and the speculative
decoding fixes (§4.5), not the architecture.

**Never `pip install -r requirements.txt` inside the serving container.** It clobbers the
pinned build's torch/vLLM pair. This is the most natural-looking wrong move an agent handed a
repo can make. If a dependency is genuinely missing, change the image, not the running
container.

Cache and workspace paths — each line is a bug someone already paid for. Mount these from the
host so they survive a container replacement:

```
VLLM_CACHE_ROOT=/opt/spark-ai/cache/vllm        # compile cache; large, keep on NVMe
TRITON_CACHE_DIR=/opt/spark-ai/cache/triton     # MUST persist across runs -- never a temp dir
TORCHINDUCTOR_CACHE_DIR, TORCH_EXTENSIONS_DIR, TORCH_HOME, XDG_CACHE_HOME
FLASHINFER_WORKSPACE_BASE=/opt/spark-ai/cache/flashinfer   # ignores XDG; defaults to $HOME
HF_HOME=/opt/spark-ai/hf                        # sparkrun syncs models here; keep one copy
VLLM_ENGINE_READY_TIMEOUT_S=2400                # 600 is not enough for a cold Triton cache
FLASHINFER_CUDA_ARCH_LIST=12.1a                 # GB10's arch string, per the reference recipe
```

---

## 3. Model policy

### 3.1 Registry

`MODEL_REGISTRY` maps key → checkpoint; `SERVED` is a strict subset. Registered-but-unserved
entries record **why**, so a settled question is not re-litigated. Every measured number cites
a run id (§9).

| key | checkpoint | size | ceiling | status |
|---|---|---:|---:|---|
| `qwen3.8_27B` | `unsloth/Qwen3.8-27B-NVFP4` | 23.4 GB | 11.7 (**measured 11.5**) | **served — default** |
| `nemotron_30b_a3b` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 21.6 GB | range (§1.3) | served — fast alt; **coding quality unmeasured**, which is what would justify promoting it |
| — draft | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark` | 1.3 GB | — | draft for the above |
| `gemma4_31B` | `nvidia/Gemma-4-31B-IT-NVFP4` | — | ~9 measured elsewhere | served — second coding option; NVIDIA's own NVFP4, and the build in NVIDIA's Spark support matrix |
| — drafter | `Doopeworld/Qwen3.8-27B-DSpark-vLLM` | 2.7 GB | — | DSpark drafter for the **default** (§4.5) |

### 3.2 Quantization on this GPU

| format | weights | activations | status |
|---|---|---|---|
| NVFP4 (W4A16) | 4-bit | 16-bit | `UNVERIFIED` — `vllm serve` the default and read the kernel line at startup |
| compressed-tensors w4a16 | 4-bit | 16-bit | `UNVERIFIED` — same, with `gemma4_31B` |
| FP8 | 8-bit | 8/16-bit | wedged under concurrent deep-context load where NVFP4 did not (field report) — the *official* sparkrun recipe for this model is FP8, so prefer the NVFP4 recipe in §4.6 |
| NVFP4 W4A4 | 4-bit | **4-bit** | `UNVERIFIED` — GB10 computes W4A16 via Marlin; there is no native FP4 tensor-core path here, so a W4A4 recipe may not dispatch |

---

## 4. Serving

### 4.1 Per-model flags

| | `qwen3.8_27B` (default) | `nemotron_30b_a3b` | `gemma4_31B` |
|---|---|---|---|
| architecture | dense 27.8B, **hybrid** GDN + attention, 262144 ctx | **hybrid** Mamba-2 + MoE, 30B/3B active, 1M ctx | dense 30.7B |
| extra serve flags | — | `--moe-backend marlin --mamba-backend flashinfer --mamba-cache-mode align` | — |
| `--reasoning-parser` | `qwen3` | `nemotron_v3` | `gemma4` |
| `--tool-call-parser` | `qwen3_coder` | `qwen3_coder` | `gemma4` |
| thinking | on by default | reasoning model | off by default — **but on whenever tools or a system message are present, i.e. every coding session** |
| speculative decoding | in-checkpoint MTP head | external DSpark draft | none |
| multimodal | yes — see §4.3 | no | yes — see §4.3 |

**Gemma's thinking costs 5–12× tokens and wall time** with no measurable gain (n=2 prompts).
Its template enables thinking in every coding session, so budget for it or suppress it per
request with `chat_template_kwargs.enable_thinking: false`; record which you chose.

**`hermes` fails silently on the default model.** A wrong tool parser does not error; it
presents as the model being stupid. Route the parser by model key, never globally.

Base invocation — these are this box's values, not anyone else's:

```
vllm serve $CHECKPOINT \
  --served-model-name $KEY \
  --host 127.0.0.1 --port 8000 \
  --enable-prefix-caching \
  --max-model-len $DERIVED \        # §4.2
  --kv-cache-memory $ABSOLUTE \     # §1.1
  --enable-auto-tool-choice \       # WITHOUT THIS the tool parser never activates (§4.1)
  $PARSER_FLAGS $SPECDEC_FLAGS $MODEL_FLAGS
```

`$MODEL_FLAGS` carries the per-model extras above, plus `--language-model-only` for the two
dense models (§4.3) — it is not a global flag.

No `--tensor-parallel-size`. No `--enforce-eager` — it was a crash workaround elsewhere, the
crash needs TP>1, and leaving it on cost ~3.9× throughput on the source system (on-box cost
`UNVERIFIED`). `--trust-remote-code` is not needed:
both architectures are pre-supported in the pinned build.

### 4.2 Context length is derived, not chosen

Set `--max-model-len` per model from the measured KV budget at install. **32768 is the floor**;
long context is the target — the default model's native window is 262144 and §1.3 shows decode
barely degrades with it. Client context declarations (§7) are generated from this number, never
picked independently.

### 4.3 Both dense models carry vision towers

A vision tower profiles at startup and can OOM **even when the language weights fit** — this
has happened. **First serve runs `--language-model-only`.** Enabling vision is a separate step,
gated on measured headroom against §1.1, and the registry records which mode is live.

### 4.4 Prefix caching — always on

A coding agent resends a growing, near-identical prompt every turn, and at ~4,000 tok/s prefill
that is ~8 s per turn at 32K and ~25 s at 100K. Prefix caching turns it into a lookup.
`UNVERIFIED`: both served models are hybrids and prefix-caching support for them on the pinned
build is unconfirmed — check the startup log on first serve and record it. Two costs:

- **Cached blocks live in the §1.1 KV budget.** There is no separate size flag: when deriving
  `--kv-cache-memory`, reserve headroom above the max-model-len KV requirement and record it.
- **vLLM reports full `prompt_tokens` regardless of cache hits.** Per-user usage logs therefore
  overstate real prefill work and cannot be read as capacity.

### 4.5 Speculative decoding is mandatory, and the mechanism differs per model

Without it the default decodes ~11.5 tok/s (measured). The reference NVFP4 recipe for this
exact checkpoint (§4.6) **targets ~75 tok/s single-stream** with a DSpark drafter. That is the
single largest lever on this box — larger than the choice between any two checkpoints.

**Default (`qwen3.8_27B`) — two options, both valid:**

```
# In-checkpoint MTP head. 15 mtp.* tensors are present in the checkpoint.
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'

# External DSpark drafter (2.7 GB, 1.36B params, 5-layer DFlash + Markov head).
# k=14 for single-stream latency (~75 tok/s); k=7 for aggregate throughput.
--speculative-config '{"method":"dspark","model":"Doopeworld/Qwen3.8-27B-DSpark-vLLM",\
                       "num_speculative_tokens":14,"draft_sample_method":"probabilistic"}'
```

**`num_speculative_tokens` must be spelled out for MTP.** vLLM reads `mtp_num_hidden_layers`
from the top-level config, but this checkpoint carries it only under `text_config`, so the
depth cannot be inferred — omit it and startup fails.

**DSpark forces the FLASH_ATTN backend, which rejects an fp8 KV cache.** Set
`--kv-cache-dtype bfloat16` explicitly when using it, or the serve dies with "Selected backend
FLASH_ATTN is not valid… kv_cache_dtype not supported". Every other recipe on the shelf sets
`--kv-cache-dtype fp8`; copying that setting into a DSpark serve is the obvious wrong move.
The drafter plus a bf16 KV cache also costs memory — the reference recipe runs
`--gpu-memory-utilization 0.55`, not the 0.8–0.88 the fp8-KV recipes use.

**Fast alternative (`nemotron_30b_a3b`)** uses its own external draft, `…-NVFP4-DSpark`, at
`num_speculative_tokens: 3`. **Do not cross the drafters** — each is trained against its own
target, and a mismatch is a vocabulary error.

Measured MTP acceptance for the default's family, from the vLLM recipe: 92.2 % (BF16), 84.8 %
(FP8), 81.5–84.8 % (NVFP4).

**Two things to confirm at install**, both `UNVERIFIED`: that the pinned build carries the
gated-delta-net speculative fix (vllm-project/vllm#51812, #51674 — the vLLM recipe notes no
released tag had it), and open issue **#46249**, which reports tool calls failing with MTP on
the **Responses API** for a different Qwen model. Our clients use `/v1/chat/completions`, so it
may not apply — enable speculative decoding, smoke-test tool calls with it **on**, and record
the result rather than pre-emptively disabling it.

### 4.6 Start from a published recipe, never from scratch

Three registries carry configurations for these models; none of their defaults are safe here
unquestioned, but their flags encode work you should not redo.

| source | what it has | caveat |
|---|---|---|
| `spark-arena/recipe-registry` | the *official* Spark recipe for this model family — but **FP8**, not NVFP4 | FP8 is the precision that wedged (§3.2) |
| `styles01/sparkrun-recipes` | `qwen-38-27b-nvfp4-dspark.yaml` — **our exact checkpoint**, NVFP4 + DSpark, GB10-tuned | third-party; container is a community GB10 build |
| `Avarok-Cybersecurity/atlas-recipes` | `gemma-4-31b-nvfp4.yaml` on `nvidia/Gemma-4-31B-IT-NVFP4` | runs the `atlas` runtime, not vLLM — take the model choice, not the launcher |
| `vllm-project/recipes` | `models/Qwen/Qwen3.8-27B.yaml` — per-variant VRAM, all three spec-decode modes, measured acceptance | its verified-hardware list covers GB300 and RTX 5090; **GB10 is deliberately absent** |
| `NVIDIA/dgx-spark-playbooks` | Spark playbooks for vLLM, Open WebUI, coding agents, NVFP4 quantization, multi-Spark | its model support matrix does not list this model |

**Every one of these defaults to `host: 0.0.0.0`.** So does vLLM when `--host` is unset. §6
forbids it. Take a recipe's flags; never run one verbatim.

---

## 4.7 Launching: sparkrun

`sparkrun` launches, manages and stops inference workloads on one or more DGX Sparks with no
scheduler. Prefer it to a bespoke launcher — it already covers run, logs, stop, status and
multi-node parallelism, and detaching from logs does not kill the job.

```bash
uvx sparkrun setup                 # wizard: cluster, SSH mesh, ConnectX-7, sudoers, earlyoom
sparkrun run <recipe>              # launch
sparkrun logs|stop|status <recipe> # manage; Ctrl+C detaches, it does not kill
```

External registries attach through a `.sparkrun/registry.yaml`, which is how the NVFP4 recipes
in §4.6 are reachable. **Fork the recipe rather than using it unmodified**: at minimum set
`host: 127.0.0.1` (§6) and replace `gpu_memory_utilization` with the budget derived in §1.1.
Keep the fork in this repo so the served configuration is versioned with the rules.

---

## 5. Serving fences

1. **Bench runs must be unmistakable.** Served-model name starts `bench-`; ports 9000–9099. The
   gateway refuses to route to a `bench-` name, and a bench serve must never write the backend
   pointer.
2. **A bench serve must never evict the resident model** while anyone is connected.
3. **One resident model; swapping takes a lock.** The reason is *not* that they do not fit — at
   4-bit all three total ~68 GB and fit together. It is that KV cache wants the leftover memory
   and a second server competes for the same scarce bandwidth. A swap requires no in-flight
   requests, takes a `flock` on `/opt/spark-ai/run/resident.lock`, and is visible: `ai status`
   prints the resident model and any user with gateway requests in the last 10 minutes.
4. **First load of any model is far slower than the second** — sm_121 has no prebuilt kernels
   and the Triton cache compiles cold. This is why `TRITON_CACHE_DIR` persists and the ready
   timeout is raised (§2). It is not a hang; say so before someone kills it.
5. **Keep long-run scratch off `/tmp`** — systemd ages it, and a benchmark that lost its scratch
   mid-run produced 33 HTTP 500s that would have scored as wrong answers. Write results as
   `<name>.partial` and atomic-rename on completion; the scorer refuses `.partial`.
6. **Record decode at t=0 and after ten minutes sustained, and publish both.** A desk box under
   sustained load throttles; a burst number alone misleads users.
7. **After any agent or background run, check for orphans**: `pgrep -af 'vllm serve'` and
   `docker ps`. A stray process holding unified memory blocks everyone.

---

## 6. Users and access

Three users. **Every listener binds `127.0.0.1`.** Browser access is over an SSH tunnel, and
the tunnel is the entire security boundary — no LAN bind, no TLS, no auth proxy.

```bash
# 3000 = Open WebUI, 8421 = gateway (vLLM itself stays on 8000, unexposed)
ssh -N -L 3000:127.0.0.1:3000 -L 8421:127.0.0.1:8421 <user>@<spark-host>
```

- **Open WebUI runs one instance with `WEBUI_AUTH=True` and three accounts.** Unauthenticated
  gives three people one shared chat history; three instances each hold their own embedding
  model in the shared pool.
- **The gateway API key is mandatory and per-user.** Per-user usage records need an identity.
- Chat history lives in a mode-700 directory under `$HOME`.

---

## 7. Clients

The gateway holds one fixed URL in front of whichever model is resident, captures each
response's `usage` to a dated JSONL, and exposes a keyless TTL-cached `/status`
(ready / loading / no_backend). Clients read three variables from `~/.ai-session/env`, mode 600
— `AISESSION_BASE_URL`, `AISESSION_API_KEY`, `AISESSION_MODEL` — so no client config contains a
path, a hostname, or a key.

- **aider** needs a litellm metadata entry per served model or it mis-sizes prompts, with keys
  duplicated with and without the `openai/` prefix. Costs are zero.
- **opencode** needs a project-local provider block using `@ai-sdk/openai-compatible` and
  `{env:AISESSION_BASE_URL}` / `{env:AISESSION_API_KEY}`.
- **Continue** needs an OpenAI-compatible provider with `apiBase` pointing at the gateway.
- **All three must declare the same context and output budget**, generated from §4.2.
- A reasoning parser puts chain-of-thought in `reasoning`, not `content`. With a tight
  `max_tokens`, `content` can return null while the model is still thinking — that is not a
  broken API.
- **Telemetry off by name**: aider `--analytics-disable`; Continue
  `allowAnonymousTelemetry: false`; opencode `share: disabled` and `autoupdate: false`; Open
  WebUI `ANONYMIZED_TELEMETRY=False`, `DO_NOT_TRACK`, `SCARF_NO_ANALYTICS`. The coding tool is
  separate software with its own telemetry, outside this service's control.
- Open WebUI's web search, URL fetch and paper search are **opt-in, default off** — their
  queries leave the machine. Everything else on the serving path does not.

---

## 8. Inference as a tool

Offline batch (`LLM()`, no server), guided/structured decoding, and an embeddings endpoint are
**designed, not proven** — mark them aspirational until a tracked run says otherwise.

Read-only MCP servers may expose box state — resident model, free unified memory, temperature,
disk. The security model is enforced in code, not convention: a whitelist of binaries checked
every call; every command an argv list with `shell=False`; arguments regex-constrained and
anchored `\A...\Z`, **not** `^...$` (Python's `$` also matches before a trailing newline and
would accept `"123\n"`); no mutating verb reachable. They live in their own environment so the
MCP SDK's dependencies cannot perturb vLLM.

---

## 9. Evidence

`_evidence/` is gitignored and holds the only copy of measured results.

- **Every serve, benchmark and scoring run gets a tracked id** in the append-only
  `_evidence/manifest.jsonl`, with date, checkpoint, exact serve argv, and result. **Every
  on-box number cites a manifest id; every off-box number cites its source and date.** No number
  in this file cites a manifest id yet, because none was measured here.
- **Whenever two numbers appear together, so do the conditions they were measured under.** The
  two coding scores in §3.1 are why: they were measured on the BF16 parents, elsewhere, with
  thinking disabled — which is Gemma's native default but suppresses Qwen's.
- Partial result files must be distinguishable from complete ones. A truncated benchmark that
  scores as a bad model is worse than a crash.
- **Commit evidence out of `_evidence/` before deleting any weights it describes.** Weights are
  re-downloadable; a scored generation set at a frozen subset is not.

Staging: set `HF_HUB_DISABLE_XET=1` (the Xet backend has no stall timeout and hangs silently)
and `HF_HUB_DOWNLOAD_TIMEOUT=60`, retry with a bound, then **verify byte-exactly** — parse each
safetensors header-length prefix, sum `filesize − 8 − header` across shards, and compare with
the index's `metadata.total_size`. A partial download that looks complete is what this catches.

---

## 10. Related

**`AGENTS.md` must not exist in this repo.** Where you have seen it, it is not a rules file — it
is a prompt-level workaround forcing `<tool_call>` tokens out of a model that could not emit
them. All three models here have working tool parsers, and that file actively degrades them.

User-facing documentation lives in `docs/`. This file is project rules for work *on* this repo
and is not user documentation.
