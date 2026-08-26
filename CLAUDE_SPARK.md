# Project rules — DGX Spark LLM service

A vLLM service on one GB10 GPU, served over an OpenAI-compatible API to a browser client and
three coding clients. Nothing on the serving path leaves this machine.

**The box is shared.** This project is one tenant among several in `spark-users`, not the owner
of the machine. Rules that read like conventions — one resident model, the `bench-` prefix, the
port ranges — bind us and cannot bind anyone else, so treat the GPU and the memory pool as
contended at all times.

Hardware values in §1.1 are **measured** as of 2026-08-26. Everything still marked `UNVERIFIED`
carries the command that settles it; when you settle one, replace the number *and* the marker in
the same edit.

---

## 1. Hardware reality

### 1.1 The platform — verified 2026-08-26

| | verified value | re-check with |
|---|---|---|
| GPU | **NVIDIA GB10, 1 GPU, compute cap 12.1 (sm_121)**, driver 580.159.03 | `nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv` |
| CUDA | **13.0** (nvcc 13.0.88; driver reports 13.0) — matches the §2 target | `nvcc --version` |
| memory | **121 GB unified, one pool shared with the OS.** `nvidia-smi` reports FB Memory `N/A`, confirming there is no separate VRAM | `free -g`; `nvidia-smi -q -d MEMORY` |
| CPU | **20-core Arm Cortex-X925, aarch64** | `nproc`; `uname -m` |
| OS | **DGX OS on Ubuntu 24.04.4** (`DGX_NAME="DGX Spark"`), host `spark-2500`, Docker present | `cat /etc/dgx-release`; `docker info` |
| disk | **3.7 TB NVMe, 2.4 TB free**; weights live in `/data/models/vllm/` | `df -h /` |
| bandwidth | ~273 GB/s vendor spec — **`UNVERIFIED`** | no read-only probe; run a STREAM-class microbenchmark and record it |

**This is a shared multi-user box.** Several people in the `spark-users` group hold sessions and
can start their own serves outside this project's conventions (§5.8). Never assume the GPU or
the memory pool is yours alone — check before allocating.

### 1.2 Memory budget — in gigabytes, and it comes first

Unified memory is not VRAM. `--gpu-memory-utilization` is a fraction of a pool the OS, the
desktop session, Open WebUI and the other resident containers are also using. On a discrete GPU
an over-commit kills the job; here it causes system-wide pressure and OOM-killed processes.
**vLLM's ordinary 0.9 default is unsafe on this box.**

Measured baseline, 2026-08-26, with `spark-open-webui`, `spark-ollama` and `geoagent-postgis`
resident and **no model loaded**:

```
total 121 GB   available 115 GB
```

Derive the serve budget from a live reading, not from this one: `budget = MemAvailable − OS
floor − margin`, measured with the resident services **already up**, never on an idle box.

- **Take a share, not the pool.** 115 GB available does not mean 115 GB is yours — others in
  `spark-users` can start a serve while yours is running. Decide the share this project claims,
  write it here as a number, and size every recipe against that rather than against
  `MemAvailable` at the moment you happen to look.
- Prefer `--kv-cache-memory` (absolute) to `--gpu-memory-utilization` (fraction) wherever the
  pinned build supports it. If it does not, compute the fraction as `budget ÷ total pool` and
  record it — never fall back to a default.
- A preflight must fail before loading weights unless `MemAvailable` ≥ weights bytes +
  `--kv-cache-memory` + 5 GB, printing all three terms. Do not let a load OOM the box.
- The three served checkpoints total ~68 GB at 4-bit, which **fits inside 121 GB** — so
  co-residency is arithmetically possible and still forbidden (§5.1), and on a shared box the
  reason is stronger: memory you hold is memory another user cannot.

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
- **A second concurrently-serving model** (§5.1), and **any LAN-bound listener** (§7).

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

Known images. Architecture and CUDA are **verified from the registry manifests**, 2026-08-26:

| image | from | os/arch | CUDA | note |
|---|---|---|---|---|
| `ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38:mtp3-20260813` | the NVFP4+DSpark reference recipe | **linux/arm64** | **13.0.2** | **GB10-specific**; start here |
| `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest` | the official Spark recipe | **linux/arm64** | **13.0.2** | community nightly |
| `vllm/vllm-openai:qwen38` | vLLM's own recipe for this model | not checked | not checked | a fork build in the 0.27 range |
| ~~`vllm/vllm-openai:v0.27.1-aarch64-cu129-ubuntu2404`~~ | upstream stable | arm64 | cu129 | **ruled out** — wrong CUDA for a 13.0 host (§1.1) |

Both Spark images are **single-arch arm64**, not multi-arch manifests. That is a safety
property: a wrong-architecture pull fails loudly with "no matching manifest for
linux/amd64" rather than silently running under emulation.

Start from the GB10-specific one, since the reference recipe's ~75 tok/s target was measured
with it. What remains `UNVERIFIED` is narrower than it was: **that the image's kernels cover
sm_121**. Confirm at first load — the startup log must select a Marlin/W4A16 kernel, with no
"no kernel image is available" error and no PTX JIT-fallback warning. Pin the digest once
pulled.

Both images carry `NVIDIA_REQUIRE_CUDA=cuda>=13.0 …driver>=535,driver<536…` — stock NVIDIA
base-image boilerplate. The host runs driver 580.159.03, so if the container toolkit ever
refuses the constraint, `NVIDIA_DISABLE_REQUIRE=1` is the documented escape hatch. Unlikely to
bite; noted so it is not mistaken for a real incompatibility.

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
VLLM_CACHE_ROOT=/data/cache/vllm        # compile cache; large, keep on NVMe
TRITON_CACHE_DIR=/data/cache/triton     # MUST persist across runs -- never a temp dir
TORCHINDUCTOR_CACHE_DIR, TORCH_EXTENSIONS_DIR, TORCH_HOME, XDG_CACHE_HOME
FLASHINFER_WORKSPACE_BASE=/data/cache/flashinfer   # ignores XDG; defaults to $HOME
HF_HOME=/data/models/vllm                       # where weights already live; keep ONE copy
                                                # -- 2.4 TB free, but it is shared with other users
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
barely degrades with it. Client context declarations are generated from this number, never
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

**Every one of these defaults to `host: 0.0.0.0`.** So does vLLM when `--host` is unset. §7
forbids it. Take a recipe's flags; never run one verbatim.

---

### 4.7 Launching: sparkrun

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
`host: 127.0.0.1` (§7) and replace `gpu_memory_utilization` with the budget derived in §1.1.
Keep the fork in this repo so the served configuration is versioned with the rules.

---

## 5. Serving discipline

sparkrun (§4.7) owns launch, logs, stop, status and the health check. These rules are the
discipline **around** it, not a replacement for it.

1. **One resident model.** The reason is *not* that they do not fit — at 4-bit all three total
   ~68 GB and co-reside comfortably. It is that KV cache wants the leftover memory and a second
   server competes for the same scarce bandwidth. A swap is `sparkrun proxy unload <a>` then
   `load <b>` (or plain `stop`/`run`), and it affects the other two users: do it with no in-flight requests, and announce it.
   `sparkrun status` is the answer to "what is resident".
2. **Bench runs must be unmistakable.** Give every benchmark its own forked recipe with a
   `bench-` prefixed `served_model_name` and a port in 9000–9099. `sparkrun benchmark <recipe>`
   runs the whole cycle (run → benchmark → stop) — use it rather than hand-driving a serve, so
   a bench workload cannot be left running. The gateway refuses to route to a `bench-` name.
3. **A bench run must never evict the resident model** while anyone is connected. On one box a
   benchmark and a live session are competing for the same memory and bandwidth, so the numbers
   are wrong *and* the users are slowed.
4. **First load of any model is far slower than the second** — sm_121 has no prebuilt kernels
   and the Triton cache compiles cold. This is why `TRITON_CACHE_DIR` persists across container
   replacements and the ready timeout is raised (§2). It is not a hang; say so before someone
   kills it. sparkrun's health check waits on the port and an HTTP probe, so let it wait.
5. **Keep long-run scratch off `/tmp`** — systemd ages it, and a benchmark that lost its scratch
   mid-run produced 33 HTTP 500s that would have scored as wrong answers. Write results as
   `<name>.partial` and atomic-rename on completion; the scorer refuses `.partial`.
6. **Record decode at t=0 and after ten minutes sustained, and publish both.** A desk box under
   sustained load throttles; a burst number alone misleads users.
7. **After any agent or background run, check for orphans**: `sparkrun status`, then `docker ps`
   for anything sparkrun did not start. A stray container holding unified memory blocks
   everyone. Note that Ctrl+C on `sparkrun logs` detaches and leaves the workload serving — that
   is the intended behaviour, and it is also how orphans happen if you assume otherwise.
8. **Coexist with the other tenants.** Everything above binds this project and nobody else —
   a `bench-` name and a 9000-range port stop *us* from confusing a benchmark for production,
   and stop no one in `spark-users` from doing anything. So before every load, look:
   ```bash
   nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # who holds the GPU
   free -g                                                        # what is actually available
   docker ps --format '{{.Names}}\t{{.Image}}'                    # whose containers are up
   sparkrun status                                                # ours specifically
   ```
   If another tenant has the memory committed, wait or ask — do not force the load. An OOM here
   lands on the whole machine, so the cost of being wrong is someone else's work, not just ours.
   Announce swaps and long benchmarks rather than surprising people with them.
9. **Keep `earlyoom` enabled.** On unified memory an over-commit takes the machine down rather
   than the job; earlyoom is the backstop that kills a process instead of letting the box wedge.
   Installing it needs root, which this project does not have — check whether it is already
   running (`systemctl status earlyoom`) and, if not, ask the box's admin. Until it is, §1.2's
   preflight is the **only** guard between a bad `--kv-cache-memory` and everyone's session.

---

## 6. The proxy, and what is left to build

**Do not build a gateway.** `sparkrun proxy` is one: a LiteLLM instance (run via `uvx`, no
permanent install) that discovers running sparkrun endpoints and fronts them with a single
OpenAI-compatible API.

```bash
sparkrun proxy start --host 127.0.0.1 --master-key sk-<token>   # docs recommend this bind
sparkrun proxy status | models
sparkrun proxy load <recipe> | unload <recipe>                  # swap through the proxy
```

It gives us, without writing any of it: live endpoint discovery, a background re-scan every
30 s, health checks via `GET /v1/models`, model aliases so clients address a friendly name,
and endpoint deduplication across interfaces. Default port is 4000. **A model swap therefore
reconfigures no client** — which was the whole point of a gateway.

Two operational facts to know before relying on it:

- **Applying a model-set change rewrites the config and restarts the proxy.** LiteLLM's runtime
  mutation endpoints need a database-backed model store that sparkrun does not provision, so
  they answer `500 No DB Connected`. A steady-state discovery sweep that finds no change skips
  the restart entirely and never interrupts serving — but a genuine swap is a proxy restart.
- **`--master-key` is one shared bearer token, not per-user keys.** LiteLLM's virtual keys and
  per-user spend tracking need that same absent database.

### What is still ours

| need | status |
|---|---|
| launch, stop, logs, status, health, image and model distribution | sparkrun |
| one fixed URL that survives a swap; model aliases; discovery | `sparkrun proxy` |
| **per-user identity** — our users behind one shared token | **not provided** (§7) |
| **per-user usage capture** for capacity | **not provided** — needs the DB LiteLLM wants |
| **authenticated browser chat** | Open WebUI (§7) |

A single `--master-key` with per-user Open WebUI accounts is defensible given the tunnel is
already the security boundary — but it means **usage logs cannot attribute a request to a
person**, and §9's tracked run ids carry that load instead. Note the key is only as private as
the box: anyone in `spark-users` who can read the proxy's config or process list can read it, so
it authenticates the service, not the human. Decide this explicitly rather than by default.

### Recipe trust

sparkrun runs shell commands from recipes: `pre_exec` and `post_exec` inside containers,
`post_commands` **on the control machine**. Trust is gated accordingly:

- Recipes from a **local path are automatically trusted** — including anything you fork into
  this repo. **Read a third-party recipe's hooks before forking it**, because forking is what
  removes the prompt.
- Third-party registries are untrusted until `sparkrun registry trust <name>`; their hooks
  prompt once per recipe per session.
- **URL recipes are never auto-trusted**, regardless of anything else.

The NVFP4 recipes in §4.6 come from third-party registries. Fork them deliberately, hooks read.

---

## 7. Users and access

**Every listener binds `127.0.0.1`.** Browser access is over an SSH tunnel, and the tunnel is
the entire security boundary — no LAN bind, no TLS, no auth proxy. On a shared box `127.0.0.1`
is a weaker boundary than it sounds: it excludes the network, not the other users logged into
this machine. Anything genuinely private needs file permissions, not just a loopback bind.

```bash
# 3000 = Open WebUI, 4000 = sparkrun proxy (vLLM itself stays on 8000, unexposed)
ssh -N -L 3000:127.0.0.1:3000 -L 4000:127.0.0.1:4000 <user>@<spark-host>
```

**Every published recipe sets `host: 0.0.0.0`, and so does vLLM when `--host` is unset.** A
forked recipe must set `127.0.0.1` before it is ever run (§4.6). This is the single easiest way
to breach the boundary above, and nothing in sparkrun will warn you.

- **Open WebUI runs one instance with `WEBUI_AUTH=True` and three accounts.** Unauthenticated
  gives three people one shared chat history; three instances each hold their own embedding
  model in the shared pool.
- **The proxy runs with `--master-key`**, shared across our users (§6). Per-user Open
  WebUI accounts are the practical substitute for per-user identity.
- Chat history lives in a mode-700 directory under `$HOME`.

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


