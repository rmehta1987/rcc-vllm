# HANDOFF — write a `CLAUDE.md` for the DGX Spark deployment

**Deliverable: one file, `CLAUDE.md`, at the root of a new repository**, governing a vLLM
service on an NVIDIA DGX Spark. Not user docs, not an install script, not a port of anyone's
code. It is the rules file: hard constraints, fences against expensive mistakes, and pointers
to evidence.

Source project: `/project/rcc/mehta5/vllm`, a vLLM service on a Slurm HPC cluster. **The Spark
cannot reach that filesystem**, so this document is self-contained and is the only copy of most
facts below. The scheduler machinery does not port; the incidents do.

---

## 0. Execution contract

**This contract binds only the agent authoring `CLAUDE.md`. It expires the moment that file
exists — the install is then governed by `CLAUDE.md`, not by this table.**

| MUST | MUST NOT |
|---|---|
| run §5's read-only verification commands | install vLLM, containers, or packages |
| read checkpoint metadata from the Hugging Face API (ids in A8) | download model weights |
| write one file, `CLAUDE.md` | start `vllm serve`, Open WebUI, or any long-running process |
| report which numbers are measured, derived, or unverified | write code, launchers, or install scripts |
| ask at the STOP conditions | commit or push unless asked |

### What "measured" means here

- **Measurable now:** §5's table — GPU name, compute capability, total and free memory, CPU
  architecture, OS, driver and CUDA version, container runtime, free disk.
- **Computable now:** decode *ceilings*, from published checkpoint sizes (A8) ÷ assumed
  bandwidth. Label as derived; name both inputs.
- **Structurally unavailable:** realized decode, load time, whether a checkpoint fits at a
  given `--max-model-len`, whether a format's kernels exist for this GPU. Each becomes a rule
  in `CLAUDE.md` marked `UNVERIFIED` with the command that settles it at install. **A file
  where most numbers are `UNVERIFIED` is the expected outcome, not a failure.**

### Order of work

1. Read this document.
2. Run §5's commands. Record actual output.
3. Pull checkpoint sizes: `/api/models/<id>/tree/main?recursive=1` (no weights downloaded).
4. Compute the memory budget (§5a) and ceilings (§5b) from steps 2–3.
5. Draft against §8's outline, applying §6, §7, §9 and A8.
6. Run §10's self-check. Report.

### STOP — ask, do not guess

1. **The arithmetic says the plan fails.** A checkpoint's computed ceiling is too low for its
   assigned workload, or checkpoint + working KV exceeds the §5a budget. Report the
   arithmetic. Substituting an unchosen model is not your call.
2. **A GPU is visible but is not GB10** — `nvidia-smi --query-gpu=name` returns something else.
   §5a–c are derived from GB10; the rules built on them may not hold.
3. **No GPU visible / `nvidia-smi` absent** — you are not on the Spark. This is a *degraded
   mode, not a halt*: write the file with every hardware number `UNVERIFIED`, use
   `128 GB × 0.85 ≈ 108 GB` as the labelled placeholder budget, and say so in your summary.
4. **Appendix ambiguity that would change a rule.**

---

## 1. You do not have the source project

`/project/rcc/mehta5/vllm` is not mounted here. Do not write paths into `CLAUDE.md` that point
at it. The repo is `rmehta1987/rcc-vllm`, branch `milestone/model-refresh`, and **may be
private — ask before assuming you can clone it**. Nothing below depends on having it.

---

## 2. Scope

Rules are needed for four workloads: install vLLM reproducibly; stage models with a registry
recording what is served and what is not, each with its reason; run Open WebUI and the coding
harness (aider, Continue, opencode); and use vLLM as an inference tool — offline batch,
structured output, embeddings, MCP.

**Workloads 1–3 are described from working systems. Workload 4 largely is not**: the source has
MCP servers and one PydanticAI example, and nothing for batch, structured output, or
embeddings. Design those from vLLM's documentation and **mark them aspirational in
`CLAUDE.md`** — presenting them as proven is a fabricated provenance.

---

## 3. What carries over

- **A registry with a served/not-served split.** The key is the stable identifier used in the
  served-model-name, logs, and client configs. The gap between registered and served records
  *why* — here, memory footprint and decode ceiling. A model that fits but decodes too slowly
  is a NO-GO recorded with its number.
- **A stable URL over a moving backend.** The endpoint stops moving; the *resident model*
  moves. The gateway remains the seam for swaps, usage capture, and the access key.
- **Presets, not knobs.** Users pick intent. Parallelism disappears entirely (TP=1); the
  remaining knobs are memory budget and quantization, both more dangerous than TP was.
- **Model choice is a measurement.** Coding candidates were scored on a frozen 60-problem
  LiveCodeBench subset, identical harness and decode, verdicts recorded with job IDs. Only the
  *coding* candidates were scored — do not attribute benchmark provenance to chat models. On
  the Spark a verdict is a pair, (quality, decode rate): **quality scores carry as priors;
  decode rates do not transfer at all.**
- **Nothing leaves the machine.** External tools opt-in and off by default; client telemetry
  disabled by name (A6). This is why the service exists.
- **Evidence outlives weights.** Commit evidence out of the gitignored directory before
  deleting weights it describes.

---

## 4. Departures from the source design

Guidance to you, the author — **not a section to reproduce in `CLAUDE.md`.** Three testers who
never used the cluster do not need its absences narrated, and §10 greps for exactly that.

No scheduler, accounts, or GPU allowlist — one owner, one GPU. No chargeback; per-request
usage logging survives for capacity only. Ship a CLI on `PATH` plus systemd user units, so
users never type an install path and the deployed copy cannot drift from the repo. Staging is a
plain download plus A9's byte-exact gate. Multi-tenancy reduces to: authenticated WebUI,
per-user usage records, a swap lock.

**`AGENTS.md` must not exist.** In the source repo it is not a rules file — it is a
prompt-level workaround forcing `<tool_call>` tokens out of a model that could not emit them.
All three models here have working tool parsers, so an inherited copy actively degrades them.
Say so in `CLAUDE.md`, because the file looks like a convention worth replicating.

---

## 5. Verify the hardware first

**Write no hardware number you have not read off the box.**

| assumed | verify with |
|---|---|
| GB10 Grace-Blackwell, one GPU, sm_121 | `nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv` |
| 128 GB LPDDR5X unified, **shared with the OS** | `free -g`, `nvidia-smi -q -d MEMORY` — same pool? |
| ~273 GB/s bandwidth (vendor spec, no read-only probe) | assumed; cite it as the input to every ceiling, and make a microbenchmark an install task |
| 20-core Arm, **aarch64** | `uname -m`, `lscpu` |
| DGX OS, container runtime present | `cat /etc/os-release`, `docker info`, `nvidia-smi` driver + CUDA |
| ~4 TB NVMe | `df -h` — the weights budget comes from actual free space |

### (a) Unified memory is not VRAM

`--gpu-memory-utilization` is a fraction of the pool the OS, the desktop, Open WebUI and the
gateway also occupy. On a discrete GPU an over-commit kills the job; here it causes system-wide
pressure and OOM-killed processes. vLLM's ordinary 0.9 default is unsafe on this box.

**State the budget in gigabytes, and derive it:** `budget = measured free unified memory − OS
floor − margin`, measured **with Open WebUI and the gateway already resident**. Prefer an
absolute KV size to a fraction where the pinned vLLM supports one. This is rule 1 of
`CLAUDE.md`; §10 checks the ordering.

A related trap (§7): a vision tower profiles at startup and can OOM even when language weights
fit comfortably.

### (b) Decode is bandwidth-bound

```
tok/s_ceiling ≈ bandwidth / bytes_read_per_token
bytes_read_per_token ≈ active_parameter_bytes + KV_bytes_read
```

**Interactive speed is governed by bytes read per token, not parameter count.** Field
measurement (A8) puts realized decode at **92–99 % of the computed ceiling** for a dense
checkpoint — treat the ceiling as the estimate, not an optimistic bound. The same measurement
shows the KV term is small: decode fell 15 % from zero context to 100K.

- **Dense model:** quantization is the primary lever on bytes-read; speculative decoding is the
  other, and on measured evidence it is larger (§7, A8).
- **Sparse (MoE):** only active experts are read, so a 30B/3B-active model reads roughly a
  tenth of what its parameter count suggests.

### (c) Quantization on GB10

- **NVFP4 is fine here.** A NO-GO recorded in the source registry, on H200/sm_90, was
  `cudaErrorUnsupportedPtxVersion` — the Marlin repack's PTX outran EOL driver 535. A
  *driver* artifact, not silicon. Without this, that verdict travels with the quality priors and
  forbids the default model's own checkpoint format.
- **Marlin is load-bearing, not legacy.** NVIDIA's hardware matrix for a Spark-targeted NVFP4
  model lists GB10 as: stored NVFP4, compute path **W4A16**, MoE backend **`marlin`**, native
  FP4 path **"No — runs via Marlin"** (GB200: "Yes"). Their own recipe sets `--moe-backend
  marlin`. Do not write a rule steering away from it.
- **Stored precision ≠ compute path**, and **weight precision ≠ activation precision.** Record
  a per-format table — NVFP4, FP8, compressed-tensors w4a16 — with columns for weight
  precision, activation precision, status `UNVERIFIED`, and the command that settles it. The
  sharpest case is on the table already: one candidate (A8) is **W4A4**, validated on GB300,
  while GB10's published path is W4A16.
- **aarch64**: packages with compiled extensions may have no Arm wheel. This is why the install
  is container-only (§6.3).

---

## 6. Rules `CLAUDE.md` must contain

1. **Memory budget in gigabytes**, per §5a, with the derivation and the OS floor. Not a
   utilization fraction. Measuring a model's resident size is a precondition of serving it, and
   a preflight must assert free unified memory before load and fail legibly rather than OOM the
   box.
2. **TP=1, unconditionally.** One GPU, no second Spark (§9.3). Do not document a two-box case:
   an aspirational escape hatch in a rules file is how `--tensor-parallel-size 2` eventually
   gets typed.
3. **Container only.** Pin `vllm/vllm-openai:v0.27.1-aarch64-cu129-ubuntu2404` — verified on
   Docker Hub as a published `linux/arm64` image (the bare `v0.27.1` tag is a multi-arch
   manifest that also resolves on arm64; the fully-qualified tag removes any ambiguity about
   architecture and CUDA minor version). Pin the digest if you can. **Do not build from
   source**: on aarch64 for sm_121 that means compiling kernels for a compute capability with
   no prebuilt artifacts — the failure in §7 row 2. **Never `pip install -r requirements.txt`
   into the serving environment** — on the cluster it pulled a driver-incompatible torch/vLLM
   pair; here it clobbers the pinned build. It is the most natural-looking wrong move an agent
   handed a repo can make.
4. **Quantization policy** — §5c's per-format table, with activation precision recorded
   separately.
5. **Prefix caching on, always.** Set on every source serve path including the hybrid-architecture
   model, so it is proven compatible with these families (compatibility on the pinned Spark
   build is `UNVERIFIED` like everything else). Measured prefill is ~4,000 tok/s for the default
   checkpoint (A8), so a 32K prompt costs ~8 s and 100K ~25 s **per turn**; decode barely
   degrades with context but prefill does not get cheaper, and a coding agent resends a growing,
   near-identical prompt every turn. Two costs: **cached blocks live in the KV memory rule 1
   budgets**, so cache size trades against context length — set it explicitly; and **vLLM
   reports full `prompt_tokens` regardless of cache hits**, so per-user usage logs overstate
   prefill and cannot be read as capacity.
6. **Context length is derived, not chosen.** Set `--max-model-len` per model from the measured
   KV budget at install: 32768 is the floor, long context is the target (§5b — context is
   nearly free at decode, and the default model's native window is 262144). Client context
   declarations (A6) are generated from that number, never picked independently.
7. **Speculative decoding is mandatory, and the mechanism differs per model.** Measured 2.3×
   on this box (A8). **Default (`qwen3.8_27B`): an in-checkpoint MTP head.** Verified present
   in `unsloth/Qwen3.8-27B-NVFP4` — its safetensors index carries 15 `mtp.*` tensors, the same
   set as the BF16 parent. Configure via `--speculative_config` with an MTP method; the exact
   form is a property of the pinned build (`UNVERIFIED` — settle at install; note
   `num_nextn_predict_layers` is absent from config, the head is carried as tensors). **Fast
   alternative: an external draft model** — `--speculative_config.model <DSpark ckpt>`, per
   NVIDIA's recipe in A8. **Do not cross these**: pointing the Nemotron draft at the Qwen
   target is a vocabulary mismatch. MTP × tool calls is open vLLM **#46249**; enable it, run a
   tool-call smoke test with it on, record the fallback condition, and do not resolve it by
   silently leaving speculative decoding off.
8. **Bench fences.** Benchmarks use a distinct port range and a `bench-` served-name prefix the
   gateway refuses to route; a bench serve must never write the backend pointer or evict the
   resident model mid-conversation.
9. **Track every experiment.** Name the gitignored evidence directory and an append-only JSONL
   manifest. Every serve, benchmark, and scoring run gets an id recorded there with its date,
   checkpoint, exact serve argv, and result; **every measured number in the registry or in
   `CLAUDE.md` cites one.** This is what replaces the source project's scheduler job ids, which
   cannot be resolved from this box. Commit evidence out of the gitignored directory before
   deleting the weights it describes — weights are re-downloadable, a scored generation set at
   a frozen subset is not.
10. **One resident model; swapping takes a lock.** The reason is **not** that they do not fit —
    at 4-bit all three total ~68 GB and fit in 128 GB together. It is that KV cache wants the
    leftover memory and a second server competes for the same scarce bandwidth. With three
    users a swap affects other people: no in-flight requests, an explicit lock, visible. The
    CLI must answer "what is resident, and who is using it".
11. **Users and access.** Three users; every listener on `127.0.0.1`; browser access over SSH
    tunnel only; authenticated Open WebUI. Put the tunnel command in `CLAUDE.md`:
    `ssh -N -L 3000:127.0.0.1:3000 -L 8421:127.0.0.1:8421 <user>@<spark-host>`. No LAN bind, no
    TLS, no auth proxy — the tunnel is the security boundary.
12. **Privacy posture.** Nothing leaves the box on the serving path; external tools opt-in and
    off by default; client telemetry disabled by name per client (A6).
13. **Thermal honesty.** Record decode at t=0 and after ten minutes sustained; publish both.
14. **After any agent or background run, check for stray `vllm serve` processes and containers
    holding unified memory.** On one box that blocks everyone.

---

## 7. Mistakes already paid for

Each cost a debugging cycle on the source system. State the Spark form in `CLAUDE.md`. The
originating run identifiers were scheduler job ids and are unresolvable here, so they are
dropped — rule 9 is the replacement: every experiment on this box gets a tracked id, and every
measured number cites one.

| incident | Spark form |
|---|---|
| Serve-time caches defaulted to `$HOME` and filled a quota mid-load | pin the vLLM compile cache, Triton cache, Torch/Inductor caches, HF cache and FlashInfer workspace to named NVMe paths. **FlashInfer ignores `XDG_CACHE_HOME`** and defaults to `$HOME` (A1) |
| A per-run Triton cache meant recompiling every kernel; on an unfamiliar compute capability that blew the 600 s engine-ready timeout | sm_121 has no prebuilt kernels — this **will** happen on first load. Persist the Triton cache; set `VLLM_ENGINE_READY_TIMEOUT_S=2400`. Expect first load to be far slower than the second, and say so, or it reads as a hang |
| A benchmark lost its scratch mid-run; 33 HTTP 500s *would have* scored as wrong answers had it not been caught | keep long-run scratch out of `/tmp` (systemd ages it); partial result files must be distinguishable from complete ones. This guards the owed LCB re-score (A8) |
| The `hermes` tool parser fails **silently** on models it does not fit | route both parsers per model key (A6, A8). Silent tool-call failure presents as the model being stupid — it is the most expensive bug in this stack |
| Gemma's thinking cost 5–12× tokens and wall time, no measurable gain on two greedy prompts | a 5–12× latency multiplier here. Set thinking **per model** (A8), never as a blanket default; carry the n=2 caveat with the multiplier |
| `--enforce-eager` was adopted as a crash workaround, then removed: CUDA graphs plus a compilation-config flag recovered 894.7 → 3492.7 tok/s, ~3.9× | never leave it on permanently. The crash cannot occur here — the failing pass only enables at TP>1 |
| An MTP draft head worth +53 % sat unused | speculative decoding trades spare compute for bandwidth, this box's exact imbalance. See rule 7 and A8 — the field figure supersedes +53 % |
| A serve OOM'd during **multimodal profiling** though language weights fit; `--language-model-only` fixed it | **both reference models are multimodal.** Record the flag; skipping the tower also drops it from the resident footprint |
| Two models compared under conditions that suited one's defaults and suppressed the other's | whenever two numbers appear together, so do the conditions they were measured under |

---

## 8. Required outline

```
# Project rules
## Hardware reality      <- §5: verified specs, memory budget in GB (rule 1), bandwidth ceiling
### What this rules out
## Install and pinning   <- container tag, CUDA/driver, aarch64, the requirements.txt rule
## Model policy          <- registry + served set, quantization table, sizing arithmetic
## Serving               <- per-model flags, speculative decoding, context derivation, prefix cache
## Serving fences        <- port range, bench- prefix, one-resident-model rule, swap lock
## Users and access      <- three users, 127.0.0.1, the tunnel command, WebUI auth
## Clients               <- Open WebUI, aider/Continue/opencode
## Inference as a tool   <- offline batch, structured output, embeddings, MCP (mark aspirational)
## Evidence              <- gitignored path, manifest, commit-before-delete
## Related               <- why AGENTS.md must not exist; where user docs live
```

Target **~300 lines**. If the material will not fit, cut reasons to one clause before cutting a
rule, and never cut a number's provenance.

---

## 9. Decisions already made

**1. Three users, internal testing.** One resident model means the box serializes (rule 10).
**Per-user decode does not divide by user count**: measured, ten concurrent streams gave
84.3 tok/s aggregate while per-user fell only 11.5 → ~8.4 (A8). Open WebUI runs **one instance
with authentication on and three accounts** — unauthenticated gives three people one shared
chat history, and three instances each hold their own embedding model in the shared pool.
Usage logging stays, per user, for capacity.

**2. SSH tunnel only.** Every listener on `127.0.0.1` (rule 11).

**3. No second Spark.** TP=1 unconditional (rule 2).

**4. Clean repo.** No code ported. The appendix is reference for *what the pieces do and why*.
The cost: each rewrite can lose a fix that survives only as a comment here — work through A1's
environment list and A6's parser routing deliberately.

**5. Three models.** Detail in A8.

- **Default: `qwen3.8_27B`, served from `unsloth/Qwen3.8-27B-NVFP4`** (23.4 GB). The checkpoint
  the field report measured at 11.5 tok/s against a computed 11.7. Native vision, 262K context,
  Apache-2.0, day-zero vLLM support. Its **BF16 parent** scored 50.00 % on the frozen LCB-60
  (cluster, TP=2, thinking suppressed — a lower bound); **the served 4-bit checkpoint is
  unscored, and re-scoring it is the owed measurement** — quantization is the one thing that
  silently destroys a coding score.
- **First serve runs `--language-model-only`.** Enabling the vision tower is a separate step
  gated on measured headroom against rule 1, and the registry records which mode is live.
  Serving the tower untested reproduces the §7 OOM on the flagship model's first load.
- **Fast alternative: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`** + `DSpark` draft
  (21.6 + 1.3 GB). 3B active against the default's 27.8B dense, and the only model here with a
  published official single-DGX-Spark recipe. Coding quality **unmeasured** — that measurement
  is what would justify promoting it.
- **Second coding option: `gemma4_31B`** from `google/gemma-4-31B-it-qat-w4a16-ct` (23.3 GB).
  Its **BF16 parent** scored 66.67 % — higher than the default's 50.00 %, and not a reason to
  swap: that comparison ran with thinking disabled, Gemma's native default but a suppression of
  Qwen's, so 50.00 % is a lower bound. The source project reached the same conclusion on the
  same evidence. The served 4-bit checkpoint is likewise unscored.
- **BF16 checkpoints of both dense models are registered, not served**: 55.6 and 62.5 GB read
  per token for a 4.4–4.9 tok/s ceiling, against 11.5 measured at 4-bit. Single-device residency
  for either is also unproven — both were only ever served TP=2 across two cards.

---

## 10. Self-check

```bash
# Present: the load-bearing rules.
grep -q '127\.0\.0\.1' CLAUDE.md              && echo ok-bind    || echo MISSING-bind
grep -q 'bench-' CLAUDE.md                    && echo ok-fence   || echo MISSING-fence
grep -qiE '[0-9]+ *GB' CLAUDE.md              && echo ok-budget  || echo MISSING-budget
grep -qi 'speculative' CLAUDE.md              && echo ok-specdec || echo MISSING-specdec
grep -qi 'NVFP4' CLAUDE.md                    && echo ok-quant   || echo MISSING-quant
grep -qi 'language-model-only' CLAUDE.md      && echo ok-mmodal  || echo MISSING-mmodal
grep -qi 'AGENTS.md' CLAUDE.md                && echo ok-agents  || echo MISSING-agents
grep -qiE 'swap|lock' CLAUDE.md               && echo ok-swap    || echo MISSING-swap
wc -l CLAUDE.md                                # target ~300

# Absent: paths into an unreachable filesystem, and cluster machinery.
grep -n '/project/rcc' CLAUDE.md                                        # expect: no hits
grep -niE 'slurm|sbatch|squeue|--qos|service unit|modulefile' CLAUDE.md # expect: no hits
grep -niE 'tensor-parallel-size [2-9]' CLAUDE.md                        # expect: no hits

# Unverified numbers are labelled.
grep -o 'UNVERIFIED' CLAUDE.md | wc -l        # expect: nonzero (see §0)
```

Then read it as the agent it governs. Every rule carries its reason. Every number is measured
with its command, derived with both inputs named, or `UNVERIFIED` with the command that settles
it. The memory budget is in gigabytes and comes first. A reader who has never seen the source
project can answer: what fits, why TP is 1, how to install vLLM reproducibly, how a benchmark
is named so it cannot be mistaken for production, who reaches the box and how, and where
evidence lives.

Report: file location, which §5 numbers you measured, which are derived, which are
`UNVERIFIED`, and any STOP you hit.

## House style

Scientist-to-scientist prose. No checkmarks, no emoji, no marketing. Exact commands, tables,
measured numbers with provenance. Prefer deleting a paragraph to hedging it.

---

## Appendix

### A1. Serve invocation

Spark values, not the cluster's:

```
vllm serve $MODEL_PATH \
  --served-model-name $MODEL_KEY \
  --host 127.0.0.1 --port $PORT \
  --enable-prefix-caching \
  --max-model-len $DERIVED \          # rule 6, from the measured KV budget
  --kv-cache-memory $ABSOLUTE \       # rule 1; fall back to a fraction only if unsupported
  --language-model-only \             # rule: both dense models carry vision towers (§7)
  $TOOL_FLAGS $REASONING_FLAG $SPEC_DECODE_FLAGS
```

No `--tensor-parallel-size` (rule 2 — do not reintroduce the flag). No `--enforce-eager`
(§7). `--trust-remote-code` was set on the cluster; both current architectures are
pre-supported in the pinned build, so decide per model rather than setting it blanket on a
privacy-first box. Build the argv **once** and reuse it for serve, smoke and bench, so the
three cannot drift — the source's launcher drifted from its benchmark harness and their
context lengths disagreed.

Environment, each line a fixed bug:

```
VLLM_CACHE_ROOT              compile cache; large, pin to NVMe
TRITON_CACHE_DIR             MUST persist across runs (§7) -- never a per-run temp dir
TORCHINDUCTOR_CACHE_DIR, TORCH_EXTENSIONS_DIR, TORCH_HOME, XDG_CACHE_HOME
FLASHINFER_WORKSPACE_BASE    FlashInfer ignores XDG and defaults to $HOME
HF_HOME                      model and tokenizer cache
VLLM_ENGINE_READY_TIMEOUT_S=2400   default 600 is not enough for a cold Triton cache on sm_121
```

### A2. Registry shape

`MODEL_REGISTRY` maps key → path; `SERVED` is a strict subset. The value is in the comments. A
real entry records: the measured score that justified it (`50.00% (30/60) vs 26.67% (16/60),
+23.33 pts, run 2026-09-02-lcb60-a`), the build it was proven on, which tool parser works and which
fails silently, whether thinking is on by default and what it costs, and what measurement is
owed. Registered-but-unserved entries record why, so nobody re-litigates a settled question.

### A3. Gateway contract

One fixed URL; the proxy re-reads a backend-pointer file, so a model swap needs no restart.

- Proxies `/v1`, `/metrics`, `/health`, `/version`, `/ping`, `/tokenize`, `/detokenize`,
  `/pooling`; drops hop-by-hop and length headers both directions.
- Captures each response's `usage` to a dated JSONL — the authoritative record; users never log
  their own tokens.
- `GET /status` keyless and TTL-cached: ready / loading / no_backend.
- A request body cap (16 MB → `413`). A rate limit is optional here — it catches a runaway
  agent loop, not a crowd.
- **The API key is mandatory and per-user on the Spark** (optional on the cluster). §9.1's
  per-user usage records need an identity.

### A4. CLI surface

Short intent verbs, no paths, no flags for the common case, no billing verbs. The load-bearing
part: after a session comes up the dispatcher writes `~/.ai-session/env`, mode 600, containing
exactly `AISESSION_BASE_URL`, `AISESSION_API_KEY`, `AISESSION_MODEL`. Every client and example
reads those, which is why no client config contains an install path, a hostname, or a key.

### A5. Open WebUI

Its own environment, never the serving one; pointed at the gateway so it needs no
reconfiguration when the resident model changes. Spark values:

```
ENABLE_OPENAI_API=True
OPENAI_API_BASE_URL=http://127.0.0.1:$GW_PORT/v1
OPENAI_API_KEY=<per-user key>
ENABLE_OLLAMA_API=False
WEBUI_AUTH=True                                # three accounts (§9.1); the cluster ran False
DATA_DIR=$HOME/.ai-session/openwebui-data      # chat history: private, mode 700
HF_HOME=<state dir>/hf_cache                   # the UI writes here
ANONYMIZED_TELEMETRY=False  DO_NOT_TRACK=true  SCARF_NO_ANALYTICS=true
```

The chat database originally sat in a group-readable tree where any user could read another's
chats — hence mode 700. On first run the UI downloads a RAG embedding model, which can outrun
its own bind-wait and look like a failed start; pre-warm the cache or expect a slow first boot.
(The cluster set `HF_HUB_OFFLINE=1` against a shared read-only cache — that is air-gap-specific;
do not copy it onto an empty cache.) Web search, URL fetch and paper search are **opt-in behind
one variable, default off**, because their queries leave the machine.

### A6. Coding clients

- **Parser routing per model key** (values in A8). Wrong tool parser = silent failure.
  Tool calling needs `--enable-auto-tool-choice --tool-call-parser X`.
- A reasoning parser puts chain-of-thought in `reasoning` instead of `content`. With a parser on
  and a tight `max_tokens`, `content` returns null while the model is still thinking — it
  presents as a broken API.
- **aider** needs a litellm metadata entry per served model or it warns "Unknown context window
  size" and mis-sizes prompts; keys duplicated with and without the `openai/` prefix. The source
  project shipped this gap for the very models it served — do not inherit it:

```json
{"openai/qwen3.8_27B": {"max_input_tokens": 28000, "max_output_tokens": 4096,
  "input_cost_per_token": 0, "output_cost_per_token": 0,
  "litellm_provider": "openai", "mode": "chat"}}
```

- **opencode** needs a project-local provider block:

```json
{"model": "spark/qwen3.8_27B", "share": "disabled", "autoupdate": false,
 "provider": {"spark": {"npm": "@ai-sdk/openai-compatible",
   "options": {"baseURL": "{env:AISESSION_BASE_URL}", "apiKey": "{env:AISESSION_API_KEY}"},
   "models": {"qwen3.8_27B": {"limit": {"context": 32768, "output": 4096}}}}}}
```

- **Continue** (`~/.continue/config.yaml`): an `openai`-compatible provider with `apiBase`
  pointing at the gateway, the same context/output numbers, and `allowAnonymousTelemetry: false`.
- All client context/output numbers are generated from rule 6's derived `--max-model-len`, and
  the clients must agree with each other — the source had aider at 4096 output and opencode at
  8192 against one window.
- Telemetry disabled by name: aider `--analytics-disable`, Continue
  `allowAnonymousTelemetry: false`, opencode `share: disabled` + `autoupdate: false`. The
  coding tool is separate software with its own telemetry, outside this service's control.

### A7. MCP tools and their security model

Read-only servers expose box state — resident model, free unified memory, temperature, disk —
without handing an agent a shell. Enforced in code, not convention:

- a hard whitelist of binaries, checked every call;
- every command an argv list with `shell=False`;
- arguments regex-constrained and anchored `\A...\Z` — **not** `^...$`, because Python's `$`
  also matches before a trailing newline and would accept `"123\n"`;
- ownership verified before returning anything; no mutating verb reachable.

They live in a dedicated environment so the MCP SDK's dependency tree cannot perturb vLLM.

### A8. The three models

Sizes from the Hub file listing; ceilings derived at the assumed 273 GB/s. **Every tok/s below
is arithmetic except where marked measured.**

| checkpoint | Hub id | weights | read/token | ceiling |
|---|---|---:|---:|---:|
| **Qwen3.8-27B NVFP4** *(default)* | `unsloth/Qwen3.8-27B-NVFP4` | 23.4 GB | 23.4 GB | 11.7 — **measured 11.5** |
| **Nemotron Lightning NVFP4** *(fast alt)* | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 21.6 GB | ~1.7–4 GB (3B active) | range, see note |
| DSpark draft | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark` | 1.3 GB | ~1.3 GB/draft pass | draft only |
| Gemma-4-31B QAT 4-bit | `google/gemma-4-31B-it-qat-w4a16-ct` | 23.3 GB | 23.3 GB | 11.7 |
| Qwen3.8-27B FP8 | `Qwen/Qwen3.8-27B-FP8` | 30.9 GB | 30.9 GB | 8.8 — **wedges under load, do not serve** |
| Qwen3.8-27B NVFP4 (alt) | `RadixArk/Qwen3.8-27B-NVFP4` | 21.9 GB | 21.9 GB | 12.5 — see risks |
| Qwen3.8-27B BF16 *(not served)* | `Qwen/Qwen3.8-27B` | 55.6 GB | 55.6 GB | 4.9 |
| Gemma-4-31B BF16 *(not served)* | `google/gemma-4-31B-it` | 62.5 GB | 62.5 GB | 4.4 |

The fast alternative's ceiling is a range, not a point: only active experts are read, but
router, attention, shared layers and Mamba state add to the 1.7 GB floor. Bracket it (active
experts only vs active + trunk), mark it derived, and measure it early.

**`RadixArk/Qwen3.8-27B-NVFP4` risks**, if the default disappoints: it is a **W4A4** recipe
validated on **GB300**, while GB10's path is W4A16 (§5c); its card names **SGLang**, not vLLM;
and its evaluation ran with an **uncalibrated FP8 KV cache** (`scales_calibrated: false`, every
scale defaulting to 1.0) — a question that attaches to any fp8-KV serve, including NVIDIA's
recipe below. Its published GSM8K of 97.27 % is real evidence but is not coding.

**Per-model serve facts:**

| | default `qwen3.8_27B` | fast alt Nemotron | `gemma4_31B` |
|---|---|---|---|
| architecture | dense 27.8B, hybrid Gated DeltaNet + attention, **multimodal**, 262144 ctx | hybrid Mamba-2 + MoE, 30B/**3B active**, 1M ctx | dense 30.7B, **multimodal** |
| `--reasoning-parser` | `qwen3` | `nemotron_v3` | `gemma4` |
| `--tool-call-parser` | `qwen3_coder` — **`hermes` fails silently** | `qwen3_coder` | `gemma4` |
| thinking | **ON** by default (`reasoning_effort` xhigh) | reasoning model | OFF by default — **but the template turns it on whenever tools or a system message are present, i.e. every coding session**; measured 5–12× tokens and wall time, no measurable gain (n=2) |
| speculative decoding | in-checkpoint **MTP head** (15 `mtp.*` tensors verified present in the pinned checkpoint) | external **DSpark** draft, plus DFlash and an MTP head | none in this checkpoint |
| LCB-60 | 50.00 % *(BF16 parent, cluster, TP=2 — not this checkpoint)* | unmeasured | 66.67 % *(BF16 parent, cluster, TP=2 — not this checkpoint)* |
| licence | Apache-2.0 | OpenMDW-1.1 | check the shipped `LICENSE` |

**Measured on a DGX Spark — Qwen3.8-27B NVFP4, one day after release.** Provenance: a field
report from the user, not measured by this project. Label it that way.

| | |
|---|---|
| decode, 0 / 32K / 100K context | **11.5 / 10.9 / 9.8 tok/s** |
| aggregate, 10 concurrent | **84.3 tok/s** (~8.4/user) |
| prefill | **~4,000 tok/s** |
| single-stream, Ollama | **26.5 tok/s** |
| availability | llama.cpp and vLLM day zero; Ollama v0.32.12 same day |

1. **The bandwidth model is validated to 1 %** — computed 11.7, measured 11.5. Use §5b to size
   any checkpoint before downloading it.
2. **Realized is 92–99 % of ceiling** for this dense checkpoint. Do not discount the ceiling.
   (For the MoE, the range note above governs.)
3. **Context is nearly free at decode** — 15 % loss across 100K. Plan for large contexts; they
   are this hardware's comparative advantage. Prefill is what scales with context, which is why
   rule 5 exists.
4. **Speculative decoding is worth ~2.3×.** Caveat: 26.5 tok/s was **Ollama running its own
   quantized artifact**, so the gap bundles runtime and checkpoint differences — treat it as
   motivation, not a target. The realized vLLM+MTP gain on the pinned checkpoint is
   `UNVERIFIED`. The only acceptance statistics on record (length 2.775, rate 0.59) were
   measured on the **RadixArk** checkpoint under SGLang, not this one. See rule 7.
5. **FP8 wedged under concurrent deep-context load; NVFP4 did not.** This outranks any
   first-party preference — NVFP4 until FP8 is re-tested on the pinned build.

**Caveats that travel with these numbers:** the two LCB scores are not like-for-like (thinking
disabled suited Gemma's default and suppressed Qwen's, so 50.00 % is a lower bound and the gap
an upper bound); the older +53 % MTP figure was measured against an `--enforce-eager` baseline
later shown ~3.9× understated, and MTP perturbed greedy output on 1 of 3 prompts; benchmarks ran
at 16384 context, production coding at 32768 — neither is a property of the models; the default
loads as `Qwen3_5ForConditionalGeneration`, absent from vLLM 0.10.2, so pin at least v0.27.1
and confirm it carries sm_121 kernels.

**NVIDIA's published single-Spark recipe (fast alternative)** — the closest thing to a
validated starting configuration:

```
vllm serve --model $MODEL_CKPT \
  --host 127.0.0.1 \                # ADDED. NVIDIA's recipe omits --host; vLLM then binds
                                    # 0.0.0.0, violating rule 11. Never run it without this.
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.85 \   # NVIDIA's value, for THIS model, on an idle box. Rule 1:
                                    # convert to an absolute GB budget measured with WebUI and
                                    # the gateway resident. UNVERIFIED for the other two models.
  --speculative_config.num_speculative_tokens 3 \
  --speculative_config.model $DSPARK_CKPT \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  --reasoning-parser nemotron_v3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
```

### A9. Staging

Downloading is easy; verifying is not, and 23–63 GB over a desk connection fails often.

- `HF_HUB_DISABLE_XET=1` — the Xet backend has no stall timeout and **hangs silently**.
- `HF_HUB_DOWNLOAD_TIMEOUT=60` — fail a dead connection fast so the retry loop resumes.
- A bounded retry loop.
- **A byte-exact gate**: parse each safetensors file's 8-byte header-length prefix, sum
  `filesize − 8 − header` across shards, compare against the index's `metadata.total_size`.
  Byte equality, not a file count — a partial download that looks complete is what this catches.

Record the verified byte total per checkpoint in the evidence manifest (rule 9); it is also the
input to every ceiling in this appendix.
