# HANDOFF — write a `CLAUDE.md` for the DGX Spark deployment

**Your deliverable is one file: `CLAUDE.md`, at the root of a new repository, for running this
service on an NVIDIA DGX Spark.** Not user documentation, not an install script, not a port of
anyone's code. `CLAUDE.md` is the project-rules file that governs work *on* that repo — the
hard constraints, the fences that prevent expensive mistakes, and the pointers to where
evidence lives.

The source project is `/project/rcc/mehta5/vllm`: a single-user-per-session vLLM service on a
Slurm HPC cluster, with a browser chat client, three coding clients, MCP tools, and
Service-Unit billing. It has been running long enough to have paid for a large number of
mistakes. **The mistakes are the valuable part of this handoff.** The Slurm machinery is not
portable; the reasoning behind it is.

**The Spark cannot reach that filesystem.** This file is self-contained: §3 and §7 carry the
design and the incidents, and the appendix carries the concrete artifacts you would otherwise
have read out of the repo. Read it end to end before writing anything.

---

## 0. Execution contract

| you MUST | you MUST NOT |
|---|---|
| run the read-only verification commands in §5 | install vLLM, containers, or packages |
| read checkpoint metadata from the Hugging Face API (A8 gives the ids) | download model weights |
| write one file, `CLAUDE.md` | start `vllm serve`, Open WebUI, or any long-running process |
| report which numbers are measured and which are not | write code, launchers, or install scripts |
| ask at the STOP conditions | commit or push anything unless asked |

You are writing the rules that govern the install, not performing it. If a step tempts you to
"just check whether it works" by loading a model, that is out of scope — write the check into
`CLAUDE.md` as a rule for whoever runs the install.

### What "measured" means on this task

This matters because it is the one place an earlier draft of this document contradicted
itself. You cannot measure decode rate, load time, or memory bandwidth without serving a
model, which §0 forbids. Therefore:

- **Measurable now:** everything in §5's table — GPU name, compute capability, total and free
  memory, CPU architecture, OS, driver and CUDA version, container runtime, free disk.
- **Computable now:** decode *ceilings*, from published checkpoint byte sizes (A8) divided by
  the assumed bandwidth. An arithmetic ceiling is not a measurement; label it as derived and
  name both inputs.
- **Structurally unavailable:** realized decode rate, load time, whether a checkpoint fits at
  a given `--max-model-len`, whether a quantization format's kernels exist for this GPU.
  Every one of these is written into `CLAUDE.md` as a rule with the command that settles it
  at install time, marked `UNVERIFIED`. That is the expected end state, not a failure.

### Order of work

1. Read this document end to end.
2. Run §5's verification commands. Record actual output; do not paraphrase.
3. Pull checkpoint sizes from the Hub for the ids in A8 (`/api/models/<id>/tree/main` gives
   file sizes without downloading weights).
4. Compute the memory budget (§5a) and the decode ceilings (§5b) from steps 2 and 3.
5. Draft `CLAUDE.md` against §8's outline, incorporating §6's rules, §7's incidents, §9's
   decisions, and A8's model facts.
6. Run §10's self-check. Report.

### STOP conditions — ask, do not guess

1. **The arithmetic says the plan cannot work.** If a checkpoint's computed ceiling is so low
   that the workload it is assigned to is untenable, or a checkpoint plus a working KV cache
   exceeds the §5a budget, stop and report the arithmetic. Substituting a model the user did
   not choose is not your call.
2. **Measured hardware contradicts §5's assumed table** — different memory size, different
   compute capability, no container runtime. §5a–c are derived from those specs; if they are
   wrong, the rules built on them are wrong.
3. **You are not on the Spark.** Then even §5 is unavailable. Write the file with every
   hardware number marked `UNVERIFIED`, and say so plainly in your summary.
4. **Something in the appendix is ambiguous enough that guessing would change a rule.**

---

## 1. You do not have the source project

`/project/rcc/mehta5/vllm` is on the RCC cluster filesystem, not mounted on the Spark. Do not
write paths into `CLAUDE.md` that point at it, and do not plan a step that reads it.

The repository is on GitHub as `rmehta1987/rcc-vllm`, branch `milestone/model-refresh`. It may
be private — ask the user before assuming you can clone it. Nothing below depends on having it.

What the source project is, in one paragraph: users load a module, type `ai-session chat`,
`code`, or `fast`, and a preset submits a Slurm job that runs `vllm serve` on cluster GPUs. A
long-lived reverse proxy on the login node gives clients one fixed URL regardless of which node
the model landed on, captures per-request token usage, and exposes a keyless `/status`. Clients
are Open WebUI in a browser and aider / Continue / opencode for coding, each configured from
three environment variables the CLI writes to `~/.ai-session/env`. Usage is charged in Service
Units. Coding-model choices are made by measurement against a frozen benchmark subset and
recorded in the registry with their job IDs.

---

## 2. Scope of the deployment

Four workloads the new `CLAUDE.md` must carry rules for:

1. **Install vLLM**, reproducibly and pinned.
2. **Stage models**, with a registry recording what is served and what is registered-but-not,
   each with its reason.
3. **The web GUI** (Open WebUI) and the **coding harness** (aider, Continue, opencode).
4. **vLLM as an inference tool** — offline batch, structured output, embeddings, MCP.

Workloads 1–3 exist in the source project and are described from working systems. **Workload 4
largely does not**: the source has MCP servers and one PydanticAI example, but no offline-batch,
structured-output, or embeddings implementation. Design it from vLLM's documentation, and mark
in `CLAUDE.md` which parts are proven and which are aspirational. A single-box Spark with no
queue in front of it is the best host this workload will ever get, which is why it is in scope.

---

## 3. The design carried over

Seven ideas. Carry the ideas; do not copy implementations (§9.4).

| idea | why it exists on the cluster | what it becomes on the Spark |
|---|---|---|
| **Registry with a served/not-served split** | keys are stable identifiers used in job names, rate tables, logs; the gap records *why* a model is not servable | same split, keyed on what fits the memory budget and clears a usable decode ceiling. A model that fits but decodes too slowly is a NO-GO, recorded with its number |
| **Stable URL over a moving backend** | the vLLM endpoint changed node and port every session | the endpoint stops moving but the **resident model** does. The gateway keeps its value as the seam for model swaps, usage capture, and the access key |
| **Presets, not knobs** | users pick intent, not tensor parallelism | keep the principle, not the preset list. Parallelism disappears entirely (TP=1); memory budget and quantization are the remaining knobs and both are more dangerous than TP was |
| **One session at a time, and something that reclaims it** | idle GPUs were billed | nothing is billed, but memory and bandwidth are shared. One resident model, an idle unload, and a swap lock (§6.8) |
| **Model choice is a measurement** | every coding candidate scored on a frozen 60-problem LiveCodeBench subset, identical harness and decode, verdicts recorded with job IDs and pass@1 deltas | keep the discipline. The Spark adds a second axis: a verdict is now a pair, (quality, decode rate). Quality scores carry as priors; **decode rates do not transfer at all** |
| **Nothing leaves the machine** | the service exists so unpublished code and data never reach a commercial provider; external tools are opt-in and off by default; client telemetry disabled by name | carry the posture verbatim, including the per-client telemetry list (A6) and the honesty about the opt-in exceptions |
| **Evidence outlives the weights** | measured results live in a gitignored dir and are the only copy | same rule, sharper: commit evidence out of the gitignored directory before deleting weights it describes. A 4 TB NVMe fills faster than a project filesystem |

Note on the fifth row: only the *coding* candidates were LCB-scored. The general-chat and
smoke models predate that discipline and were never benchmarked.

---

## 4. What is NOT carried over

State this in `CLAUDE.md` so a reader who knows the source project stops looking for it. Phrase
it as a short "departures" section — see §10's note about wording.

- **The scheduler, the accounts, the queue, and the GPU allowlist.** The allowlist exists
  because the cluster has hardware owned by other research groups. A Spark has one owner and
  one GPU. Do not port it. **Do port the *shape*** — a short table of hard constraints, each
  with its reason and the incident behind it. That shape is what makes a rules file work.
- **All chargeback machinery** — the Service-Unit formula, rate table, reservation floor, the
  sweep, the central ledger. One fragment survives in a different form: per-request usage
  logging, now for capacity and for answering "why was it slow", not for billing.
- **Environment Modules and modulefile deployment.** Replace with a CLI on `PATH` plus systemd
  user units. Keep the principle: users never type an install path, and the deployed copy
  cannot drift from the source copy in the repo.
- **The air-gapped-compute-node assumption.** Every awkward staging script in the source repo
  exists because cluster compute nodes have no internet. **The Spark has internet**, so staging
  is a plain download — but keep the verification gate (A9), which matters more over a desk
  connection, not less.
- **Cluster-scale multi-tenancy.** Three users (§9.1) need far less than a campus, but more
  than nothing: authenticated browser chat, per-user usage records, a swap lock. Reduce, do
  not delete.
- **`AGENTS.md`.** In the source repo this is *not* a rules file; it is a prompt-level
  workaround for a model that could not emit `<tool_call>` tokens. All three models here have
  working tool parsers, so that file should not exist — and `CLAUDE.md` should say so, because
  an inherited copy is actively counterproductive with models that parse tool calls correctly.

---

## 5. Verify the hardware first

**Do not write a hardware number into `CLAUDE.md` that you have not read off the box.** The
table is the assumed specification; every row is a hypothesis with a verification command.

| assumed | verify with |
|---|---|
| GB10 Grace-Blackwell, one GPU, compute capability sm_121 | `nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv` |
| 128 GB LPDDR5X unified, coherent CPU/GPU, **shared with the OS** | `free -g` and `nvidia-smi -q -d MEMORY` — note whether they report the same pool |
| ~273 GB/s memory bandwidth | no read-only command gives this. Treat as assumed, cite it as the input to every ceiling you compute, and write the microbenchmark into `CLAUDE.md` as an install-time task |
| 20-core Arm CPU, **aarch64** | `uname -m`, `lscpu` |
| DGX OS (Ubuntu-derived), container runtime present | `cat /etc/os-release`, `docker info`, `nvidia-smi` driver + CUDA version |
| ~4 TB NVMe | `df -h` — set the weights budget from what is actually free |

### (a) Unified memory is not VRAM

`--gpu-memory-utilization` is a fraction of a pool the OS, the desktop session, and your own
processes are also using. On a discrete-GPU node, over-committing kills the job; here it causes
system-wide memory pressure — allocation failures and OOM-killed processes, up to and including
an unusable box. Establish the real usable budget by measurement, **state it in gigabytes**,
and prefer an absolute KV-cache size to a utilization fraction where the pinned vLLM supports
one. This is the most likely way to make the Spark unusable, so it is the rule that goes first
in `CLAUDE.md` — the structural slot the GPU allowlist occupies in the source file.

A related trap, and a real incident (§7): a model with a vision tower profiles its multimodal
path at startup and can OOM there even when the language weights fit comfortably.

### (b) Decode is bandwidth-bound

One GB10 has roughly an order of magnitude less memory bandwidth than an H100. Decode is capped
by bytes read per token:

```
tok/s_ceiling  ≈  memory_bandwidth / bytes_read_per_token
bytes_read_per_token  ≈  active_parameter_bytes  +  KV_bytes_read
```

**What governs interactive speed here is bytes read per token, not parameter count.** That
sentence belongs in `CLAUDE.md`; A8 works the arithmetic for every candidate checkpoint.
Realized rates are typically 40–70 % of ceiling. Two consequences:

- **For a dense model, quantization is the only lever** — it moves bytes-read linearly.
- **For a sparse (MoE) model, only the active experts are read**, so a 30B model with 3B
  active reads about a tenth of what its parameter count suggests. This is why the default
  model in §9.5 is an MoE.

### (c) Quantization on GB10 — read this before assuming anything

An earlier draft of this document got this backwards in both directions. The corrected facts:

- The source project's NVFP4 NO-GO was **not** an Ampere hardware limitation. It was job
  53069684 on **H200 (sm_90)**, where the Marlin FP4 expert repack aborted with
  `cudaErrorUnsupportedPtxVersion` — the kernel's PTX was newer than the cluster's EOL driver
  535. It is void on the Spark because of the **driver**, not the silicon.
- **GB10 does not necessarily take a native FP4 tensor-core path.** NVIDIA's own hardware
  matrix for a Spark-targeted NVFP4 model lists, for DGX Spark (GB10): stored precision NVFP4,
  compute path **W4A16**, MoE backend **`marlin`**, native FP4 path **"No — runs via Marlin"**
  — while GB200 gets "Yes". So **Marlin is load-bearing on this box**, not a legacy path to be
  avoided. Do not write a rule that steers away from Marlin.
- The practical rule for `CLAUDE.md`: a checkpoint's *stored* precision and its *compute* path
  are different things, and which kernels exist is a property of the pinned vLLM build. Record
  a per-format table — NVFP4, FP8, compressed-tensors w4a16 — each row marked `UNVERIFIED`
  with the command that settles it at install time.
- **aarch64** is the other half of this: packages needing compiled extensions may have no Arm
  wheel, which is the main argument for a container-first install.

---

## 6. Rules the new `CLAUDE.md` must contain

Each is a hard rule with its reason, in the style of the source file's hardware section.

1. **Memory budget in gigabytes.** An absolute ceiling for a serve and an absolute floor
   reserved for the OS — not a utilization fraction, which is precisely what is dangerous here
   (§5a). Measuring a model's resident size is a precondition of serving it.
2. **TP=1, unconditionally.** One GPU, one box, no second Spark (§9.3). Do not document a
   two-box case as a possibility.
3. **Container-first install, with a pin.** Name the exact image and the CUDA and driver
   versions, as a *candidate* marked `UNVERIFIED` until first load — you cannot verify it under
   §0. NVIDIA's published Spark recipe names `vllm/vllm-openai:v0.27.1` (A8); that is the
   obvious candidate. Carry the source project's hard-won rule: *do not
   `pip install -r requirements.txt` into the serving environment* — it pulls a torch/vLLM pair
   the driver cannot run.
4. **Quantization policy**, with §5c's per-format table and the corrected Marlin story. No
   claim about native FP4 without evidence from this box.
5. **Bench and smoke fences.** The source's fences exist so a test run cannot be mistaken for
   production by discovery or billing. There is no billing here, but the hazards — a benchmark
   being picked up as the live backend, or evicting the resident model mid-conversation — are
   real. Define a distinct port range and a `bench-` served-name prefix the gateway refuses to
   route to.
6. **Evidence rule** (§3): name the gitignored directory and an append-only JSONL manifest of
   scored runs; commit evidence out before deleting the weights it describes.
7. **Thermal and power honesty.** A desk box under sustained load throttles. If sustained
   throughput differs from burst, users need both numbers.
8. **One resident model; swapping takes a lock.** The reason is *not* that they cannot
   co-reside — at 4-bit all three models in A8 fit in 128 GB together. The reasons are that KV
   cache is what makes a long context usable and it wants the leftover memory, and that a
   second model serving concurrently competes for the same scarce bandwidth. With three users
   a swap affects other people: it requires no in-flight requests, takes an explicit lock, and
   is visible. The CLI must answer "what is resident, and who is using it".
9. **Users and access.** Three users; every listener bound to `127.0.0.1`; browser access over
   SSH tunnel only; authenticated Open WebUI. Put the tunnel command in `CLAUDE.md` (user docs
   inherit it later). No LAN bind, no TLS, no auth proxy — the tunnel is the security boundary.
10. **Privacy posture.** Nothing leaves the box on the serving path; external tools opt-in and
    off by default; client telemetry disabled by name per client (A6). This is the reason the
    service exists and it needs a rule, not just a sentence.
11. **After any agent, subagent, or background run, check for orphaned processes.** The source
    rule is `squeue -u $USER`; here it is stray `vllm serve` processes and containers holding
    unified memory. On one box that blocks everyone.

---

## 7. Mistakes already paid for

Each cost the source project a debugging cycle or a wasted allocation. State the Spark form in
`CLAUDE.md` so they are not re-discovered.

| source incident | Spark form |
|---|---|
| Serve-time caches filled a 35 GB home quota and killed a load mid-profiling (job 53061850) | point the vLLM compile cache, Triton cache, Torch/Inductor caches, HF cache and FlashInfer workspace at explicit NVMe paths. Name them. Several do **not** honour `XDG_CACHE_HOME` (A1) |
| Triton cache in a per-job temp dir meant recompiling every kernel; on an unfamiliar compute capability that blew the 600 s engine-ready timeout (job 53539504) | sm_121 is exactly such an architecture. Make the Triton cache **persistent** and raise `VLLM_ENGINE_READY_TIMEOUT_S`. Expect the first load of any model to be far slower than the second, and say so |
| A concurrent job's cleanup deleted a running job's `TMPDIR`, killing FlashAttention 27 problems into a 60-problem benchmark; the 33 resulting HTTP 500s *would have* scored as wrong answers had it not been caught (job 53544726) | namespace scratch per process, and make partial result files distinguishable from complete ones. A truncated benchmark that scores as a bad model is worse than a crash |
| The `hermes` tool-call parser fails **silently** on models it does not fit; the launcher routes both parsers by model key | keep per-model parser routing in the registry (A6, A8). Silent tool-call failure is the most expensive bug in this stack because it presents as the model being stupid |
| Gemma's thinking cost 5–12× tokens and wall time with no measurable answer improvement — on two greedy prompts (job 53587542) | on a bandwidth-bound box that is a 5–12× latency multiplier. Set thinking per model, per the registry (A8) — **not** a blanket default, since the three models differ. Record the multiplier and the n=2 caveat together |
| `--enforce-eager` was adopted as a crash workaround, then removed: CUDA graphs plus a compilation-config flag recovered 894.7 → 3492.7 tok/s decode, ~3.9× (job 53729212) | never leave `--enforce-eager` as a permanent default; record it as a workaround with the condition for removing it. The specific crash cannot occur here — the failing pass only enables at TP>1 (A8) |
| An MTP draft head worth +53 % decode sat unused | speculative decoding trades spare compute for bandwidth, which is exactly this box's imbalance. **This is a headline lever on the Spark**, and the default model ships a purpose-built draft (A8). Carry the caveats with the number |
| Slurm reported 2 GPUs while the cgroup exposed 1; vLLM died after the allocation was spent (job 53539505) | write a preflight that asserts actual free unified memory before loading weights and fails with a legible message instead of OOMing the box |
| Two models' scores were compared under conditions that suited one model's defaults and suppressed the other's | whenever two numbers appear together, the conditions they were measured under appear too. Carry this as a documentation rule |
| A serve attempt OOM'd during **multimodal profiling** even though the language weights fit; `--language-model-only` fixed it (jobs 53738460 → 53742254) | a vision tower profiles at startup and can OOM on a memory-tight unified box. Record the flag next to any multimodal checkpoint (A8) |

---

## 8. Required outline for the new `CLAUDE.md`

Hard rules first with reasons attached, fences second, evidence third, related files last.

```
# Project rules
## Hardware reality        <- §5: verified specs, the memory budget in GB, the bandwidth ceiling
### What this rules out    <- the analogue of "everything else is off limits"
## Install and pinning     <- container pin, CUDA/driver, aarch64, the requirements.txt rule
## Model policy            <- registry + served set, quantization table, sizing arithmetic
## Serving fences          <- port range, bench- prefix, one-resident-model rule, swap lock
## Users and access        <- three users, 127.0.0.1, the SSH tunnel command, WebUI auth
## Clients                 <- Open WebUI, aider/Continue/opencode, per-model serve flags
## Inference as a tool     <- offline batch, structured output, embeddings, MCP
## Evidence                <- the gitignored path, the manifest, commit-before-delete
## Related                 <- what AGENTS.md is and is not; where user docs live
```

Keep it under ~200 lines. If the material will not fit, cut reasons down to one clause each
before you cut a rule, and never cut a number's provenance.

---

## 9. Decisions the user has made

Constraints, not defaults. Each belongs in `CLAUDE.md` as a rule with its consequence.

**1. Three users, internal testing.** Consequences:

- One resident model and three people means the box serializes; see §6.8 for the swap lock.
- **Do not assume per-user decode rate simply divides.** In the bandwidth-bound regime the
  weights are read once per step for the whole batch, so continuous batching amortizes the
  dominant cost and per-user rates hold up better than intuition suggests. The real three-user
  cost is prefill contention stealing decode steps. Measure the three-user case; do not model
  it from the single-user number in either direction.
- **Open WebUI runs one instance with authentication on and three accounts.** Unauthenticated
  is a single-user posture that would give three people one shared chat history — the same
  class of bug the source project fixed once by moving its chat database out of a
  group-readable tree. One instance, not three: each holds its own embedding model and unified
  memory is the scarce resource.
- Usage logging stays, per user, for capacity — not chargeback. "Who is holding the model" is
  a real question at three users and the CLI should answer it.

**2. SSH tunnel for the web GUI.** Every listener on `127.0.0.1`, no LAN bind, no VPN
dependency. This is the simplifying answer: no TLS, no auth proxy, no exposure beyond SSH,
which is already how these users reach the box.

**3. No second Spark.** TP=1 is unconditional. Delete the two-box case rather than leaving it
as a documented possibility — an aspirational escape hatch in a rules file is how
`--tensor-parallel-size 2` eventually gets typed.

**4. Clean repo.** No code is ported. The appendix is reference for *what the pieces do and
why*, not source to copy. The cost worth naming in `CLAUDE.md`: each rewrite is a chance to
lose a fix that survives only as a comment here. Work through A1's environment list and A6's
parser routing deliberately rather than rediscovering them.

**5. Three models: one default, two references.** Full detail in A8.

- **Default: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`**, with its `DSpark` draft model for
  speculative decoding. Chosen because it is the only candidate with a published, official
  single-DGX-Spark vLLM recipe, and because its architecture matches what this box is short of:
  30B total but **3B active**, so it reads roughly a tenth of a comparable dense model per
  token. It serves chat, coding, and batch inference.
- **References: `qwen3.8_27B` and `gemma4_31B`.** These are the two models whose coding quality
  this project actually measured on a frozen 60-problem subset (50.00 % and 66.67 %). Keep them
  registered: they cost nothing to retain, and they are the only yardstick you have for judging
  whether the default is good enough at coding. The default's quality on that subset is
  **unknown** — measuring it is the first real experiment on the box.
- Both references are **dense**, so for them quantization is mandatory, not preferred (§5b).
  Their BF16 checkpoints are on the wrong side of the bandwidth arithmetic; A8 lists the
  quantized checkpoints and their ceilings.
- Registry status: the default is served; the references are served only if their quantized
  checkpoints clear the ceiling test; BF16 checkpoints of either are registered-but-not-served
  with the arithmetic as the recorded reason.

---

## 10. Self-check before reporting

```bash
# Positive: the load-bearing rules are present.
grep -q '127\.0\.0\.1' CLAUDE.md            && echo ok-bind      || echo MISSING-bind
grep -q 'bench-'       CLAUDE.md            && echo ok-fence     || echo MISSING-fence
grep -qi 'tensor-parallel' CLAUDE.md        && echo ok-tp        || echo MISSING-tp
grep -qiE 'GB|gigabyte' CLAUDE.md           && echo ok-budget    || echo MISSING-budget
wc -l CLAUDE.md                             # target: under ~200

# Negative: no path into the unreachable source project.
grep -n '/project/rcc' CLAUDE.md            # expect: no hits

# Review-each-hit: cluster machinery should appear ONLY in the departures section (§4).
grep -niE 'slurm|sbatch|squeue|partition|--qos|service unit|modulefile' CLAUDE.md
# Every hit must be inside "what is not carried over". A hit anywhere else is a defect.

# Every unverified number is labelled.
grep -o 'UNVERIFIED' CLAUDE.md | wc -l      # expect: nonzero -- see §0, most numbers
                                            # cannot be measured under this contract
```

Then read it as the agent who will be governed by it:

- Every hard rule carries the reason it exists.
- Every number is measured with its command, or derived with both inputs named, or marked
  `UNVERIFIED` with the command that would settle it.
- The memory budget is in gigabytes and appears before any other rule.
- A reader who has never seen the source project can answer: what fits in memory, why TP is
  fixed at 1, how to install vLLM reproducibly, what a benchmark run must be named so it cannot
  be mistaken for production, who may reach the box and how, and where evidence lives.
- The reversals from the cluster are stated as reversals: there is no billing; quantization is
  mandatory for the dense models; and the FP4 NO-GO does not transfer — for the driver reason,
  not a silicon one (§5c).
- §9's five decisions each appear as a rule with its consequence.

Report: the file's location, which §5 numbers you measured, which are derived, which are
`UNVERIFIED`, and any STOP condition you hit.

## House style

Scientist-to-scientist prose. No checkmarks, no emoji, no marketing language. Numbered steps,
exact commands, tables, and measured numbers with their provenance. Prefer deleting a paragraph
to hedging it. State a constraint once, with its reason, and do not repeat it.

---

## Appendix — the artifacts

Reference material, not a template to port. Read what each piece is *for*, then decide what the
Spark version should be.

### A1. The canonical serve invocation

```
vllm serve $MODEL_PATH \
  --served-model-name $MODEL_KEY \
  --host 0.0.0.0 \          # cluster posture. On the Spark this MUST be 127.0.0.1 (§9.2)
  --port $PORT \
  --tensor-parallel-size $TP \      # always 1 here (§9.3)
  --trust-remote-code \
  --enable-prefix-caching \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEM_UTIL \
  $TOOL_FLAGS $REASONING_FLAG $EAGER_FLAG $COMPILATION_FLAG
```

`--enable-prefix-caching` is always on and matters more here than on the cluster: coding
clients resend a large, mostly-identical prompt every turn, and prefill is the compute-bound
half of a workload whose decode half is already bandwidth-starved.

The benchmark harness builds this argv **once** into an array and reuses it for the serve,
smoke, and bench paths, so the three cannot drift and produce numbers measured under different
conditions. Keep that property. (The production launcher built its own argv separately and did
drift — its context default is 32768 while every benchmark ran at 16384.)

Environment that had to be set around the serve command, each line a fixed bug:

```
VLLM_CACHE_ROOT              compile cache -- large, must not sit on a quota'd filesystem
TRITON_CACHE_DIR             MUST persist across runs (§7) -- never a per-job temp dir
TORCHINDUCTOR_CACHE_DIR, TORCH_EXTENSIONS_DIR, TORCH_HOME, XDG_CACHE_HOME
FLASHINFER_WORKSPACE_BASE    FlashInfer ignores XDG and defaults to $HOME
HF_HOME                      model and tokenizer cache
VLLM_ENGINE_READY_TIMEOUT_S=2400   default 600 is not enough for a cold Triton cache on a
                                   compute capability with no prebuilt kernels
```

### A2. The registry shape

```python
MODEL_REGISTRY = {                        # key -> path. The key is the stable identifier used
    "<key>": f"{MODELS_ROOT}/<dir>",      # in the served-model-name, logs, and client configs.
    ...                                   # Registered-but-unserved entries stay here WITH the
}                                         # reason recorded in a comment.

SERVED = {"<key>", ...}                   # strict subset of MODEL_REGISTRY
```

The value is in the comments, not the code. A real entry records: the measured score that
justified it (`50.00% (30/60) vs 26.67% (16/60), +23.33 pts, score job 53531932`), the hardware
and parallelism it was proven on, which tool-call parser works and which fails silently,
whether thinking is on by default and what that costs, and what measurement is still owed.
Registered-but-unserved entries record why, so nobody re-litigates a settled question. Write the
Spark registry the same way, with memory footprint and decode ceiling as the reasons.

### A3. The gateway contract

One fixed URL in front of a backend that moves. The current backend is written to a JSON file
when a session starts and cleared when it ends; the proxy re-reads it with an mtime cache, so
it follows a new backend without a restart.

- Proxies verbatim: `/v1`, `/metrics`, `/health`, `/version`, `/ping`, `/tokenize`,
  `/detokenize`, `/pooling`. Drops hop-by-hop and length headers both directions.
- Captures every response's `usage` block to a dated JSONL. This is the authoritative usage
  record — users never log their own tokens.
- `GET /status` is **keyless** and TTL-cached, returning ready / loading / no_backend, so a
  status check cannot fan out into a storm of backend health probes.
- Per-client token-bucket rate limit (30 rps default, `429` + `Retry-After`) and a body cap
  (16 MB default, `413`), each disabled by setting its variable `<= 0`.
- API key optional on the cluster. **On the Spark make it mandatory and per-user** — §9.1's
  per-user usage records need an identity, and an optional key does not provide one.

### A4. The user-facing CLI surface

The source verbs are `chat`, `code`, `fast`, `status`, `connect`, `env`, `models`, `receipt`,
`mcp config|run`, `stop`. Do not copy that list: `fast` has no model behind it here, and
`receipt` is billing (§4). Copy the *shape* — short intent verbs, no paths, no flags for the
common case.

The load-bearing part is this: after a session comes up the dispatcher writes `~/.ai-session/env`,
mode 600, containing exactly three variables — `AISESSION_BASE_URL`, `AISESSION_API_KEY`,
`AISESSION_MODEL`. Every client and example reads those, which is why no client config in the
project contains an install path or a hostname. Preserve this. It is the highest-leverage piece
of UX in the source project and costs almost nothing.

### A5. Browser chat (Open WebUI)

Runs in its **own** environment, never the serving one, pointed at the gateway so it needs no
reconfiguration when the backend changes:

```
ENABLE_OPENAI_API=True
OPENAI_API_BASE_URL=http://localhost:$GW_PORT/v1
OPENAI_API_KEY=<session key>
ENABLE_OLLAMA_API=False
DATA_DIR=$HOME/.ai-session/openwebui-data     # chat history: PRIVATE, mode 700
HF_HOME=<per-user state dir>/hf_cache          # the UI writes here; never the shared install
ANONYMIZED_TELEMETRY=False  DO_NOT_TRACK=true  SCARF_NO_ANALYTICS=true
WEBUI_AUTH=False                               # cluster demo posture -- MUST be True here (§9.1)
```

Two details worth carrying: the chat database originally sat in a group-readable project tree
where any other user could read someone's chats — it belongs in a mode-700 directory under
`$HOME`. And the RAG embedding model is resolved from a shared read-only cache with
`HF_HUB_OFFLINE=1` when present, because letting every user download it into an empty cache on
first run was slow enough to overrun the UI's bind-wait.

Web search, URL fetch and paper search are **opt-in** behind one variable, default off, because
their queries leave the machine. Say so plainly rather than burying it.

### A6. The coding-client contract

- **Per-model tool-call and reasoning parser routing**, by model key — see A8 for the three
  models' values. The wrong tool parser fails *silently*: the model appears stupid rather than
  broken. Tool calling is enabled with `--enable-auto-tool-choice --tool-call-parser X`.
- A reasoning parser puts chain-of-thought in a separate `reasoning` field instead of
  contaminating `content`. With a parser on and a tight `max_tokens`, `content` can come back
  null while the model is still thinking — budget tokens generously or disable thinking.
- **aider** needs a litellm metadata file declaring the context window, or it warns "Unknown
  context window size" and mis-sizes prompts; keys must be duplicated with and without the
  `openai/` prefix. Note the source project's file has **no entry for any model it currently
  serves** — it lists only retired ones, so aider warns today. Do not inherit that gap: write
  entries for whatever you serve. Split the window so prompt + `max_tokens` cannot exceed
  `--max-model-len` (the source uses 28000 in / 4096 out against 32768).
- **opencode** needs a project-local config declaring a custom provider:

```json
{
  "model": "spark/<model_key>",
  "share": "disabled",
  "autoupdate": false,
  "provider": { "spark": {
    "npm": "@ai-sdk/openai-compatible",
    "options": { "baseURL": "{env:AISESSION_BASE_URL}", "apiKey": "{env:AISESSION_API_KEY}" },
    "models": { "<model_key>": { "limit": { "context": 32768, "output": 8192 } } }
  }}
}
```

  Note the two clients disagree in the source: aider is capped at 4096 output, opencode declares
  8192, against the same 32768 window. Pick one budget per model and make both clients agree.
- Client telemetry is disabled by name in each: aider `--analytics-disable`, Continue
  `allowAnonymousTelemetry: false`, opencode `share: disabled` and `autoupdate: false`. The docs
  state that the coding tool is separate software with its own telemetry, outside the service's
  control — an honest boundary rather than a blanket privacy claim.

### A7. MCP tools and their security model

Two read-only MCP servers let a coding agent answer "what are my jobs doing?" and "how much have
I used?" without being handed a shell. The security model is enforced in code, not by
convention, and is the part to carry:

- a hard whitelist of binaries, checked on every call;
- every command built as an argv list with `shell=False`, so no argument is shell-interpreted;
- arguments constrained by regex, anchored `\A...\Z` — **not** `^...$`, because in Python `$`
  also matches before a trailing newline and would accept `"123\n"`;
- ownership verified before returning anything;
- no mutating verb reachable at all.

They live in a **dedicated environment**, deliberately not the serving one, so the MCP SDK's
dependency tree can never perturb vLLM. The Spark analogue exposes box state — resident model,
free unified memory, temperature, disk — under the same rules.

### A8. The three models, exactly

Hub ids, sizes pulled from the Hub file listing, ceilings derived at the assumed 273 GB/s.
**Every tok/s figure here is arithmetic, not measurement** (§0).

| checkpoint | Hub id | weights | read/token | ceiling |
|---|---|---:|---:|---:|
| **Nemotron Lightning NVFP4** *(default)* | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 21.6 GB | ~1.7 GB (3B active) | high — see note |
| **DSpark draft** | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark` | 1.3 GB | 967M params | draft only |
| Qwen3.8-27B BF16 | `Qwen/Qwen3.8-27B` | 55.6 GB | 55.6 GB | 4.9 tok/s |
| Qwen3.8-27B FP8 | `Qwen/Qwen3.8-27B-FP8` | 30.9 GB | 30.9 GB | 8.8 tok/s |
| Qwen3.8-27B NVFP4 | `unsloth/Qwen3.8-27B-NVFP4` | 23.4 GB | 23.4 GB | 11.7 tok/s |
| Qwen3.8-27B NVFP4 | `RadixArk/Qwen3.8-27B-NVFP4` | 21.9 GB | 21.9 GB | 12.5 tok/s |
| Gemma-4-31B-it BF16 | `google/gemma-4-31B-it` | 63.4 GB | 63.4 GB | 4.3 tok/s |
| Gemma-4-31B-it QAT 4-bit | `google/gemma-4-31B-it-qat-w4a16-ct` | 23.6 GB | 23.6 GB | 11.6 tok/s |

Note on the default's ceiling: only the active experts are read per token, so the naive
weights/bandwidth figure does not apply — but router, attention and shared layers add to the
1.7 GB, and the Mamba state has its own traffic. Compute a range rather than a point, mark it
derived, and make measuring it the first install-time task. Note also that the two Qwen NVFP4
checkpoints are **third-party**; the FP8 is first-party. Quantization quality is a measurement,
not an assumption — that is the whole discipline in §3.

**Per-model serve facts:**

| | Nemotron Lightning | `qwen3.8_27B` | `gemma4_31B` |
|---|---|---|---|
| architecture | hybrid Mamba-2 + MoE, 30B total / **3B active**, 1M context | dense 27.8B, hybrid Gated DeltaNet + attention | dense 30.7B, **multimodal prefix-LM** (vision tower) |
| `--reasoning-parser` | `nemotron_v3` | `qwen3` | `gemma4` |
| `--tool-call-parser` | `qwen3_coder` | `qwen3_coder` — **`hermes` fails silently** | `gemma4` |
| thinking | reasoning model | **ON** by default (`reasoning_effort` xhigh) | OFF by default — **but the template turns it on whenever tools or a system message are present, i.e. exactly in coding sessions**. Measured cost when on: 5–12× tokens and wall time, no measurable gain on two greedy prompts |
| speculative decoding | **DSpark draft, plus MTP and DFlash heads** | ships an MTP head, +53 % decode measured — see caveats | none in this checkpoint |
| known trap | — | — | first single-GPU attempt **OOM'd during multimodal profiling**; `--language-model-only` fixed it |
| frozen LCB-60 | **unmeasured** | 50.00 % (30/60) | **66.67 %** (40/60); hard subset 14/30 vs 6/30 |
| licence | OpenMDW-1.1 | check the shipped `LICENSE` | check the shipped `LICENSE` |

**Caveats that must travel with these numbers:**

- **The two reference scores are not like-for-like.** Both were measured with thinking
  disabled, which is Gemma's native default but suppresses Qwen's `xhigh` default — so 50.00 %
  is a lower bound and the gap is an upper bound. Wherever the two appear together, so does this.
- **The +53 % MTP figure was measured under `--enforce-eager`**, a baseline later shown to be
  ~3.9× understated, so the gain over a CUDA-graphs baseline is unestablished. MTP also
  perturbed greedy output on 1 of 3 prompts, and its interaction with tool calls is an open
  vLLM issue — which matters here, because coding is a tool-heavy workload.
- **CUDA graphs are ON in current production** for the Qwen model, with a compilation-config
  flag disabling one fusion pass. That pass only enables at TP>1, so the crash that once forced
  `--enforce-eager` **cannot occur on this box**. Do not inherit the workaround.
- Benchmarks ran at `--max-model-len 16384`; 32768 is the production coding default. Neither is
  a property of the models.
- Both reference models were only ever served **TP=2** across two cards ranging 40–141 GB.
  Single-device residency for either BF16 checkpoint is unproven. The one single-card datapoint
  on record is the Gemma QAT 4-bit checkpoint on a 48 GB A40. The default model, by contrast, is
  published as a single-GB10 deployment.
- Qwen3.8-27B loads as `Qwen3_5ForConditionalGeneration`, an architecture vLLM 0.10.2 does not
  have; the cluster needed 0.26.0. NVIDIA's Spark recipe names 0.27.1. Pin at least that, and
  confirm it carries kernels for this GPU.

**NVIDIA's published single-Spark recipe for the default model** — the closest thing to a
validated starting configuration anyone has:

```
vllm serve --model $MODEL_CKPT \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.85 \
  --speculative_config.num_speculative_tokens 3 \
  --speculative_config.model $DSPARK_CKPT \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  --reasoning-parser nemotron_v3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
```

Read every flag against §5: `--moe-backend marlin` is the W4A16 compute path (§5c),
`--kv-cache-dtype fp8` is buying KV headroom in a tight memory budget (§5a), and
`--gpu-memory-utilization 0.85` is NVIDIA's own answer to §6.1 — worth treating as a starting
point to verify, not a number to copy blindly, since your OS floor may differ.

### A9. The staging recipe

Downloading is trivial on a box with internet; verifying is not, and 22–63 GB over a desk
connection fails more often than over a datacentre link. The source project's stager, worth
reproducing in spirit:

- `HF_HUB_DISABLE_XET=1` — the Xet backend has no stall timeout and hangs.
- `HF_HUB_DOWNLOAD_TIMEOUT=60` — fail a dead connection fast so the retry loop resumes.
- A bounded retry loop around the download.
- **A verification gate**: parse each safetensors file's 8-byte header-length prefix, sum
  `filesize - 8 - header` across shards, and compare against the index's `metadata.total_size`.
  Exact byte equality, not a file count. A partial download that looks complete is the failure
  this catches.

Record the verified byte total per checkpoint in the evidence manifest (§6.6) — it is also the
input to every decode-ceiling calculation in A8.
