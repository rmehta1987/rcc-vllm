# HANDOFF — write a `CLAUDE.md` for the DGX Spark deployment

**Your deliverable is one file: `CLAUDE.md` for a new repository that installs and runs the
same service on an NVIDIA DGX Spark.** Not user documentation, not an install script, not a
port of this repo's code. `CLAUDE.md` is the project-rules file that governs work *on* that
repo — the hard constraints, the fences that prevent expensive mistakes, and the pointers to
where evidence lives. Everything else follows from getting that file right.

The source project is `/project/rcc/mehta5/vllm` (branch `milestone/model-refresh`, green
floor `1849a95`): a single-user-per-session vLLM service on a Slurm HPC cluster, serving
open-weight models over an OpenAI-compatible API, with a browser chat client, three coding
clients, MCP tools, and Service-Unit billing. It has been running long enough to have paid
for a large number of mistakes. **The mistakes are the valuable part of this handoff.** The
Slurm machinery is not portable; the reasoning behind it is.

Read this document end to end before writing anything. If you have the source repo mounted,
read the files named in §1 as well — they are the primary sources and this document is a
summary of them.

---

## 1. What to read in the source project

| file | what it gives you |
|---|---|
| `CLAUDE.md` | the shape your output should have: hard rules first, each with the reason it exists and the incident that produced it |
| `ai-session/README.md` | the operator guide — layout, runtime environment, packaging, every component and why it exists |
| `ai-session/server.py` | `MODEL_REGISTRY` / `PHASE1_SERVED` — the registry pattern, and ~90 lines of commentary recording why each model is in or out |
| `ai-session/gateway.py` (header) | the stable-URL-over-ephemeral-backend design |
| `ai-session/run_openwebui.sh` | the browser-chat client, private data dir, telemetry-off posture, opt-in external tools |
| `ai-session/CODING_AGENTS.md` | the coding-client contract (aider, Continue, opencode) |
| `tools/serve_cu129.sbatch` | the canonical serve invocation, and eight comment blocks each recording one production failure |
| `docs/index.md`, `docs/coding/overview.md` | the user-facing framing: presets, not knobs |
| `billing/su_formula.py`, `ai-session/BILLING_POLICY.md` | the accounting model (mostly *not* carried over — see §4) |

If you do not have the repo, §3 and §7 carry enough of the design to proceed.

---

## 2. What the Spark deployment must do

Four workloads, all of which exist in the source project and all of which must be covered by
rules in the new `CLAUDE.md`:

1. **Install vLLM** on the Spark and keep the install reproducible and pinned.
2. **Install and stage models**, with a registry that records which ones are actually
   servable on this box and which are registered-but-not-served, with the reason.
3. **The same web GUI** (Open WebUI) and **the same coding harness** (aider, Continue,
   opencode) pointed at the served endpoint.
4. **vLLM/the models as an inference tool** — offline batch inference, structured output,
   embeddings, and MCP tools, called from scripts rather than by a human in a chat window.
   This is the workload the source project supports least well and the one a single-box
   Spark is best at, because there is no queue between you and the GPU.

---

## 3. The design being carried over

Seven load-bearing ideas. For each: why it exists, and what it becomes on the Spark. These
are the ideas — do not copy the implementations.

**1. A model registry with a served/not-served distinction.** `MODEL_REGISTRY` maps a stable
key to a path; `PHASE1_SERVED` is the subset users may actually start. The gap between them
is deliberate and carries the reasons: one model is registered but unserved because its FP8
weights need Hopper, another because its production serve path routes to an environment that
cannot load its architecture. *On Spark:* the same two-set structure, with the gap keyed on
**what fits in 128 GB of unified memory and runs acceptably at the box's memory bandwidth**
(§5). A model that fits but decodes at 4 tok/s is a NO-GO for interactive use and should be
recorded as one, with the measured number.

**2. A stable URL in front of an ephemeral backend.** The gateway exists because the vLLM
endpoint moved every session and clients hate that. *On Spark:* the endpoint no longer moves,
but the **model behind it does** — 128 GB holds one large model at a time, so serving a
second means unloading the first. Keep the gateway: it makes model swaps invisible to Open
WebUI, aider, and any script holding a base URL, and it is the seam where request logging
and the per-client key live. Its value moved from *address stability* to *model swaps and
observability*, and it should be justified that way in the new file, not by inheritance.

**3. Presets, not knobs.** Users type `ai-session chat` / `code` / `fast`; the preset picks
the model, GPU count, and parallelism. Nobody configures tensor parallelism by hand. *On
Spark:* preserve exactly. The knobs that remain are memory split and quantization, which are
more dangerous than TP was — a bad `--gpu-memory-utilization` on unified memory takes the
desktop down, not just the job (§5).

**4. One session at a time, and something that reclaims it.** The source project enforces one
session per user and runs an idle reaper because idle GPUs are billed. *On Spark:* nothing is
billed, but 128 GB is shared with the operating system and with every other thing on the box.
The constraint is harder, not softer: **one resident model, and an idle unload.** Two
`vllm serve` processes each claiming 60 % of unified memory is the Spark's version of the
floor-billing bug — cheap to write, expensive to hit.

**5. Model choice is a measurement, not an opinion.** Every model in the source registry was
chosen against a frozen 60-problem LiveCodeBench subset with identical harness, decode
settings, and serve environment; verdicts are recorded with job IDs and pass@1 deltas. This
is why the registry commentary can say "50.00 % vs 26.67 %, +23.33 pts, score job 53531932"
instead of "seemed better". *On Spark:* keep the frozen-subset discipline and keep recording
the provenance. The Spark adds a second axis the cluster never had — a model can win on
quality and lose on tokens/second — so a verdict is now a **pair** (quality, decode rate),
and both must be measured on this box. Cluster numbers do not transfer.

**6. Nothing leaves the machine, and that is the point.** The service exists because prompts,
code, and unpublished data must not go to a commercial provider. External tools (web search,
paper search) are opt-in, off by default, and documented as an explicit tradeoff; client
telemetry is disabled by name in each client's configuration. *On Spark:* this gets stronger
— the box is physically yours and has no shared filesystem. Carry the posture verbatim,
including the client-telemetry list, and carry the honesty about the opt-in exceptions.

**7. Evidence outlives the weights.** `_scratch/` is gitignored and holds the only copy of
measured results. The rule: commit evidence out of the gitignored directory before deleting
any model weights it describes. Weights are re-downloadable; a scored generation set at a
frozen subset SHA is not. *On Spark:* the same rule with sharper teeth — a 4 TB local NVMe
fills faster than a project filesystem, so weight deletion will happen sooner and more often.

---

## 4. What is NOT carried over

State this explicitly in the new `CLAUDE.md`, because a reader who knows the source project
will otherwise look for it:

- **Slurm, partitions, accounts, QOS, and the entire GPU allowlist.** The allowlist rule
  exists because the cluster has hardware owned by other research groups. A Spark has one
  owner and one GPU. Do not port the allowlist. **Do port the *shape* of the rule** — a short
  table of hard constraints, each with the reason and the incident behind it — because that
  shape is what makes `CLAUDE.md` work as a rules file.
- **Service-Unit billing, the rate table, the reservation floor, the billing sweep, and the
  central ledger.** No chargeback on a desk box. Two fragments are worth keeping in a
  different form: **per-request usage logging** (now for capacity planning and for answering
  "why is it slow", not for charging) and the **floor concept**, which becomes "a load that
  fails still cost you the unload/reload cycle and the wall time".
- **Environment Modules and the modulefile deployment.** Replace with a CLI on `PATH` plus
  systemd user units. Keep the principle that made the modulefile worth it: users never type
  an install path, and the deployed copy cannot drift from the source copy in the repo.
- **The air-gapped-compute-node assumption.** Every awkward staging script in the source repo
  exists because cluster compute nodes have no internet. The Spark has internet. Model
  staging becomes a plain download and the staging sbatch files have no analogue.
- **Multi-tenancy machinery** (per-user state dirs, one-session-per-user guards, the
  shared-install non-owner write paths). Reduce to whatever the actual user count is — ask
  (§9). Do not carry per-user isolation the box does not need.
- **`AGENTS.md`.** In the source repo this is *not* a rules file; it is a prompt-level
  workaround for a model that could not emit `<tool_call>` tokens. If the Spark serves models
  with working tool parsers, that file should not exist, and the new `CLAUDE.md` should say
  so — an inherited `AGENTS.md` is actively counterproductive with models that parse tool
  calls correctly.

---

## 5. Verify the hardware before you write a single number

**Do not write any number into the new `CLAUDE.md` that you have not read off the box.** The
table below is the assumed specification; treat every row as a hypothesis with a verification
command next to it. If you cannot run these — you are not on the Spark yet — then write the
file with the numbers marked `UNVERIFIED` and a verification block at the top, and say so in
your summary. Do not silently promote an assumption to a fact.

| assumed | verify with |
|---|---|
| GB10 Grace-Blackwell, one GPU, compute capability sm_121 | `nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv` |
| 128 GB LPDDR5X unified, coherent CPU/GPU, **shared with the OS** | `free -g`, `nvidia-smi -q -d MEMORY` — note whether they report the same pool |
| ~273 GB/s memory bandwidth | a bandwidth microbenchmark; or derive it from a measured decode rate on a known-size model and record it as derived |
| 20-core Arm CPU, **aarch64** | `uname -m`, `lscpu` |
| DGX OS (Ubuntu-derived), NVIDIA container toolkit present | `cat /etc/os-release`, `docker info \| grep -i runtime`, `nvidia-smi` driver + CUDA version |
| ~4 TB NVMe | `df -h`, and decide the model-weights budget from what is actually free |
| dual QSFP ConnectX-7, for pairing two Sparks | `ip link`, `ibstat` if present — only relevant if a second box exists (§9) |

### The three consequences that reshape every other decision

**(a) Unified memory is not VRAM.** `--gpu-memory-utilization` is a fraction of a pool the
operating system, the desktop session, and your own Python process are also using. On a
discrete-GPU cluster node, over-committing kills the job. Here it can kill the box. Establish
the real usable budget by measurement, write it into `CLAUDE.md` as a hard number, and prefer
an explicit KV-cache size over a utilization fraction if the vLLM version supports it. This
is the single most likely way to make the Spark unusable, so it is the rule that goes first
in the new file — the structural slot the GPU allowlist occupies in the source `CLAUDE.md`.

**(b) Decode is bandwidth-bound, and the ceiling is low.** One GB10 has roughly an order of
magnitude less memory bandwidth than an H100. Decode throughput is capped by how many bytes
must be read per token:

```
tok/s_ceiling  ≈  memory_bandwidth / bytes_read_per_token
bytes_read_per_token  ≈  active_parameter_bytes  +  KV_bytes_read
```

At ~273 GB/s, a 27B **dense** model in BF16 (~55 GB of weights) cannot exceed ~5 tok/s no
matter what else is true. The same model at 4-bit (~14 GB) tops out near ~19 tok/s. A
**Mixture-of-Experts** model with ~3B active parameters at 4-bit reads under 2 GB per token
and has a ceiling an order of magnitude higher. Realized rates are typically 40–70 % of the
ceiling. **Verify these arithmetic examples against measurement before publishing them.**

The design consequence is large and must be stated plainly in the new file: **on the Spark,
prefer MoE architectures with small active-parameter counts, and prefer quantized weights,
even at some quality cost.** This *reverses* a source-project conclusion — the cluster
measured a 30B-A3B MoE as a NO-GO on quality and picked dense models, which was correct on
A100/H100 where dense 27B decodes fine. That verdict does not transfer. Re-measure, and
record that you are knowingly overturning it and why.

**(c) Blackwell changes the quantization answer.** The source project recorded a clean NO-GO
for an NVFP4 checkpoint because the FP4→Marlin repack failed against the cluster's driver 535
on Ampere. sm_121 has native FP4 and FP8 support, so **that NO-GO does not apply here and the
result should be revisited first, not inherited.** Conversely, do not assume the Ampere-era
kernels the source project relies on (AWQ/GPTQ Marlin paths) have sm_121 support in whatever
vLLM version you install — verify per-quantization-format and record a table of what actually
loads. Add to the same table what wheels exist for **aarch64**: several packages in the source
environment (FlashInfer, and anything needing a compiled extension) may have no Arm build,
which is the main argument for a container-first install.

---

## 6. Rules the new `CLAUDE.md` must contain

Each of these is a hard rule with a stated reason, in the style of the source file's hardware
section. Fill in the real numbers from §5.

1. **Memory budget.** The maximum unified-memory fraction any serve may claim, the absolute
   floor to leave for the OS, and the requirement to measure resident size before adding a
   model to the served set. One resident model at a time; a second serve must unload the
   first, not race it.
2. **TP=1.** One GPU means no tensor parallelism inside the box. Any `--tensor-parallel-size`
   above 1 is a bug unless a second Spark is paired over ConnectX-7 (§9), in which case say
   what was actually verified and nothing more.
3. **Container-first install, with a pin.** Name the exact image or wheel set and the CUDA and
   driver versions it was verified against, the way the source project pins its conda env and
   states that it "runs against driver 535 through CUDA minor-version compatibility". State
   the rule that stopped a recurring failure there: *do not `pip install -r requirements.txt`
   into the serving environment* — it pulls a torch/vLLM pair the driver cannot run.
4. **Quantization-first model policy**, with the per-format support table from §5(c), and the
   explicit note that the cluster's FP4 NO-GO is void here.
5. **Fences for benchmark and smoke runs.** The source project's fences (`mrefresh-nest*` job
   names, `bench-` served-model names) exist so a test run cannot be mistaken for production
   by the billing sweep or by service discovery. There is no billing here, but the underlying
   hazard — a benchmark process being picked up as the live backend, or evicting the resident
   model mid-conversation — is real. Define the Spark's equivalent: a distinct port range and
   a `bench-` served-name prefix that the gateway will refuse to route to.
6. **Evidence rule**, per §3.7, naming the actual gitignored path and the manifest file.
7. **Thermal and power honesty.** A desk box under sustained load throttles. If sustained
   throughput differs from burst throughput, that is a number users need, and it belongs in
   the file next to the decode rates.
8. **After any subagent or workflow run, check for orphaned processes.** The source rule is
   `squeue -u $USER`; here it is stray `vllm serve` processes and containers holding unified
   memory. Same failure mode, and on one box it blocks everything.

---

## 7. Mistakes already paid for — carry them forward

Each of these cost the source project a debugging cycle or a wasted allocation. Every one has
a Spark analogue; state them in the new file so they are not re-discovered.

| source incident | Spark form |
|---|---|
| Serve-time caches filled a 35 GB home quota and killed a load mid-profiling | point the vLLM compile cache, Triton cache, Torch/Inductor caches, HF cache, and FlashInfer workspace at explicit paths on the NVMe. Name them. Several of these do **not** honour `XDG_CACHE_HOME` |
| Triton cache in a per-job temp dir meant recompiling every kernel; on an unfamiliar compute capability that blew the 600 s engine-ready timeout | sm_121 is exactly such an unfamiliar architecture. Make the Triton cache **persistent**, and raise `VLLM_ENGINE_READY_TIMEOUT_S` for first load per model. Expect the first load of any model to be dramatically slower than the second, and say so |
| A concurrent job's cleanup deleted a running job's `TMPDIR`, killing FlashAttention after 27 of 60 problems and silently scoring 33 failures as wrong answers | namespace scratch per process, and make partial-result files distinguishable from complete ones. A truncated benchmark that scores as a bad model is worse than a crash |
| The `hermes` tool-call parser silently fails on some models; the launcher routes `--tool-call-parser` and `--reasoning-parser` by model key | keep per-model parser routing in the registry. Silent tool-call failure is the most expensive bug in this stack because it looks like the model being stupid |
| Enabling one model's "thinking" cost 5–12× tokens and wall time with no measured quality gain | on a bandwidth-starved box this is a 5–12× *latency* multiplier. Thinking off by default, opt-in per request, with the measured multiplier recorded |
| `--enforce-eager` was a safety default; recovering CUDA graphs later gave 3.9× decode | do not leave `--enforce-eager` on as a permanent default. Record it as a temporary workaround with the condition for removing it, and re-test on every version bump |
| One model ships an unused MTP draft head worth +53 % decode | **on the Spark this is not a footnote, it is a headline.** Speculative decoding and draft heads trade spare compute for memory bandwidth, which is precisely the trade this box wants. Any model with a draft head, and speculative decoding generally, should be evaluated early rather than left unused |
| Slurm reported 2 GPUs while the cgroup exposed 1; vLLM died after the reservation was spent | the analogue is a preflight that asserts the box's actual free unified memory before loading weights, and fails fast with a legible message instead of OOMing the desktop |
| A benchmark comparison between two models was not like-for-like (thinking disabled suited one model's default and suppressed the other's) | whenever two numbers appear together, the conditions under which they were measured appear too. Carry this as a documentation rule |

---

## 8. Required outline for the new `CLAUDE.md`

Mirror the source file's structure — hard rules first, reasons attached, fences second,
evidence third, related-files last. Suggested sections:

```
# Project rules
## Hardware reality            <- §5: verified specs, the memory budget, the bandwidth ceiling
### What this rules out        <- the analogue of "everything else is off limits"
## Install and pinning         <- container/wheel pin, CUDA/driver, aarch64 constraints
## Model policy                <- registry + served set, quantization table, sizing arithmetic
## Serving fences              <- port ranges, bench- prefix, one-resident-model rule
## Clients                     <- Open WebUI, aider/Continue/opencode, what each needs from the server
## Inference as a tool         <- offline batch, structured output, embeddings, MCP
## Evidence                    <- the gitignored path and the commit-before-delete rule
## Related                     <- what AGENTS.md is and is not; where user docs live
```

The **Inference as a tool** section deserves more than the source project gives it, because
it is what a dedicated box is for: vLLM's offline `LLM()` API for batch jobs that never need
a server, guided/structured decoding for extraction pipelines, an embeddings endpoint, and
MCP tools exposing the box's own state (resident model, free memory, thermals) to a coding
agent — the same read-only, whitelisted-argv, no-shell security model as the source project's
Slurm MCP server. Say which of these were actually verified and which are aspirational.

---

## 9. Ask the user before assuming

These change the design materially and none of them is inferable from the source project:

1. **How many people use the box, and how do they reach it?** One person at a desk, or a
   small group over the network? This decides whether the gateway needs per-user keys, whether
   Open WebUI keeps `WEBUI_AUTH=False`, and whether usage logging is per-user.
2. **Remote access posture** — SSH tunnel (as on the cluster), Tailscale/VPN, or LAN-bound?
   The source project binds everything to `127.0.0.1` and tunnels; a desk box on a LAN invites
   a looser default that should be a deliberate decision, not a drift.
3. **Is there a second Spark to pair?** Two boxes over ConnectX-7 change the model-size
   ceiling. Without a confirmed second box, write TP=1 as an unconditional rule.
4. **Should the Spark repo reuse this project's code, fork it, or start clean?** The gateway,
   metering, registry, and MCP servers are portable Python; the launcher and billing are not.
   A fork inherits the Slurm assumptions silently — that is the risk to name.
5. **Which models matter most** — coding, general chat, or batch inference? At 128 GB you
   are choosing, not collecting.

Do the parts that do not depend on the answers first. Ask when you reach the parts that do.

---

## 10. Definition of done

- `CLAUDE.md` exists in the new repo, is under ~200 lines, and every hard rule carries the
  reason it exists.
- Every number in it is either measured on the Spark with the command that produced it, or
  explicitly marked `UNVERIFIED` with the verification command next to it. No number is
  inherited from the cluster without being re-measured or flagged.
- Reading it, someone who has never seen the source project can answer: what fits in memory,
  what parallelism is legal, how to install vLLM reproducibly, what a benchmark run must be
  named so it cannot be mistaken for production, and where the evidence lives.
- The three reversals from the cluster are stated as reversals, not silently changed: MoE and
  quantization are now preferred over dense BF16; FP4 is now viable; there is no billing.
- Nothing in it depends on Slurm, an account, a partition, or a QOS.

## House style

Scientist-to-scientist prose. No checkmarks, no emoji, no marketing language. Numbered steps,
exact commands, tables, and measured numbers with their provenance. Prefer deleting a
paragraph to hedging it. State a constraint once, with its reason, and do not repeat it.
