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

Read this document end to end before writing anything. **The Spark cannot reach that
filesystem**, so this file is self-contained: §3 and §7 carry the design and the incidents,
and Appendix A carries the concrete artifacts — the serve invocation, the registry shape, the
gateway contract, and the client configurations — that you would otherwise have read out of
the repo.

---

## 0. Execution contract

**Produce exactly one file: `CLAUDE.md`, at the root of the new Spark repository.** Nothing
else is in scope. Specifically:

| you MUST | you MUST NOT |
|---|---|
| run the read-only verification commands in §5 and record their real output | install vLLM, Docker images, or any package |
| look up the two models' checkpoint metadata (§9.5) to size them | download model weights |
| write the file, and report which numbers are measured vs `UNVERIFIED` | start `vllm serve`, Open WebUI, or any long-running process |
| ask the user at the STOP conditions below | write code, launchers, or install scripts |
| | commit or push anything unless the user asks |

You are writing the rules that govern the install, not performing it. If a step tempts you to
"just check whether it works" by loading a model, that is out of scope — write the check into
`CLAUDE.md` as a rule instead.

### Order of work

1. Read this document end to end.
2. Run §5's verification commands. Record actual output; do not paraphrase it.
3. Resolve the model identities in §9.5 — exact repository id, parameter count, dtype, and
   whether a quantized checkpoint exists. **STOP condition 1** applies here.
4. Compute the memory budget (§5a) and the decode ceilings (§5b) from *your* measured
   numbers, not from this document's examples.
5. Draft `CLAUDE.md` against the outline in §8, incorporating §6's rules, §7's incidents, and
   §9's decisions.
6. Self-check against §10 and report.

### STOP conditions — ask the user, do not guess

1. **The measured decode rate is unusable, or a checkpoint does not fit.** Both models are
   pinned (§9.5, A8) — do not substitute a different one. But if BF16 measures unusably slow
   and no quantized checkpoint of that model loads on sm_121, stop and report the arithmetic
   and what you tried. Swapping in a model the user did not choose is not your call.
2. **§5's measured hardware contradicts §5's assumed table** — different memory size,
   different compute capability, no container runtime. The design consequences in §5a–c are
   derived from those specs; if they are wrong, the rules built on them are wrong.
3. **You are not on the Spark.** Then no number can be measured. Write the file with every
   number marked `UNVERIFIED`, list the commands that would settle each, and say so in your
   summary. Do not silently promote an assumption to a fact.
4. **You cannot reach the source repository** and something in Appendix A is ambiguous enough
   that guessing would change a rule. Ask rather than invent.

---

## 1. You do not have the source project

`/project/rcc/mehta5/vllm` lives on the RCC cluster filesystem and is not mounted on the
Spark. Do not write paths into the new `CLAUDE.md` that point at it, and do not plan any step
that involves reading it.

If the box has internet and you have credentials for it, the repository is on GitHub as
`rmehta1987/rcc-vllm`, branch `milestone/model-refresh` — clone it if you can, since the file
commentary is richer than any summary. **Ask the user before assuming it is reachable or
public.** If you cannot clone it, proceed without it: nothing below depends on having it.

What the source project is, in one paragraph: users load a module, type `ai-session chat`,
`code`, or `fast`, and a preset submits a Slurm job that runs `vllm serve` on cluster GPUs.
A long-lived reverse proxy on the login node gives clients one fixed URL regardless of which
node the model landed on, captures per-request token usage, and exposes a keyless `/status`.
Clients are Open WebUI in a browser, and aider / Continue / opencode for coding, each
configured from three environment variables the CLI writes to `~/.ai-session/env`. Usage is
charged in Service Units. Model choices are made by measurement against a frozen benchmark
subset and recorded in the registry with their job IDs.

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

**1. A model registry with a served/not-served distinction (shape in A2).** `MODEL_REGISTRY` maps a stable
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
Spark:* preserve the **principle**, not the preset list — there are two models now (§9.5) and
one GPU, so a three-preset menu over two models would be theatre. Define presets that name a
real distinction, and if a preset would only ever pick the same model as another, do not
create it. Parallelism disappears as a knob entirely (TP=1). The knobs that remain are memory
budget and quantization, which are more dangerous than TP ever was — a bad
`--gpu-memory-utilization` on unified memory takes the desktop down, not just the job (§5a).

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
- **Cluster-scale multi-tenancy machinery** (shared-install non-owner write paths, per-user
  state dirs sized for a whole campus). Three users (§9.1) need far less than that — but they
  need more than nothing: authenticated browser chat, per-user usage records, and a model-swap
  lock. Reduce, do not delete.
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

The rule that follows from this, and that must appear in the new file: **what governs
interactive speed here is bytes read per token, not parameter count.** Two consequences, and
they are not equally available to you:

- **Quantization is mandatory.** Both chosen models are dense (§9.5), so 4-bit weights are the
  only lever that moves a 4–5 tok/s ceiling into usable range. This is a hard rule, not a
  preference.
- **MoE would be the bigger lever, but it is not on the menu.** A ~3B-active MoE at 4-bit
  reads under 2 GB per token — an order of magnitude better ceiling. The source project
  measured a 30B-A3B MoE as a quality NO-GO and chose dense models, which was correct on
  A100/H100 where dense 27B decodes fine and is *not* obviously correct here. The user has
  chosen two dense models (§9.5); do not overturn that. Record the tradeoff in `CLAUDE.md` as
  a known, deliberate one, so that if the measured dense rate proves unusable the alternative
  is already written down with its reasoning.

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

1. **Memory budget, stated in gigabytes.** An absolute ceiling for a serve and an absolute
   floor reserved for the OS — not a utilization fraction, which is what makes this dangerous
   on unified memory (§5a). Give the fraction only as a derived convenience, if at all.
   Measuring a model's resident size is a precondition of adding it to the served set.
2. **TP=1, unconditionally.** One GPU, one box, no second Spark (§9.3). Any
   `--tensor-parallel-size` above 1 is a bug. Do not document the two-box case as a
   possibility.
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
8. **One resident model, and swapping it takes a lock.** A second serve must unload the
   first, never race it. With three users (§9.1) a swap is an operation that affects other
   people: it requires no in-flight requests, it takes an explicit lock, and it is visible.
   The CLI must be able to answer "what is resident and who is using it".
9. **Users and access.** Three named users; every listener bound to `127.0.0.1`; browser
   access over SSH tunnel only; authenticated Open WebUI. Write the tunnel command into the
   file. No LAN bind, no TLS, no auth proxy — the tunnel is the security boundary (§9.2).
10. **After any agent, subagent, or background run, check for orphaned processes.** The source
   rule is `squeue -u $USER`; here it is stray `vllm serve` processes and containers holding
   unified memory. Same failure mode, and on one box it blocks everyone.

---

## 7. Mistakes already paid for — carry them forward

Each of these cost the source project a debugging cycle or a wasted allocation. Every one has
a Spark analogue; state them in the new file so they are not re-discovered.

| source incident | Spark form |
|---|---|
| Serve-time caches filled a 35 GB home quota and killed a load mid-profiling | point the vLLM compile cache, Triton cache, Torch/Inductor caches, HF cache, and FlashInfer workspace at explicit paths on the NVMe. Name them. Several of these do **not** honour `XDG_CACHE_HOME` |
| Triton cache in a per-job temp dir meant recompiling every kernel; on an unfamiliar compute capability that blew the 600 s engine-ready timeout | sm_121 is exactly such an unfamiliar architecture. Make the Triton cache **persistent**, and raise `VLLM_ENGINE_READY_TIMEOUT_S` for first load per model. Expect the first load of any model to be dramatically slower than the second, and say so |
| A concurrent job's cleanup deleted a running job's `TMPDIR`, killing FlashAttention after 27 of 60 problems and silently scoring 33 failures as wrong answers | namespace scratch per process, and make partial-result files distinguishable from complete ones. A truncated benchmark that scores as a bad model is worse than a crash |
| The `hermes` tool-call parser silently fails on some models; the launcher routes `--tool-call-parser` and `--reasoning-parser` by model key | keep per-model parser routing in the registry (A6). Silent tool-call failure is the most expensive bug in this stack because it looks like the model being stupid |
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
## Serving fences              <- port ranges, bench- prefix, one-resident-model rule, swap lock
## Users and access            <- the three users, 127.0.0.1 binding, SSH tunnel, WebUI auth
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
Slurm MCP server (A7). Say which of these were actually verified and which are aspirational.

---

## 9. Decisions the user has already made

These were open questions; they are now answered. Treat them as constraints, not defaults,
and write each into the new `CLAUDE.md` as a rule with its consequence.

**1. Three users, internal testing.** Not one person at a desk and not a service. The
consequences are specific:

- One resident model and three people means the box **serializes**. Two users wanting
  different models cannot both be served; a model swap must therefore require no in-flight
  requests, take a lock, and be visible to the other two. Design the swap as an explicit,
  announced operation, not something a client triggers implicitly.
- Continuous batching holds aggregate throughput up under three concurrent users, but
  **per-user decode rate divides**. On a box whose decode ceiling is already low (§5b),
  three simultaneous coding sessions will feel materially slower than one. Measure the
  three-user case, not just the single-user case, and publish both numbers.
- `WEBUI_AUTH=False` (A5) is a single-user demo posture and is now wrong: three people
  sharing one unauthenticated Open WebUI share one chat history. This is the same class of
  bug the source project already fixed once by moving the chat database out of a
  group-readable tree. **Recommended: one Open WebUI instance with `WEBUI_AUTH=True` and
  three accounts**, rather than three instances — each instance holds its own embedding model
  and unified memory is the scarce resource here. Record whichever you choose and why.
- Usage logging stays, per user, for capacity and for answering "why was it slow at 3pm" —
  not for chargeback. Three users is enough that "who is holding the model" is a real
  question the CLI should answer.

**2. SSH tunnel for the web GUI.** Same posture as the cluster: bind every listener to
`127.0.0.1`, no LAN bind, no VPN dependency, and document the exact tunnel command in the
user-facing docs the way the source project does. This is the *simplifying* answer — it means
no TLS, no auth proxy, and no exposure surface beyond SSH itself, which is already how these
three users reach the box.

**3. No second Spark.** TP=1 is unconditional. Delete the ConnectX-7 branch from the rules
entirely rather than leaving it as a documented possibility — an aspirational escape hatch in
a rules file is how `--tensor-parallel-size 2` eventually gets typed. Model sizing is bounded
by one box's 128 GB, full stop.

**4. Clean repo.** No code is ported from the source project. Appendix A is reference
material for *what the pieces do and why*, not source to copy. This is the right call and it
has a cost worth naming in the new `CLAUDE.md`: the gateway, the registry, the metering, and
the MCP servers get rewritten, and each rewrite is an opportunity to lose a fix that is only
recorded as a comment in Appendix A. Work through A1's environment list and A6's parser
routing deliberately rather than rediscovering them.

**5. Two models, pinned: `qwen3.8_27B` and `gemma4_31B`.** Qwen3.8-27B (dense 27.8B) and
Gemma-4-31B-it (dense 30.7B), covering inference, chat, and coding between them. No separate
small/fast preset and no general-chat 72B. That is a sound choice for a box that holds one
model at a time: fewer swaps, and both are strong enough to be the only thing resident.
Full operational detail for both — parsers, thinking behaviour, measured scores, serve flags
— is in **A8**, which you should treat as the starting registry entry for each.

Three consequences to write into the new `CLAUDE.md` as hard constraints:

- **Both are dense, so quantization is mandatory rather than preferred.** §5b's arithmetic
  applies directly: a dense 27–31B in BF16 is 55–61 GB read per token, capping decode near
  4–5 tok/s at ~273 GB/s — resident, but practically unusable for interactive coding. The
  same models at 4-bit read ~14–16 GB and raise the ceiling to roughly 17–19 tok/s. The
  architectural lever (§5b) is unavailable, so the whole margin has to come from
  quantization, speculative decoding, and prefix caching. Measure, and if 4-bit still does
  not clear a usable interactive rate, report that as a finding (STOP 1).
- **Neither model has ever been served on one GPU.** Every cluster job for both ran TP=2
  across two 40–80 GB cards (A8). TP=1 is now mandatory (§9.3), which means single-device
  residency for a 55–61 GB BF16 checkpoint is *unproven*, not merely untested — and it is the
  first thing to establish, before any quality work.
- **The quality numbers transfer; the speed numbers do not.** Both models were scored on the
  same frozen 60-problem LiveCodeBench subset under one harness (A8). Carry those scores into
  the registry as priors with their caveat, and re-measure only decode rate on the Spark.
  Re-running the benchmark is not required to start; re-running it after quantization is,
  because that is the one number quantization can silently destroy.

One concrete lead: the source project staged **`Gemma-4-31B-it-qat-w4a16`** — a
quantization-aware-trained 4-bit checkpoint, ~22 GB on disk, which passed a serve-and-accept
check on a single 48 GB A40. That is the only evidence on record of either model serving on
one card. It was never advertised to cluster users because its coding quality is unmeasured
and Google publishes no quantized-vs-BF16 comparison. For the Spark it is the most promising
starting point available: QAT 4-bit is exactly the shape this box wants, and the quality gap
the cluster never needed to measure is now worth measuring. Treat that gap as an open task,
not as a reason to skip the checkpoint.

---

## 10. Definition of done — self-check before reporting

Run these against the file you wrote. Each is checkable; do not report done on a failing line.

```bash
# 1. Nothing cluster-shaped survived.
grep -niE 'slurm|sbatch|squeue|sacct|partition|--qos|service unit|\bSU\b|modulefile|beagle3|midway3' CLAUDE.md
# expect: no hits

# 2. No parallelism above 1, and no two-box escape hatch.
grep -niE 'tensor-parallel-size [2-9]|connectx|second spark' CLAUDE.md
# expect: no hits

# 3. No path into the unreachable source project.
grep -n '/project/rcc' CLAUDE.md
# expect: no hits

# 4. Every unverified number is labelled as one.
grep -nc 'UNVERIFIED' CLAUDE.md
# expect: 0 if you measured everything on the box; otherwise one per unmeasured number
```

Then read it as the agent who will be governed by it, and confirm each of these:

- It is under ~200 lines and every hard rule carries the reason it exists.
- Each number is either measured, with the command that produced it, or marked `UNVERIFIED`
  with the command that would settle it. No number is inherited from the cluster unmarked.
- The memory budget is in gigabytes and appears before any other rule.
- A reader who has never seen the source project can answer: what fits in memory, why TP is
  fixed at 1, how to install vLLM reproducibly, what a benchmark run must be named so it
  cannot be mistaken for production, who may reach the box and how, and where evidence lives.
- The three reversals from the cluster are stated as reversals, not silently applied: FP4 is
  viable on sm_121; there is no billing; quantization is mandatory rather than optional.
- §9's five decisions each appear as a rule with its consequence, not as background.

Report to the user: the file's location, which §5 numbers you measured versus assumed, any
STOP condition you hit, and anything in §9.5 you could not pin down.

## House style

Scientist-to-scientist prose. No checkmarks, no emoji, no marketing language. Numbered steps,
exact commands, tables, and measured numbers with their provenance. Prefer deleting a
paragraph to hedging it. State a constraint once, with its reason, and do not repeat it.

---

## Appendix A — the artifacts, extracted

Everything here is copied from the source project so you do not need it mounted. It is
reference material, not a template to port: read what each piece is *for*, then decide what
the Spark version should be.

### A1. The canonical serve invocation

```
vllm serve $MODEL_PATH \
  --served-model-name $MODEL_KEY \
  --host 0.0.0.0 --port $PORT \
  --tensor-parallel-size $TP \
  --trust-remote-code \
  --enable-prefix-caching \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEM_UTIL \
  $AGENT_FLAGS $REASONING_FLAG $EAGER_FLAG $COMPILATION_FLAG
```

`--enable-prefix-caching` is always on and matters more on the Spark than it did on the
cluster: coding clients resend a large, mostly-identical prompt every turn, and prefill is
the compute-bound half of a workload whose decode half is already bandwidth-starved.

The serve argv is built **once** into an array and reused by the serve, smoke, and benchmark
paths, specifically so those three cannot drift apart and produce numbers measured under
different conditions. Keep that property.

Environment that had to be set around the serve command, each line a fixed bug:

```
VLLM_CACHE_ROOT       compile cache -> large, must not sit on a quota'd home
TRITON_CACHE_DIR      MUST persist across runs (see §7) -- not a temp dir
TORCHINDUCTOR_CACHE_DIR, TORCH_EXTENSIONS_DIR, TORCH_HOME
XDG_CACHE_HOME
FLASHINFER_WORKSPACE_BASE   FlashInfer ignores XDG and defaults to $HOME
HF_HOME               model/tokenizer cache
VLLM_ENGINE_READY_TIMEOUT_S=2400   default 600 is not enough for a cold Triton cache
                                   on a compute capability with no prebuilt kernels
```

### A2. The registry shape

```python
MODELS_ROOT = "..."

MODEL_REGISTRY = {                       # key -> path. The key is the stable identifier
    "qwen2.5_72B": f"{MODELS_ROOT}/...", # used in the served-model-name, the rate table,
    "qwen3.8_27B": f"{MODELS_ROOT}/...", # the usage logs, and the client configs.
    "qwen3.5_122B": f"{MODELS_ROOT}/...",  # registered but NOT served -- reason in a comment
    "qwen2.5_0.5B": f"{MODELS_ROOT}/...",  # smoke only, never a benchmark reference
}

PHASE1_SERVED = {"qwen2.5_72B", "qwen3.8_27B", "gemma4_31B", "qwen3_4b"}
```

The value is in the comments, not the code. A real entry's commentary records: the measured
score that justified it (`50.00% (30/60) vs 26.67% (16/60), +23.33 pts, score job 53531932`),
the exact hardware and parallelism it was proven on, which tool-call parser works and which
one *silently* fails on it, whether thinking is on by default and what that costs, and what
measurement is still owed. Anything registered-but-unserved says why in the same place —
"FP8 weights, needs Hopper", "the production serve path cannot load this architecture" — so
nobody re-litigates a settled question. Write the Spark's registry the same way, with the
reasons being memory footprint and measured decode rate.

### A3. The gateway contract

One fixed URL in front of a backend that moves. The current backend is written to
`logs/gateway/upstream.json` when a session starts and cleared when it ends; the proxy
re-reads that file with an mtime cache, so it needs no restart to follow a new backend.

- Proxies verbatim: `/v1`, `/metrics`, `/health`, `/version`, `/ping`, `/tokenize`,
  `/detokenize`, `/pooling`. Drops hop-by-hop and length headers in both directions.
- Captures every response's `usage` block to `logs/gateway/usage-YYYYMMDD.jsonl`. This is the
  authoritative usage record — users never log their own tokens.
- `GET /status` is **keyless** and TTL-cached, returning ready / loading / no_backend, so a
  status check cannot fan out into a storm of backend health probes.
- Per-client token-bucket rate limit (default 30 rps, `429` with `Retry-After`) and a request
  body cap (default 16 MB, `413`), both disableable by setting their env var to `<= 0`.
- Optional API key; clients send it as the OpenAI API key.

On the Spark the moving part is the resident model rather than the address, but the usage
capture, the keyless `/status`, and the body cap all keep their value. The `/status` verb is
what the CLI shows users instead of raw scheduler output.

### A4. The user-facing CLI surface

```
ai-session chat      general chat in the browser
ai-session code      coding session for aider / Continue / opencode
ai-session fast      small, cheap, quick to start
ai-session status    ready, still loading, or stopped?
ai-session connect   client settings: URL, access key, per-client commands
ai-session env       prints `export AISESSION_...` lines; use as eval "$(ai-session env)"
ai-session models    list the presets
ai-session receipt   re-print the newest usage receipt
ai-session mcp config | mcp run NAME
ai-session stop      stop, free the GPUs, print the charge
```

This verb list is the source project's, not a specification: `fast` has no model behind it
under the two-model decision (§9.5) and `receipt` is billing, which does not exist here (§4).
Copy the *shape* — short intent verbs, no paths, no flags for the common case.

After a session comes up, the dispatcher writes `~/.ai-session/env`, mode 600, containing
exactly three variables: `AISESSION_BASE_URL`, `AISESSION_API_KEY`, `AISESSION_MODEL`. Every
client and example reads those instead of a hand-edited value, which is why no client config
in the project contains an install path or a hostname. Preserve this — it is the single
highest-leverage piece of UX in the source project and it costs almost nothing.

### A5. Browser chat (Open WebUI)

Runs in its **own** virtual environment, never the serving one, and points at the gateway so
it needs no reconfiguration when the backend changes:

```
ENABLE_OPENAI_API=True
OPENAI_API_BASE_URL=http://localhost:$GW_PORT/v1
OPENAI_API_KEY=<session key>
ENABLE_OLLAMA_API=False
DATA_DIR=$HOME/.ai-session/openwebui-data     # chat history: PRIVATE, mode 700
HF_HOME=<per-user state dir>/hf_cache          # UI writes here; never the shared install
ANONYMIZED_TELEMETRY=False  DO_NOT_TRACK=true  SCARF_NO_ANALYTICS=true
WEBUI_AUTH=False                               # demo posture only -- see §9.1
```

Two details worth carrying: the chat database was originally in a group-readable project
tree, where any other user could read someone's chats — it belongs in a mode-700 directory
under `$HOME`. And the RAG embedding model (`all-MiniLM-L6-v2`) is resolved from a shared
read-only cache with `HF_HUB_OFFLINE=1` when present, because letting every user download it
into an empty cache on first run was slow enough to overrun the UI's bind-wait.

Web search, URL fetch, and academic paper search are **opt-in** behind one environment
variable, default off, because their queries leave the machine. That is stated plainly in the
user docs rather than buried.

### A6. The coding-client contract

- **Per-model tool-call parser routing**, by model key, in the launcher:
  `gemma4* -> gemma4`, `qwen3.5*|qwen3.6*|qwen3.8* -> qwen3_coder`, older Qwen -> `hermes`.
  Tool calling is only enabled in agent mode: `--enable-auto-tool-choice --tool-call-parser X`.
  The wrong parser fails *silently* — the model appears stupid rather than broken.
- **Reasoning parser routing** likewise (`qwen3`, `gemma4`), so chain-of-thought comes back in
  a separate `reasoning` field instead of contaminating `content`. With a parser on and a
  tight `max_tokens`, `content` can come back null while the model is still thinking —
  budget tokens generously or disable thinking.
- **aider** needs a litellm metadata file declaring the context window, or it warns
  "Unknown context window size" and mis-sizes prompts. Keys must be duplicated with and
  without the `openai/` prefix. Costs are zero there because billing happens elsewhere; on
  the Spark they are genuinely zero. Split the window so prompt + `max_tokens` cannot exceed
  `--max-model-len` (28000 in / 4096 out against a 32768 window).
- **opencode** needs a project-local config declaring a custom provider:

```json
{
  "model": "rcc/<model_key>",
  "share": "disabled",
  "autoupdate": false,
  "provider": { "rcc": {
    "npm": "@ai-sdk/openai-compatible",
    "options": { "baseURL": "{env:AISESSION_BASE_URL}", "apiKey": "{env:AISESSION_API_KEY}" },
    "models": { "<model_key>": { "limit": { "context": 32768, "output": 8192 } } }
  }}
}
```

- Client telemetry is disabled by name in each: aider `--analytics-disable`, Continue
  `allowAnonymousTelemetry: false`, opencode `share: disabled` and `autoupdate: false`. The
  docs state that the coding tool is separate software with its own telemetry, outside the
  service's control — an honest boundary rather than a blanket privacy claim.

### A7. MCP tools, and their security model

Two read-only MCP servers let a coding agent answer "what are my jobs doing?" and "how much
have I used?" without being handed a shell. The security model is enforced in code, not by
convention, and is the part to carry:

- a hard whitelist of binaries, checked on every call;
- every command built as an argv list, `shell=False`, so no argument is ever shell-interpreted;
- arguments constrained by regex (`^[0-9_]+$` for a job id, so `"1; scancel 999"` is refused
  before any process spawns);
- ownership verified before returning anything;
- no mutating verb reachable at all.

They live in a **dedicated virtual environment**, deliberately not the serving one, so the
MCP SDK's dependency tree can never perturb vLLM. The Spark analogue exposes box state —
resident model, free unified memory, temperature, disk — under the same rules.

### A8. The two models, exactly

Everything the source project measured about the two pinned models (§9.5). This is the
starting content for the Spark registry — carry the facts, re-measure the rates.

| | `qwen3.8_27B` | `gemma4_31B` |
|---|---|---|
| weights | Qwen3.8-27B, dense 27.8B, BF16 | Gemma-4-31B-it, dense 30.7B, BF16 |
| architecture | hybrid Gated DeltaNet + attention | dense transformer |
| `--reasoning-parser` | `qwen3` | `gemma4` |
| `--tool-call-parser` | `qwen3_coder` — **`hermes` fails silently on it** | `gemma4` |
| thinking | **ON** by default (`reasoning_effort` xhigh) | **OFF** by default; opt in via `chat_template_kwargs.enable_thinking` |
| cost of thinking | (the default) | measured **5–12× tokens and wall time**, no measurable quality gain |
| draft head | ships an **MTP head worth +53 % decode**, unused on the cluster | none in this checkpoint (verified on disk) |
| CUDA graphs | forced `--enforce-eager` by a crash — a workaround, not a setting | not required |
| proven on | TP=2, H100 / H200 | TP=2 on H100 NVL, A100 80 GB, A100 40 GB, A40 |
| frozen LCB-60 | 50.00 % (30/60) | **66.67 %** (40/60); hard subset 14/30 vs 6/30 |
| quantized checkpoint on record | none | `Gemma-4-31B-it-qat-w4a16`, ~22 GB, served on one A40, quality unmeasured |

**The two scores are not like-for-like.** They were measured with `enable_thinking: false`,
which is Gemma's native default but suppresses Qwen3.8's `xhigh` default — so 50.00 % is a
lower bound for Qwen and the 16.67-point gap is an upper bound on the difference. Wherever
those two numbers appear together, that caveat appears too. The models also differ in
character, not just score: Qwen thinks by default and suits hard problems; Gemma is faster
and cheaper by default. Offering both is the point.

Serve details that carried across every cluster job for both: 32768 context for coding
sessions, `--enable-prefix-caching` on, and an aider metadata split of 28000 in / 4096 out so
prompt plus `max_tokens` cannot exceed the window.

Three things to establish on the Spark before either is declared servable:

1. **Single-device residency at TP=1** — never done for either model (§9.5).
2. **Minimum vLLM version.** Qwen3.8-27B loads as `Qwen3_5ForConditionalGeneration`, an
   architecture vLLM 0.10.2 does **not** have; the cluster needed 0.26.0. Whatever you pin
   must be at least that new, and must have sm_121 kernels. Check both models load before
   pinning the version.
3. **The MTP head.** Unused on the cluster because it was not needed there. On a
   bandwidth-bound box a +53 % decode draft head is one of the few large levers available
   (§7) — evaluate it early rather than treating it as a footnote.

Licensing: verify the `LICENSE` file shipped with each checkpoint rather than trusting a
table. The source project's user docs list one served model with real obligations and treat
the rest as permissive, but that page describes a fleet that no longer exists.
