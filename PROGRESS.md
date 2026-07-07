# vLLM Project — Progress Report

_Last updated: 2026-07-07_

## What this project is

We are standing up a local large-language-model service on the university's
research computing cluster (RCC beagle3). The idea is simple: instead of sending
data and paying a commercial provider, faculty and students run open Qwen models
on the GPUs we already have. A user starts a session, gets a web address that
speaks the same language as the OpenAI API, chats or codes against it, and then
shuts it down. Usage is charged back in a fair, measured way.

Everything lives under `/project/rcc/mehta5/vllm/`.

## Where things stand

The service is built and has been run end-to-end with the real production model.
The core pieces all work. What remains is mostly optional cross-checks and
polish, plus deciding how widely to open it up to other people.

In plain terms:

- A person can start a model session on a GPU, talk to it through a browser or a
  coding tool, and stop it — and the system correctly figures out what to charge.
- This has been done for real with the large 72-billion-parameter model on the
  fastest GPUs available, not just in a dry run.
- The billing math has been tested and the "price list" for several GPU types is
  filled in from actual measurements.

## The main building blocks (all working)

- **The session tool (`ai-session/`).** One set of scripts that launches a model
  on a GPU, hands the user a stable web address, keeps track of the session, and
  tears it down cleanly when they're done. Because sessions come and go, there's a
  small proxy (a "gateway") that gives users one unchanging address even though the
  actual model server behind it is temporary. The gateway also quietly counts how
  many words go in and out, which is what billing is based on.

- **The billing logic (`billing/`).** The rule for what a session costs. The
  short version: you're charged for the GPU time you hold, weighted by how
  expensive that GPU is. Cheaper hardware costs less; more expensive hardware costs
  more. There's a floor so that reserving a whole GPU and leaving it idle still
  costs something. Token-by-token charges are a small top-up on heavy use. The math
  is covered by an automated test suite (22 checks, all passing).

- **The "price list" (`billing/rate_table.json`).** For each combination of model,
  GPU type, and configuration, we ran a real benchmark to measure how fast it
  processes text, and stored that. This is what turns "you used the GPU for 20
  minutes" into an actual charge. It currently holds five measured records.

- **The models on disk (`models/`).** Staged and ready to serve, no downloads
  needed (the compute nodes have no internet): the general 72B chat model, a
  dedicated 32B coding model, a small 4B model, a tiny 0.5B model for quick tests,
  and a Llama 70B model kept for comparison.

## The user-facing ways in (all proven)

- **Browser chat (Open WebUI).** A familiar chat window in the browser. Proven
  working against the real 72B model on the fastest GPUs — a user chatted with it
  live and the session billed correctly on teardown.

- **Coding assistant (aider).** A command-line coding tool that edits files in your
  own repository using the local model. Proven working: it made real code edits
  through the service and billed correctly. The default is now a dedicated 32B
  coding model, which turned out to be both better at code and cheaper to run than
  the general 72B model.

- **One-command launchers.** Single scripts bring the whole stack up and down
  (`run_browser_demo.sh`, `run_coding_agent.sh`), print the exact command to
  connect from a laptop, and show a charge summary when you stop. These are safe
  for any staff member to run, and each person's session is kept separate.

## What we've measured so far

The benchmark runs have filled in the price list for the large 72B model on three
GPU types, plus the smaller models. The headline finding, in plain terms:

- **The value/everyday choice is the A100 GPU.** It has the lowest cost to hold,
  so short and interactive use is cheapest there.
- **For sustained, heavy generation, the H200 GPU is cheaper per word** even though
  it costs more to hold, because it's fast enough to make up the difference.
- **The H100 GPU has no sweet spot for this model** — it's both dear to hold and
  dear per word, so we don't steer people to it.
- **The 32B coding model is cheaper per word than the 72B model on every GPU**, and
  is now the default for coding.

## Recent additions (July 2026)

- **Everything is reached through modules now.** `module load ai-session` gives the
  `ai-session` command (start, status, connect, stop, receipts); `module load
  opencode` gives the opencode coding agent from a shared install. Nobody types a
  project path. The modulefiles live in the repo and are symlinked into the
  deployed location, so source and deployment cannot drift.
- **Bring-your-own fine-tune.** `ai-session <preset> --lora NAME=PATH` serves a
  user's LoRA adapter next to the base model; the adapter is validated before any
  GPU is reserved. A design note (`ai-session/LORA_TRAINING_DESIGN.md`) proposes how
  a supported training path should work; training itself is not offered yet.
- **Agent tools without paths.** `ai-session mcp config` prints the block a coding
  agent needs to use the two built-in read-only tool servers (job queue, usage);
  `ai-session mcp run jobs|usage` starts them. Agent configurations no longer
  contain any install path.
- **A second de-jargoning pass over the user docs**, cutting the internal component
  vocabulary ("gateway", the serving software's name, "OpenAI-compatible") in the
  user-flow pages roughly in half and defining what remains in plain words.

## What's left to do

Nothing here is blocking; the service works today. These are the open items.

Optional measurements (each one uses GPU time, so they're run deliberately, not
automatically):

- Benchmark the cheaper A40 GPU tier, to give people a low-cost option.
- Benchmark the Llama 70B model as a cross-check against Qwen.
- An H100 two-GPU configuration, only relevant on the larger-memory H100 nodes.

Small polish items:

- One benchmark record stores a placeholder port value in its provenance notes;
  cosmetic, worth cleaning up.

Decisions and rollout (the bigger "what next"):

- Decide whether to open this to real users beyond the initial single-user,
  PI-owned-node phase, and what authentication and access controls that needs. The
  gateway was built with this future step in mind.
- Confirm one billing detail: whether the shared test partition hands out whole
  nodes or fractional GPUs, since that changes the GPU count used in every charge.
- Write down and publish the final usage policy for actual users (a draft policy
  document already exists at `ai-session/BILLING_POLICY.md`).

Housekeeping (waiting on a go-ahead, since these weren't created by this work):

- An old container image (`vllm-v0.7.3.sif`) is superseded and can be removed.
- A large Llama checkpoint in its original format (~132 GB) is archival and could
  be pruned to reclaim space.

## Where to read more

- `ai-session/README.md` — how to operate the service.
- `ai-session/CODING_AGENTS.md` — how to connect coding tools.
- `ai_session_notes.md` — plain-language build notes and the measured cost tables.
- `HANDOFF_NEXT.md` — detailed technical status from the last working session.
