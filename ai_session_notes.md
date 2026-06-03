# ai-session on RCC — build notes & cost guide

This is a living document. It records, in plain language, how we built the
**ai-session** service (a private AI chat/coding endpoint running on the
University of Chicago RCC cluster), what we set up, which tests we ran and where
their logs are, and how the cost ("Service Units", or **SU**) is worked out.

A few terms up front, no jargon:

- **AI model** — the program that answers prompts (here, Qwen — an open model we
  run ourselves, not a paid outside service).
- **GPU** — the specialized chip that runs the model. Big models need several at
  once. We have a few kinds (A100, A40, H100, H200), and the fancier ones cost
  more to use.
- **Token** — a chunk of text, roughly ¾ of a word. Models read input tokens and
  write output tokens; we count both.
- **Service Unit (SU)** — the cluster's internal "credit". We define **1 SU = one
  hour of one A100 GPU**. Everything is priced in this single unit.
- **Session** — one period where a user has the model running for them, from
  "start" to "end".

---

## What we used (and what we did *not* install)

We deliberately installed **nothing new** for the service itself. There was
already a working software environment on the cluster
(`/project/rcc/mehta5/conda-envs/vllm-probe`) containing the AI-serving software
(vLLM 0.10.2) proven to run on these GPUs. Re-installing or "upgrading" it is the
one thing that reliably breaks it, so we left it alone and built everything on
top of it.

What we *added* were our own small programs (plain text files of code and
configuration — see "The pieces we built"). They need no special installation;
they run with the existing environment.

For the user-facing chat apps (optional, future): **Open WebUI** (a private
ChatGPT-style web page) and **aider** (a coding assistant) would each be a small
one-time install in their *own* separate environment on a login computer — never
into the AI-serving environment.

---

## Cluster facts

| Item | Value |
|---|---|
| Account | `rcc-staff` |
| Queue we use | `test` |
| Software environment | `/project/rcc/mehta5/conda-envs/vllm-probe` (vLLM 0.10.2) |
| Project folder | `/project/rcc/mehta5/vllm` |
| Models on disk | Qwen3-4B, Qwen2.5-72B, Llama-3.1-70B, Qwen2.5-0.5B (`models/`) |
| GPU naming gotcha | GPU labels are **case-sensitive**. `a100` (lowercase) and `A100` (uppercase) are *different* machines. |
| `a100` (lowercase) | beagle3 + some midway3 nodes — **40 GB** per GPU (confirmed: "A100-PCIE-40GB") |
| `A100` (uppercase) | midway3 nodes with 512 GB system memory — **80 GB** per GPU |
| Why it matters | The 72-billion model needs ~145 GB; it fits comfortably on four 80 GB GPUs, but is too cramped on four 40 GB ones. So we benchmark/serve the big model on the 80 GB (`A100`) machines. |

---

## The pieces we built

In plain terms, the service has four parts:

1. **The price rules** (`billing/`). A short, tested calculator that turns "tokens
   used" and "hours the GPUs were held" into a number of SU, plus the official
   price policy written out for users (`BILLING_POLICY.md`).
2. **The speed test** (`benchmark/`). A measurement job that starts the model,
   pushes a fixed, repeatable workload through it, and records how fast it reads
   and writes tokens. Those measured speeds feed the price rules.
3. **The session manager** (`ai-session/ai_session.py`). One command to **start** a
   private model for a user, **check** on it, print **connect** instructions, and
   **end** it (which works out the bill and frees the GPUs).
4. **The front door** (`ai-session/gateway.py`). A small relay that gives every
   chat app one unchanging web address, even though the actual model moves to a
   different machine each session. It also quietly counts each request's tokens,
   so the bill is exact and the user doesn't have to track anything.

---

## What we tested, and where the logs are

| What we checked | How | Result | Evidence / log |
|---|---|---|---|
| The price math is correct | 22 automated checks of the SU formula (worked examples, the "pay for the GPUs you hold" floor, free failed requests, etc.) | **All 22 pass** | run `pytest billing/` (prints to screen) |
| The front door relays correctly and counts tokens | Simulated a model behind the relay and sent normal and streaming requests | **Pass** (routing, token counting, "no model running" message all correct) | ran in terminal (no saved log) |
| First real speed-test attempt | Submitted the measurement job | **Failed in 3 seconds** — the folder for its log file didn't exist yet | job `50383944` (no log produced — that was the symptom) |
| Speed test works on a real GPU (small model) | Ran the full measurement on **Qwen3-4B**, 1× A100 | **Success, 6 minutes** — speeds recorded, sanity checks passed | `benchmark/logs/bench_billing-50383949.out`, model log `benchmark/results/serve_qwen3_4b_a100_tp1.log`, result saved to `billing/rate_table.json` |
| Speed test on the real production model | Ran the measurement on **Qwen2.5-72B**, 4× 80 GB A100 | **Success, 21 minutes** — read ~2,900 / write ~1,120 tokens/sec; price record saved | job `50384019` on `midway3-0377` → `benchmark/logs/bench_billing-50384019.out` |
| Speed test on the fast GPU tier | Ran the measurement on **Qwen2.5-72B**, 2× H200 | **Success, 11 minutes** — read ~7,600 / write ~2,330 tokens/sec; price record saved | job `50399550` on `midway3-0605` → `benchmark/logs/bench_billing-50399550.out` |
| The whole service, start to finish | Started a private model, sent 9 real requests (chat + streaming) through the front door, then ended it | **Success** — every request's tokens were captured, the bill matched the model's own count exactly, and the machine was released | session `50384737`; front-door log `logs/gateway/usage-20260602.jsonl`; bill `logs/usage/mehta5_50384737_*_summary.json` |

### What the small-model speed test actually found (job 50383949)

- It reads input about **20,000 tokens/second** and writes output about **4,100
  tokens/second** on one A100. Writing is slower than reading (about 4.9× slower)
  — this is expected, and it's exactly why output is priced higher than input.
- A built-in double-check passed: the model's own internal token counter and the
  test's token tally agreed (the tiny difference is the one warm-up request).

---

## How much it costs (SU breakdown)

Two things can be charged, and **you pay whichever is larger**:

1. **Holding the GPUs** ("the floor"). Because a session reserves a whole machine
   that nobody else can use, you pay for the time you hold it, period. This is
   `price-of-the-GPU × number-of-GPUs × hours`.
2. **Tokens used.** Your reading and writing, priced from the measured speeds.

On these reserved machines, **the floor is almost always the whole bill** — the
token charge only pulls ahead if you keep the GPUs busier than our standard test
workload for the entire reservation.

### Table 1 — Cost of *holding* the GPUs (the usual charge)

This is exact and does not depend on how much you use the model.

| Setup | Price × GPUs | SU per hour | SU for a 2-hour session |
|---|---|---:|---:|
| Small model, 1× A100 | 1.0 × 1 | **1.0** | 2.0 |
| 72B model, 4× A100 | 1.0 × 4 | **4.0** | 8.0 |
| 72B model, 4× H100 | 2.0 × 4 | **8.0** | 16.0 |
| 72B model, 2× H200 | 3.0 × 2 | **6.0** | 12.0 |
| 4× A40 | 0.5 × 4 | **2.0** | 4.0 |

> H100 is shown at **4× (TP=4)**, not 2×: the 72B model's ~145 GB of weights do not
> fit two 80 GB H100s (2 × 80 × 0.9 ≈ 144 GB), so it needs four — measured 2026-06-02
> on H100 **NVL** (94 GB) nodes. Holding cost is `w_gpu × N`, so because H100 needs
> **four** GPUs while the roomier H200 needs only **two**, the 4× H100 session is the
> priciest to hold (8.0/hr) even though a single H200 is the "fancier" chip.

A fancier GPU always costs more to hold — an H200 session costs more than the
same A100 session. (This is the fairness fix over the old pricing, where a faster
GPU could look "cheaper".)

### Table 2 — Cost of *tokens* (all measured on real GPUs)

| You use… | Qwen3-4B (1× A100) | Qwen2.5-72B (4× A100) | Qwen2.5-72B (4× H100) | Qwen2.5-72B (2× H200) |
|---|---:|---:|---:|---:|
| 1,000 input tokens | 0.000014 | 0.000383 | 0.000587 | 0.000219 |
| 1,000 output tokens | 0.000067 | 0.000990 | 0.001227 | 0.000716 |
| A typical chat turn (≈500 in / 300 out) | 0.000027 | 0.00049 | 0.00066 | 0.00032 |
| A long document (≈8,000 in / 1,000 out) | 0.00018 | 0.00405 | 0.00592 | 0.00247 |
| One million output tokens | 0.067 | 0.99 | 1.23 | 0.72 |

The 72B model costs about **15× more per output token** than the 4B — it's a far
bigger model running on four GPUs instead of one. Even so, the token cost is tiny
next to the holding cost: a 2-hour 72B session that writes **3 million** tokens
runs about 3.1 SU of tokens versus an **8.0 SU** holding charge — so you still pay
the 8.0 (the floor).

**A100 vs H200 for the same 72B model — a surprising twist.** Per *token*, the
H200 is actually **cheaper** (about 0.72× the A100), because two H200s write text
about twice as fast as four A100s, which more than makes up for the H200 being a
pricier chip to hold. But per *hour held*, the H200 is **more expensive** (6.0
SU/hour vs 4.0). So the rule of thumb: for short, interactive sessions the A100 is
the cheaper choice (you mostly pay the holding floor, which is lower); for long,
flat-out, high-volume work the H200 can come out ahead (you'd pay mostly for
tokens, which are cheaper there). In everyday chat use, the A100 is the value
pick.

**Where does the H100 land? Dead last for this model — on both axes.** Measured on
the cluster's H100 **NVL** (94 GB) nodes, serving 72B needs four of them (TP=4), so a
session both **holds dearest** (8.0 SU/hour — more than the H200's 6.0, because it ties
up four GPUs vs two) **and** costs the **most per token** (0.001227 per 1k out, vs the
A100's 0.000990 and the H200's 0.000716). The H100 NVL's decode speed (≈1,810 tok/s
across four cards) simply isn't fast enough to overcome the 2.0 weight × 4 GPUs. The
upshot: for *this* 72B workload there is no regime where the H100 tier is the
economical pick — the **A100 wins on holding** (interactive) and the **H200 wins on
tokens** (sustained throughput). (This isn't a knock on H100s in general; it's the
specific combination of this model needing 4 cards and the chosen w_gpu weights.)

### Table 3 — Example sessions (small model, one A100; billed = the larger of the two)

| Session | Tokens written | Token SU | Holding SU | **You pay** | Why |
|---|---:|---:|---:|---:|---|
| 30 min of light chatting (~30k out) | 30,000 | 0.002 | 0.50 | **0.50** | holding |
| 2 hours of heavy use (5 million out) | 5,000,000 | 0.34 | 2.00 | **2.00** | holding |
| 1 hour, GPU pinned at 100% (~14.8M out) | 14,800,000 | 1.01 | 1.00 | **1.01** | tokens (just barely) |

**The key idea in one line:** running a GPU flat-out for an hour costs almost
exactly the 1-hour holding price — so the holding charge already *is* the cost of
full use. You only pay extra by being busier than our standard test, which is
hard to do in normal interactive use. **Practical advice: end your session as
soon as you're done**, because the clock — not your typing — is the main cost.

---

## Decisions / changes log

- **2026-06-02 — Built the price rules and proved them.** Wrote the SU calculator
  and the price policy, with 22 automated checks based on the agreed worked
  examples. All pass. Decided one A100-hour = 1 SU, and that you pay the larger of
  "tokens used" or "GPUs held".
- **2026-06-02 — Added the front door (gateway) and connection helper.** So chat
  apps use one stable web address across sessions, and token counts are captured
  automatically for billing. Chose **Open WebUI** for general chat and **aider**
  for coding (both work as-is); coding *agents* like opencode need an extra
  "agent mode" switch we added.
- **2026-06-02 — Discovered the GPU-label case trap.** `a100` (40 GB) and `A100`
  (80 GB) are different machines, and the labels are case-sensitive. The big model
  must use the 80 GB (`A100`) machines.
- **2026-06-02 — First speed-test submission failed in 3 seconds (job 50383944).**
  The cluster couldn't find the folder to write the job's log into. Fix: create
  the `benchmark/logs/` folder before submitting (the script created it too late).
- **2026-06-02 — Small-model speed test GREEN (job 50383949).** Qwen3-4B on one
  A100, 6 minutes. Measured the read/write speeds, the internal and external
  token counts agreed, and the first real price record was saved. This proved the
  whole measurement pipeline works on real hardware.
- **2026-06-02 — Production-model speed test GREEN (job 50384019).** Qwen2.5-72B
  across four 80 GB A100s on `midway3-0377`, 21 minutes. Confirmed the card was the
  80 GB version ("A100 80GB PCIe"), so the model had room to run normally. Measured
  read ~2,900 tokens/sec and write ~1,120 tokens/sec; the read/write speeds and the
  token double-check all passed. The real price record for the production model is
  now saved (`billing/rate_table.json`). Both models on the A100 tier are done; the
  remaining tiers to measure are H200 and H100.
- **2026-06-02 — Fast-tier (H200) speed test GREEN (job 50399550).** Qwen2.5-72B
  on two H200 GPUs (`midway3-0605`), 11 minutes — about half the time the four-A100
  run took. Measured read ~7,600 tokens/sec and write ~2,330 tokens/sec (roughly
  2–2.6× the A100 speeds); the read/write order and the token double-check both
  passed, and the price record for the H200 tier is now saved. This surfaced the
  A100-vs-H200 trade-off now written into Table 2: the H200 is cheaper per token
  but more expensive per hour held. Two of the four 72B serving tiers (A100, H200)
  are now measured; H100 is the remaining one.
- **2026-06-02 — Whole-service dry run GREEN (session 50384737).** Started Qwen3-4B
  on one A100, ran the front door (gateway) on a login computer, and sent 9 real
  requests through it (6 normal, 3 streaming). Confirmed three things end to end:
  (1) a login computer **can** reach the model's machine over the network;
  (2) the front door captured the token counts for **all** 9 requests — including the
  streaming ones — and the total (151 in / 860 out) **matched the model's own internal
  counter exactly**; (3) ending the session produced a correct bill (0.058 SU, charged
  on time-held since the few tokens cost almost nothing) and released the machine. The
  service works start-to-finish on real hardware.
- **2026-06-02 — Real-model whole-service run GREEN (session 50400678).** Repeated the
  whole-service flow with the **actual production model** — Qwen2.5-72B on **two H200
  GPUs** (`midway3-0605`) — not the small stand-in. Ran the front door (gateway) on a
  login computer and sent **14 real requests** (9 normal, 5 streaming), with a
  deliberate spread: short chats, long answers, a legacy text-completion, and two
  large ~7,400-token document prompts. Result: the front door captured **all 14**
  requests and its token total (**17,260 in / 1,668 out**) matched the model's own
  internal counter **exactly** (no drift — this was a fresh engine with no warm-up
  traffic). The bill came out to **0.567 SU**, charged on **time-held** (the 5.7-minute
  reservation of 2× H200 at 3.0 each) — the token cost (0.005 SU) was, as expected,
  a rounding error next to it. The machine was released cleanly (`sacct` shows the job
  CANCELLED, GPUs freed). This is the last open end-to-end check: the service now works
  start-to-finish **on the real 72B model and the priciest GPU tier**, with the token
  price drawn from the real H200 speed record. (Note: an idle reserved node bills the
  same time-held floor whether busy or not, so the run was ended as soon as the varied
  workload was sent rather than padding it out — the floor, not the typing, is the cost.)
- **2026-06-02 — H100 tier speed test GREEN (job 50401160).** Measured Qwen2.5-72B on
  **four H100 GPUs** (`midway3-0426`, which turned out to be H100 **NVL**, 94 GB), 14
  minutes (after a ~14-minute queue wait for four free cards). We used four, not two:
  the 72B's ~145 GB of weights don't fit two 80 GB H100s, so two would OOM on load.
  Measured read ~3,787 tok/s and write ~1,811 tok/s; read > write held and the token
  double-check reconciled. The H100 price record is saved, **completing all three 72B
  serving tiers** (A100, H100, H200). Headline finding (now in Tables 1–2): for serving
  *this* model the H100 tier is the **least economical of the three** — it both holds
  dearest (8.0 SU/hr, because it needs four cards) and costs the most per token — so the
  A100 stays the value pick for interactive use and the H200 for sustained throughput.
- **2026-06-03 — Real client stood up: Open WebUI GREEN end-to-end.** Installed Open
  WebUI (the private ChatGPT-style web page) in its **own** Python-3.11 environment on a
  login computer — `/project/rcc/mehta5/openwebui-env`, kept entirely separate from the
  model-serving environment — and pointed it at the **front door (gateway)**, not at any
  one machine, so it never needs reconfiguring as sessions come and go. Launch helper:
  `ai-session/run_openwebui.sh` (data + caches live in project space, login wall off for
  the demo). Proof: started a small model (Qwen3-4B on one A100), and through Open WebUI
  the model **appeared in the menu**, a normal chat **and** a streaming chat both came
  back correctly, and the front door **counted the tokens** for billing (37 in / 100 out,
  matching the model's own counter exactly). Ending the session billed **0.039 SU** (a
  couple of minutes of one A100) and freed the machine. Net: a real user-facing client now
  works start-to-finish over the stable URL — chat in, tokens metered, model swappable per
  session without touching the client.
