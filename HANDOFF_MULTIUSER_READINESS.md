# Handoff: making ai-session ready for non-staff users

Status as of 2026-07-08. This is a work prompt for a later session. It lists the
remaining blockers to letting ARBITRARY UChicago RCC users (not in `rcc-staff`,
no write access to `/project/rcc/mehta5`) use the service. Two of the six items a
prior review found are already fixed; the four below are not.

## How to use this document

Pick up the items in priority order (2 → 3 → 5 → 6 in the review's numbering,
kept here for continuity). Each item states the problem, the on-disk evidence
(`file:line`), a proposed fix, acceptance criteria, and constraints. Verify each
claim against the current code before acting — line numbers drift.

## Standing constraints (do not violate)

- **Never submit a Slurm job** (`sbatch`/`srun`/`salloc`, or `ai-session
  chat|code|fast`, or any wrapper `up`/`start`) without the user's explicit
  go-ahead. A GPU session costs SU. A prior build agent orphaned a ~4-GPU job.
  Test with a stubbed `sbatch` or read-only checks instead.
- **Do not modify** `/project/rcc/mehta5/conda-envs/vllm-probe` (the serving
  env) — installing into it risks vLLM's pinned deps. New Python deps go in a
  dedicated venv (see how `mcp-env` was done for the MCP servers).
- Keep `mkdocs build --strict` green (the Pages deploy depends on it); build with
  `/project/rcc/mehta5/mkdocs-env/bin/mkdocs`.
- Docs style: scientist-to-scientist prose, no emoji/marketing, exact commands,
  measured numbers.
- Do not delete `vllm-v0.7.3.sif` or the Llama `original/` checkpoint.

## Context: the deployment model

A user runs `module load ai-session` then `ai-session chat|code|fast`, which
launches a vLLM server as THAT USER's own Slurm job on a GPU node, fronted by a
per-user gateway (reverse proxy) on the login node, with SU billing. The whole
install lives under `/project/rcc/mehta5` and is currently world-readable
(`/project/rcc` is `drwxrws--x`, traversable; everything the runtime needs below
is `o+rx`). A permission audit confirmed every runtime read-path (dispatcher,
`conda-envs/vllm-probe`, `mcp-env`, `modulefiles`, `opencode`, `models/`) is
readable by outside users; the only unreadable files are dead `apptainer_cache/`
build junk. So the READ side is fine as-is; the open items are writable paths,
per-user isolation, billing operationalization, and durability.

## Already fixed (2026-07-08) — do not redo

- **State dir off the project tree.** `AISESSION_STATE_DIR` now defaults to
  `$HOME/.ai-session/state` (`bin/ai-session`), so logs, session key, and
  receipts write to the user's own space. Commit `caeb683`.
- **Writable caches off the project tree (review item 1, BLOCKER).** HF and
  torch-inductor caches now default under the per-user state dir in both
  `launch_ai_session.sh` and `run_openwebui.sh`. Commit `7298c7e`.
- **Identity from the real uid (review item 4).** `_real_user()` (real uid via
  `pwd.getpwuid`) replaces `$USER` in `ai_session.py` and `metering.py` for all
  receipt/session/license-ack/ledger attribution. Commit `7298c7e`.

---

## Item 2 (HIGH — availability + credential leak): UID-derived ports

**Problem.** The per-user gateway and Open WebUI ports are deterministic:
`GW_PORT = 8400 + UID%90` (`bin/ai-session`), `OWUI_PORT = 3000 + UID%90`
(`run_browser_demo.sh`, `run_coding_agent.sh`). Only 90 slots exist, so across an
arbitrary-user population collisions mod 90 are guaranteed. Two consequences:

1. **Availability:** if a co-tenant on the same login node already holds the
   derived port, the start aborts (`port_busy` in `run_browser_demo.sh`) and the
   second user has no documented recovery.
2. **Credential leak (worse):** UID is public (`id <user>`), so the port a given
   user will use is predictable. A hostile co-tenant can pre-bind (squat) that
   port on the shared login-node loopback. Because the victim's labmates tunnel
   to `localhost:<GW_PORT>` and send the **shared session key**, a rogue listener
   there harvests the key and prompt contents — a man-in-the-middle. The
   `127.0.0.1` bind and key gate do NOT prevent this, because the attacker owns
   the listener.

**Fix (proposed).** Allocate an ephemeral free port per session (bind to port 0,
read back the assigned port, or scan upward from the UID-derived value for the
first free port) and record it in `~/.ai-session/env` and the state dir, which
clients already read (`AISESSION_BASE_URL`). Never rely on a
publicly-derivable port for isolation. Keep the UID value only as a starting
hint. Update: `bin/ai-session` (GW_PORT/OWUI_PORT derivation), the wrappers, and
any doc that tells users the port is `8400+UID%90` (`docs/reference.md`,
`docs/getting-started.md`, `docs/coding/overview.md`).

**Acceptance.** Two users on one login node can each start a session without
collision; the chosen port is discoverable only from the user's own private env
file, not computable from their UID; a squatted port does not silently redirect a
labmate's key.

**Watch out.** The port must be stable for the life of ONE session (clients hold
it) but can differ between sessions. Preserve the "labmate shares your session"
flow (they need your current port + key). The gateway health check and
`ai-session status`/`connect` must read the actual chosen port, not recompute it.

---

## Item 3 (HIGH): central billing does not capture non-staff users

**Problem.** Billing has two write paths and NEITHER currently records an outside
user's spend:

1. **Per-`end` central write** (`metering.write_central_billing_record`) targets
   `/project/rcc/mehta5/ai-session-billing` (`metering.py`, `BILLING_DIR_DEFAULT`),
   which is `drwxrws--- mehta5:rcc-staff` — no world write. The write is
   deliberately fail-safe (a `PermissionError` is caught and swallowed so `end`
   never breaks), so an outside user's session records NOTHING centrally; their
   only receipt is in their own HOME, which staff cannot read.
2. **Staff `sacct` sweep** (`billing_sweep.py`) is the authoritative,
   tamper-proof path (reconstructs the floor charge from Slurm accounting, which
   users cannot edit). It works for all users REGARDLESS of write access — but it
   is NOT SCHEDULED. `crontab -l` is empty; the only systemd unit shipped is the
   gateway service; `billing_sweep`/`OnCalendar` appear only in docs.

**Fix (proposed).**
- Schedule `billing_sweep.py` on a staff-owned cron or systemd timer. A ready
  cron line is in `ai-session/README.md` (search `billing_sweep`). Run it under
  an account that can `sacct -a` and write the ledger dir. Verify it writes
  `source="sweep"` records for a NON-staff user's job.
- Decide the per-`end` path for outside users: either make the ledger a
  write-only drop-box (`chmod 1733` so any user can drop a file but not read
  others'), or accept that the sweep is the sole central record for non-staff
  users (the floor is authoritative and dominates interactive billing, so this is
  defensible — document it).

**Acceptance.** After a non-staff user runs and ends a session, a staff query
(or `su_usage_mcp` run as staff) shows that session's floor charge in the central
ledger, produced by the scheduled sweep.

**Watch out.** The sweep must correctly identify ai-session jobs among all
cluster jobs (it keys off the job-name pattern `model_key:port`). Confirm it does
not mis-attribute or double-count against per-`end` records (records are keyed by
job id; `source="end"` vs `source="sweep"`).

---

## Item 5 (MEDIUM/HIGH): no spend cap or central reaper

**Problem.** Billing is accounting-only; nothing enforces a ceiling. The
one-session-per-user guard (`ai_session.py`) is bypassable with
`--allow-multiple`. `idle_reaper.py` is an opt-in tool the user must schedule
themselves (login-node cron/tmux) — nothing centrally reaps a forgotten session.
The only hard backstop is the Slurm walltime (`TIME_LIMIT` default `02:00:00`)
and the PI's Slurm allocation. A careless user who never runs `stop`, or sets a
long `--time`, floor-bills a whole node for that duration; nothing stops repeated
starts.

**Fix (decision needed first).**
- Confirm whether RCC's Slurm enforces per-account GPU-hour limits. If it does,
  that is the real backstop and this item is largely a documentation task
  (make the limit and its consequences explicit to users).
- If not, add service-side guardrails: a maximum `--time` the dispatcher will
  accept, a cap on concurrent sessions per user, and/or a staff-run central
  reaper (a scheduled `idle_reaper`-style sweep that ends sessions idle beyond a
  threshold — reuse `idle_reaper.py` logic but run centrally against all users'
  sessions).
- Treat printed SU numbers as advisory until a quota mechanism exists; say so in
  the docs.

**Acceptance.** There is a defined, enforced ceiling on how much a single user
can spend without staff intervention, OR an explicit documented statement that
the Slurm allocation is the ceiling and how it behaves.

---

## Item 6 (MEDIUM — structural): single-person single point of failure

**Problem.** Every shared component — code, `conda-envs/vllm-probe`, `mcp-env`,
`models/`, `modulefiles` (`set root /project/rcc/mehta5/vllm` in
`modulefiles/ai-session/1.0`), the billing ledger, the caches — is under one
person's (`mehta5`) personal allocation and ownership. Outside-user access
depends on mehta5's world `r-x` bits staying set and the allocation not lapsing.
If mehta5 departs or the allocation is cleaned, the whole service and its billing
history vanish.

**Fix (the real "make it a service" step; needs RCC).** Relocate the install,
venvs, models, modulefiles, and ledger to a location owned by a SERVICE ACCOUNT
or an rcc-staff-collective group, independent of any individual — ideally the
`/software` central install (`drwxrwsr-x root:rcc-software`, world-readable,
where central modules live). This is also the Tier-3 module step that removes the
`module use /project/rcc/mehta5/modulefiles` line from every doc page. Coordinate
with RCC: what path, which owning group/account, and how the modulefile `root`
and all hardcoded `/project/rcc/mehta5/...` paths get parameterized. Grep the
tree for `/project/rcc/mehta5` to find every path that must move or become
configurable (there are many: `bin/ai-session` `PY`/`MCP_PY`,
`launch_ai_session.sh` `REPO`/`ENV_PATH`, `metering.py` `BILLING_DIR_DEFAULT`,
`server.py` model root, the MCP servers' fallback dirs).

**Acceptance.** The service runs from a non-personal, RCC-sanctioned location and
survives any individual's departure; `module load ai-session` works with no
`module use` line.

---

## Suggested order and effort

1. **Item 3 sweep scheduling** — small, unblocks central billing for everyone.
2. **Item 2 ephemeral ports** — medium, closes a real security hole; touches the
   dispatcher, wrappers, and a few doc pages.
3. **Item 5 spend policy** — starts as a decision (Slurm quota?) then small code.
4. **Item 6 /software relocation** — large, RCC-coordinated; the durability step.

Items 1 and 4 (writable caches; real-uid identity) are already done — see
"Already fixed" above.
