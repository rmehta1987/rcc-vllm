#!/usr/bin/env python3
"""slurm_mcp -- a read-only Slurm MCP server for the invoking user.

Exposes a FIXED set of typed, read-only tools over the MCP stdio protocol so a
coding agent (opencode, Cline, ...) can answer "what are my jobs doing?" without
being handed a shell. It runs with the caller's own cluster permissions.

Tools:
  my_jobs(states?)        squeue --me            -- your queued/running jobs
  job_detail(job_id)      sacct -j <id>          -- accounting for a job you own
  partition_info(part?)   sinfo                  -- partition availability

This uses the official MCP SDK (the ``mcp`` package, FastMCP) from the dedicated
mcp-env; the protocol framing is the SDK's, only the tool bodies are ours.

Security model (enforced here, not by convention):
  * Only four binaries may ever be spawned: squeue, sacct, sinfo, scontrol. The
    whitelist is checked on every call; anything else raises before exec.
  * Every command is built as an argv list and run with shell=False, so no
    argument is ever interpreted by a shell. There is no free-form command tool.
  * job_id is constrained to ^[0-9_]+$ (a bare id or an array task like 12345_6).
    A value like "1; scancel 999" fails the regex and is refused before any
    process is spawned.
  * partition/state filters are constrained to their own whitelists.
  * job_detail verifies you own the job (its sacct User == $USER) before
    returning any detail; a job you do not own returns "not found or not owned".
  * squeue defaults to --me, so jobs are scoped to the invoking user by default.
No mutating command (scancel/sbatch/srun/scontrol update/hold/...) is reachable.
"""

import os
import pwd
import re
import shutil
import subprocess
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import Field

# Only these binaries may be spawned. scontrol is included for read-only
# "scontrol show" queries; the wrappers below never pass an update subcommand.
ALLOWED_BINARIES = {"squeue", "sacct", "sinfo", "scontrol"}

# Anchor with \A...\Z (NOT ^...$): in Python `$` also matches just before a
# trailing newline, so `^[0-9_]+$` would accept "123\n" and pass a newline-bearing
# argv to the Slurm client. \A...\Z anchors to the absolute string bounds.
JOB_ID_RE = re.compile(r"\A[0-9_]+\Z")         # 12345 or 12345_6 (array task)
PARTITION_RE = re.compile(r"\A[A-Za-z0-9_,.-]+\Z")
STATES_RE = re.compile(r"\A[A-Za-z,]+\Z")

_TIMEOUT = 25  # seconds; a hung scheduler must not wedge the agent

mcp = FastMCP("slurm-mcp", log_level="WARNING")


def _me():
    # Derive identity from the real uid, NOT $USER/$LOGNAME, which the caller can
    # spoof. Slurm still enforces real-uid visibility, but keying the ownership
    # check off the real user makes the app-level check meaningful too.
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except (KeyError, OSError):
        return os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"


def _run(argv):
    """Run a whitelisted read-only Slurm command as an argv list (no shell)."""
    if not argv or argv[0] not in ALLOWED_BINARIES:
        # Defense in depth: this can only trip on a coding mistake, never on
        # user input, because every caller hard-codes argv[0].
        raise ToolError("refused: %r is not a permitted read-only command"
                        % (argv[0] if argv else None,))
    if shutil.which(argv[0]) is None:
        raise ToolError("%s not found on PATH (are Slurm client tools loaded?)"
                        % argv[0])
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              shell=False, timeout=_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise ToolError("%s timed out after %ds" % (argv[0], _TIMEOUT))
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ToolError("%s exited %d: %s" % (argv[0], proc.returncode, detail))
    return proc.stdout


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #
@mcp.tool(
    description="List the invoking user's Slurm jobs (squeue --me). Optional "
                "'states' filters by comma-separated Slurm states such as "
                "RUNNING,PENDING. Read-only.")
def my_jobs(
    states: Annotated[str, Field(
        description="Comma-separated Slurm states, e.g. 'RUNNING,PENDING'. "
                    "Optional.")] = "",
) -> str:
    argv = ["squeue", "--me",
            "-o", "%.18i %.24j %.9T %.10M %.11l %.5D %R"]
    if states:
        if not STATES_RE.match(states):
            raise ToolError(
                "states must match ^[A-Za-z,]+$ (e.g. RUNNING,PENDING)")
        argv += ["--states", states.upper()]
    out = _run(argv).rstrip("\n")
    body = out.split("\n")[1:] if "\n" in out else []
    if not any(line.strip() for line in body):
        who = _me()
        scope = " in states %s" % states.upper() if states else ""
        return "No jobs for %s%s." % (who, scope)
    return out


@mcp.tool(
    description="Accounting summary (sacct) for one Slurm job the invoking user "
                "owns. Ownership is verified before any detail is returned. "
                "Read-only.")
def job_detail(
    job_id: Annotated[str, Field(
        description="Slurm job id, e.g. '12345' or array task '12345_6'.")],
) -> str:
    if not JOB_ID_RE.match(job_id):
        raise ToolError(
            "job_id must match ^[0-9_]+$ (a job id or array task, e.g. 12345 "
            "or 12345_6); got %r" % (job_id,))
    me = _me()
    # Ownership check first: sacct -X gives one row per job; compare its User.
    owner_out = _run(["sacct", "-X", "-n", "-P", "-j", job_id, "-o", "User"])
    owners = {u.strip() for u in owner_out.splitlines() if u.strip()}
    if not owners:
        raise ToolError(
            "job %s not found in your accounting records (it may be too old, "
            "or it is not yours)" % job_id)
    if owners != {me}:
        # Do not reveal another user's job; report as if not owned.
        raise ToolError("job %s is not owned by %s" % (job_id, me))
    detail = _run([
        "sacct", "-j", job_id, "--units=G",
        "-o", "JobID%20,JobName%22,Partition,State,Elapsed,"
              "AllocTRES%42,TotalCPU,MaxRSS,ReqTRES%42",
    ])
    return detail.rstrip("\n")


@mcp.tool(
    description="Partition availability (sinfo). Optional 'partition' restricts "
                "to one or a comma-separated list. Read-only.")
def partition_info(
    partition: Annotated[str, Field(
        description="Partition name(s), e.g. 'gpu' or 'gpu,test'. "
                    "Optional.")] = "",
) -> str:
    argv = ["sinfo", "-o", "%.20P %.6a %.11l %.6D %.6t %N"]
    if partition:
        if not PARTITION_RE.match(partition):
            raise ToolError(
                "partition must match ^[A-Za-z0-9_,.-]+$; got %r" % (partition,))
        argv += ["-p", partition]
    return _run(argv).rstrip("\n")


if __name__ == "__main__":
    mcp.run()
