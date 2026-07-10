"""Run untrusted, LLM-written Python in a fresh, single-use, tightly-isolated Docker
container. Not a pool (that's a throughput optimization Cadence doesn't need, and
container reuse adds a state-leakage surface). Each call is one `docker run` with a
strict flag set; the program reads its input as JSON on stdin and writes its result
as JSON on stdout — no host directory is mounted, so the container can't see the repo
or the database. The `runner` seam lets tests fake execution (CI never runs Docker);
the real path is exercised by scripts/sandbox_smoke.py manually."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

_IMAGE = "cadence-sandbox:latest"  # built from Dockerfile.sandbox (slim + pandas);
#   pin by digest for prod. CI fakes the runner, so this image is only needed for the
#   manual smoke check.
# nobody:nogroup — the container process is unprivileged.
_ISOLATION_FLAGS = [
    "--rm",                              # remove the container when it exits
    "--network=none",                    # no network: no exfiltration / outbound calls
    "--memory=256m",                     # memory cap
    "--cpus=1",                          # cpu cap
    "--pids-limit=64",                   # process cap: no fork bomb
    "--read-only",                       # read-only root filesystem
    "--tmpfs", "/work:rw,noexec,nosuid,nodev,size=64m",  # hardened scratch dir
    "--workdir", "/work",                # run in the writable scratch dir
    "--cap-drop=ALL",                    # drop every Linux capability
    "--security-opt=no-new-privileges",  # block privilege escalation
    "--user", "65534:65534",             # nobody:nogroup, unprivileged
]


@dataclass
class SandboxResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    error: str = ""


def build_sandbox_command(program: str, *, image: str = _IMAGE) -> list[str]:
    # -i keeps the container's stdin open so the input JSON is actually delivered
    # (docker run closes stdin without it). Not an isolation flag, so it lives here,
    # not in _ISOLATION_FLAGS. No -t: a TTY is neither needed nor wanted.
    return ["docker", "run", "-i", *_ISOLATION_FLAGS, image, "python", "-c", program]


_MAX_OUTPUT_BYTES = 1_000_000   # cap captured stdout/stderr: a print-bomb inside the
#   container shouldn't be able to grow the host-side buffer without bound. The
#   container's --memory/--pids/timeout limits already bound how much it can emit;
#   this truncates what we hand downstream. (A fully streaming early-kill reader is a
#   possible future hardening if print-bombs become part of the real threat model.)


def _subprocess_runner(cmd: list[str], stdin_text: str, timeout: float):
    proc = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True,
                          timeout=timeout)
    return subprocess.CompletedProcess(
        proc.args, proc.returncode,
        stdout=(proc.stdout or "")[:_MAX_OUTPUT_BYTES],
        stderr=(proc.stderr or "")[:_MAX_OUTPUT_BYTES])


def run_in_sandbox(program: str, stdin_data: dict, *, timeout: float = 10.0,
                   image: str = _IMAGE, runner=None) -> SandboxResult:
    runner = runner or _subprocess_runner
    cmd = build_sandbox_command(program, image=image)
    try:
        proc = runner(cmd, json.dumps(stdin_data), timeout)
    except subprocess.TimeoutExpired:
        return SandboxResult(False, error="sandbox timed out")
    except FileNotFoundError:
        return SandboxResult(False, error="docker not available")
    if proc.returncode != 0:
        return SandboxResult(False, stdout=proc.stdout, stderr=proc.stderr,
                             error="sandbox exited non-zero")
    return SandboxResult(True, stdout=proc.stdout, stderr=proc.stderr)
