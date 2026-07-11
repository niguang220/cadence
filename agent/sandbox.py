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
import uuid
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


def build_sandbox_command(program: str, *, image: str = _IMAGE,
                          name: str | None = None) -> list[str]:
    # -i keeps the container's stdin open so the input JSON is actually delivered
    # (docker run closes stdin without it). Not an isolation flag, so it lives here,
    # not in _ISOLATION_FLAGS. No -t: a TTY is neither needed nor wanted.
    # --name lets a timed-out run be killed by name: `docker run` talks to the daemon,
    # so killing the CLI on timeout would otherwise leave the container running.
    named = ["--name", name] if name else []
    return ["docker", "run", "-i", *named, *_ISOLATION_FLAGS, image, "python", "-c", program]


_MAX_OUTPUT_CHARS = 1_000_000   # cap captured stdout/stderr (character count): a print-bomb inside the
#   container shouldn't be able to grow the host-side buffer without bound. The
#   container's --memory/--pids/timeout limits already bound how much it can emit;
#   this truncates what we hand downstream. (A fully streaming early-kill reader is a
#   possible future hardening if print-bombs become part of the real threat model.)


def _kill_container(name: str) -> None:
    """Best-effort: stop a container that outran its wall-clock timeout. `docker run`
    talks to the daemon, so killing the CLI on timeout leaves the container alive; an
    explicit `docker kill` on the named container is what actually stops it (`--rm`
    then reaps it). Cleanup failures are swallowed -- the container's own resource caps
    still bound it, and there is nothing useful to do if the daemon is unreachable."""
    try:
        subprocess.run(["docker", "kill", name], capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        pass


def _subprocess_runner(cmd: list[str], stdin_text: str, timeout: float):
    proc = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True,
                          timeout=timeout)
    return subprocess.CompletedProcess(
        proc.args, proc.returncode,
        stdout=(proc.stdout or "")[:_MAX_OUTPUT_CHARS],
        stderr=(proc.stderr or "")[:_MAX_OUTPUT_CHARS])


def run_in_sandbox(program: str, stdin_data: dict, *, timeout: float = 10.0,
                   image: str = _IMAGE, runner=None, kill=None) -> SandboxResult:
    runner = runner or _subprocess_runner
    kill = kill or _kill_container
    name = f"cadence-sandbox-{uuid.uuid4().hex}"
    cmd = build_sandbox_command(program, image=image, name=name)
    try:
        proc = runner(cmd, json.dumps(stdin_data), timeout)
    except subprocess.TimeoutExpired:
        kill(name)                       # stop the container, not just the killed CLI
        return SandboxResult(False, error="sandbox timed out")
    except FileNotFoundError:
        return SandboxResult(False, error="docker not available")
    if proc.returncode != 0:
        return SandboxResult(False, stdout=proc.stdout, stderr=proc.stderr,
                             error="sandbox exited non-zero")
    return SandboxResult(True, stdout=proc.stdout, stderr=proc.stderr)
