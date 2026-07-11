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
import threading
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


_MAX_OUTPUT_CHARS = 1_000_000   # keep at most this many chars of stdout/stderr; excess
#   is drained and discarded (see _read_capped) so a streaming print-bomb can't grow the
#   host-side buffer without bound.
_CHUNK = 65536


def _read_capped(stream, cap: int, sink: list) -> None:
    """Drain a text stream, keeping at most `cap` chars (the result is appended to
    `sink`). Excess is read and discarded rather than buffered, so a runaway producer
    can't balloon host memory; the pipe is still fully drained so it never fills and
    blocks the child. Runs in a thread per stream so stdout and stderr can't deadlock."""
    kept: list[str] = []
    total = 0
    for chunk in iter(lambda: stream.read(_CHUNK), ""):
        if total < cap:
            kept.append(chunk[:cap - total])
        total += len(chunk)
    sink.append("".join(kept))


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
    # Handle all three pipes off the main thread (like subprocess.communicate): reader
    # threads drain stdout/stderr with a cap, and a writer thread feeds stdin. Writing
    # stdin on the main thread would block on a larger-than-pipe-buffer payload if the
    # child never reads it, so the wall-clock timeout below -- which triggers the
    # container kill -- would never fire.
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    out: list[str] = []
    err: list[str] = []

    def _write_stdin():
        try:
            proc.stdin.write(stdin_text)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass                         # child exited before reading its input

    threads = [
        threading.Thread(target=_write_stdin),
        threading.Thread(target=_read_capped, args=(proc.stdout, _MAX_OUTPUT_CHARS, out)),
        threading.Thread(target=_read_capped, args=(proc.stderr, _MAX_OUTPUT_CHARS, err)),
    ]
    for t in threads:
        t.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()                      # stop the CLI; run_in_sandbox kills the container
        proc.wait()
        raise
    finally:
        for t in threads:
            t.join()
    return subprocess.CompletedProcess(cmd, proc.returncode,
                                       stdout=out[0] if out else "",
                                       stderr=err[0] if err else "")


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
