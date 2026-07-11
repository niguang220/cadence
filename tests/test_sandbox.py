import json
import subprocess
import sys
import agent.sandbox as sandbox
from agent.sandbox import build_sandbox_command, run_in_sandbox, SandboxResult

_ISOLATION = ["--rm", "--network=none", "--memory=256m", "--cpus=1",
              "--pids-limit=64", "--read-only", "--workdir", "/work",
              "--cap-drop=ALL", "--security-opt=no-new-privileges"]

def test_command_pins_the_full_isolation_contract():
    cmd = build_sandbox_command("print(1)", image="python:3.11-slim")
    for flag in _ISOLATION:
        assert flag in cmd, f"missing isolation flag: {flag}"
    # tmpfs is hardened (noexec/nosuid/nodev/size) and the user is unprivileged
    assert "--tmpfs" in cmd and any("noexec" in a and "nosuid" in a for a in cmd)
    assert "65534:65534" in cmd
    # no host directory is ever mounted into the sandbox
    assert "-v" not in cmd and "--volume" not in cmd
    # -i is present so container stdin receives the JSON; -t (tty) is not
    assert "-i" in cmd
    assert "-t" not in cmd and "--tty" not in cmd
    assert cmd[:2] == ["docker", "run"]
    assert cmd[-3:] == ["python", "-c", "print(1)"]

def _fake_runner(result):
    def runner(cmd, stdin_text, timeout):
        runner.seen = {"cmd": cmd, "stdin": stdin_text, "timeout": timeout}
        return result
    return runner

def test_stdin_data_passed_as_json_and_stdout_returned():
    proc = subprocess.CompletedProcess([], 0, stdout='{"mrr": 42}', stderr="")
    runner = _fake_runner(proc)
    res = run_in_sandbox("prog", {"rows": [[1]], "columns": ["x"]}, runner=runner)
    assert res.ok and res.stdout == '{"mrr": 42}'
    assert json.loads(runner.seen["stdin"]) == {"rows": [[1]], "columns": ["x"]}
    assert runner.seen["timeout"] == 10.0   # default timeout passed through to the runner

def test_nonzero_exit_is_error():
    proc = subprocess.CompletedProcess([], 1, stdout="", stderr="Traceback...")
    res = run_in_sandbox("prog", {}, runner=_fake_runner(proc))
    assert not res.ok and res.stderr == "Traceback..." and "non-zero" in res.error

def test_timeout_is_error():
    def runner(cmd, stdin_text, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)
    res = run_in_sandbox("prog", {}, runner=runner, kill=lambda name: None)
    assert not res.ok and "timed out" in res.error

def test_command_carries_a_unique_container_name():
    # a named container is what lets a timed-out run be killed (docker run alone
    # leaves the container on the daemon after the CLI is killed).
    cmd = build_sandbox_command("print(1)", name="cadence-sandbox-abc")
    assert "--name" in cmd and "cadence-sandbox-abc" in cmd

def test_timeout_kills_the_container():
    # teeth: on wall-clock timeout the container must be killed, not just the CLI.
    killed = []
    def runner(cmd, stdin_text, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)
    res = run_in_sandbox("prog", {}, runner=runner, kill=killed.append)
    assert not res.ok and "timed out" in res.error
    assert len(killed) == 1 and killed[0].startswith("cadence-sandbox-")

def test_container_name_in_command_matches_the_one_killed():
    # the name docker sees and the name we kill must be identical, or the kill misses.
    seen = {}
    killed = []
    def runner(cmd, stdin_text, timeout):
        seen["name"] = cmd[cmd.index("--name") + 1]
        raise subprocess.TimeoutExpired(cmd, timeout)
    run_in_sandbox("prog", {}, runner=runner, kill=killed.append)
    assert killed == [seen["name"]]

def test_docker_not_available_is_error():
    def runner(cmd, stdin_text, timeout):
        raise FileNotFoundError("docker")
    res = run_in_sandbox("prog", {}, runner=runner)
    assert not res.ok and "docker not available" in res.error

def test_read_capped_bounds_memory_and_discards_excess():
    # teeth: a streaming print-bomb must not balloon host memory. The reader keeps at
    # most `cap` chars and drains (discards) the rest, so kept memory is bounded even
    # when the producer emits far more.
    import io
    kept = []
    sandbox._read_capped(io.StringIO("x" * 5000), 100, kept)
    assert kept == ["x" * 100]

def test_subprocess_runner_round_trips_stdin_and_caps_output(monkeypatch):
    # exercise the real Popen + reader-thread plumbing without docker (a local python):
    # stdin must reach the child, and stdout beyond the cap must be dropped.
    monkeypatch.setattr(sandbox, "_MAX_OUTPUT_CHARS", 50)
    echo = sandbox._subprocess_runner(
        [sys.executable, "-c", "import sys; print(sys.stdin.read().strip().upper())"],
        "hello", 5)
    assert echo.returncode == 0 and echo.stdout.strip() == "HELLO"
    flood = sandbox._subprocess_runner(
        [sys.executable, "-c", "print('x' * 100000)"], "", 5)
    assert flood.returncode == 0 and len(flood.stdout) <= 50
