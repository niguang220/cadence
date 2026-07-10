import json
import subprocess
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
    res = run_in_sandbox("prog", {}, runner=runner)
    assert not res.ok and "timed out" in res.error

def test_docker_not_available_is_error():
    def runner(cmd, stdin_text, timeout):
        raise FileNotFoundError("docker")
    res = run_in_sandbox("prog", {}, runner=runner)
    assert not res.ok and "docker not available" in res.error
