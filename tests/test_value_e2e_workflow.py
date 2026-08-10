"""Static safety checks for the real, paid value-e2e workflow. These are guardrails against an
accidental edit that would auto-trigger a paid run, leak the API key, or run off a feature branch."""
import re
from pathlib import Path

_WF = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "value-e2e.yml"
_TEXT = _WF.read_text(encoding="utf-8")
# the trigger block: everything from `on:` up to the next top-level key (`permissions:`)
_ON_BLOCK = _TEXT.split("\non:", 1)[1].split("\npermissions:", 1)[0]


def test_manual_only_trigger():
    assert "workflow_dispatch" in _ON_BLOCK
    for auto in ("pull_request", "push:", "schedule", "labeled", "label"):
        assert auto not in _ON_BLOCK, f"auto-trigger {auto!r} must not be present"


def test_main_only_guard():
    assert "if: github.ref == 'refs/heads/main'" in _TEXT


def test_environment_scoped_secret():
    assert "environment: real-eval" in _TEXT
    assert "secrets.DEEPSEEK_API_KEY" in _TEXT           # sourced from the environment secret


def test_secret_is_never_echoed():
    for leak in ('echo "$DEEPSEEK_API_KEY"', "echo $DEEPSEEK_API_KEY",
                 'echo "${DEEPSEEK_API_KEY}"', "echo ${DEEPSEEK_API_KEY}"):
        assert leak not in _TEXT, f"secret must never be echoed ({leak!r})"


def test_least_privilege_permissions():
    assert "permissions:\n  contents: read" in _TEXT


def test_real_elasticsearch_service():
    assert "docker.elastic.co/elasticsearch/elasticsearch:8.19.0" in _TEXT


def test_runs_the_official_e2e_command():
    assert "python -m evals.value_e2e" in _TEXT
    assert "CADENCE_ES_URL: http://localhost:9200" in _TEXT


def test_actions_pinned_to_full_sha():
    uses = re.findall(r"uses:\s*(actions/\S+)", _TEXT)
    assert uses, "no actions found"
    for u in uses:
        assert re.search(r"@[0-9a-f]{40}$", u), f"action not pinned to a 40-hex SHA: {u}"


def test_no_cancel_and_bounded_timeout():
    assert "cancel-in-progress: false" in _TEXT
    assert "timeout-minutes: 90" in _TEXT


def test_uploads_report_artifact():
    assert "docs/reliability/value_e2e_*.json" in _TEXT
    assert "if-no-files-found: error" in _TEXT
