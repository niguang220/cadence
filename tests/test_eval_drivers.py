"""Fake-based tests for the manual real-API drivers (no real API, no real Docker).

These pin the reconcile-critical paths that otherwise run only under a real model/Docker
-- and that a review caught precisely because they were untested: the sandbox driver's
Docker-preflight abort (infrastructure must never masquerade as a model miss), its
model-failure-in-denominator classification, computation-only filtering, provenance
JSON-serializability, and the consistency driver validating a fixture BEFORE the judge.
"""
import json

import pytest

import evals.run_sandbox as rs
import evals.scorecard as sc
from agent.db.build_saas_db import build
from agent.db.introspect import introspect
from agent.sandbox import SandboxResult
from evalharness.consistency_eval import ConsistencyOutcome
from evalharness.golden import ConsistencyCase, SandboxCase
from evalharness.sandbox_eval import SandboxOutcome


class _ProgModel:
    """Fake model: returns a fixed program string for generate_python."""
    def __init__(self, program="import sys, json; print(json.dumps({'v': 1}))"):
        self._p = program

    def invoke(self, prompt):
        return type("R", (), {"content": self._p})()

    def bind(self, **_):
        return self


def _comp(cid="c"):
    return SandboxCase(cid, "compute v", {"columns": ["a"], "rows": [[1]]},
                       expected_output={"v": 1}, tolerance=0.01)


def _adv(cid="adv"):
    return SandboxCase(cid, "compute v", {"columns": ["a"], "rows": [[1]]},
                       expected_output={"v": 1}, tolerance=0.01,
                       wrong_program="import sys, json; print(json.dumps({'v': 2}))")


# --- sandbox driver: infra vs. model failure --------------------------------------

def test_sandbox_preflight_aborts_when_docker_down(monkeypatch):
    # a down daemon / missing image surfaces as "sandbox exited non-zero"; it must ABORT
    # (no score), never be counted as a model miss producing a fake match_rate=0.0.
    monkeypatch.setattr(rs, "run_in_sandbox",
                        lambda *a, **k: SandboxResult(False, error="sandbox exited non-zero",
                                                      stderr="Cannot connect to the Docker daemon"))
    with pytest.raises(RuntimeError, match="preflight"):
        rs.run_sandbox([_comp()], _ProgModel())


def test_sandbox_model_failure_stays_in_denominator(monkeypatch):
    # preflight ok, then the model's program fails -> matched=False, still counted (pass@1)
    monkeypatch.setattr(rs, "_preflight_docker", lambda: None)
    monkeypatch.setattr(rs, "run_in_sandbox",
                        lambda *a, **k: SandboxResult(False, error="sandbox exited non-zero"))
    outcomes = rs.run_sandbox([_comp("x")], _ProgModel())
    assert len(outcomes) == 1 and outcomes[0].matched is False and outcomes[0].error


def test_sandbox_measures_computation_cases_only(monkeypatch):
    monkeypatch.setattr(rs, "_preflight_docker", lambda: None)
    monkeypatch.setattr(rs, "run_in_sandbox", lambda *a, **k: SandboxResult(True, stdout='{"v": 1}'))
    outcomes = rs.run_sandbox([_comp("keep"), _adv("skip")], _ProgModel())
    assert [o.case_id for o in outcomes] == ["keep"]      # the adv_* fixture is never measured
    assert outcomes[0].matched is True


# --- scorecard provenance ----------------------------------------------------------

def test_real_api_report_is_json_serializable_with_provenance(monkeypatch):
    monkeypatch.setattr("evals.run_consistency.run_consistency",
                        lambda db, tables, cases, model: [ConsistencyOutcome("a1", True, True)])
    monkeypatch.setattr("evals.run_sandbox.run_sandbox",
                        lambda cases, model: [SandboxOutcome("nrr", True)])
    report = sc.real_api_report(_ProgModel())
    json.dumps(report)                                    # must be serializable (per-case outcomes included)
    assert report["measured"] is True
    assert set(report["golden_sha256"]) == {"consistency", "sandbox"}
    assert report["consistency"]["outcomes"] and report["sandbox"]["outcomes"]


# --- consistency driver: validate the fixture before the judge ---------------------

def test_consistency_driver_validates_fixture_before_judge(tmp_path):
    from evals.run_consistency import run_consistency

    db = str(build(tmp_path / "saas.db"))
    tables = introspect(db)

    class _Judge:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            return type("R", (), {"content": '{"ok": true}'})()

        def bind(self, **_):
            return self

    # a "clean" fixture (expected_caught=False) whose candidate actually DIVERGES from gold
    broken = ConsistencyCase("broken", "how many accounts?",
                             candidate_sql="SELECT COUNT(*) FROM account",
                             gold_sql='SELECT COUNT(*) FROM "user"',
                             category="", expected_caught=False)
    judge = _Judge()
    with pytest.raises(RuntimeError, match="divergence"):
        run_consistency(db, tables, [broken], judge)
    assert judge.calls == 0                               # the judge never saw a broken fixture
