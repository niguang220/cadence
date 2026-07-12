"""Scorecard tier gating.

The deterministic tier runs with no API and no Docker and must NOT contain a catch-rate
or match-rate field (those are real-API-only, and CI must never present teeth as a
measured score). The real-api tier hard-fails when no model is available rather than
silently emitting a complete-looking report.
"""
import pytest

from evals.scorecard import deterministic_report, run


def test_deterministic_report_has_no_measured_scores():
    rep = deterministic_report()
    assert rep["gate"]["routing_accuracy"] == 1.0
    assert rep["teeth"]["consistency_passed"] >= 1 and rep["teeth"]["sandbox_passed"] >= 1
    flat = str(rep)
    assert "catch_rate" not in flat and "match_rate" not in flat


def test_real_api_tier_hard_fails_without_a_model():
    with pytest.raises(SystemExit):
        run("real-api", model_factory=None)


def test_deterministic_tier_runs_without_a_model():
    rep = run("deterministic", model_factory=None)
    assert rep["tier"] == "deterministic" and "gate" in rep
