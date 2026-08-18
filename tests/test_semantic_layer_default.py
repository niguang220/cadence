"""Semantic governance defaults ON, and stays safe on schemas the registry does not govern.

Defaulting governance on is only sound if it degrades on a foreign database instead of crashing.
The metric registry is SaaS-specific: on any other schema every metric's required tables are
absent. The rule pinned here is all-or-nothing --

* no metric resolves  -> the registry does not govern this database; governance is inert, traced,
  and the run proceeds exactly as it would with governance off;
* every metric resolves -> governed as before;
* some resolve and some do not -> genuine misconfiguration of a governed database, still fails
  fast (the G4 invariant is preserved where it actually means something).
"""
from __future__ import annotations

import inspect

import pytest

from agent.db.introspect import introspect
from agent.retrieval.metric_match import registry_governs, validate_all_metrics
from agent.semantic_layer import MetricDef, MetricRegistry


@pytest.fixture(scope="module")
def saas_tables(tmp_path_factory):
    from agent.db.build_saas_db import build
    return introspect(build(tmp_path_factory.mktemp("s") / "saas.db"))


@pytest.fixture(scope="module")
def foreign_tables(tmp_path_factory):
    from agent.db.build_demo_db import build
    return introspect(build(tmp_path_factory.mktemp("c") / "chinook.db"))


# --- the default itself -------------------------------------------------------------------

@pytest.mark.parametrize(
    "module, func",
    [("agent.graph", "run_agent"), ("agent.graph", "start_agent_session"),
     ("agent.pipeline", "answer_question"), ("agent.pipeline", "start_question_session")],
)
def test_public_entry_points_default_semantic_governance_on(module, func):
    import importlib

    fn = getattr(importlib.import_module(module), func)
    assert inspect.signature(fn).parameters["semantic_layer"].default is True


def test_cli_defaults_governance_on_with_an_explicit_opt_out():
    from agent.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["ask", "q"]).semantic_layer is True
    assert parser.parse_args(["ask", "q", "--no-semantic-layer"]).semantic_layer is False
    assert parser.parse_args(["ask", "q", "--semantic-layer"]).semantic_layer is True
    assert parser.parse_args(["retrieve", "q"]).semantic_layer is True
    assert parser.parse_args(["retrieve", "q", "--no-semantic-layer"]).semantic_layer is False


# --- registry_governs -----------------------------------------------------------------------

def test_registry_governs_the_saas_schema(saas_tables):
    assert registry_governs(MetricRegistry.load(), saas_tables) is True


def test_registry_does_not_govern_a_foreign_schema(foreign_tables):
    assert registry_governs(MetricRegistry.load(), foreign_tables) is False


def test_a_partially_resolving_registry_is_still_governed_and_still_fails_fast(saas_tables):
    good = MetricRegistry.load().metrics[0]
    broken = MetricDef(name="broken", aliases=["broken"], definition="d", measure="m",
                       grain="g", required_filters=[], common_mistake="c",
                       required_tables=["no_such_table"], required_columns=[])
    registry = MetricRegistry([good, broken])
    assert registry_governs(registry, saas_tables) is True      # some resolve -> governed
    with pytest.raises(ValueError):
        validate_all_metrics(registry, saas_tables)             # G4 fail-fast preserved


def test_empty_registry_does_not_govern(saas_tables):
    assert registry_governs(MetricRegistry([]), saas_tables) is False


# --- end to end: the default must not crash on a foreign schema ---------------------------

def test_default_run_on_a_foreign_schema_does_not_crash_and_traces_the_degradation(tmp_path):
    from agent.db.build_demo_db import build
    from agent.pipeline import answer_question
    from tests.conftest import PlanningFakeModel

    db = build(tmp_path / "chinook.db")
    res = answer_question(db, "how many tracks are there",
                          model=PlanningFakeModel("SELECT COUNT(*) FROM track"))
    assert res.sql and res.execution.ok
    preflight = next(t for t in res.trace if t.get("node") == "preflight_context")
    assert preflight["semantic_metrics"] == []
    assert preflight.get("governance") == "registry_does_not_govern_schema"


def test_default_run_on_the_governed_schema_still_binds_metrics(saas_db):
    from agent.graph import run_agent
    from tests.conftest import PlanningFakeModel

    res = run_agent(saas_db, "what is our total MRR?",
                    model=PlanningFakeModel("SELECT SUM(mrr) FROM subscription"))
    preflight = next(t for t in res.trace if t.get("node") == "preflight_context")
    assert preflight["semantic_metrics"] == ["mrr"]
    assert "governance" not in preflight
