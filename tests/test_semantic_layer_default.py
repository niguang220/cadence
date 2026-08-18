"""Semantic governance defaults ON, and stays safe on schemas the registry does not govern.

Defaulting governance on is only sound if it degrades on a foreign database instead of crashing.
The metric registry is SaaS-specific: on any other schema every metric's required tables are
absent. The rule pinned here separates APPLICABILITY from VALIDATION --

* applicability is decided by the registry's whole ``required_tables`` footprint. The built-in
  registry applies to a schema only when that schema contains EVERY table the registry governs.
  A foreign database that happens to have some of those tables -- ``account`` and ``subscription``
  are ordinary names -- is ungoverned: the semantic layer is inert and the degradation is traced.
* once the full table footprint is present the registry does apply, and ``validate_all_metrics``
  keeps failing fast on required columns and governance metadata, so real misconfiguration of a
  genuinely governed database is still caught.

Deciding applicability on tables alone is deliberate: a partial-footprint match is the collision
case, and resolving a single metric must not be enough to claim the whole database.
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


def test_a_registry_naming_an_absent_table_does_not_govern(saas_tables):
    """A metric whose TABLE is missing means the footprint is not covered -> not applicable."""
    good = MetricRegistry.load().metrics[0]
    absent = MetricDef(name="absent", aliases=["absent"], definition="d", measure="m",
                       grain="g", required_filters=[], common_mistake="c",
                       required_tables=["no_such_table"], required_columns=[])
    assert registry_governs(MetricRegistry([good, absent]), saas_tables) is False


def test_full_table_footprint_still_fails_fast_on_a_bad_required_column(saas_tables):
    """Applicability is decided on tables; columns are still validated fail-fast. A registry whose
    tables are all present but which names a nonexistent column is real misconfiguration of a
    governed database and must not be silently tolerated."""
    good = MetricRegistry.load().metrics[0]
    bad_column = MetricDef(name="bad_column", aliases=["bad column"], definition="d", measure="m",
                           grain="g", required_filters=[], common_mistake="c",
                           required_tables=["subscription"],
                           required_columns=["subscription.no_such_column"])
    registry = MetricRegistry([good, bad_column])
    assert registry_governs(registry, saas_tables) is True      # footprint covered -> applicable
    with pytest.raises(ValueError):
        validate_all_metrics(registry, saas_tables)             # G4 fail-fast preserved


def test_empty_registry_does_not_govern(saas_tables):
    assert registry_governs(MetricRegistry([]), saas_tables) is False


def _foreign_schema_with_mrr_tables(path):
    """A foreign database that fully satisfies the `mrr` metric -- both its required tables AND
    every required column -- while containing none of the rest of the governed catalog. Under a
    resolve-any rule this schema would be claimed as governed and then crash validation."""
    import sqlite3

    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE account (account_id INTEGER PRIMARY KEY, account_type TEXT);
        CREATE TABLE subscription (
            subscription_id INTEGER PRIMARY KEY,
            account_id INTEGER REFERENCES account(account_id),
            mrr REAL, ended_on TEXT, phase TEXT);
        CREATE TABLE warehouse (warehouse_id INTEGER PRIMARY KEY, city TEXT);
        """
    )
    conn.commit()
    conn.close()
    return path


def test_a_foreign_schema_that_satisfies_one_metric_is_still_ungoverned(tmp_path):
    """The collision case. `account` and `subscription` are ordinary table names; a foreign
    database having them must not be claimed by the built-in SaaS registry."""
    from agent.retrieval.metric_match import _catalog, _validate_metric_deps

    tables = introspect(_foreign_schema_with_mrr_tables(tmp_path / "foreign.db"))
    registry = MetricRegistry.load()

    # precondition: this schema genuinely satisfies `mrr` completely, tables and columns
    mrr = next(m for m in registry.metrics if m.name == "mrr")
    names, cols = _catalog(tables)
    _validate_metric_deps(mrr, names, cols)          # must not raise -- otherwise the test is vacuous

    assert registry_governs(registry, tables) is False


def test_the_collision_schema_runs_without_crashing_under_the_default(tmp_path):
    from agent.pipeline import answer_question
    from tests.conftest import PlanningFakeModel

    db = _foreign_schema_with_mrr_tables(tmp_path / "foreign.db")
    res = answer_question(db, "how many accounts are there",
                          model=PlanningFakeModel("SELECT COUNT(*) FROM account"))
    assert res.execution.ok
    preflight = next(t for t in res.trace if t.get("node") == "preflight_context")
    assert preflight["semantic_metrics"] == []
    assert preflight.get("governance") == "registry_does_not_govern_schema"


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
