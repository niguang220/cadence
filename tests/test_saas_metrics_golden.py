"""Loader + fixture-integrity tests for the saas_metrics golden set (service-free)."""
import pytest

from agent.db.build_saas_db import build
from agent.db.introspect import introspect
from agent.execution import run_query
from evalharness.golden import SAAS_METRICS_PATH, SaasMetricsCase, load_saas_metrics


def test_loads_thirty_cases_24_traps_6_controls():
    cases = load_saas_metrics()
    assert len(cases) == 30
    controls = [c for c in cases if c.category == "control"]
    traps = [c for c in cases if c.category != "control"]
    assert len(controls) == 6 and len(traps) == 24
    assert all(isinstance(c, SaasMetricsCase) for c in cases)


def test_every_case_has_required_tables_and_gold_sql():
    for c in load_saas_metrics():
        assert c.required_tables, f"{c.id}: required_tables must be non-empty"
        assert c.gold_sql.strip(), f"{c.id}: gold_sql must be non-empty"


def test_unknown_field_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('[{"id": "x", "category": "mrr", "question": "q", "gold_sql": "SELECT 1",'
                   ' "surprise": 1}]', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown field"):
        load_saas_metrics(bad)


def test_all_gold_sql_execute_on_saas_db(tmp_path):
    # Fixture integrity: every gold_sql must run on the real saas.db, else the baseline's
    # gold side is broken before the agent is even involved. Deterministic, zero-API.
    db = str(build(tmp_path / "saas.db"))
    tables = introspect(db)
    for c in load_saas_metrics():
        res = run_query(db, c.gold_sql, tables=tables)
        assert res.ok, f"{c.id}: gold_sql failed to execute: {res.error}"
