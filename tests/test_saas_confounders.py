"""The confounder expansion adds >=12 same-domain distractor tables (so retrieval is no longer
saturated at 8 <= candidate_k) WITHOUT touching the original 8 tables, their data, the 30-case golden,
or gold-SQL results, and WITHOUT introducing any searchable column (the value channel stays inert)."""
from agent.db.build_saas_db import build
from agent.db.introspect import introspect
from agent.execution import run_query
from agent.retrieval.value_policy import resolve_searchable_columns
from evalharness.golden import load_saas_metrics

_ORIGINAL_8 = {"account", "user", "plan", "subscription", "mrr_movement", "activity_event",
               "invoice", "revenue_recognition"}


def _names(db):
    return {t.name for t in introspect(db)}


def test_default_build_is_unchanged_eight_tables(tmp_path):
    assert _names(build(tmp_path / "base.db")) == _ORIGINAL_8


def test_confounders_expand_to_at_least_twenty_tables(tmp_path):
    names = _names(build(tmp_path / "exp.db", confounders=True))
    assert _ORIGINAL_8 <= names                          # original 8 preserved
    assert len(names) >= 20                              # >= 12 confounders added
    assert "weather" not in names and "hockey" not in names   # same-domain only


def test_confounders_add_no_searchable_columns(tmp_path):
    # value must stay inert on the saas domain -> no new searchable column may sneak in
    assert resolve_searchable_columns(introspect(build(tmp_path / "exp.db", confounders=True))) == []


def test_confounders_do_not_change_gold_sql_results(tmp_path):
    base = build(tmp_path / "base.db")
    exp = build(tmp_path / "exp.db", confounders=True)
    base_tables, exp_tables = introspect(base), introspect(exp)
    for case in load_saas_metrics():
        b = run_query(base, case.gold_sql, tables=base_tables)
        e = run_query(exp, case.gold_sql, tables=exp_tables)
        assert b.ok and e.ok, f"{case.id}: gold_sql failed"
        assert b.rows == e.rows, f"{case.id}: confounders changed the gold result"
