"""Tests for the deterministic FeasibilityAssessment gate.

The ONLY reliable deterministic signal is "did retrieval find any relevant
tables" -- the semantic retriever already judges "is this about our data"
better than lexical word-matching could. This gate refuses only on empty
recall and traces other signals (e.g. a missing direct join edge) as risk,
never as a refusal. Zero LLM.
"""
from agent.feasibility import assess_feasibility
from agent.db.build_saas_db import build
from agent.db.introspect import introspect


def test_refuses_when_no_recalled_tables(tmp_path):
    tables = introspect(build(tmp_path / "s.db"))
    v = assess_feasibility("anything", tables, [], [], [])
    assert not v.feasible and v.reason_code == "no_recalled_tables"


def test_normal_question_with_recalled_tables_is_feasible(tmp_path):
    # no brittle word-matching: any non-empty recall is on-topic -> feasible.
    tables = introspect(build(tmp_path / "s.db"))
    names = [t.name for t in tables]
    v = assess_feasibility("how many rows are there", tables, names, [], [])
    assert v.feasible


def test_many_recalled_no_direct_path_traces_risk_not_refusal(tmp_path):
    tables = introspect(build(tmp_path / "s.db"))
    names = [t.name for t in tables]
    v = assess_feasibility("q", tables, names, [], [])   # >1 recalled, paths=[]
    assert v.feasible and "possible_missing_join" in v.risks
