"""value_linking golden self-check: the set loads under the strict loader, and it is coherent with
the fixture — positive cases' seeded values link to their expected owner table (deterministic fake
backend), negatives produce no admitting value hit and never touch a PII column."""
from agent.db.build_value_db import build
from agent.db.introspect import introspect
from agent.execution import run_query
from agent.retrieval.channels import ValueChannel
from agent.retrieval.value_backend import FakeValueBackend
from agent.retrieval.value_index import build_value_index
from evalharness.golden import load_value_linking

_HIGH_CONF = {"exact_keyword", "exact_phrase"}


def _setup(tmp_path):
    db = build(tmp_path / "v.db")
    tables = introspect(db)
    b = FakeValueBackend()
    build_value_index(tables, db, b)
    return tables, b


def test_golden_shape_by_role_and_category():
    from collections import Counter
    roles = Counter(c.role for c in load_value_linking())
    assert roles["primary"] >= 12 and roles["negative"] >= 6
    assert roles["diagnostic"] == 1 and roles["safety"] == 2
    prim_cats = {c.category for c in load_value_linking() if c.role == "primary"}
    assert {"en", "zh", "code", "homonym"} <= prim_cats           # >= 4 primary categories


def test_primary_values_link_to_expected_owner_table(tmp_path):
    tables, b = _setup(tmp_path)
    for c in load_value_linking():
        if c.role != "primary":
            continue
        owners = {s.table for s in ValueChannel(b).signals(c.question, tables)}
        assert c.expected_table in owners, f"{c.id}: linked {owners}, expected {c.expected_table}"


def test_fuzzy_diagnostic_uses_the_fuzzy_tier(tmp_path):
    tables, b = _setup(tmp_path)
    case = next(c for c in load_value_linking() if c.role == "diagnostic")
    sigs = ValueChannel(b).signals(case.question, tables)
    assert any(s.table == "company" and s.match_type == "fuzzy" for s in sigs)


def test_negative_cases_have_no_admitting_hit_and_no_pii(tmp_path):
    tables, b = _setup(tmp_path)
    for c in load_value_linking():
        if c.role != "negative":
            continue
        sigs = ValueChannel(b).signals(c.question, tables)
        assert not any(s.match_type in _HIGH_CONF for s in sigs), f"{c.id}: admitting value hit"
        assert all(s.column not in {"email", "full_name", "phone"} for s in sigs), f"{c.id}: PII"


def test_primary_and_diagnostic_gold_sql_run_nonempty(tmp_path):
    db = build(tmp_path / "v.db")
    tables = introspect(db)
    for c in load_value_linking():
        if c.role in ("primary", "diagnostic"):
            res = run_query(db, c.gold_sql, tables=tables)
            assert res.ok and res.rows, f"{c.id}: gold_sql failed or empty ({res.error})"
