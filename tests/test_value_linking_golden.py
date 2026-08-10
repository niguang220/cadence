"""value_linking golden self-check: the set loads under the strict loader, and it is coherent with
the fixture — positive cases' seeded values link to their expected owner table (deterministic fake
backend), negatives produce no admitting value hit and never touch a PII column."""
from agent.db.build_value_db import build
from agent.db.introspect import introspect
from agent.retrieval.channels import ValueChannel
from agent.retrieval.value_backend import FakeValueBackend
from agent.retrieval.value_index import build_value_index
from evalharness.golden import load_value_linking

_POSITIVE = {"en", "zh", "fuzzy"}
_HIGH_CONF = {"exact_keyword", "exact_phrase"}


def _setup(tmp_path):
    db = build(tmp_path / "v.db")
    tables = introspect(db)
    b = FakeValueBackend()
    build_value_index(tables, db, b)
    return tables, b


def test_golden_loads_all_categories():
    cases = load_value_linking()
    assert len(cases) == 8
    assert {c.category for c in cases} == {"en", "zh", "fuzzy", "no_hit", "pii", "off_topic"}


def test_positive_values_link_to_expected_owner(tmp_path):
    tables, b = _setup(tmp_path)
    for c in load_value_linking():
        if c.category not in _POSITIVE:
            continue
        owners = {s.table for s in ValueChannel(b).signals(c.question, tables)}
        assert owners & set(c.required_tables), f"{c.id}: linked {owners}, need {c.required_tables}"


def test_fuzzy_case_actually_uses_the_fuzzy_tier(tmp_path):
    tables, b = _setup(tmp_path)
    case = next(c for c in load_value_linking() if c.category == "fuzzy")
    sigs = ValueChannel(b).signals(case.question, tables)
    assert any(s.table == "company" and s.match_type == "fuzzy" for s in sigs)


def test_negative_cases_have_no_admitting_hit_and_no_pii(tmp_path):
    tables, b = _setup(tmp_path)
    for c in load_value_linking():
        if c.category in _POSITIVE:
            continue
        sigs = ValueChannel(b).signals(c.question, tables)
        assert not any(s.match_type in _HIGH_CONF for s in sigs), f"{c.id}: admitting value hit"
        assert all(s.column not in {"email", "full_name", "phone"} for s in sigs), f"{c.id}: PII"
