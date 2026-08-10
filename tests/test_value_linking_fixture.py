"""Stage 2 value-linking fixture self-check: > candidate_k tables (non-saturated), a correct
searchable/PII policy split, and ingestion that indexes searchable values (incl. Chinese) while
never indexing or returning PII. Deterministic, no ES."""
from agent.db.build_value_db import build
from agent.db.introspect import introspect
from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.value_backend import FakeValueBackend
from agent.retrieval.value_index import build_value_index
from agent.retrieval.value_policy import resolve_searchable_columns


def _setup(tmp_path):
    db = build(tmp_path / "value.db")
    return introspect(db), db


def test_fixture_is_not_candidate_saturated(tmp_path):
    tables, _ = _setup(tmp_path)
    assert len(tables) == 16
    assert len(tables) > RetrievalConfig.rrf_hybrid().candidate_k    # > 15: breaks Stage-1 saturation


def test_searchable_allowlist_and_pii_split(tmp_path):
    tables, _ = _setup(tmp_path)
    allow = set(resolve_searchable_columns(tables))
    assert {("company", "company_name"), ("catalog", "name"), ("catalog", "sku"),
            ("agreement", "external_id"), ("deal", "name"), ("vendor", "name"),
            ("plan_tier", "name"), ("shipment", "tracking_no"), ("campaign", "name")} == allow
    for pii in [("person", "email"), ("person", "full_name"), ("person", "phone"),
                ("sales_rep", "email"), ("sales_rep", "full_name")]:
        assert pii not in allow


def test_ingestion_indexes_searchable_not_pii(tmp_path):
    tables, db = _setup(tmp_path)
    b = FakeValueBackend()
    build_value_index(tables, db, b)
    allow = resolve_searchable_columns(tables)
    hits = b.search("how many open tickets for Globex Corporation", allowed=allow)
    assert any(h.value == "Globex Corporation" and h.table == "company" for h in hits)
    # PII value never indexed / returned
    pii = b.search("john.smith@globex.com", allowed=allow)
    assert all(h.column not in ("email", "full_name", "phone") for h in pii)
    assert all("@" not in h.value for h in pii)


def test_chinese_company_name_is_searchable(tmp_path):
    tables, db = _setup(tmp_path)
    b = FakeValueBackend()
    build_value_index(tables, db, b)
    hits = b.search("北京数据科技有限公司 有多少合同", allowed=resolve_searchable_columns(tables))
    assert any(h.value == "北京数据科技有限公司" and h.table == "company" for h in hits)
