"""Real Elasticsearch value backend (opt-in tier).

Skipped unless CADENCE_ES_URL points at a running ES (the es-integration CI job, or a local Docker
ES via docker-compose.es.yml). The channel/pipeline/ingestion logic is fully covered elsewhere by
the FakeValueBackend; these tests prove the ES wire behavior and fake<->ES parity."""
import pytest

from agent.retrieval.value_backend import (ElasticsearchValueBackend, FakeValueBackend,
                                            ValueBackendError, ValueDoc)

pytestmark = pytest.mark.es_integration

_DOCS = [
    ValueDoc("d1", "product", "name", "Widget"),
    ValueDoc("d2", "account", "company_name", "Acme Corp"),
    ValueDoc("d3", "account", "contact_email", "ceo@acme.com"),   # pii column, not in allowlist
]
ALLOW = [("product", "name"), ("account", "company_name")]


def _reindex(backend):
    backend.upsert(_DOCS)
    backend.prune({d.document_id for d in _DOCS})


def test_es_index_preflight(es_backend):
    assert es_backend.index_exists() is True


def test_es_exact_keyword_hit(es_backend):
    _reindex(es_backend)
    hits = es_backend.search("do we sell Widget", allowed=ALLOW)
    assert any(h.value == "Widget" and h.match_type == "exact_keyword" for h in hits)


def test_es_fuzzy_near_spelling(es_backend):
    _reindex(es_backend)
    hits = es_backend.search("orders for Widgett", allowed=ALLOW)
    assert any(h.value == "Widget" and h.match_type == "fuzzy" for h in hits)


def test_es_allowlist_excludes_pii_column(es_backend):
    es_backend.upsert(_DOCS)                                  # d3 is a pii column, indexed here but...
    hits = es_backend.search("ceo@acme.com Acme Corp", allowed=ALLOW)
    assert all(h.column != "contact_email" for h in hits)    # ...never searched (not in allowlist)


def test_es_prune_removes_stale(es_backend):
    es_backend.upsert(_DOCS)
    es_backend.prune({"d1"})
    hits = es_backend.search("Acme Corp Widget", allowed=ALLOW)
    assert {h.value for h in hits} == {"Widget"}


def test_es_upsert_raises_on_partial_bulk_failure(es_backend):
    """A bulk where SOME items index and SOME are rejected must fail loud. We force a deterministic
    per-item rejection with a test-only incompatible mapping (value as a long): a numeric value
    coerces and indexes, a text value is rejected -> errors=true with a mixed items list. The old
    implementation swallowed this (bulk returned errors=true but never raised)."""
    es = es_backend._es
    idx = "cadence-values-itest-partial"
    es.indices.delete(index=idx, ignore_unavailable=True)
    es.indices.create(index=idx, mappings={"properties": {
        "table": {"type": "keyword"}, "column": {"type": "keyword"},
        "value": {"type": "long"}}})                              # incompatible on purpose
    backend = ElasticsearchValueBackend(es, idx)
    docs = [ValueDoc("ok", "product", "name", "123"),             # coerces to long -> indexes
            ValueDoc("bad", "product", "name", "Widget")]         # not numeric -> rejected
    try:
        with pytest.raises(ValueBackendError) as ei:
            backend.upsert(docs)
        msg = str(ei.value)
        assert "upsert" in msg and "1/2" in msg                   # op type + failed/total
        assert "Widget" not in msg and "product" not in msg       # no raw value / document source
        assert "mapper_parsing" not in msg and "reason" not in msg  # no full ES error reason
    finally:
        es.indices.delete(index=idx, ignore_unavailable=True)


def test_fake_es_parity_on_merged_cjk_value(es_backend):
    """A long CJK company value merged into the question (no whitespace — the shape query_enhance
    produced) must resolve to the SAME owner/tier/rank on real ES and the fake: ES recalls via
    per-character unigrams, then the shared _classify assigns the exact_phrase tier on both."""
    import pathlib
    import tempfile

    from agent.db.build_value_db import build as build_value_db
    from agent.db.introspect import introspect
    from agent.retrieval.value_index import _collect_docs
    from agent.retrieval.value_policy import resolve_searchable_columns
    with tempfile.TemporaryDirectory() as wd:
        db = build_value_db(pathlib.Path(wd) / "v.db")
        allowed = resolve_searchable_columns(introspect(db))
        docs = _collect_docs(db, allowed)
    es_backend.upsert(docs)
    es_backend.prune({d.document_id for d in docs})
    fake = FakeValueBackend()
    fake.upsert(docs)
    q = "上海云图信息技术有限公司有多少个未解决的工单"          # merged, no space (the bad rewrite)

    def owner_hits(hits):                                       # ordered -> compares owner, tier AND rank
        return [(h.value, h.match_type) for h in hits if h.table == "company"]

    es_hits = owner_hits(es_backend.search(q, allowed=allowed))
    assert es_hits == owner_hits(fake.search(q, allowed=allowed))
    assert ("上海云图信息技术", "exact_phrase") in es_hits


def test_fake_and_es_agree_on_owner_and_tier(es_backend):
    _reindex(es_backend)
    fake = FakeValueBackend()
    fake.ensure_index()
    fake.upsert(_DOCS)
    q = "orders for Acme Corp and a Widget"

    def norm(hits):
        return sorted((h.table, h.column, h.value, h.match_type) for h in hits)

    assert norm(es_backend.search(q, allowed=ALLOW)) == norm(fake.search(q, allowed=ALLOW))
