"""Real Elasticsearch value backend (opt-in tier).

Skipped unless CADENCE_ES_URL points at a running ES (the es-integration CI job, or a local Docker
ES via docker-compose.es.yml). The channel/pipeline/ingestion logic is fully covered elsewhere by
the FakeValueBackend; these tests prove the ES wire behavior and fake<->ES parity."""
import pytest

from agent.retrieval.value_backend import FakeValueBackend, ValueDoc

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


def test_fake_and_es_agree_on_owner_and_tier(es_backend):
    _reindex(es_backend)
    fake = FakeValueBackend()
    fake.ensure_index()
    fake.upsert(_DOCS)
    q = "orders for Acme Corp and a Widget"

    def norm(hits):
        return sorted((h.table, h.column, h.value, h.match_type) for h in hits)

    assert norm(es_backend.search(q, allowed=ALLOW)) == norm(fake.search(q, allowed=ALLOW))
