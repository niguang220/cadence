"""ValueBackend contract + FakeValueBackend (no ES).

The fake is a deterministic in-memory stand-in so the channel/pipeline/ingestion logic is fully
tested without Docker: it produces the same four match types the ES backend will
(exact_keyword > exact_phrase > token_match > fuzzy) and restricts search to the allowlist."""
from agent.retrieval.value_backend import (FakeValueBackend, ValueBackendError, ValueDoc,
                                            ValueHit)


def _docs():
    return [
        ValueDoc(document_id="d1", table="account", column="company_name", value="Acme Corp"),
        ValueDoc(document_id="d2", table="product", column="name", value="Widget"),
        ValueDoc(document_id="d3", table="contract", column="external_id", value="CT-2025-0042"),
        ValueDoc(document_id="d4", table="account", column="contact_email", value="a@x.com"),
    ]


def _built():
    b = FakeValueBackend()
    b.ensure_index()
    b.upsert(_docs())
    return b


ALLOWED = [("account", "company_name"), ("product", "name"), ("contract", "external_id")]


def test_exact_keyword_single_token_value():
    hits = _built().search("do we sell Widget anywhere", allowed=ALLOWED)
    h = next(h for h in hits if h.value == "Widget")
    assert h.match_type == "exact_keyword" and h.table == "product" and h.document_id == "d2"


def test_exact_phrase_multi_token_value():
    hits = _built().search("how many invoices for Acme Corp", allowed=ALLOWED)
    h = next(h for h in hits if h.value == "Acme Corp")
    assert h.match_type == "exact_phrase" and h.column == "company_name"


def test_token_match_partial_overlap():
    hits = _built().search("show me the acme account", allowed=ALLOWED)
    h = next(h for h in hits if h.value == "Acme Corp")
    assert h.match_type == "token_match" and 0 < h.score < 1.0


def test_fuzzy_near_spelling():
    hits = _built().search("orders for Widgett", allowed=ALLOWED)   # one extra char
    h = next(h for h in hits if h.value == "Widget")
    assert h.match_type == "fuzzy"


def test_search_is_restricted_to_allowlist():
    # 'a@x.com' lives in a pii column not in ALLOWED -> it must never be searched/returned.
    hits = _built().search("email a@x.com", allowed=ALLOWED)
    assert all(h.column != "contact_email" for h in hits)


def test_hits_bucket_ranked_exact_over_fuzzy():
    b = _built()
    hits = b.search("Widget", allowed=ALLOWED)   # exact_keyword on product.name
    assert hits and hits[0].match_type == "exact_keyword"


def test_prune_removes_stale_docs():
    b = _built()
    b.prune({"d2"})                                  # keep only Widget
    hits = b.search("Acme Corp Widget", allowed=ALLOWED)
    assert {h.value for h in hits} == {"Widget"}


def test_upsert_is_idempotent_by_document_id():
    b = _built()
    b.upsert([ValueDoc(document_id="d2", table="product", column="name", value="Widget")])
    hits = b.search("Widget", allowed=ALLOWED)
    assert len([h for h in hits if h.value == "Widget"]) == 1


def test_backend_error_type_exists():
    assert issubclass(ValueBackendError, Exception)
    assert isinstance(ValueHit("t", "c", "v", "exact_keyword", 1.0, "d"), ValueHit)
