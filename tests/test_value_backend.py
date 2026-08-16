"""ValueBackend contract + FakeValueBackend (no ES).

The fake is a deterministic in-memory stand-in so the channel/pipeline/ingestion logic is fully
tested without Docker: it produces the same four match types the ES backend will
(exact_keyword > exact_phrase > token_match > fuzzy) and restricts search to the allowlist."""
import pytest

from agent.retrieval.value_backend import (ElasticsearchValueBackend, FakeValueBackend,
                                            ValueBackendError, ValueDoc, ValueHit,
                                            _bulk_error_summary, _classify)


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


def test_es_index_exists_uses_schema_specific_index():
    class Indices:
        def __init__(self):
            self.seen = None

        def exists(self, *, index):
            self.seen = index
            return False

    class Client:
        indices = Indices()

    backend = ElasticsearchValueBackend(Client(), "cadence-values-test")
    assert backend.index_exists() is False
    assert Client.indices.seen == "cadence-values-test"


# --- long-CJK contiguous-substring high-confidence matching (the query_enhance merge case) --------

def _is_high_conf(res):
    return res is not None and res[0] in ("exact_keyword", "exact_phrase")


def test_cjk_long_value_matches_when_merged_no_space():
    # LLM rewrite merged the name with following chars (no whitespace delimiter)
    v, q = "上海云图信息技术", "上海云图信息技术有限公司有多少个未解决的工单"
    assert _classify(v, q) == ("exact_phrase", 1.0)


def test_cjk_long_value_matches_in_raw_spaced_question():
    v, q = "广州天河数据服务", "广州天河数据服务 有几份合同?"     # space-isolated -> existing exact path
    assert _is_high_conf(_classify(v, q))


def test_cjk_value_matches_when_original_question_has_no_space():
    v, q = "北京数据科技有限公司", "北京数据科技有限公司有几份合同"   # user's own question, no space
    assert _classify(v, q) == ("exact_phrase", 1.0)


def test_cjk_short_value_is_not_high_confidence():
    v, q = "数据", "北京数据科技有限公司有多少份合同"            # 2 CJK common word
    assert not _is_high_conf(_classify(v, q))


def test_cjk_partial_value_is_not_high_confidence():
    v, q = "上海云图信息技术", "上海云图 有多少工单"              # only part of the name present
    assert not _is_high_conf(_classify(v, q))


def test_cjk_scrambled_value_is_not_high_confidence():
    v, q = "上海云图信息技术", "技术信息图云海上 有多少工单"        # same chars, wrong order
    assert not _is_high_conf(_classify(v, q))


def test_cjk_nfkc_fullwidth_normalisation():
    v, q = "ＡＢＣ数据服务", "abc数据服务公司有多少合同"           # fullwidth+CJK vs halfwidth
    assert _classify(v, q) == ("exact_phrase", 1.0)


# --- English tiers must be unchanged (no substring rule leaks into English) ----------------------

def test_english_exact_keyword_unchanged():
    assert _classify("Widget", "do we sell Widget") == ("exact_keyword", 1.0)


def test_english_exact_phrase_unchanged():
    assert _classify("Acme Corp", "invoices for Acme Corp today") == ("exact_phrase", 1.0)


def test_english_token_match_unchanged():
    assert _classify("Acme Corp", "show the acme account")[0] == "token_match"


def test_english_fuzzy_unchanged():
    assert _classify("Widget", "orders for Widgett")[0] == "fuzzy"


def test_english_substring_is_not_a_phrase_hit():
    # 'Wid' is a substring of 'Widget' but English has NO substring rule -> not high-conf
    assert not _is_high_conf(_classify("Wid", "do we sell Widget"))


# --- bulk partial-failure detection (pure function; no ES) --------------------------------------

def test_bulk_error_summary_none_when_no_errors():
    assert _bulk_error_summary({"errors": False, "items": []}) is None


def test_bulk_error_summary_flags_index_item_failures():
    body = {"errors": True, "items": [
        {"index": {"_id": "a", "status": 201}},
        {"index": {"_id": "b", "status": 400,
                   "error": {"type": "mapper_parsing_exception", "reason": "leaky value here"}}}]}
    assert _bulk_error_summary(body) == (1, 2, [400])


def test_bulk_error_summary_flags_delete_item_failures():
    body = {"errors": True, "items": [
        {"delete": {"_id": "a", "status": 200}},
        {"delete": {"_id": "b", "status": 409,
                    "error": {"type": "version_conflict_engine_exception"}}}]}
    assert _bulk_error_summary(body) == (1, 2, [409])


def test_bulk_error_summary_ignores_not_found_delete():
    # a delete of a missing doc is status 404 with no error key and errors=False -> not a failure
    body = {"errors": False, "items": [{"delete": {"_id": "a", "status": 404, "result": "not_found"}}]}
    assert _bulk_error_summary(body) is None


class _StubES:
    """Minimal stand-in for the ES client: prune's match_all search + a canned bulk response."""

    def __init__(self, search_ids, bulk_body):
        self._search_ids = search_ids
        self._bulk_body = bulk_body
        self.bulk_ops = None

    def search(self, **_kw):
        return {"hits": {"hits": [{"_id": i} for i in self._search_ids]}}

    def bulk(self, *, operations, **_kw):
        self.bulk_ops = operations
        return self._bulk_body


def test_prune_raises_on_delete_bulk_partial_failure():
    stub = _StubES(search_ids=["stale1", "keep1"],
                   bulk_body={"errors": True, "items": [
                       {"delete": {"_id": "stale1", "status": 409, "error": {"type": "x"}}}]})
    backend = ElasticsearchValueBackend(stub, "idx")
    with pytest.raises(ValueBackendError) as ei:
        backend.prune(keep_ids={"keep1"})
    msg = str(ei.value)
    assert "prune" in msg and "1/1" in msg           # op type + failed/total
    assert "stale1" not in msg and "409" in msg      # no doc id; http status allowed


def test_prune_succeeds_when_deletes_clean():
    stub = _StubES(search_ids=["stale1", "keep1"],
                   bulk_body={"errors": False, "items": [{"delete": {"_id": "stale1", "status": 200}}]})
    backend = ElasticsearchValueBackend(stub, "idx")
    backend.prune(keep_ids={"keep1"})                # must not raise
    assert stub.bulk_ops == [{"delete": {"_index": "idx", "_id": "stale1"}}]
