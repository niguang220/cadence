"""Value channel wired into run_retrieval: value_ablation is executable, value evidence enters
the RRF candidate set, a backend error degrades (typed) instead of vanishing, and only a
high-confidence (exact) value hit can admit an otherwise off-topic question."""
import pytest

from agent.db.introspect import Column, Table
from agent.retrieval.contracts import RetrievalConfig, UnsupportedRetrievalCapability
from agent.retrieval.pipeline import run_retrieval
from agent.retrieval.value_backend import FakeValueBackend, ValueBackendError, ValueDoc


def _vtables():
    return [
        Table(name="product", columns=[
            Column(name="product_id", type="INTEGER", pk=True, notnull=True),
            Column(name="name", type="TEXT", pk=False, notnull=False, policy="searchable")]),
        Table(name="orders", columns=[
            Column(name="order_id", type="INTEGER", pk=True, notnull=True),
            Column(name="qty", type="INTEGER", pk=False, notnull=False)]),
    ]


def _backend(value="Widget"):
    b = FakeValueBackend()
    b.ensure_index()
    b.upsert([ValueDoc("d1", "product", "name", value)])
    return b


def test_value_ablation_is_executable():
    res = run_retrieval("do we sell Widget", _vtables(), RetrievalConfig.value_ablation(),
                        k=5, value_backend=_backend())
    assert res.config_name == "value_ablation"          # no UnsupportedRetrievalCapability


def test_full_rag_still_unsupported_on_llm_only():
    with pytest.raises(UnsupportedRetrievalCapability) as ei:
        run_retrieval("q", _vtables(), RetrievalConfig.full_rag(), k=5)
    assert set(ei.value.capabilities) == {"llm"}         # es now supported; llm still not


def test_value_hit_enters_candidates_and_signals():
    res = run_retrieval("do we sell Widget", _vtables(), RetrievalConfig.value_ablation(),
                        k=5, value_backend=_backend())
    assert "product" in [c.table for c in res.candidates]
    assert any(s.channel == "value" and s.match_type == "exact_keyword" for s in res.signals)


def test_value_backend_error_becomes_value_degraded_lexical_survives():
    class Boom:
        def search(self, *a, **k):
            raise ValueBackendError("es down")

    res = run_retrieval("how many product", _vtables(), RetrievalConfig.value_ablation(),
                        k=5, value_backend=Boom())
    assert any(e.stage == "channel" and e.event == "value_degraded" for e in res.stage_events)
    assert res.candidates                                 # lexical (table word 'product') survives


def test_high_confidence_value_admits_offtopic():
    # no table/column word, no metric -> only an exact value hit can admit
    res = run_retrieval("Widget", _vtables(), RetrievalConfig.value_ablation(),
                        k=5, value_backend=_backend())
    assert res.candidates
    assert not any(e.event == "admission_rejected" for e in res.stage_events)


def test_token_or_fuzzy_value_alone_does_not_admit():
    # value 'Acme Widget' vs question 'widget' -> token_match only (not exact whole-value)
    res = run_retrieval("widget", _vtables(), RetrievalConfig.value_ablation(),
                        k=5, value_backend=_backend("Acme Widget"))
    assert res.candidates == []
    assert any(e.event == "admission_rejected" for e in res.stage_events)


def test_value_query_routes_only_to_value_channel_and_records_provenance():
    # the value is in value_query (the original question), NOT in the enhanced `question`
    res = run_retrieval("some unrelated enhanced rewrite", _vtables(),
                        RetrievalConfig.value_ablation(), k=5, value_backend=_backend("Widget"),
                        value_query="do we sell Widget")
    val = [s for s in res.signals if s.channel == "value"]
    assert val and val[0].table == "product" and val[0].match_type == "exact_keyword"
    assert val[0].query_term == "do we sell Widget"       # provenance = the value_query actually used


def test_value_query_none_falls_back_to_question():
    res = run_retrieval("do we sell Widget", _vtables(), RetrievalConfig.value_ablation(),
                        k=5, value_backend=_backend("Widget"), value_query=None)
    val = [s for s in res.signals if s.channel == "value"]
    assert val and val[0].query_term == "do we sell Widget"


def test_enhanced_only_entity_never_reaches_value_channel():
    # `question` (enhanced) fabricates 'Gadget'; value_query (original) does not mention it, so the
    # indexed 'Gadget' must produce NO value signal (a model-introduced entity cannot admit).
    res = run_retrieval("do we sell Gadget", _vtables(), RetrievalConfig.value_ablation(),
                        k=5, value_backend=_backend("Gadget"), value_query="how many things")
    assert not [s for s in res.signals if s.channel == "value"]
