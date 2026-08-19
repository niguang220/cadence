import pytest

from agent.db.build_demo_db import build
from agent.db.introspect import introspect
from agent.semantic_layer import MetricRegistry, MetricRetrievalHit
from agent.retrieval.backends import DenseBackendError
from agent.retrieval.contracts import RetrievalConfig, UnsupportedRetrievalCapability
from agent.retrieval.pipeline import run_retrieval


def _tables(tmp_path):
    return introspect(build(tmp_path / "t.db"))


def _saas_tables(tmp_path):
    from agent.db.build_saas_db import build as build_saas
    return introspect(build_saas(tmp_path / "s.db"))


class _FakeDense:              # deterministic, no fastembed
    def __init__(self, rows): self._rows = rows
    def column_scores(self, q, tables): return self._rows


class _RaisingDense:
    def column_scores(self, q, tables): raise DenseBackendError("model down")


def _mrr_hit():
    mrr = next(m for m in MetricRegistry.load().metrics if m.name == "mrr")
    return MetricRetrievalHit(mrr, "alias", 1.0)


def test_collects_all_missing_capabilities(tmp_path):
    # es is now built (value backend); full_rag's only unbuilt capability is the llm selector.
    with pytest.raises(UnsupportedRetrievalCapability) as ei:
        run_retrieval("q", _tables(tmp_path), RetrievalConfig.full_rag(), k=5)
    assert set(ei.value.capabilities) == {"llm"}
    # collect-ALL (don't stop at the first) still holds for a config with two unbuilt capabilities
    cfg = RetrievalConfig(name="qdrant_llm", lexical=True, dense_backend="qdrant",
                          value_backend=None, selector="llm", fusion="rrf",
                          relation_strategy="shortest_path")
    with pytest.raises(UnsupportedRetrievalCapability) as ei2:
        run_retrieval("q", _tables(tmp_path), cfg, k=5)
    assert set(ei2.value.capabilities) == {"qdrant", "llm"}


def test_admission_rejects_offtopic_dense_only(tmp_path):
    # off-topic: no lexical footing, no metric; even with dense scores -> reject (G2)
    res = run_retrieval("xyzzy quux frobnicate", _tables(tmp_path), RetrievalConfig.rrf_hybrid(),
                        k=5, dense_backend=_FakeDense([("track", "name", 0.9)]))
    assert res.candidates == []
    assert any(e.event == "admission_rejected" for e in res.stage_events)


def test_lexical_false_skips_lexical_channel(tmp_path):
    cfg = RetrievalConfig(name="dense_only", lexical=False, dense_backend="memory",
                          value_backend=None, selector=None, fusion="rrf",
                          relation_strategy="shortest_path")
    res = run_retrieval("how many tracks", _tables(tmp_path), cfg, k=5,
                        dense_backend=_FakeDense([("track", "name", 0.9)]))
    assert not any(s.channel == "lexical" for s in res.signals)   # lexical channel did not run
    assert res.candidates == []                                    # dense-only -> admission rejects


def test_dense_error_becomes_stage_event_and_lexical_survives(tmp_path):
    res = run_retrieval("how many tracks are in each genre", _tables(tmp_path),
                        RetrievalConfig.rrf_hybrid(), k=5, dense_backend=_RaisingDense())
    assert any(e.event == "dense_degraded" for e in res.stage_events)
    assert res.candidates                                          # lexical still produced candidates


def test_rrf_protects_metric_required_tables(tmp_path):
    # mrr's required_tables (account, subscription) live in the SaaS catalog, not the demo one —
    # MetricMatchProvider validates hits against the passed-in catalog, so this must run on the
    # saas DB (the demo DB would fail fast with an unknown-table ValueError before the pipeline
    # even gets to the RRF/selection logic being tested here).
    res = run_retrieval("how many active subscriptions", _saas_tables(tmp_path),
                        RetrievalConfig.rrf_hybrid(), k=5, metric_hits=[_mrr_hit()],
                        dense_backend=_FakeDense([]))
    assert {"account", "subscription"} <= set(res.selection.anchor_tables)   # protected, from mrr


def test_runtime_k_controls_selected_anchor_count(tmp_path):
    tables = _tables(tmp_path)
    res = run_retrieval("tracks invoices customers genres", tables,
                        RetrievalConfig.lexical_baseline(), k=2)
    assert len(res.selection.anchor_tables) <= 2
