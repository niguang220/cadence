"""Governed metric binding precision.

Contract: only ``match_type="alias"`` hits BIND -- they drive the MUST-apply prompt metrics AND
become protected anchors. ``match_type="dense"`` is discovery/telemetry only: a fuzzy similarity
must never inject MUST filters or protect required_tables over a plain structural query.

Alias matching is tested deterministically with a zero-vector embed stub so the dense channel
never fires (no fastembed). The preflight-boundary tests use the real registry to prove dense
hits are dropped from binding even when they DO fire.
"""
import pytest

import agent.graph as graph
from agent.db.build_saas_db import build
from agent.db.introspect import introspect
from agent.graph import _metric_registry, _preflight_context, _semantic_metrics
from agent.retrieval.metric_match import MetricMatchProvider, deserialize_hits
from agent.retrieval.selector import protected_anchors
from agent.semantic_layer import MetricRegistry, MetricRetrievalHit
from evalharness.golden import load_saas_metrics

_CASES = load_saas_metrics()
_CONTROLS = [c for c in _CASES if c.category == "control"]
_TRAPS = [c for c in _CASES if c.category != "control"]


def _zero_embed(texts):
    # cos() with a zero vector is 0.0 < any positive threshold -> the dense pass adds nothing.
    return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


def _alias_metric_names(question):
    hits = MetricRegistry.load(embed=_zero_embed).retrieve_matches(question, threshold=0.5)
    assert all(h.match_type == "alias" for h in hits)   # zero-embed guarantees no dense hit
    return sorted({h.metric.name for h in hits})


def _tables(tmp_path):
    return introspect(build(tmp_path / "s.db"))


def _state(tables, question, **kw):
    return {"question": question, "db_path": "unused", "tables": tables,
            "semantic_layer": True, "threshold": 0.5, **kw}


# --- requirement 2: every trap's expected governed metric is reachable by a deterministic alias
@pytest.mark.parametrize("case", _TRAPS, ids=[c.id for c in _TRAPS])
def test_trap_expected_metric_binds_via_alias(case):
    assert case.metric in _alias_metric_names(case.question)


# --- requirement 1 (alias layer): no control question matches any governed-metric alias
@pytest.mark.parametrize("case", _CONTROLS, ids=[c.id for c in _CONTROLS])
def test_control_matches_no_alias(case):
    assert _alias_metric_names(case.question) == []


# --- requirement 1 (binding boundary, real registry incl. dense): controls bind NOTHING
def test_preflight_controls_bind_no_metric_even_if_dense_fires(tmp_path):
    tables = _tables(tmp_path)
    for c in _CONTROLS:
        out = _preflight_context(_state(tables, c.question))
        assert [m["name"] for m in out["semantic_metrics"]] == [], f"{c.id} bound prompt metrics"
        assert out["semantic_metric_hits"] == [], f"{c.id} produced pipeline hits/anchors"


# --- requirement 3: a dense hit does NOT enter prompt-bound metrics; alias does; dense -> telemetry
def test_preflight_filters_dense_from_binding_keeps_telemetry(tmp_path, monkeypatch):
    tables = _tables(tmp_path)
    reg = _metric_registry()
    mrr = next(m for m in reg.metrics if m.name == "mrr")
    billed = next(m for m in reg.metrics if m.name == "billed_revenue")
    mixed = [MetricRetrievalHit(mrr, "alias", 1.0), MetricRetrievalHit(billed, "dense", 0.71)]
    monkeypatch.setattr(reg, "retrieve_matches", lambda *a, **k: mixed)
    out = _preflight_context(_state(tables, "irrelevant question text"))
    assert [m["name"] for m in out["semantic_metrics"]] == ["mrr"]
    assert [h["match_type"] for h in out["semantic_metric_hits"]] == ["alias"]
    assert [h["metric"]["name"] for h in out["semantic_metric_hits"]] == ["mrr"]
    # dense retained as discovery/telemetry only (metric/match_type/score), no second retrieval
    assert out["trace"][0]["dense_discovery"] == [
        {"metric": "billed_revenue", "match_type": "dense", "score": 0.71}]


# --- requirement 4: a dense-only hit produces NO protected anchor
def test_dense_only_hit_produces_no_protected_anchor(tmp_path, monkeypatch):
    tables = _tables(tmp_path)
    reg = _metric_registry()
    billed = next(m for m in reg.metrics if m.name == "billed_revenue")
    monkeypatch.setattr(reg, "retrieve_matches",
                        lambda *a, **k: [MetricRetrievalHit(billed, "dense", 0.9)])
    out = _preflight_context(_state(tables, "how many invoices were issued in 2025"))
    assert out["semantic_metric_hits"] == []
    hits = deserialize_hits(out["semantic_metric_hits"])
    assert protected_anchors(MetricMatchProvider(tables).from_hits(hits)) == []


# --- requirement 5: an alias hit still produces a protected anchor (required_tables protected)
def test_alias_hit_produces_protected_anchor(tmp_path, monkeypatch):
    tables = _tables(tmp_path)
    reg = _metric_registry()
    mrr = next(m for m in reg.metrics if m.name == "mrr")
    monkeypatch.setattr(reg, "retrieve_matches", lambda *a, **k: [MetricRetrievalHit(mrr, "alias", 1.0)])
    out = _preflight_context(_state(tables, "what is our mrr"))
    hits = deserialize_hits(out["semantic_metric_hits"])
    anchors = protected_anchors(MetricMatchProvider(tables).from_hits(hits))
    assert set(mrr.required_tables) <= set(anchors)


# --- requirement 6: preflight retrieves exactly once (single-retrieval ownership unchanged)
def test_preflight_retrieves_exactly_once(tmp_path, monkeypatch):
    tables = _tables(tmp_path)
    reg = _metric_registry()
    calls = {"n": 0}
    orig = reg.retrieve_matches

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(reg, "retrieve_matches", counting)
    _preflight_context(_state(tables, "what is our mrr"))
    assert calls["n"] == 1


# --- fallback path (_semantic_metrics without preflight) is alias-only too, for consistency
def test_fallback_semantic_metrics_is_alias_only(monkeypatch):
    reg = _metric_registry()
    billed = next(m for m in reg.metrics if m.name == "billed_revenue")
    monkeypatch.setattr(reg, "retrieve_matches",
                        lambda *a, **k: [MetricRetrievalHit(billed, "dense", 0.9)])
    # no "semantic_metrics" key -> fallback path; dense-only -> nothing bound
    assert _semantic_metrics({"question": "q", "semantic_layer": True, "threshold": 0.5}) == []
