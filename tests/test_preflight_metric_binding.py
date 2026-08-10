import tempfile
from pathlib import Path

from agent.db.build_saas_db import build
from agent.db.introspect import introspect
import agent.graph as graph
from agent.graph import _preflight_context, _metric_registry


def _tables(tmp_path):
    return introspect(build(tmp_path / "s.db"))


def _state(tables, question, semantic_layer):
    return {"question": question, "db_path": "unused", "tables": tables,
            "semantic_layer": semantic_layer, "threshold": 0.5}


def test_binds_typed_hits_and_keeps_semantic_metrics(tmp_path):
    out = _preflight_context(_state(_tables(tmp_path), "what is our mrr", True))
    hits = out["semantic_metric_hits"]
    assert hits and all("match_type" in h and "score" in h for h in hits)
    assert all("required_tables" in h["metric"] and "required_columns" in h["metric"] for h in hits)
    # prompt-facing metrics unchanged: names line up with the hits' metrics
    assert [m["name"] for m in out["semantic_metrics"]] == [h["metric"]["name"] for h in hits]


def test_no_hits_when_semantic_off(tmp_path):
    out = _preflight_context(_state(_tables(tmp_path), "what is our mrr", False))
    assert out["semantic_metric_hits"] == [] and out["semantic_metrics"] == []


def test_retrieval_happens_exactly_once(tmp_path, monkeypatch):
    reg = _metric_registry()
    calls = {"n": 0}
    orig = reg.retrieve_matches
    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    monkeypatch.setattr(reg, "retrieve_matches", counting)
    _preflight_context(_state(_tables(tmp_path), "what is our mrr", True))
    assert calls["n"] == 1
