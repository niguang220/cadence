import tempfile
from pathlib import Path

import pytest

from agent.db.build_saas_db import build
from agent.db.introspect import introspect
from agent.semantic_layer import MetricDef, MetricRegistry, MetricRetrievalHit
from agent.retrieval.metric_match import (MetricMatchProvider, deserialize_hits,
                                          serialize_hits, validate_all_metrics)


def _tables(tmp_path):
    return introspect(build(tmp_path / "s.db"))


def _bad_metric():
    return MetricDef("x", ["x"], "d", "m", "g", [], "c",
                     required_tables=["nope"], required_columns=["nope.c"])


def test_provider_fails_fast_on_unknown_table(tmp_path):
    hit = MetricRetrievalHit(_bad_metric(), "alias", 1.0)
    with pytest.raises(ValueError, match="nope"):
        MetricMatchProvider(_tables(tmp_path)).from_hits([hit])


def test_from_hits_copies_validated_deps(tmp_path):
    reg = MetricRegistry.load()
    mrr = next(m for m in reg.metrics if m.name == "mrr")
    [mm] = MetricMatchProvider(_tables(tmp_path)).from_hits([MetricRetrievalHit(mrr, "alias", 1.0)])
    assert mm.metric == "mrr" and mm.match_type == "alias" and mm.score == 1.0
    assert set(mm.required_tables) == set(mrr.required_tables)
    assert set(mm.required_columns) == set(mrr.required_columns)


def test_empty_hits_yield_no_matches(tmp_path):
    assert MetricMatchProvider(_tables(tmp_path)).from_hits([]) == []


def test_validate_all_metrics_catches_unmatched_bad_metric(tmp_path):
    reg = MetricRegistry([_bad_metric()])
    with pytest.raises(ValueError, match="nope"):
        validate_all_metrics(reg, _tables(tmp_path))       # performs NO retrieval


def test_real_registry_validates_clean(tmp_path):
    validate_all_metrics(MetricRegistry.load(), _tables(tmp_path))   # must not raise


def test_serialize_roundtrip(tmp_path):
    hits = MetricRegistry.load().retrieve_matches("what is mrr")
    back = deserialize_hits(serialize_hits(hits))
    assert [h.metric.name for h in back] == [h.metric.name for h in hits]
    assert back[0].match_type == hits[0].match_type and back[0].score == hits[0].score
    assert back[0].metric.required_tables == hits[0].metric.required_tables
