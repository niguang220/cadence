"""Per-channel RRF weights as explicit, serialisable configuration.

Weighted RRF always supported a per-channel weight, but the pipeline never set one, so every
channel was implicitly 1.0 and the knob was unreachable from production. These tests pin the
weights as real config: they reach weighted_rrf, they round-trip through serde, they change
ranking, and -- critically -- they cannot switch the admission gate off, because admission is
decided before fusion from channel PRESENCE, not from fused score.
"""
from __future__ import annotations

import pytest

from agent.db.introspect import introspect
from agent.retrieval.contracts import ChannelTableResult, RetrievalConfig, RetrievalSignal
from agent.retrieval.fusion import weighted_rrf
from agent.retrieval.pipeline import run_retrieval
from agent.retrieval.serde import deserialize_config, serialize_config


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    from agent.db.build_saas_db import build
    return introspect(build(tmp_path_factory.mktemp("w") / "saas.db"))


def test_weights_default_to_one_for_every_channel():
    c = RetrievalConfig.default()
    assert c.lexical_weight == 1.0 and c.dense_weight == 1.0


def test_weights_round_trip_through_serde():
    c = RetrievalConfig.default().with_weights(lexical=0.25, dense=1.0)
    assert deserialize_config(serialize_config(c)) == c
    assert serialize_config(c)["lexical_weight"] == 0.25


def test_with_weights_returns_a_new_frozen_config():
    base = RetrievalConfig.default()
    tuned = base.with_weights(lexical=0.5, dense=1.0)
    assert base.lexical_weight == 1.0                 # original untouched
    assert tuned.lexical_weight == 0.5
    assert tuned.name == base.name and tuned.fusion == base.fusion


def test_lexical_weight_must_stay_positive():
    """Weight 0 is dense-only ranking, not hybrid fusion. It is rejected as configuration."""
    with pytest.raises(ValueError):
        RetrievalConfig.default().with_weights(lexical=0.0, dense=1.0)
    with pytest.raises(ValueError):
        RetrievalConfig.default().with_weights(lexical=-1.0, dense=1.0)
    with pytest.raises(ValueError):
        RetrievalConfig.default().with_weights(lexical=1.0, dense=0.0)


def test_weighted_rrf_actually_consumes_the_weights():
    def ctr(channel, table, rank):
        return ChannelTableResult(channel=channel, table=table, raw_table_score=1.0,
                                  channel_rank=rank, signals=[])
    # lexical ranks A first, dense ranks B first; the heavier channel must win
    results = {"lexical": [ctr("lexical", "A", 1), ctr("lexical", "B", 2)],
               "dense": [ctr("dense", "B", 1), ctr("dense", "A", 2)]}
    lex_heavy = weighted_rrf(results, weights={"lexical": 1.0, "dense": 0.25})
    dense_heavy = weighted_rrf(results, weights={"lexical": 0.25, "dense": 1.0})
    assert lex_heavy[0].table == "A"
    assert dense_heavy[0].table == "B"


def test_pipeline_threads_config_weights_into_fusion(tables, monkeypatch):
    seen = {}
    import agent.retrieval.pipeline as pipeline
    real = pipeline.weighted_rrf

    def spy(channel_results, **kwargs):
        seen.update(kwargs)
        return real(channel_results, **kwargs)

    monkeypatch.setattr(pipeline, "weighted_rrf", spy)
    cfg = RetrievalConfig.default().with_weights(lexical=0.25, dense=1.0)
    run_retrieval("how many invoices", tables, cfg, k=5)
    assert seen["weights"] == {"lexical": 0.25, "dense": 1.0, "value": 1.0}


def test_weights_cannot_disable_the_offtopic_refusal(tables):
    """Admission is decided BEFORE fusion from channel presence, so no weight setting can let
    an off-topic question through. This is what makes the weight knob safe to tune."""
    for lexical_weight in (0.25, 0.5, 1.0):
        cfg = RetrievalConfig.default().with_weights(lexical=lexical_weight, dense=1.0)
        result = run_retrieval("what is the weather in singapore today", tables, cfg, k=5)
        assert result.candidates == []
        assert any(e.event == "admission_rejected" for e in result.stage_events)
