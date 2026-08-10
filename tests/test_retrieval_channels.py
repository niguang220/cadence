"""Tests for retrieval channels (signal production) and the in-memory dense backend.

Service-free: no fastembed. LexicalChannel is exercised against the demo DB (pure
lexical logic, no embeddings). DenseChannel is exercised against fake backends.
InMemoryDenseBackend is exercised with stub EmbeddingEncoders (fixed embed_fn).
"""
from __future__ import annotations

import numpy as np
import pytest

from agent.db.build_demo_db import build
from agent.db.introspect import introspect
from agent.retrieval.backends import DenseBackendError, InMemoryDenseBackend
from agent.retrieval.channels import DenseChannel, LexicalChannel
from agent.retrieval.encoder import EmbeddingEncoder


def _tables(tmp_path):
    return introspect(build(tmp_path / "t.db"))


# ---------------------------------------------------------------------------
# LexicalChannel
# ---------------------------------------------------------------------------

def test_lexical_channel_signals_shape(tmp_path):
    tables = _tables(tmp_path)
    signals = LexicalChannel().signals("which customers are from Singapore", tables)

    assert signals, "expected at least one signal"
    for sig in signals:
        assert sig.channel == "lexical"
        assert sig.table  # non-empty

    # "Singapore" is a sample value of customer.country -> a value-target hit
    # that carries the matched column (not flattened to a generic table hit).
    value_hits = [s for s in signals if s.target_type == "value"]
    assert value_hits
    for hit in value_hits:
        assert hit.column is not None


def test_lexical_channel_name_hit_has_no_column(tmp_path):
    tables = _tables(tmp_path)
    signals = LexicalChannel().signals("list the albums", tables)

    name_hits = [s for s in signals if s.match_type == "name"]
    assert name_hits
    for hit in name_hits:
        assert hit.target_type == "table"
        assert hit.column is None


def test_lexical_channel_has_real_per_term_signal(tmp_path):
    tables = _tables(tmp_path)
    signals = LexicalChannel().signals("total sales by billing country", tables)

    # "billing country" is a phrase-alias hit on invoice; "total" is a real
    # column term hit on invoice.total.
    assert any(s.table == "invoice" and s.query_term == "total" for s in signals)


# ---------------------------------------------------------------------------
# DenseChannel
# ---------------------------------------------------------------------------

class _FakeBackend:
    def __init__(self, hits):
        self._hits = hits

    def column_scores(self, question, tables):
        return self._hits


def test_dense_channel_wraps_backend_hits(tmp_path):
    tables = _tables(tmp_path)
    backend = _FakeBackend([("account", "name", 0.9), ("subscription", "mrr", 0.4)])
    signals = DenseChannel(backend).signals("mrr", tables)

    assert len(signals) == 2
    by_col = {(s.table, s.column): s for s in signals}
    assert by_col[("account", "name")].raw_score == 0.9
    assert by_col[("subscription", "mrr")].raw_score == 0.4
    for sig in signals:
        assert sig.channel == "dense"
        assert sig.target_type == "column"


class _RaisingBackend:
    def column_scores(self, question, tables):
        raise DenseBackendError("model unavailable")


def test_dense_channel_propagates_backend_error(tmp_path):
    tables = _tables(tmp_path)
    backend = _RaisingBackend()
    with pytest.raises(DenseBackendError):
        DenseChannel(backend).signals("mrr", tables)


# ---------------------------------------------------------------------------
# InMemoryDenseBackend
# ---------------------------------------------------------------------------

def test_cache_is_partitioned_by_encoder_id(tmp_path):
    tables = introspect(build(tmp_path / "t.db"))
    encA = EmbeddingEncoder(embed_fn=lambda ts: np.ones((len(ts), 384), np.float32), id="A")
    encB = EmbeddingEncoder(embed_fn=lambda ts: np.full((len(ts), 384), 2.0, np.float32), id="B")
    a = InMemoryDenseBackend(encA).column_scores("mrr", tables)
    b = InMemoryDenseBackend(encB).column_scores("mrr", tables)
    assert a and b and a != b        # different encoders -> different vectors, no reuse
