from __future__ import annotations

import numpy as np
import pytest


def _fastembed_available():
    try:
        from agent.hybrid_retriever import _model
        _model()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _fastembed_available(), reason="fastembed model not available offline")
def test_encoder_matches_private_embed():
    from agent.hybrid_retriever import _embed
    from agent.retrieval.encoder import EmbeddingEncoder
    texts = ["monthly recurring revenue", "account name"]
    np.testing.assert_array_equal(EmbeddingEncoder().embed(texts), _embed(texts))


def test_encoder_is_injectable_stub():
    from agent.retrieval.encoder import EmbeddingEncoder
    enc = EmbeddingEncoder(embed_fn=lambda t: np.zeros((len(t), 3), dtype="float32"))
    assert enc.embed(["a", "b"]).shape == (2, 3)


def test_default_encoder_delegates_to_private_embed(monkeypatch):
    # Offline proof of byte-identity: the default encoder is a pure passthrough to _embed.
    import agent.hybrid_retriever as hr
    from agent.retrieval.encoder import EmbeddingEncoder
    sentinel = np.array([[1.0, 2.0]], dtype="float32")
    monkeypatch.setattr(hr, "_embed", lambda texts: sentinel)
    assert EmbeddingEncoder().embed(["x"]) is sentinel


def test_semantic_layer_default_embed_routes_through_encoder(monkeypatch):
    # _default_embed must go through the encoder seam, still reaching _embed unchanged.
    import agent.hybrid_retriever as hr
    from agent.semantic_layer import _default_embed
    marker = np.array([[9.0]], dtype="float32")
    monkeypatch.setattr(hr, "_embed", lambda texts: marker)
    assert _default_embed(["q"]) is marker
