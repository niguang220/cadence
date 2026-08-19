"""Embedding encoder seam shared by retrieval backends and the metric registry.

BGE-via-fastembed is the encoder; later storage/search backends (in-memory numpy, Qdrant)
and the MetricRegistry all depend on THIS interface. Injecting ``embed_fn`` lets
tests avoid loading the model. No caching/backends here — that is a later task.
"""
from __future__ import annotations

import numpy as np


class EmbeddingEncoder:
    def __init__(self, embed_fn=None, id: str | None = None):
        self._embed_fn = embed_fn
        self._id = id

    @property
    def id(self) -> str:
        if self._id is not None:
            return self._id
        return "bge-small-en-v1.5" if self._embed_fn is None else f"custom-{id(self._embed_fn):x}"

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._embed_fn is not None:
            return self._embed_fn(texts)
        from agent.retrieval.embedding import embed
        return embed(texts)


_DEFAULT = EmbeddingEncoder()


def default_encoder() -> EmbeddingEncoder:
    return _DEFAULT
