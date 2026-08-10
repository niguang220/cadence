"""Dense embedding backends. A backend owns vector storage/search; a channel (channels.py)
turns backend hits into RetrievalSignals. Keeping them separate means a later Qdrant backend
drops in behind the same DenseChannel without touching signal production."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from agent.db.introspect import Table
from agent.hybrid_retriever import _column_doc, _schema_fingerprint
from agent.retrieval.encoder import EmbeddingEncoder, default_encoder


class DenseBackend(Protocol):
    def column_scores(self, question: str, tables: list[Table]) -> list[tuple[str, str, float]]: ...


class DenseBackendError(Exception):
    """Embedding backend failed (model unavailable / runtime error). The pipeline catches this
    and records a dense_degraded stage event; the channel must let it propagate."""


class _ColumnIndex:
    def __init__(self, tables: list[Table], encoder: EmbeddingEncoder):
        self._docs_meta: list[tuple[str, str]] = []   # (table, column) per embedded doc
        docs: list[str] = []
        for t in tables:
            for c in t.columns:
                docs.append(_column_doc(t, c))
                self._docs_meta.append((t.name, c.name))
        self._embs = encoder.embed(docs) if docs else np.zeros((0, 384), np.float32)

    def column_scores(self, question: str, encoder: EmbeddingEncoder) -> list[tuple[str, str, float]]:
        if self._embs.shape[0] == 0:
            return []
        q = encoder.embed([question])[0]
        sims = self._embs @ q
        return [(tbl, col, float(s)) for (tbl, col), s in zip(self._docs_meta, sims)]


# cache keyed by (schema fingerprint, encoder id) so different encoders never reuse vectors (G5)
_INDEX_CACHE: dict[tuple, _ColumnIndex] = {}


class InMemoryDenseBackend:
    def __init__(self, encoder: EmbeddingEncoder | None = None):
        self._encoder = encoder or default_encoder()

    def column_scores(self, question: str, tables: list[Table]) -> list[tuple[str, str, float]]:
        try:
            key = (_schema_fingerprint(tables), self._encoder.id)
            index = _INDEX_CACHE.get(key)
            if index is None:
                index = _INDEX_CACHE[key] = _ColumnIndex(tables, self._encoder)
            return index.column_scores(question, self._encoder)
        except Exception as e:  # model download/runtime failure -> typed, so pipeline can degrade
            raise DenseBackendError(str(e)) from e
