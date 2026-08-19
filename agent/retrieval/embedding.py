"""Shared local embedding primitives for dense schema retrieval and index identity."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from agent.db.introspect import Column, Table

_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def model():
    from fastembed import TextEmbedding

    return TextEmbedding(_MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    vecs = np.asarray(list(model().embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-12, None)


def column_document(table: Table, column: Column) -> str:
    parts = [table.name, column.name]
    if column.description:
        parts.append(column.description)
    return " ".join(parts)


def schema_fingerprint(tables: list[Table]) -> tuple:
    """Hashable identity of the schema content used by retrieval indexes."""
    return tuple(sorted(column_document(table, column)
                        for table in tables for column in table.columns))
