"""Value-index ingestion lifecycle — a SEPARATE step, never run at query time.

Builds/refreshes the entity-value index from ONLY the searchable allowlist (pii/public excluded),
with a stable ``document_id`` for idempotent upsert, stale-doc pruning, and a data-fingerprint skip
so an unchanged database is a no-op instead of a rebuild.
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from agent.db.introspect import Table
from agent.retrieval.embedding import schema_fingerprint
from agent.retrieval.value_backend import ValueBackend, ValueDoc
from agent.retrieval.value_policy import resolve_searchable_columns


def index_name(tables: list[Table]) -> str:
    """Schema-fingerprinted index name. The fingerprint is HASHED to a lowercase hex digest so the
    name is a valid Elasticsearch index name (the raw fingerprint tuple contains spaces/commas/
    parens, which ES rejects — a bug the no-op FakeValueBackend could not surface)."""
    fp = hashlib.sha1(repr(schema_fingerprint(tables)).encode("utf-8")).hexdigest()[:16]
    return f"cadence-values-{fp}"


def document_id(table: str, column: str, value: str) -> str:
    """Stable id for (table, column, value) — same value ⇒ same id ⇒ idempotent upsert."""
    return hashlib.sha1("\x00".join([table, column, value]).encode("utf-8")).hexdigest()


def data_fingerprint(docs: list[ValueDoc]) -> str:
    h = hashlib.sha256()
    for did in sorted(d.document_id for d in docs):
        h.update(did.encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True)
class BuildResult:
    index: str
    fingerprint: str
    doc_count: int
    skipped: bool


def _collect_docs(db_path: str | Path, allowed: list[tuple[str, str]]) -> list[ValueDoc]:
    conn = sqlite3.connect(str(db_path))
    try:
        docs: list[ValueDoc] = []
        for table, column in allowed:                       # allowlist names are our own schema, not user input
            rows = conn.execute(
                f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL'
            ).fetchall()
            for (val,) in rows:
                value = str(val)
                docs.append(ValueDoc(document_id=document_id(table, column, value),
                                     table=table, column=column, value=value))
        return docs
    finally:
        conn.close()


def build_value_index(tables: list[Table], db_path: str | Path, backend: ValueBackend, *,
                      known_fingerprint: str | None = None) -> BuildResult:
    """Rebuild the value index from ONLY the searchable allowlist. Idempotent upsert + stale prune.
    Returns early (no writes) when the data fingerprint matches ``known_fingerprint`` — so callers
    skip unchanged rebuilds, and this never runs implicitly at query time."""
    allowed = resolve_searchable_columns(tables)            # validates (fail-closed); pii/public excluded
    docs = _collect_docs(db_path, allowed)
    fp = data_fingerprint(docs)
    name = index_name(tables)
    if known_fingerprint is not None and known_fingerprint == fp:
        return BuildResult(index=name, fingerprint=fp, doc_count=len(docs), skipped=True)
    backend.ensure_index()
    backend.upsert(docs)
    backend.prune({d.document_id for d in docs})
    return BuildResult(index=name, fingerprint=fp, doc_count=len(docs), skipped=False)
