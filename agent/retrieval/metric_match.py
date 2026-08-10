from __future__ import annotations

from dataclasses import asdict

from agent.db.introspect import Table
from agent.retrieval.contracts import MetricMatch


def _catalog(tables: list[Table]) -> tuple[set[str], set[str]]:
    names = {t.name for t in tables}
    cols = {f"{t.name}.{c.name}" for t in tables for c in t.columns}
    return names, cols


def _validate_metric_deps(metric, names: set[str], cols: set[str]) -> None:
    bad_t = sorted(set(metric.required_tables) - names)
    if bad_t:
        raise ValueError(f"metric {metric.name!r}: unknown required table(s) {bad_t}")
    bad_c = sorted(set(metric.required_columns) - cols)
    if bad_c:
        raise ValueError(f"metric {metric.name!r}: unknown required column(s) {bad_c}")


def validate_all_metrics(registry, tables: list[Table]) -> None:
    """G4: fail-fast on ANY invalid governed metric (matched or not). Performs NO retrieval, so it
    does not violate single-retrieval ownership. Safe to call once per catalog fingerprint."""
    names, cols = _catalog(tables)
    for m in registry.metrics:
        _validate_metric_deps(m, names, cols)


class MetricMatchProvider:
    def __init__(self, tables: list[Table]):
        self._names, self._cols = _catalog(tables)

    def from_hits(self, hits) -> list[MetricMatch]:
        out = []
        for h in hits:
            m = h.metric
            _validate_metric_deps(m, self._names, self._cols)   # fail-fast on bad governance metadata
            out.append(MetricMatch(
                metric=m.name, match_type=h.match_type, score=h.score,
                required_tables=list(m.required_tables),
                required_columns=list(m.required_columns),
                required_filters=list(m.required_filters)))
        return out


def serialize_hits(hits) -> list[dict]:
    """Checkpoint-safe representation of MetricRetrievalHits (plain dicts) for LangGraph state."""
    return [{"metric": asdict(h.metric), "match_type": h.match_type, "score": h.score} for h in hits]


def deserialize_hits(dicts) -> list:
    from agent.semantic_layer import MetricDef, MetricRetrievalHit
    return [MetricRetrievalHit(MetricDef(**d["metric"]), d["match_type"], d["score"]) for d in dicts]
