"""Plain-dict serialization for RetrievalConfig / RetrievalResult so both can cross a
LangGraph checkpoint (which requires JSON/pickle-safe state, not arbitrary dataclasses)."""
from __future__ import annotations

from dataclasses import asdict

from agent.retrieval.contracts import (ChannelTableResult, MetricMatch, RelationEdge, RelationPlan,
                                        RetrievalConfig, RetrievalResult, RetrievalSignal,
                                        RetrievalStageEvent, SelectionDecision, TableCandidate)


def serialize_config(c: RetrievalConfig) -> dict:
    return asdict(c)


def deserialize_config(d: dict) -> RetrievalConfig:
    return RetrievalConfig(**d)


def serialize_result(r: RetrievalResult) -> dict:
    return asdict(r)                         # asdict recurses nested dataclasses in lists/dicts


def _signal(s): return RetrievalSignal(**s)


def _ctr(v):
    return ChannelTableResult(channel=v["channel"], table=v["table"],
                              raw_table_score=v["raw_table_score"], channel_rank=v["channel_rank"],
                              signals=[_signal(s) for s in v["signals"]])


def _candidate(c):
    return TableCandidate(table=c["table"],
                          channel_results={k: _ctr(v) for k, v in c["channel_results"].items()},
                          fusion_score=c["fusion_score"], fusion_rank=c["fusion_rank"])


def _relation_plan(p):
    return RelationPlan(strategy=p["strategy"], anchors=p["anchors"], bridges=p["bridges"],
                        context_tables=p["context_tables"],
                        edges=[RelationEdge(**e) for e in p["edges"]],
                        unconnected_anchors=p["unconnected_anchors"],
                        ambiguous_paths=[list(x) for x in p["ambiguous_paths"]])


def deserialize_result(d: dict) -> RetrievalResult:
    return RetrievalResult(
        config_name=d["config_name"],
        signals=[_signal(s) for s in d["signals"]],
        candidates=[_candidate(c) for c in d["candidates"]],
        metric_matches=[MetricMatch(**m) for m in d["metric_matches"]],
        selection=SelectionDecision(**d["selection"]),
        relation_plan=_relation_plan(d["relation_plan"]),
        stage_events=[RetrievalStageEvent(**e) for e in d["stage_events"]])
