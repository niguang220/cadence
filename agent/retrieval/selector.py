from __future__ import annotations

from typing import Protocol

from agent.retrieval.contracts import (MetricMatch, RetrievalStageEvent, SelectionDecision,
                                       TableCandidate)


def protected_anchors(metric_matches: list[MetricMatch]) -> list[str]:
    """Union of every matched governed metric's required tables (protected — undroppable)."""
    out: set[str] = set()
    for m in metric_matches:
        out.update(m.required_tables)
    return sorted(out)


def _merge(selected: list[str], protected: list[str]) -> list[str]:
    out = list(selected)
    for t in protected:
        if t not in out:
            out.append(t)
    return out


class Selector(Protocol):
    def select(self, candidates: list[TableCandidate], protected: list[str], *,
               context_anchor_k: int) -> SelectionDecision: ...


class TopKSelector:
    def select(self, candidates, protected, *, context_anchor_k):
        ranked = sorted(candidates, key=lambda c: c.fusion_rank)
        selected = [c.table for c in ranked[:context_anchor_k]]
        anchors = _merge(selected, protected)
        anchor_set = set(anchors)
        dropped = [c.table for c in candidates if c.table not in anchor_set]
        return SelectionDecision(anchor_tables=anchors, dropped_tables=dropped,
                                selector="topk", model_reason={})


class NoOpSelector:
    def select(self, candidates, protected, *, context_anchor_k):
        anchors = _merge([c.table for c in candidates], protected)
        return SelectionDecision(anchor_tables=anchors, dropped_tables=[],
                                selector="noop", model_reason={})


def validate_structured_selection(raw, candidates: list[TableCandidate]) -> list[str] | None:
    """A future LLMSelector's structured output. Returns the validated table subset, or None
    when the output is empty / not a list / names a table that isn't a candidate — the caller
    then falls back deterministically."""
    if not isinstance(raw, list) or not raw:
        return None
    valid = {c.table for c in candidates}
    if not all(isinstance(t, str) and t in valid for t in raw):
        return None
    return list(raw)


def fallback_topk(candidates, protected, *, context_anchor_k) -> tuple[SelectionDecision,
                                                                       RetrievalStageEvent]:
    """Deterministic degrade target for a failed/empty/illegal structured selection."""
    dec = TopKSelector().select(candidates, protected, context_anchor_k=context_anchor_k)
    ev = RetrievalStageEvent(stage="selection", event="selector_fallback",
                             detail={"context_anchor_k": context_anchor_k})
    return dec, ev
