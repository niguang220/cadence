"""RetrievalPipeline: assembles channels, aggregate, fusion, selector, relation, and metric
matches behind a single ``run_retrieval`` entrypoint. Routes ONLY on config STRATEGY fields
(fusion / relation_strategy / lexical / dense_backend / ...) — never on ``config.name``, so a
caller can build a custom RetrievalConfig and still get correct routing. ``config.fusion`` and
``config.relation_strategy`` are independent knobs: fusion picks candidate production (RRF vs.
legacy min-max) and relation_strategy picks the closure (shortest_path vs. legacy_one_hop) --
any of the four combinations is valid."""
from __future__ import annotations

from agent.db.introspect import Table
from agent.retrieval.aggregate import aggregate
from agent.retrieval.backends import DenseBackendError, InMemoryDenseBackend
from agent.retrieval.channels import DenseChannel, LexicalChannel
from agent.retrieval.contracts import (RelationPlan, RetrievalConfig, RetrievalResult,
                                        RetrievalStageEvent, SelectionDecision,
                                        UnsupportedRetrievalCapability)
from agent.retrieval.fusion import legacy_candidates, weighted_rrf
from agent.retrieval.metric_match import MetricMatchProvider
from agent.retrieval.relation import legacy_one_hop_plan, plan_relations
from agent.retrieval.selector import NoOpSelector, TopKSelector, protected_anchors


def _missing_capabilities(config: RetrievalConfig) -> list[str]:
    """Collect every unbuilt capability the config requests (never stop at the first one)."""
    missing = []
    if config.value_backend == "es":
        missing.append("es")
    if config.dense_backend == "qdrant":
        missing.append("qdrant")
    if config.selector == "llm":
        missing.append("llm")
    return missing


def run_retrieval(question: str, tables: list[Table], config: RetrievalConfig, *, k: int,
                  metric_hits=None, dense_backend=None) -> RetrievalResult:
    metric_hits = metric_hits or []
    missing = _missing_capabilities(config)
    if missing:
        raise UnsupportedRetrievalCapability(missing)
    matches = MetricMatchProvider(tables).from_hits(metric_hits)

    if config.fusion == "legacy_minmax":
        candidates = legacy_candidates(question, tables, k=k)   # fusion_score=None (Fix 2)
        anchors = [c.table for c in candidates]                 # retrieve() already applied top-k
        # metric matches are TELEMETRY ONLY on the legacy path — never added to anchors (parity).
        selection = SelectionDecision(anchor_tables=anchors, dropped_tables=[],
                                      selector="noop", model_reason={})
        signals, events = [], []
    else:
        candidates, selection, signals, events, rejected = _rrf_fusion(
            question, tables, config, matches, dense_backend)
        if rejected:
            return RetrievalResult(config_name=config.name, signals=signals, candidates=[],
                                   metric_matches=matches, selection=selection,
                                   relation_plan=_empty_plan(config.relation_strategy),
                                   stage_events=events)

    plan = _closure(config, tables, selection.anchor_tables, events)   # relation_strategy chooses (Fix 1/3)
    return RetrievalResult(config_name=config.name, signals=signals, candidates=candidates,
                           metric_matches=matches, selection=selection, relation_plan=plan,
                           stage_events=events)


def _empty_plan(strategy):
    return RelationPlan(strategy, [], [], [], [], [], [])


def _closure(config, tables, anchors, events):
    if config.relation_strategy == "legacy_one_hop":
        return legacy_one_hop_plan(tables, anchors)
    plan = plan_relations(tables, anchors, max_hops=config.max_bridge_hops)
    if plan.unconnected_anchors:                                   # Fix 3: emit relation failure event
        events.append(RetrievalStageEvent(stage="relation", event="unconnected_anchor",
                                          detail={"unconnected": list(plan.unconnected_anchors)}))
    return plan


def _rrf_fusion(question, tables, config, matches, dense_backend):
    """Channels + aggregate + admission gate + weighted_rrf + protected anchors + selector.
    NO relation planning here (that's ``_closure``). Returns
    ``(candidates, selection, signals, events, rejected)``; on admission rejection ``rejected``
    is True and ``candidates``/``selection`` are the empty/no-anchor placeholders."""
    events: list[RetrievalStageEvent] = []
    signals = []
    channel_results = {}

    if config.lexical:
        lex = LexicalChannel().signals(question, tables)
        signals += lex
        if lex:
            channel_results["lexical"] = aggregate("lexical", lex)

    if config.dense_backend == "memory":
        backend = dense_backend if dense_backend is not None else InMemoryDenseBackend()
        try:
            dense = DenseChannel(backend).signals(question, tables)
            signals += dense
            if dense:
                channel_results["dense"] = aggregate("dense", dense)
        except DenseBackendError as e:
            events.append(RetrievalStageEvent(stage="channel", event="dense_degraded",
                                              detail={"error": str(e)}))

    # Admission gate (G2): dense similarity alone can't admit an off-topic question.
    has_lexical = bool(channel_results.get("lexical"))
    has_exact_metric = any(m.match_type == "alias" for m in matches)
    if not (has_lexical or has_exact_metric):
        events.append(RetrievalStageEvent(stage="admission", event="admission_rejected",
                                          detail={"reason": "no lexical or exact-metric footing"}))
        return [], SelectionDecision([], [], "topk", {}), signals, events, True

    candidates = weighted_rrf(channel_results, rrf_constant=config.rrf_constant,
                              candidate_k=config.candidate_k)
    protected = protected_anchors(matches)                        # RRF only (B)
    selector = NoOpSelector() if len(candidates) <= config.context_anchor_k else TopKSelector()
    selection = selector.select(candidates, protected, context_anchor_k=config.context_anchor_k)
    return candidates, selection, signals, events, False
