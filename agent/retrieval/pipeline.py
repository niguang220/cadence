"""Typed retrieval pipeline: channels -> RRF -> selection -> relation planning."""
from __future__ import annotations

from agent.db.introspect import Table
from agent.retrieval.aggregate import aggregate
from agent.retrieval.backends import DenseBackendError, InMemoryDenseBackend
from agent.retrieval.channels import DenseChannel, LexicalChannel, ValueChannel
from agent.retrieval.contracts import (RelationPlan, RetrievalConfig, RetrievalResult,
                                        RetrievalStageEvent, SelectionDecision,
                                        UnsupportedRetrievalCapability)
from agent.retrieval.fusion import weighted_rrf
from agent.retrieval.lexical_backends import lexical_backend_for
from agent.retrieval.metric_match import MetricMatchProvider
from agent.retrieval.relation import plan_relations
from agent.retrieval.selector import NoOpSelector, TopKSelector, protected_anchors
from agent.retrieval.value_backend import ValueBackendError

_HIGH_CONFIDENCE_VALUE = {"exact_keyword", "exact_phrase"}


def _missing_capabilities(config: RetrievalConfig) -> list[str]:
    """Collect every unbuilt capability the config requests (never stop at the first one).
    ``value_backend == "es"`` is now built (an ES failure degrades at runtime, it is not an
    unsupported capability)."""
    missing = []
    if config.dense_backend == "qdrant":
        missing.append("qdrant")
    if config.selector == "llm":
        missing.append("llm")
    return missing


def _value_backend_for(config: RetrievalConfig, injected):
    """Return the caller-injected value backend.

    Backend construction and index lifecycle stay outside the retrieval pipeline. A value-enabled
    config without an injected backend therefore records a typed degradation instead of silently
    behaving like an empty index.
    """
    if injected is not None:
        return injected
    raise ValueBackendError("no value backend was injected")


def run_retrieval(question: str, tables: list[Table], config: RetrievalConfig, *, k: int,
                  metric_hits=None, dense_backend=None, value_backend=None,
                  value_query: str | None = None) -> RetrievalResult:
    """``question`` drives lexical + dense (the graph passes the enhanced rewrite here). ``value_query``
    drives ONLY entity-value linking; the graph passes the ORIGINAL user question so a model-invented
    entity in the enhanced rewrite can never become a value admission. ``value_query=None`` falls back
    to ``question`` (direct callers and existing tests are unaffected)."""
    metric_hits = metric_hits or []
    value_query = question if value_query is None else value_query
    missing = _missing_capabilities(config)
    if missing:
        raise UnsupportedRetrievalCapability(missing)
    matches = MetricMatchProvider(tables).from_hits(metric_hits)

    candidates, selection, signals, events, rejected = _rrf_fusion(
        question, tables, config, matches, dense_backend, value_backend, value_query,
        context_anchor_k=k)
    if rejected:
        return RetrievalResult(config_name=config.name, signals=signals, candidates=[],
                               metric_matches=matches, selection=selection,
                               relation_plan=_empty_plan(), stage_events=events)

    plan = _closure(config, tables, selection.anchor_tables, events)
    return RetrievalResult(config_name=config.name, signals=signals, candidates=candidates,
                           metric_matches=matches, selection=selection, relation_plan=plan,
                           stage_events=events)


def _empty_plan():
    return RelationPlan("shortest_path", [], [], [], [], [], [])


def _closure(config, tables, anchors, events):
    plan = plan_relations(tables, anchors, max_hops=config.max_bridge_hops)
    if plan.unconnected_anchors:
        events.append(RetrievalStageEvent(stage="relation", event="unconnected_anchor",
                                          detail={"unconnected": list(plan.unconnected_anchors)}))
    return plan


def _rrf_fusion(question, tables, config, matches, dense_backend, value_backend=None,
                value_query=None, context_anchor_k=None):
    """Channels + aggregate + admission gate + weighted_rrf + protected anchors + selector.
    NO relation planning here (that's ``_closure``). ``question`` feeds lexical/dense; ``value_query``
    feeds ONLY the value channel (defaults to ``question``). Returns
    ``(candidates, selection, signals, events, rejected)``; on admission rejection ``rejected``
    is True and ``candidates``/``selection`` are the empty/no-anchor placeholders."""
    value_query = question if value_query is None else value_query
    events: list[RetrievalStageEvent] = []
    signals = []
    channel_results = {}

    if config.lexical:
        lex = LexicalChannel(lexical_backend_for(config.lexical_backend)).signals(question, tables)
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

    value_signals = []
    if config.value_backend == "es":
        try:
            vb = _value_backend_for(config, value_backend)
            value_signals = ValueChannel(vb).signals(value_query, tables)   # original question, not enhanced
            signals += value_signals
            if value_signals:
                channel_results["value"] = aggregate("value", value_signals)
        except ValueBackendError as e:
            events.append(RetrievalStageEvent(stage="channel", event="value_degraded",
                                              detail={"error": str(e)}))

    # Admission gate (G2): dense similarity alone can't admit an off-topic question. A
    # high-confidence (exact) value hit CAN admit; token/fuzzy value hits cannot admit alone.
    has_lexical = bool(channel_results.get("lexical"))
    has_exact_metric = any(m.match_type == "alias" for m in matches)
    has_high_conf_value = any(s.match_type in _HIGH_CONFIDENCE_VALUE for s in value_signals)
    if not (has_lexical or has_exact_metric or has_high_conf_value):
        events.append(RetrievalStageEvent(stage="admission", event="admission_rejected",
                                          detail={"reason": "no lexical, exact-metric, or "
                                                            "high-confidence value footing"}))
        return [], SelectionDecision([], [], "topk", {}), signals, events, True

    candidates = weighted_rrf(channel_results, rrf_constant=config.rrf_constant,
                              weights=config.channel_weights(),
                              candidate_k=config.candidate_k)
    protected = protected_anchors(matches)
    selection_k = config.context_anchor_k if context_anchor_k is None else context_anchor_k
    selector = NoOpSelector() if len(candidates) <= selection_k else TopKSelector()
    selection = selector.select(candidates, protected, context_anchor_k=selection_k)
    return candidates, selection, signals, events, False
