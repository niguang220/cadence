from __future__ import annotations

import dataclasses

import pytest


PRESET_EXPECTATIONS = {
    "lexical_baseline": {
        "lexical": True,
        "dense_backend": None,
        "value_backend": None,
        "selector": None,
        "fusion": "rrf",
        "relation_strategy": "shortest_path",
        "candidate_k": 15,
        "context_anchor_k": 5,
        "rrf_constant": 60,
        "max_bridge_hops": 3,
    },
    "current_hybrid": {
        "lexical": True,
        "dense_backend": "memory",
        "value_backend": None,
        "selector": None,
        "fusion": "legacy_minmax",
        "relation_strategy": "legacy_one_hop",
        "candidate_k": 15,
        "context_anchor_k": 5,
        "rrf_constant": 60,
        "max_bridge_hops": 3,
    },
    "rrf_hybrid": {
        "lexical": True,
        "dense_backend": "memory",
        "value_backend": None,
        "selector": None,
        "fusion": "rrf",
        "relation_strategy": "shortest_path",
        "candidate_k": 15,
        "context_anchor_k": 5,
        "rrf_constant": 60,
        "max_bridge_hops": 3,
    },
    "value_ablation": {
        "lexical": True,
        "dense_backend": None,
        "value_backend": "es",
        "selector": None,
        "fusion": "rrf",
        "relation_strategy": "shortest_path",
        "candidate_k": 15,
        "context_anchor_k": 5,
        "rrf_constant": 60,
        "max_bridge_hops": 3,
    },
    "full_rag": {
        "lexical": True,
        "dense_backend": "memory",
        "value_backend": "es",
        "selector": "llm",
        "fusion": "rrf",
        "relation_strategy": "shortest_path",
        "candidate_k": 15,
        "context_anchor_k": 5,
        "rrf_constant": 60,
        "max_bridge_hops": 3,
    },
}


@pytest.mark.parametrize("preset_name", sorted(PRESET_EXPECTATIONS))
def test_preset_strategy_fields(preset_name):
    from agent.retrieval.contracts import RetrievalConfig as C

    config = getattr(C, preset_name)()
    expected = PRESET_EXPECTATIONS[preset_name]
    actual = {field_name: getattr(config, field_name) for field_name in expected}
    assert actual == expected


def test_unsupported_capability_lists_missing():
    from agent.retrieval.contracts import UnsupportedRetrievalCapability

    e = UnsupportedRetrievalCapability(["es"])
    assert e.capabilities == ["es"] and isinstance(e, Exception)


def test_retrieval_config_is_frozen():
    from agent.retrieval.contracts import RetrievalConfig

    config = RetrievalConfig(name="frozen_check")
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.name = "mutated"


def test_stage_event_admission_stage():
    from agent.retrieval.contracts import RetrievalStageEvent

    event = RetrievalStageEvent(stage="admission", event="admission_rejected")
    assert event.stage == "admission"


def test_table_candidate_accepts_none_fusion_score():
    from agent.retrieval.contracts import TableCandidate

    candidate = TableCandidate(table="orders", channel_results={}, fusion_score=None,
                               fusion_rank=1)
    assert candidate.fusion_score is None


def test_new_types_exist_and_compose():
    from agent.retrieval.contracts import (
        RelationPlan,
        RetrievalResult,
        RetrievalStageEvent,
        SelectionDecision,
    )

    plan = RelationPlan(
        strategy="shortest_path",
        anchors=["orders"],
        bridges=[],
        context_tables=["orders"],
        edges=[],
        unconnected_anchors=[],
        ambiguous_paths=[],
    )
    assert plan.strategy == "shortest_path"

    event = RetrievalStageEvent(stage="fusion", event="dense_degraded")

    result = RetrievalResult(
        config_name="rrf_hybrid",
        signals=[],
        candidates=[],
        metric_matches=[],
        selection=SelectionDecision(
            anchor_tables=["orders"],
            dropped_tables=[],
            selector="topk",
            model_reason={},
        ),
        relation_plan=plan,
        stage_events=[event],
    )
    assert result.stage_events == [event]
