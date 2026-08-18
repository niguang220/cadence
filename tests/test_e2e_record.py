"""Pure derivation of an EvalRecord from an AnswerResult + gold rows (service-free)."""
import pytest

from agent.execution import ExecutionResult
from agent.generation import AnswerResult
from agent.retrieval.contracts import (RelationPlan, RetrievalConfig, RetrievalResult,
                                        RetrievalStageEvent, SelectionDecision, TableCandidate)
from agent.retrieval.serde import serialize_config
from evalharness.e2e_eval import EvalRecord, record_run
from evalharness.golden import SaasMetricsCase


def _case():
    return SaasMetricsCase(id="mrr_x", category="mrr", metric="mrr", question="q?",
                           gold_sql="SELECT 42", required_tables=["subscription"])


def _rr(config_name, tables=("subscription",), events=()):
    return RetrievalResult(config_name=config_name, signals=[],
        candidates=[TableCandidate(t, {}, None, i + 1) for i, t in enumerate(tables)],
        metric_matches=[], selection=SelectionDecision(list(tables), [], "topk", {}),
        relation_plan=RelationPlan("legacy_one_hop", list(tables), [], list(tables), [], [], []),
        stage_events=list(events))


def _answer(rows, *, sql="SELECT 42", ok=True, trace=None, usage=None, clarification=None,
           retrieval_result=None):
    return AnswerResult(
        question="q?", retrieved_tables=["subscription"], sql=sql,
        execution=ExecutionResult(ok, columns=["c"], rows=rows) if ok
                  else ExecutionResult(False, error="boom"),
        answer="a", clarification=clarification,
        trace=trace or [], usage=usage or {},
        retrieval_result=retrieval_result if retrieval_result is not None else _rr("legacy_minmax"))


def test_value_hit_tier_recorded_from_value_signals():
    from agent.retrieval.contracts import RetrievalSignal
    rr = _rr("dense_value")
    rr.signals = [RetrievalSignal(channel="value", target_type="value", table="account",
                                  column="name", query_term="q", raw_score=1.0,
                                  match_type="exact_phrase", matched_value="Acme"),
                  RetrievalSignal(channel="lexical", target_type="table", table="subscription",
                                  column=None, query_term="q", raw_score=1.0, match_type="alias")]
    r = record_run(_case(), _answer([(42,)], retrieval_result=rr), gold_rows=[(42,)],
                   semantic_layer=False, config=RetrievalConfig.dense_value(), k=5, repeat_index=0)
    assert r.value_hit is True and r.value_hit_tier == "exact_phrase"


def test_no_value_hit_when_no_value_signals():
    r = record_run(_case(), _answer([(42,)]), gold_rows=[(42,)], semantic_layer=False,
                   config=RetrievalConfig.legacy_minmax(), k=5, repeat_index=0)
    assert r.value_hit is False and r.value_hit_tier is None


def test_exec_match_true_when_rows_match_gold():
    r = record_run(_case(), _answer([(42,)]), gold_rows=[(42,)], semantic_layer=True,
                   config=RetrievalConfig.legacy_minmax(), k=5, repeat_index=0)
    assert r.exec_match is True and r.failure_stage is None
    assert r.semantic_layer is True
    assert r.retrieval_config == serialize_config(RetrievalConfig.legacy_minmax())
    assert r.gold_tables == ["subscription"]


def test_exec_match_false_sets_answer_mismatch_stage():
    r = record_run(_case(), _answer([(41,)]), gold_rows=[(42,)], semantic_layer=False,
                   config=RetrievalConfig.legacy_minmax(), k=5, repeat_index=0)
    assert r.exec_match is False and r.failure_stage == "answer_mismatch"


def test_first_try_from_first_execute_entry_and_repair_count():
    trace = [
        {"node": "generate_sql", "attempt": 1},
        {"node": "execute", "ok": False, "rows": 0},
        {"node": "validate", "verdict": "repair"},
        {"node": "generate_sql", "attempt": 2},
        {"node": "execute", "ok": True, "rows": 1},
    ]
    r = record_run(_case(), _answer([(42,)], trace=trace), gold_rows=[(42,)], semantic_layer=True,
                   config=RetrievalConfig.legacy_minmax(), k=5, repeat_index=0)
    assert r.sql_valid_first_try is False   # first execute failed
    assert r.sql_valid_final is True        # final execution ok
    assert r.repair_attempts == 1           # two generate_sql entries -> one repair


def test_usage_and_declined_maps_to_no_sql():
    usage = {"latency_ms": 1234.5, "input_tokens": 100, "output_tokens": 20}
    r = record_run(_case(), _answer([], sql="", ok=False, usage=usage),
                   gold_rows=[(42,)], semantic_layer=False,
                   config=RetrievalConfig.legacy_minmax(), k=5, repeat_index=0)
    assert r.latency_ms == 1234.5 and r.prompt_tokens == 100 and r.completion_tokens == 20
    assert r.failure_stage == "no_sql"


def test_clarification_maps_to_clarified_stage():
    r = record_run(_case(), _answer([], sql="", ok=False, clarification="which metric?"),
                   gold_rows=[(42,)], semantic_layer=False,
                   config=RetrievalConfig.legacy_minmax(), k=5, repeat_index=0)
    assert r.failure_stage == "clarified"


def test_record_run_fails_if_retrieval_result_missing():
    a = _answer([(42,)])
    a.retrieval_result = None
    with pytest.raises(ValueError, match="retrieval_result is missing"):
        record_run(_case(), a, [(42,)], semantic_layer=True,
                   config=RetrievalConfig.legacy_minmax(), k=5, repeat_index=0)


def test_clarified_run_without_retrieval_result_is_recorded_not_raised():
    # Real clarify path: clarify_check precedes retrieval, so a legitimately clarified run has
    # retrieval_result=None WITH a clarification. Record it as "clarified" (not a wiring bug),
    # and never fake the un-run retrieval recall as 0.
    a = _answer([], sql="", ok=False, clarification="which metric do you mean?")
    a.retrieval_result = None
    r = record_run(_case(), a, [(42,)], semantic_layer=True,
                   config=RetrievalConfig.rrf_hybrid(), k=7, repeat_index=2)
    assert r.failure_stage == "clarified"
    assert r.exec_match is False
    assert r.candidate_recall is None
    assert r.selection_recall is None
    assert r.context_recall is None
    assert r.retrieval_stage_events == []
    # real provenance preserved (config / k / repeat / gold), NOT the un-run retrieval
    assert r.retrieval_config == serialize_config(RetrievalConfig.rrf_hybrid())
    assert r.retrieval_k == 7 and r.repeat_index == 2
    assert r.gold_tables == ["subscription"]


def test_record_run_fails_on_config_name_mismatch():
    a = _answer([(42,)])   # retrieval_result config_name defaults to "legacy_minmax"
    with pytest.raises(ValueError, match="config_name"):
        record_run(_case(), a, [(42,)], semantic_layer=True,
                   config=RetrievalConfig.rrf_hybrid(), k=5, repeat_index=0)   # names differ


def test_record_run_captures_full_config_k_repeat_and_stage_events():
    event = RetrievalStageEvent(stage="selection", event="selector_fallback", detail={"why": "noop"})
    cfg = RetrievalConfig.rrf_hybrid()
    a = _answer([(42,)], retrieval_result=_rr(cfg.name, events=[event]))
    r = record_run(_case(), a, [(42,)], semantic_layer=True, config=cfg, k=9, repeat_index=3)
    assert r.retrieval_config == serialize_config(cfg)
    assert isinstance(r.retrieval_config, dict) and r.retrieval_config["name"] == cfg.name
    assert r.retrieval_k == 9
    assert r.repeat_index == 3
    assert r.retrieval_stage_events == [
        {"stage": "selection", "event": "selector_fallback", "detail": {"why": "noop"}}
    ]
