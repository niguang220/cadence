"""Pure derivation of an EvalRecord from an AnswerResult + gold rows (service-free)."""
from agent.execution import ExecutionResult
from agent.generation import AnswerResult
from evalharness.e2e_eval import EvalRecord, record_run
from evalharness.golden import SaasMetricsCase


def _case():
    return SaasMetricsCase(id="mrr_x", category="mrr", metric="mrr", question="q?",
                           gold_sql="SELECT 42", required_tables=["subscription"])


def _answer(rows, *, sql="SELECT 42", ok=True, trace=None, usage=None, clarification=None):
    return AnswerResult(
        question="q?", retrieved_tables=["subscription"], sql=sql,
        execution=ExecutionResult(ok, columns=["c"], rows=rows) if ok
                  else ExecutionResult(False, error="boom"),
        answer="a", clarification=clarification,
        trace=trace or [], usage=usage or {})


def test_exec_match_true_when_rows_match_gold():
    r = record_run(_case(), _answer([(42,)]), gold_rows=[(42,)], semantic_layer=True)
    assert r.exec_match is True and r.failure_stage is None
    assert r.semantic_layer is True and r.retrieval_config == "lexical"
    assert r.gold_tables == ["subscription"]


def test_exec_match_false_sets_answer_mismatch_stage():
    r = record_run(_case(), _answer([(41,)]), gold_rows=[(42,)], semantic_layer=False)
    assert r.exec_match is False and r.failure_stage == "answer_mismatch"


def test_first_try_from_first_execute_entry_and_repair_count():
    trace = [
        {"node": "generate_sql", "attempt": 1},
        {"node": "execute", "ok": False, "rows": 0},
        {"node": "validate", "verdict": "repair"},
        {"node": "generate_sql", "attempt": 2},
        {"node": "execute", "ok": True, "rows": 1},
    ]
    r = record_run(_case(), _answer([(42,)], trace=trace), gold_rows=[(42,)], semantic_layer=True)
    assert r.sql_valid_first_try is False   # first execute failed
    assert r.sql_valid_final is True        # final execution ok
    assert r.repair_attempts == 1           # two generate_sql entries -> one repair


def test_usage_and_declined_maps_to_no_sql():
    usage = {"latency_ms": 1234.5, "input_tokens": 100, "output_tokens": 20}
    r = record_run(_case(), _answer([], sql="", ok=False, usage=usage),
                   gold_rows=[(42,)], semantic_layer=False)
    assert r.latency_ms == 1234.5 and r.prompt_tokens == 100 and r.completion_tokens == 20
    assert r.failure_stage == "no_sql"


def test_clarification_maps_to_clarified_stage():
    r = record_run(_case(), _answer([], sql="", ok=False, clarification="which metric?"),
                   gold_rows=[(42,)], semantic_layer=False)
    assert r.failure_stage == "clarified"
