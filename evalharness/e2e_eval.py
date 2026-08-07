"""End-to-end capability scorer for the saas_metrics set.

Runs the FULL agent (``run_agent``) per case and scores it against the gold result with
the same ``execution_match`` oracle used everywhere else, then derives one flat
``EvalRecord`` (locked schema) per run. Record derivation is pure -- given an
``AnswerResult`` and the gold rows it computes every field with no I/O -- so it is unit
tested with hand-built results and never needs a real API. ``run_case`` adds the two
executions (gold + agent) and a fixture self-check; ``summarize`` aggregates a flat list
of records into the roadmap's Step-1 report shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from evalharness.oracle import execution_match


@dataclass
class EvalRecord:
    id: str
    category: str
    question: str
    semantic_layer: bool
    retrieval_config: str
    predicted_sql: str
    gold_sql: str
    exec_match: bool
    sql_valid_first_try: bool
    sql_valid_final: bool
    repair_attempts: int
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    retrieved_tables: list[str] = field(default_factory=list)
    gold_tables: list[str] | None = None
    failure_stage: str | None = None


def _first_try_valid(trace: list[dict]) -> bool:
    for entry in trace:
        if entry.get("node") == "execute":
            return bool(entry.get("ok"))
    return False


def _repair_attempts(trace: list[dict]) -> int:
    generations = sum(1 for e in trace if e.get("node") == "generate_sql")
    return max(0, generations - 1)


def _failure_stage(result, exec_match: bool) -> str | None:
    if exec_match:
        return None
    if result.clarification is not None:
        return "clarified"
    if not result.sql:
        return "no_sql"
    if not result.execution.ok:
        return "execution_error"
    return "answer_mismatch"


def record_run(case, result, gold_rows, *, semantic_layer: bool,
               retrieval_config: str = "lexical") -> EvalRecord:
    match = execution_match(result.execution.rows, gold_rows, ordered=False)
    usage = result.usage or {}
    return EvalRecord(
        id=case.id, category=case.category, question=case.question,
        semantic_layer=semantic_layer, retrieval_config=retrieval_config,
        predicted_sql=result.sql, gold_sql=case.gold_sql, exec_match=match,
        sql_valid_first_try=_first_try_valid(result.trace),
        sql_valid_final=result.execution.ok,
        repair_attempts=_repair_attempts(result.trace),
        latency_ms=float(usage.get("latency_ms", 0.0)),
        prompt_tokens=int(usage.get("input_tokens", 0)),
        completion_tokens=int(usage.get("output_tokens", 0)),
        retrieved_tables=list(result.retrieved_tables),
        gold_tables=list(case.required_tables),
        failure_stage=_failure_stage(result, match),
    )
