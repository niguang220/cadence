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

import math
from dataclasses import dataclass, field

from agent.execution import run_query
from agent.graph import run_agent
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


def run_case(db_path, tables, case, model, *, semantic_layer: bool,
             retrieval_config: str = "lexical") -> EvalRecord:
    """Execute the gold SQL (fixture self-check), run the full agent, score one record."""
    gold = run_query(db_path, case.gold_sql, tables=tables)
    if not gold.ok:
        raise RuntimeError(f"{case.id}: gold_sql failed to execute: {gold.error}")
    result = run_agent(db_path, case.question, model=model, tables=tables,
                       semantic_layer=semantic_layer)
    return record_run(case, result, gold.rows, semantic_layer=semantic_layer,
                      retrieval_config=retrieval_config)


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (small-n friendly); 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def _rate(flags: list[bool]) -> dict:
    return {"exec_match_rate": (sum(flags) / len(flags)) if flags else 0.0,
            "matched": sum(flags), "n": len(flags)}


def summarize(records: list[EvalRecord]) -> dict:
    traps = [r for r in records if r.category != "control"]
    controls = [r for r in records if r.category == "control"]

    def split(rows):
        return {"on": _rate([r.exec_match for r in rows if r.semantic_layer]),
                "off": _rate([r.exec_match for r in rows if not r.semantic_layer])}

    # A control "diverges" if any (id) run disagrees on exec_match between ON and OFF --
    # the semantic layer changed an answer it should have left alone.
    diverging = sorted({
        r.id for r in controls
        if any(o.id == r.id and o.semantic_layer != r.semantic_layer
               and o.exec_match != r.exec_match for o in controls)
    })

    lat = [r.latency_ms for r in records]
    return {
        "n_records": len(records),
        "traps": split(traps),
        "controls": {**split(controls), "diverging_ids": diverging},
        "sql_valid_first_try_rate": (sum(r.sql_valid_first_try for r in records) / len(records)) if records else 0.0,
        "sql_valid_after_repair_rate": (sum(r.sql_valid_final for r in records) / len(records)) if records else 0.0,
        "avg_repair_attempts": (sum(r.repair_attempts for r in records) / len(records)) if records else 0.0,
        "latency_ms": {"p50": _percentile(lat, 50), "p95": _percentile(lat, 95)},
        "avg_prompt_tokens": (sum(r.prompt_tokens for r in records) / len(records)) if records else 0.0,
        "avg_completion_tokens": (sum(r.completion_tokens for r in records) / len(records)) if records else 0.0,
    }
