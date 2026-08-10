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

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field

from agent.execution import run_query
from agent.graph import run_agent
from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.serde import serialize_config
from evalharness.oracle import execution_match


@dataclass
class EvalRecord:
    id: str
    category: str
    question: str
    semantic_layer: bool
    retrieval_config: dict            # serialize_config(config) -- full canonical, NOT just a name
    retrieval_k: int                  # runtime k (RetrievalConfig does not carry run_agent's k)
    repeat_index: int
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
    candidate_recall: float | None = None
    selection_recall: float | None = None
    context_recall: float | None = None
    retrieval_stage_events: list[dict] = field(default_factory=list)


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


def three_layer_recall(retrieval_result, gold_tables, *, case_id: str) -> dict:
    """|gold_set ∩ stage_set| / |gold_set| for candidates / selection anchors / context tables.
    gold is mandatory -- an empty/None gold raises (recording 0 or a fake perfect score would be a
    silent lie)."""
    if not gold_tables:
        raise ValueError(f"{case_id}: required_tables is mandatory for recall (got empty/None)")
    gold = set(gold_tables)
    def recall(stage):                                  # set() -> duplicate table names can't change it
        return len(gold & set(stage)) / len(gold)
    return {
        "candidate_recall": recall(c.table for c in retrieval_result.candidates),
        "selection_recall": recall(retrieval_result.selection.anchor_tables),
        "context_recall": recall(retrieval_result.relation_plan.context_tables),
    }


def _clarified_record(case, result, *, semantic_layer: bool, config: RetrievalConfig, k: int,
                      repeat_index: int) -> EvalRecord:
    """A legitimately clarified run has no retrieval_result -- clarify_check precedes retrieval,
    so it returns before any channel runs. Record it as ``failure_stage="clarified"`` with
    exec_match=False and the three recalls left None: retrieval never ran, and scoring an un-run
    stage as 0 would be a silent lie. Real provenance (config / k / repeat / gold) is preserved."""
    usage = result.usage or {}
    return EvalRecord(
        id=case.id, category=case.category, question=case.question,
        semantic_layer=semantic_layer, retrieval_config=serialize_config(config),
        retrieval_k=k, repeat_index=repeat_index,
        predicted_sql=result.sql, gold_sql=case.gold_sql, exec_match=False,
        sql_valid_first_try=_first_try_valid(result.trace),
        sql_valid_final=result.execution.ok,
        repair_attempts=_repair_attempts(result.trace),
        latency_ms=float(usage.get("latency_ms", 0.0)),
        prompt_tokens=int(usage.get("input_tokens", 0)),
        completion_tokens=int(usage.get("output_tokens", 0)),
        retrieved_tables=list(result.retrieved_tables),
        gold_tables=list(case.required_tables),
        failure_stage="clarified",
        candidate_recall=None, selection_recall=None, context_recall=None,
        retrieval_stage_events=[],
    )


def record_run(case, result, gold_rows, *, semantic_layer: bool, config: RetrievalConfig, k: int,
               repeat_index: int) -> EvalRecord:
    rr = result.retrieval_result
    if rr is None:
        # Clarify precedes retrieval, so a legitimately clarified run has rr=None WITH a
        # clarification -- record it (see _clarified_record). Only rr=None WITHOUT a clarification
        # is a real graph/harness wiring bug.
        if result.clarification is not None:
            return _clarified_record(case, result, semantic_layer=semantic_layer, config=config,
                                     k=k, repeat_index=repeat_index)
        raise ValueError(f"{case.id}: result.retrieval_result is missing (harness/graph wiring bug)")
    if rr.config_name != config.name:
        raise ValueError(f"{case.id}: retrieval_result.config_name={rr.config_name!r} != "
                         f"config.name={config.name!r} (config wiring bug)")
    recall = three_layer_recall(rr, case.required_tables, case_id=case.id)
    match = execution_match(result.execution.rows, gold_rows, ordered=False)
    usage = result.usage or {}
    return EvalRecord(
        id=case.id, category=case.category, question=case.question,
        semantic_layer=semantic_layer, retrieval_config=serialize_config(config),
        retrieval_k=k, repeat_index=repeat_index,
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
        candidate_recall=recall["candidate_recall"],
        selection_recall=recall["selection_recall"],
        context_recall=recall["context_recall"],
        retrieval_stage_events=[dict(vars(e)) for e in rr.stage_events],
    )


def run_case(db_path, tables, case, model, *, semantic_layer: bool, config: RetrievalConfig, k: int,
             repeat_index: int) -> EvalRecord:
    """Execute the gold SQL (fixture self-check), run the full agent, score one record."""
    gold = run_query(db_path, case.gold_sql, tables=tables)
    if not gold.ok:
        raise RuntimeError(f"{case.id}: gold_sql failed to execute: {gold.error}")
    result = run_agent(db_path, case.question, model=model, tables=tables,
                       semantic_layer=semantic_layer, k=k, retrieval_config=config)
    return record_run(case, result, gold.rows, semantic_layer=semantic_layer, config=config,
                      k=k, repeat_index=repeat_index)


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

    # A control "diverges" if its ON/OFF pair -- same id, repeat, canonical config, and k --
    # disagrees on exec_match. Pairing (not any-vs-any) keeps run-to-run model randomness
    # across repeats from masquerading as a semantic-layer regression.
    def _pair_key(r):
        return (r.id, r.repeat_index, json.dumps(r.retrieval_config, sort_keys=True), r.retrieval_k)

    by_pair: dict[tuple, dict[bool, EvalRecord]] = defaultdict(dict)
    for r in controls:
        by_pair[_pair_key(r)][r.semantic_layer] = r
    diverging = sorted({
        pair[True].id for pair in by_pair.values()
        if True in pair and False in pair and pair[True].exec_match != pair[False].exec_match
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
