"""The consistency surface scorer + deterministic teeth helpers.

The judge sees only {question, sql, result} -- no metric block -- so fixtures cover only
judge-observable mismatches (measure/grain/entity/dropped_filter), never governed
definitions. Both candidate and gold run through the PRODUCTION path
(``run_query(db, sql, tables=tables)``: safety on, governance runs) and must be
governance-clean, so the judge never sees a result the graph would have blocked. The
scorer reports BOTH catch-rate (recall on adversarial) and false-positive-rate (flag
rate on clean) -- a check that flags everything scores 100% catch and is useless.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.execution import ExecutionResult, run_query
from agent.governance import check_result_governance
from evalharness.oracle import execution_match


@dataclass
class ConsistencyOutcome:
    case_id: str
    expected_caught: bool
    caught: bool


def execute_case(case, db_path, tables) -> tuple[ExecutionResult, ExecutionResult]:
    """Run candidate and gold through the production path (governance runs)."""
    cand = run_query(db_path, case.candidate_sql, tables=tables)
    gold = run_query(db_path, case.gold_sql, tables=tables)
    return cand, gold


def governance_clean(result: ExecutionResult, tables) -> bool:
    return check_result_governance(result.columns, tables).ok


def diverges(candidate: ExecutionResult, gold: ExecutionResult) -> bool:
    return not execution_match(candidate.rows, gold.rows)


def score_consistency(outcomes: list[ConsistencyOutcome]) -> dict:
    adversarial = [o for o in outcomes if o.expected_caught]
    clean = [o for o in outcomes if not o.expected_caught]
    catch_rate = sum(o.caught for o in adversarial) / len(adversarial) if adversarial else 0.0
    fp_rate = sum(o.caught for o in clean) / len(clean) if clean else 0.0
    return {
        "catch_rate": catch_rate,
        "fp_rate": fp_rate,
        "adversarial_support": len(adversarial),
        "clean_support": len(clean),
    }
