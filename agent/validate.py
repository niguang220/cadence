"""validate_result: cheap structural checks that flag a result as suspicious even
when the SQL ran without error -- the "runs but answers wrong" class.

Each rule is deliberately conservative (a false flag costs a wasted repair and can
turn a correct answer into a wrong one), so the triggers are narrowed per the
design review:
- empty result is only suspicious when the query has a JOIN (an empty single-table
  filter is usually the correct answer);
- a "ranking" question whose query has no ORDER BY is suspicious;
- an "aggregation" question whose query neither aggregates nor groups, yet returns
  many rows, probably forgot to aggregate.

Detection uses the sqlglot AST, not regex.
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from agent.execution import ExecutionResult

# "first "/"last " were removed: they read as ordinals/adverbs ("first name of
# customer 1", "list X first") far more often than as ranking intent (design review).
_RANKING_WORDS = (
    "top ", "longest", "shortest", "highest", "lowest", "most ", "least ",
    "largest", "smallest", "best", "worst", "ranked", "rank",
)
# Known limitation: " per " also matches enumeration ("list customers per country"),
# which can false-flag a correct non-aggregated query. Harmless on the current
# golden set (those "per" questions get a GROUP BY); revisit with Phase 3 hard cases.
_AGG_WORDS = ("how many", "count", "total", "sum", "average", "avg", "number of", " per ")


@dataclass
class Verdict:
    ok: bool
    repair_kind: str = ""
    reason: str = ""


def _parse(sql: str):
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None


def _has(sql: str, node_type) -> bool:
    tree = _parse(sql)
    return bool(tree and tree.find(node_type))


def has_join(sql: str) -> bool:
    return _has(sql, exp.Join)


def has_order_by(sql: str) -> bool:
    return _has(sql, exp.Order)


def has_group_by(sql: str) -> bool:
    return _has(sql, exp.Group)


def has_aggregate(sql: str) -> bool:
    return _has(sql, exp.AggFunc)


def wants_ranking(question: str) -> bool:
    q = question.lower()
    return any(w in q for w in _RANKING_WORDS)


def wants_aggregation(question: str) -> bool:
    q = question.lower()
    return any(w in q for w in _AGG_WORDS)


def validate_result(question: str, sql: str, result: ExecutionResult) -> Verdict:
    """Judge a result; ``ok=False`` means re-generate with ``reason`` as the hint."""
    if not result.ok:
        if result.error.startswith("governance violation:"):
            return Verdict(False, "governance_block", result.error)
        return Verdict(False, "exec_error", result.error or "the query failed to execute")

    if not result.rows and has_join(sql):
        return Verdict(False, "empty_join",
                       "the query returned no rows; check the JOIN conditions and filters")

    if wants_ranking(question) and not has_order_by(sql):
        return Verdict(False, "missing_order",
                       "the question asks for a ranking/top-N but the query has no ORDER BY")

    if (wants_aggregation(question) and not has_group_by(sql)
            and not has_aggregate(sql) and len(result.rows) > 1):
        return Verdict(False, "missing_aggregate",
                       "the question expects an aggregate but the query neither aggregates nor groups")

    return Verdict(True)
