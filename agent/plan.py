"""The plan a PlannerNode emits: an ordered list of typed steps, plus a
deterministic structural validator. A step is either a SQL step (fetch rows) or a
Python step (compute/plot on prior rows). validate_plan is zero-LLM so it's cheap and
fully testable; an invalid plan sends the graph back to the planner (bounded)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

_KINDS = ("sql", "python")


@dataclass
class Step:
    kind: str
    instruction: str


@dataclass
class Plan:
    steps: list[Step]


@dataclass
class PlanVerdict:
    ok: bool
    reason: str = ""


def serialize_plan(plan: Plan) -> list[dict]:
    return [asdict(s) for s in plan.steps]


def deserialize_plan(data: list[dict]) -> Plan:
    return Plan([Step(**d) for d in data])


def validate_plan(plan: Plan) -> PlanVerdict:
    if not plan.steps:
        return PlanVerdict(False, "plan is empty")
    for s in plan.steps:
        if s.kind not in _KINDS:
            return PlanVerdict(False, f"invalid step kind: {s.kind!r}")
        if not s.instruction.strip():
            return PlanVerdict(False, "a step has an empty instruction")
    # The runtime supports exactly one SQL step, optionally followed by one Python step.
    # Step and Plan stay general so richer plans can be added without changing the data
    # model; the validator deliberately enforces the smaller executable subset. This
    # is what makes `state["result"]` an unambiguous handle for the Python step (there
    # is at most one SQL result) and lets `respond` aggregate deterministically.
    kinds = [s.kind for s in plan.steps]
    if kinds not in (["sql"], ["sql", "python"]):
        return PlanVerdict(
            False, f"unsupported plan shape {kinds}; expected [sql] or [sql, python]")
    return PlanVerdict(True)
