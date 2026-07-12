"""The gate correctness surface: deterministic routing accuracy of the two gates.

``route_case`` calls the PRODUCTION ``classify_intent`` and ``assess_feasibility``
directly (never a re-implementation), running intent first and feasibility only when
intent returns "data" -- against the case's INJECTED ``recalled_tables`` so retrieval
flakiness never enters this number. The combined ``routing_accuracy`` is the per-case
terminal route; the per-gate metrics attribute errors. Feasibility metrics are computed
only over cases that actually reach feasibility (see ``evaluate_gate``).
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.feasibility import assess_feasibility
from agent.intent import classify_intent
from evalharness.classification_metrics import ClassMetrics, accuracy, binary_metrics
from evalharness.golden import GateCase


def route_case(case: GateCase) -> str:
    """The terminal route: out_of_scope | feasibility_refuse | proceed."""
    if classify_intent(case.question).kind == "out_of_scope":
        return "out_of_scope"
    # tables=[]/metrics=[] are a deliberate no-op: assess_feasibility ignores them today
    # (only recalled + paths drive it). FORWARD RISK: if feasibility is ever extended to
    # read tables/metrics, this surface would silently diverge from production -- that
    # change must feed real context here (and ship a test that catches the divergence).
    verdict = assess_feasibility(case.question, [], case.recalled_tables, [], case.paths)
    return "proceed" if verdict.feasible else "feasibility_refuse"


@dataclass
class GateReport:
    routing_accuracy: float
    intent: ClassMetrics          # positive = out_of_scope (a refusal at the intent gate)
    feasibility: ClassMetrics     # positive = refuse (over cases that reached feasibility)
    n: int


def evaluate_gate(cases: list[GateCase]) -> GateReport:
    routes_true = [c.expected_route for c in cases]
    routes_pred = [route_case(c) for c in cases]

    # Intent (positive = out_of_scope), over ALL cases. An out_of_scope expected route
    # is an intent-refusal; anything else means intent should let it through ("data").
    intent_true = ["out_of_scope" if r == "out_of_scope" else "data" for r in routes_true]
    intent_pred = [classify_intent(c.question).kind for c in cases]

    # Feasibility (positive = refuse), ONLY over cases that reached feasibility: those
    # designed to test it (expected_route in {feasibility_refuse, proceed}) whose intent
    # actually passed. Intent-short-circuited cases are excluded, never scored here.
    feas = [c for c in cases
            if c.expected_route in {"feasibility_refuse", "proceed"}
            and classify_intent(c.question).kind == "data"]
    feas_true = ["refuse" if c.expected_route == "feasibility_refuse" else "feasible" for c in feas]
    feas_pred = ["feasible" if assess_feasibility(c.question, [], c.recalled_tables, [], c.paths).feasible
                 else "refuse" for c in feas]

    return GateReport(
        routing_accuracy=accuracy(routes_true, routes_pred),
        intent=binary_metrics(intent_true, intent_pred, positive="out_of_scope"),
        feasibility=binary_metrics(feas_true, feas_pred, positive="refuse"),
        n=len(cases),
    )
