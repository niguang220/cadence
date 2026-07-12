"""The gate correctness surface (deterministic, CI-enforced).

route_case calls the PRODUCTION gates -- the eval never re-implements gate logic. The
golden set is deterministic, so its value is a machine-checked routing spec + a
zero-false-refusal / full-refuse-recall property whose teeth are the adversarial
boundary near-misses (a greeting-prefixed or meta-containing data question that must
NOT refuse; an out-of-domain input that must). A regression drops one of the three
CI invariants.
"""
from agent.feasibility import assess_feasibility
from evalharness.gate_eval import GateReport, evaluate_gate, route_case
from evalharness.golden import GateCase, load_gate


def test_route_case_uses_production_gates():
    assert route_case(GateCase("a", "", "out_of_scope")) == "out_of_scope"
    assert route_case(GateCase("b", "how many accounts?", "proceed", recalled_tables=["account"])) == "proceed"
    assert route_case(GateCase("c", "what's the weather?", "feasibility_refuse", recalled_tables=[])) == "feasibility_refuse"


def test_gate_golden_is_perfect_and_has_zero_false_refusals():
    report = evaluate_gate(load_gate())
    assert isinstance(report, GateReport)
    assert report.routing_accuracy == 1.0                 # the routing spec holds
    assert report.intent.precision == 1.0 and report.intent.recall == 1.0
    assert report.feasibility.precision == 1.0 and report.feasibility.recall == 1.0
    # teeth are non-vacuous only because the boundary near-misses are in the set:
    assert report.intent.support >= 1 and report.feasibility.support >= 1


def test_missing_join_case_traces_a_risk_but_still_proceeds():
    case = next(c for c in load_gate() if c.id == "risk_missing_join")
    assert route_case(case) == "proceed"
    verdict = assess_feasibility(case.question, [], case.recalled_tables, [], case.paths)
    assert "possible_missing_join" in verdict.risks
