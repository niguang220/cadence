"""End-to-end wiring check for the Plan 3 node graph (capstone, test-only).

Confirms the six new Plan-3 nodes -- intent_recognition, query_enhance, schema_recall,
table_relation, feasibility_assessment, semantic_consistency -- are wired into the full
graph in the right relative order on a normal data question, and that an out-of-scope
question refuses at intent_recognition, short-circuiting before any of the other five
ever run.
"""
from agent.graph import run_agent
from conftest import PlanningFakeModel


def test_full_graph_e2e_normal_question_visits_plan3_nodes_in_order(saas_db):
    model = PlanningFakeModel("SELECT COUNT(*) FROM account")
    res = run_agent(saas_db, "how many accounts?", model=model)

    assert res.answer
    assert res.execution.ok

    nodes = [s["node"] for s in res.trace if isinstance(s, dict) and "node" in s]

    # Teeth: .index() raises ValueError (failing the test) if a node is MISSING from
    # the trace; the chained "<" comparison fails if a node is present but MIS-ORDERED
    # relative to the others (e.g. table_relation running before schema_recall).
    i_intent = nodes.index("intent_recognition")
    i_enhance = nodes.index("query_enhance")
    i_schema = nodes.index("schema_recall")
    i_relation = nodes.index("table_relation")
    i_feasibility = nodes.index("feasibility_assessment")
    i_consistency = nodes.index("semantic_consistency")
    assert i_intent < i_enhance < i_schema < i_relation < i_feasibility < i_consistency, (
        f"Plan-3 nodes out of order in trace: {nodes}"
    )

    assert "respond" in nodes and nodes[-1] == "respond"
    assert "plan_approval" not in nodes  # non-HITL run never pauses for approval


def test_out_of_scope_question_refuses_at_intent(saas_db):
    model = PlanningFakeModel("SELECT COUNT(*) FROM account")  # never invoked
    res = run_agent(saas_db, "hello", model=model)

    nodes = [s["node"] for s in res.trace if isinstance(s, dict) and "node" in s]
    assert "intent_recognition" in nodes

    intent_entry = next(
        s for s in res.trace if isinstance(s, dict) and s.get("node") == "intent_recognition"
    )
    assert intent_entry.get("refused") is True

    # the refusal short-circuits BEFORE any of the five downstream Plan-3 nodes run
    downstream = ("query_enhance", "schema_recall", "table_relation",
                 "feasibility_assessment", "semantic_consistency")
    assert not any(n in nodes for n in downstream)

    assert not res.sql
    assert not res.execution.ok
