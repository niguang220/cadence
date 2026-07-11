"""Tests for the plan-approval HITL gate (edit re-validation state machine).

After ``plan_validate`` and ONLY in HITL, the graph pauses so a human can approve,
edit, or reject the validated plan before it executes. An EDIT is re-validated
(``validate_plan`` + ``assess_feasibility``) -- never trusted just because the
original plan passed -- and a still-invalid edit re-interrupts with a reason,
BOUNDED by ``MAX_APPROVAL_ATTEMPTS`` so there is no infinite human ping-pong. A
non-HITL run never enters this node (baseline byte-identical).

A NON-ambiguous question is used so the FIRST interrupt is ``plan_approval`` (not
``clarify_check``). ``PlanningFakeModel`` answers both the enhance and planner prompts,
so the graph reaches ``plan_validate`` without a real model.
"""
import agent.graph as graph
from agent.pipeline import answer_question, resume_question_session, start_question_session
from conftest import PlanningFakeModel

_Q = "how many accounts?"                      # non-ambiguous -> first interrupt is plan_approval


def _sandbox_spy(monkeypatch):
    calls: list = []
    monkeypatch.setattr(graph, "run_in_sandbox",
                        lambda *a, **k: calls.append(1) or graph.SandboxResult(True, stdout="{}"))
    return calls


def test_first_interrupt_is_an_editable_plan(saas_db):
    model = PlanningFakeModel("SELECT COUNT(*) FROM account")
    _, first = start_question_session(saas_db, _Q, model=model)
    assert isinstance(first, dict)                 # paused for approval, did not run
    assert first.get("plan")                       # the payload is an EDITABLE plan
    assert "approve" in first["message"].lower()


def test_approve_executes_the_plan(saas_db):
    model = PlanningFakeModel("SELECT COUNT(*) FROM account")
    tid, first = start_question_session(saas_db, _Q, model=model)
    assert isinstance(first, dict)

    _, result = resume_question_session(tid, {"decision": "approve"})

    assert result.execution.ok
    nodes = [t.get("node") for t in result.trace]
    assert "plan_approval" in nodes
    assert "generate_sql" in nodes                 # the plan actually executed
    assert result.answer
    approval = [t for t in result.trace if t.get("node") == "plan_approval"][-1]
    assert approval.get("decision") == "approve"


def test_reject_refuses_and_never_executes(saas_db, monkeypatch):
    sandbox = _sandbox_spy(monkeypatch)
    model = PlanningFakeModel("SELECT COUNT(*) FROM account")
    tid, first = start_question_session(saas_db, _Q, model=model)
    assert isinstance(first, dict)

    _, result = resume_question_session(tid, {"decision": "reject"})

    assert not result.execution.ok
    assert "won't run this plan" in result.answer.lower()   # the decline sentence survives respond
    nodes = [t.get("node") for t in result.trace]
    assert "generate_sql" not in nodes and "execute" not in nodes
    assert sandbox == []                           # the sandbox was never reached
    approval = [t for t in result.trace if t.get("node") == "plan_approval"][-1]
    assert approval.get("refused") is True


def test_valid_edit_runs_the_new_plan(saas_db):
    model = PlanningFakeModel("SELECT COUNT(*) FROM account")
    tid, first = start_question_session(saas_db, _Q, model=model)
    edited = [{"kind": "sql", "instruction": "count only active accounts"}]

    _, result = resume_question_session(tid, {"decision": "edit", "plan": edited})

    assert result.execution.ok
    # teeth: the EDITED instruction drove SQL generation, not the original planner step
    assert "count only active accounts" in model.last_prompt
    approval = [t for t in result.trace if t.get("node") == "plan_approval"][-1]
    assert approval.get("decision") == "edit"


def test_invalid_shape_edit_reinterrupts_with_a_reason(saas_db, monkeypatch):
    sandbox = _sandbox_spy(monkeypatch)
    model = PlanningFakeModel("SELECT COUNT(*) FROM account")
    tid, first = start_question_session(saas_db, _Q, model=model)

    # a [python]-first plan is a structurally invalid shape (validate_plan rejects it)
    _, again = resume_question_session(
        tid, {"decision": "edit", "plan": [{"kind": "python", "instruction": "x"}]})

    assert isinstance(again, dict)                  # re-interrupted (paused), did NOT execute
    assert again.get("reason")                      # told the human why the edit was rejected
    assert sandbox == []


def test_empty_instruction_edit_reinterrupts_with_a_reason(saas_db):
    model = PlanningFakeModel("SELECT COUNT(*) FROM account")
    tid, first = start_question_session(saas_db, _Q, model=model)

    _, again = resume_question_session(
        tid, {"decision": "edit", "plan": [{"kind": "sql", "instruction": "   "}]})

    assert isinstance(again, dict)                  # empty instruction re-interrupts, not executed
    assert again.get("reason")


def test_persistent_invalid_edit_is_bounded_and_refuses(saas_db, monkeypatch):
    sandbox = _sandbox_spy(monkeypatch)
    model = PlanningFakeModel("SELECT COUNT(*) FROM account")
    tid, first = start_question_session(saas_db, _Q, model=model)
    bad = {"decision": "edit", "plan": [{"kind": "python", "instruction": "x"}]}

    _, again = resume_question_session(tid, bad)    # 1st invalid edit -> re-interrupt with reason
    assert isinstance(again, dict) and again.get("reason")

    _, result = resume_question_session(tid, bad)   # 2nd invalid edit -> BOUNDED refuse, no loop

    assert not isinstance(result, dict)             # completed instead of pinging forever
    assert not result.execution.ok
    nodes = [t.get("node") for t in result.trace]
    assert "generate_sql" not in nodes
    assert sandbox == []
    approval = [t for t in result.trace if t.get("node") == "plan_approval"][-1]
    assert approval.get("refused") is True


def test_non_hitl_run_never_enters_plan_approval(saas_db):
    # baseline guard: a non-HITL run routes plan_validate straight to dispatch, so the
    # plan_approval node is never on its trace and the pre-Task-5 behavior is preserved.
    model = PlanningFakeModel("SELECT COUNT(*) FROM account")
    res = answer_question(saas_db, _Q, model=model)
    nodes = [t.get("node") for t in res.trace]
    assert "plan_approval" not in nodes
    assert res.execution.ok
