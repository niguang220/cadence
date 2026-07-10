from agent.graph import run_agent
import agent.graph as graph
from agent.db.build_demo_db import build as build_demo_db


class _ScriptModel:
    """Returns queued responses in order; the LAST response is sticky (repeats), so a
    'bad SQL forever' test can exhaust the repair budget without counting attempts."""
    def __init__(self, *responses): self._q = list(responses)
    def invoke(self, _prompt):
        content = self._q.pop(0) if len(self._q) > 1 else self._q[0]
        return type("R", (), {"content": content})()
    def bind(self, **_): return self
    # no bind_tools -> graph uses the plain generation path


def test_sql_only_plan_runs_end_to_end(saas_db):
    model = _ScriptModel(
        '[{"kind":"sql","instruction":"count accounts"}]',   # planner
        "SELECT COUNT(*) FROM account",                       # generate_sql
    )
    res = run_agent(saas_db, "how many accounts?", model=model)
    assert res.sql == "SELECT COUNT(*) FROM account"
    assert res.answer  # non-empty


def test_sql_plus_python_plan_runs_python_step(saas_db, monkeypatch):
    # fake the sandbox so no real Docker runs
    monkeypatch.setattr(graph, "run_in_sandbox",
                        lambda prog, data, **kw: graph.SandboxResult(True, stdout='{"growth": 0.1}'))
    model = _ScriptModel(
        '[{"kind":"sql","instruction":"pull mrr"},{"kind":"python","instruction":"growth"}]',
        "SELECT mrr FROM subscription",                       # generate_sql
        "import sys,json; print(json.dumps({'growth':0.1}))", # generate_python
    )
    res = run_agent(saas_db, "mrr growth?", model=model)
    nodes = [s.get("node") for s in res.trace if isinstance(s, dict)]
    assert "python_analyze" in nodes


def test_invalid_plan_then_refusal(saas_db):
    # planner keeps emitting an empty/invalid plan -> bounded replans -> refuse.
    # The question must retrieve tables so the graph reaches the planner (a question
    # that retrieves nothing would short-circuit at retrieve_schema and never plan).
    model = _ScriptModel("no plan here", "still no plan", "and again")
    res = run_agent(saas_db, "how many accounts?", model=model)
    assert "couldn't" in res.answer.lower() or "can't" in res.answer.lower()


def test_sql_decline_short_circuits_without_execute(saas_db):
    # planner emits a valid SQL plan, but generation declines (CANNOT_ANSWER)
    model = _ScriptModel('[{"kind":"sql","instruction":"count"}]', "CANNOT_ANSWER")
    res = run_agent(saas_db, "how many accounts?", model=model)
    assert "couldn't" in res.answer.lower()
    nodes = [s.get("node") for s in res.trace if isinstance(s, dict)]
    assert "execute" not in nodes and "step_advance" not in nodes


def test_sql_repair_then_python_still_runs(saas_db, monkeypatch):
    monkeypatch.setattr(graph, "run_in_sandbox",
                        lambda prog, data, **kw: graph.SandboxResult(True, stdout='{"g":1}'))
    model = _ScriptModel(
        '[{"kind":"sql","instruction":"pull"},{"kind":"python","instruction":"g"}]',
        "SELECT * FROM no_such_table",             # bad SQL -> exec error -> repair
        "SELECT mrr FROM subscription",            # repaired SQL
        "import sys,json; print(json.dumps({'g':1}))",
    )
    res = run_agent(saas_db, "mrr growth?", model=model)
    nodes = [s.get("node") for s in res.trace if isinstance(s, dict)]
    assert nodes.count("generate_sql") >= 2 and "python_analyze" in nodes


def test_sql_repair_exhausted_refuses_without_python(saas_db):
    # SQL never becomes valid -> MAX_ATTEMPTS -> refuse; python step never reached
    model = _ScriptModel(
        '[{"kind":"sql","instruction":"pull"},{"kind":"python","instruction":"g"}]',
        "SELECT * FROM no_such_table",             # sticky: repeats to exhaust the budget
    )
    res = run_agent(saas_db, "mrr growth?", model=model)
    nodes = [s.get("node") for s in res.trace if isinstance(s, dict)]
    assert "python_generate" not in nodes and "couldn't" in res.answer.lower()


def test_python_null_analysis_still_marked_as_python_step(saas_db, monkeypatch):
    # A python step whose stdout is literally `null` parses to analysis=None. respond
    # must still recognize a python step RAN (keyed on python_analysis presence, not a
    # None sentinel), else a null result is silently misreported as SQL-only.
    monkeypatch.setattr(graph, "run_in_sandbox",
                        lambda prog, data, **kw: graph.SandboxResult(True, stdout='null'))
    model = _ScriptModel(
        '[{"kind":"sql","instruction":"pull mrr"},{"kind":"python","instruction":"x"}]',
        "SELECT mrr FROM subscription",
        "import sys,json; print('null')",
    )
    res = run_agent(saas_db, "mrr?", model=model)
    nodes = [s.get("node") for s in res.trace if isinstance(s, dict)]
    assert "python_analyze" in nodes                     # the python step did run
    respond = [s for s in res.trace if isinstance(s, dict) and s.get("node") == "respond"][-1]
    assert respond.get("python_analysis") is True        # not misclassified as SQL-only


def test_governed_sql_result_never_dispatches_python(tmp_path, monkeypatch):
    # Cross-component regression guard: a PII-governed SQL result must refuse BEFORE any
    # python step runs, so untrusted compute never touches governed rows. Enforced today
    # by run_query's result-governance gate (governed result -> ok=False -> validate
    # refuses as governance_block -> python never dispatched). This locks the invariant
    # regardless of WHICH component enforces it: if a future change removes
    # result-governance from run_query or reroutes governance_block, the sandbox would
    # run on governed data and this test turns red.
    db = str(build_demo_db(tmp_path / "demo.db"))
    sandbox_calls = []
    def _spy(prog, data, **kw):
        sandbox_calls.append(1)
        return graph.SandboxResult(True, stdout='{"x": 1}')
    monkeypatch.setattr(graph, "run_in_sandbox", _spy)
    model = _ScriptModel(
        # a [sql, python] plan: absent the guarantee, python WOULD be dispatched
        '[{"kind":"sql","instruction":"emails"},{"kind":"python","instruction":"count"}]',
        # sql-governance passes (sees only customer_id); the RESULT column 'email' is a
        # PII name, so run_query's result-governance blocks it (ok=False).
        "SELECT customer_id AS email FROM customer",
    )
    res = run_agent(db, "list customer emails", model=model)
    nodes = [s.get("node") for s in res.trace if isinstance(s, dict)]
    assert sandbox_calls == []                            # sandbox never called
    assert "python_generate" not in nodes                # python step never dispatched
    assert "python_analyze" not in nodes
    assert "governance violation" in res.answer           # refused via the governance gate
