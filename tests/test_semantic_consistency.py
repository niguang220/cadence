"""Tests for the bounded SemanticConsistency check (the 2nd and last LLM node).

Unit tests pin ``check_semantic_consistency`` with a tiny fake model (parse the
model's JSON verdict; fail OPEN to ``ok=True`` on a broken judge). Graph tests give
the check teeth: a measure mismatch repairs once then proceeds, a persistent mismatch
refuses (never reaching Python), an ``ok`` verdict goes straight to step_advance, and a
governance-blocked result NEVER reaches this LLM judge (Plan 2 invariant). No API key.
"""
import agent.graph as graph
from agent.graph import MAX_ATTEMPTS, run_agent
from agent.db.build_demo_db import build as build_demo_db
from agent.execution import ExecutionResult
from agent.semantic_consistency import check_semantic_consistency


class _Fake:
    def __init__(self, reply): self._r = reply
    def invoke(self, p): return type("R", (), {"content": self._r})()
    def bind(self, **_): return self


# --- unit: check_semantic_consistency ------------------------------------------

def test_ok_verdict():
    m = _Fake('{"ok": true}')
    v = check_semantic_consistency("q", "SELECT 1", ExecutionResult(True), [], m)
    assert v.ok


def test_structured_mismatch_verdict():
    m = _Fake('{"ok": false, "mismatch_kind": "measure", "expected": "average", '
              '"observed": "sum", "evidence": "SUM(...)", "repair_hint": "use AVG"}')
    v = check_semantic_consistency("avg price?", "SELECT SUM(price) FROM t",
                                   ExecutionResult(True), [], m)
    assert not v.ok and v.mismatch_kind == "measure" and v.repair_hint == "use AVG"


def test_non_boolean_ok_fails_open():
    # a malformed verdict ({"ok": null/0/"false"}) is a broken judge -> fail-open ok=True,
    # not an explicit mismatch (would burn the repair budget) nor a fake success.
    for bad in ('{"ok": null}', '{"ok": 0}', '{"ok": "false"}'):
        v = check_semantic_consistency("q", "SELECT 1", ExecutionResult(True), [], _Fake(bad))
        assert v.ok, f"{bad!r} should fail open"


def test_unparseable_defaults_ok():
    # a broken judge must not block a query -> default ok (fail open, bounded elsewhere)
    v = check_semantic_consistency("q", "SELECT 1", ExecutionResult(True), [], _Fake("junk"))
    assert v.ok


def test_catalog_tables_reach_the_prompt():
    # the fix's mechanism: the schema must actually be in the prompt the judge sees.
    from agent.db.introspect import Column, Table
    captured = {}

    class _Capture:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return type("R", (), {"content": '{"ok": true}'})()

        def bind(self, **_):
            return self

    tables = [
        Table(name="account", columns=[Column(name="account_id", type="INTEGER", pk=True, notnull=True)]),
        Table(name="user", columns=[Column(name="user_id", type="INTEGER", pk=True, notnull=True)]),
    ]
    check_semantic_consistency("how many accounts?", 'SELECT COUNT(*) FROM "user"',
                               ExecutionResult(True), tables, _Capture())
    assert "DATABASE TABLES" in captured["prompt"]
    assert "account" in captured["prompt"] and "user" in captured["prompt"]


# --- a purpose-made fake that SCRIPTS the consistency verdict -------------------

class _JudgeModel:
    """Scripts the consistency verdict (in order; the last one is sticky) on the
    consistency prompt, while returning valid SQL on generation. Enhance and planner
    prompts are recognized and answered WITHOUT consuming a scripted verdict, so the
    verdict script maps one-to-one onto the SQL steps. Records that the judge actually
    received the consistency prompt in ``saw_consistency`` (teeth: not ok-by-luck)."""

    def __init__(self, sql, *verdicts, plan=None):
        self._sql = sql
        self._verdicts = list(verdicts) or ['{"ok": true}']
        self._vi = 0
        self._plan = plan or '[{"kind": "sql", "instruction": "count"}]'
        self.saw_consistency = False

    def invoke(self, prompt):
        text = prompt if isinstance(prompt, str) else str(prompt)
        if "semantic-consistency judge" in text:          # the LAST model call, side-scripted
            self.saw_consistency = True
            verdict = self._verdicts[min(self._vi, len(self._verdicts) - 1)]
            self._vi += 1
            return type("R", (), {"content": verdict})()
        if "governed metric terms" in text:               # query_enhance passthrough
            return type("R", (), {"content": '{"enhanced_question": ""}'})()
        if text.rstrip().endswith("JSON:") and "Output a JSON array of steps" in text:
            return type("R", (), {"content": self._plan})()
        return type("R", (), {"content": self._sql})()

    def bind(self, **_):
        return self


# --- graph teeth ---------------------------------------------------------------

def test_measure_mismatch_repairs_once_then_proceeds(saas_db):
    # a measure mismatch triggers ONE repair; a subsequent ok verdict proceeds to answer.
    model = _JudgeModel(
        "SELECT COUNT(*) FROM account",
        '{"ok": false, "mismatch_kind": "measure", "repair_hint": "use AVG not SUM"}',
        '{"ok": true}',
    )
    res = run_agent(saas_db, "how many accounts?", model=model)
    assert model.saw_consistency                          # the judge actually ran
    nodes = [s.get("node") for s in res.trace if isinstance(s, dict)]
    assert nodes.count("generate_sql") == 2               # one repair driven by the mismatch
    assert nodes.count("semantic_consistency") == 2       # judged the draft AND the repair
    assert "step_advance" in nodes and res.answer         # proceeded past the check
    mismatch = [s for s in res.trace
                if isinstance(s, dict) and s.get("node") == "semantic_consistency"][0]
    assert mismatch["ok"] is False and mismatch["mismatch_kind"] == "measure"


def test_persistent_mismatch_refuses_without_reaching_python(saas_db, monkeypatch):
    # a persistent mismatch to budget exhaustion REFUSES (routes to respond, not
    # step_advance) and the Python step is NEVER reached.
    sandbox_calls = []
    monkeypatch.setattr(graph, "run_in_sandbox",
                        lambda *a, **k: sandbox_calls.append(1) or graph.SandboxResult(True, stdout="{}"))
    model = _JudgeModel(
        "SELECT COUNT(*) FROM account",
        '{"ok": false, "mismatch_kind": "measure", "repair_hint": "wrong measure"}',  # sticky
        plan='[{"kind": "sql", "instruction": "pull"}, {"kind": "python", "instruction": "x"}]',
    )
    res = run_agent(saas_db, "how many accounts?", model=model)
    assert model.saw_consistency
    nodes = [s.get("node") for s in res.trace if isinstance(s, dict)]
    assert nodes.count("generate_sql") == MAX_ATTEMPTS    # spent the whole shared budget
    assert "python_generate" not in nodes                 # a known semantic error never reaches Python
    assert "step_advance" not in nodes                    # refused, did not advance the step
    assert sandbox_calls == []                            # the sandbox was never called
    assert "couldn't answer" in res.answer.lower()        # respond refused
    assert "wrong measure" in res.answer                  # surfaced the repair hint as the reason


def test_ok_verdict_goes_straight_to_step_advance(saas_db):
    model = _JudgeModel("SELECT COUNT(*) FROM account", '{"ok": true}')
    res = run_agent(saas_db, "how many accounts?", model=model)
    assert model.saw_consistency
    nodes = [s.get("node") for s in res.trace if isinstance(s, dict)]
    assert nodes.count("generate_sql") == 1               # no repair on an ok verdict
    assert "semantic_consistency" in nodes and "step_advance" in nodes
    assert res.answer


def test_empty_repair_hint_mismatch_still_repairs_not_step_advance(saas_db):
    # An explicit not-ok verdict with NO repair_hint key (-> repair_hint == "") must still
    # route to repair, not silently step_advance ("" is falsy, so a naive `if not error`
    # route would treat a judge-flagged mismatch as consistent -- the exact bug this guards).
    model = _JudgeModel(
        "SELECT COUNT(*) FROM account",
        '{"ok": false, "mismatch_kind": "measure"}',       # no repair_hint key -> ""
        '{"ok": true}',
    )
    res = run_agent(saas_db, "how many accounts?", model=model)
    assert model.saw_consistency
    nodes = [s.get("node") for s in res.trace if isinstance(s, dict)]
    assert nodes.count("generate_sql") == 2                # the empty-hint mismatch triggered a repair
    assert nodes.count("semantic_consistency") == 2        # judged the draft AND the repair
    assert "step_advance" in nodes and res.answer          # proceeded past the check once ok
    mismatch = [s for s in res.trace
                if isinstance(s, dict) and s.get("node") == "semantic_consistency"][0]
    assert mismatch["ok"] is False and mismatch["mismatch_kind"] == "measure"
    assert mismatch["repair_hint"] == ""                   # trace still records the REAL (empty) hint


def test_governance_blocked_result_never_reaches_semantic_consistency(tmp_path, monkeypatch):
    # Plan 2 invariant: a PII/result-governance-blocked result must NEVER be fed to the
    # LLM judge and must NEVER reach Python. The governance_block routes validate ->
    # respond, structurally bypassing semantic_consistency (it lives on the ok branch).
    db = str(build_demo_db(tmp_path / "demo.db"))
    sandbox_calls = []
    monkeypatch.setattr(graph, "run_in_sandbox",
                        lambda *a, **k: sandbox_calls.append(1) or graph.SandboxResult(True, stdout="{}"))
    model = _JudgeModel(
        # sql-governance passes (sees only customer_id); the RESULT column 'email' is a
        # PII name, so run_query's result-governance blocks it (ok=False) before any judge.
        "SELECT customer_id AS email FROM customer",
        '{"ok": true}',                                   # would pass IF the judge ran (it must not)
        plan='[{"kind": "sql", "instruction": "emails"}, {"kind": "python", "instruction": "count"}]',
    )
    res = run_agent(db, "list customer emails", model=model)
    nodes = [s.get("node") for s in res.trace if isinstance(s, dict)]
    assert not model.saw_consistency                      # the judge NEVER received the result
    assert "semantic_consistency" not in nodes            # node absent from the trace
    assert "python_generate" not in nodes                 # governed result never reaches Python
    assert sandbox_calls == []
    assert "governance violation" in res.answer           # refused via the governance gate
