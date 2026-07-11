from agent.planner import plan_query
from agent.plan import Step

class _FakeModel:
    def __init__(self, content): self._content = content
    def invoke(self, _prompt):
        class R: pass
        r = R(); r.content = self._content; return r

def test_parses_sql_only_plan():
    m = _FakeModel('[{"kind": "sql", "instruction": "count accounts"}]')
    plan = plan_query("how many accounts?", "SCHEMA", m)
    assert plan.steps == [Step("sql", "count accounts")]

def test_parses_sql_then_python_plan():
    m = _FakeModel('[{"kind":"sql","instruction":"pull mrr rows"},'
                   '{"kind":"python","instruction":"plot monthly trend"}]')
    plan = plan_query("show the mrr trend", "SCHEMA", m)
    assert [s.kind for s in plan.steps] == ["sql", "python"]

def test_json_in_fenced_block_is_extracted():
    m = _FakeModel('```json\n[{"kind":"sql","instruction":"x"}]\n```')
    assert plan_query("q", "S", m).steps == [Step("sql", "x")]

def test_parses_valid_plan_despite_trailing_bracket_prose():
    # teeth: a greedy [.*] would grab through a later '[ok]' and fail to parse; the
    # valid leading array must still be extracted.
    from agent.planner import _parse_steps
    text = 'Here is the plan: [{"kind": "sql", "instruction": "x"}]\nNotes: [ok]'
    assert _parse_steps(text) == [{"kind": "sql", "instruction": "x"}]

def test_unparseable_returns_empty_plan():
    m = _FakeModel("I cannot plan this")
    assert plan_query("q", "S", m).steps == []
