import agent.graph as graph
from agent.graph import run_agent

class _ScriptModel:
    def __init__(self, *r): self._q = list(r)
    def invoke(self, _p):
        class R: pass
        x = R(); x.content = self._q.pop(0); return x
    def bind(self, **_): return self

def test_two_step_plan_produces_analysis(saas_db, monkeypatch):
    monkeypatch.setattr(graph, "run_in_sandbox",
                        lambda prog, data, **kw: graph.SandboxResult(True, stdout='{"trend": "up"}'))
    model = _ScriptModel(
        '[{"kind":"sql","instruction":"pull mrr by month"},'
        '{"kind":"python","instruction":"describe the trend"}]',
        "SELECT started_on, mrr FROM subscription",
        "import sys,json; print(json.dumps({'trend':'up'}))",
    )
    res = run_agent(saas_db, "what's the mrr trend?", model=model)
    assert res.answer
    assert any(isinstance(s, dict) and s.get("node") == "python_analyze" for s in res.trace)
    # the python step's analysis propagated end-to-end into the user-facing answer
    assert "Analysis:" in res.answer and "trend" in res.answer
