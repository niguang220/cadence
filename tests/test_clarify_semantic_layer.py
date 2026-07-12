import agent.graph as graphmod
from agent.pipeline import answer_question
from agent.db.build_saas_db import build
from agent.semantic_layer import format_metrics, load_metrics


class StubMetricRegistry:
    def retrieve(self, *a, **k):          # stub: pretend a governed metric matches
        return [m for m in load_metrics() if m.name == "active_user"]

    def format(self, metrics):
        return format_metrics(metrics)


def _active_registry():
    return StubMetricRegistry()

class Fake:
    saw_consistency = False
    def invoke(self, p):
        text = p if isinstance(p, str) else str(p)
        # semantic_consistency is the LAST model call on a validated SQL step; recognize
        # it as a pure SIDE-CHANNEL passthrough (keeps this fake robust if its SQL ever
        # reaches the validate-ok path).
        if "semantic-consistency judge" in text:
            self.saw_consistency = True
            return type("R", (), {"content": '{"ok": true}'})()
        # query_enhance runs first on the proceed path; a passthrough keeps generation
        # byte-identical.
        if "governed metric terms" in text:
            return type("R", (), {"content": '{"enhanced_question": ""}'})()
        # plan-aware: the first model call is the planner; yield one SQL step so
        # generation still runs and reaches generate_sql.
        if text.rstrip().endswith("JSON:") and "Output a JSON array of steps" in text:
            return type("R", (), {"content": '[{"kind": "sql", "instruction": "answer the question"}]'})()
        return type("R", (), {"content": "SELECT 1"})()

ACTIVE_Q = "How many active users did we have as of 2025-06-30?"

def test_off_still_clarifies(tmp_path):
    db = str(build(tmp_path / "saas.db"))
    r = answer_question(db, ACTIVE_Q, model=Fake(), semantic_layer=False)
    assert r.clarification is not None          # unchanged OFF behavior: it asks

def test_on_with_governed_metric_does_not_clarify(tmp_path, monkeypatch):
    monkeypatch.setattr(graphmod, "_metric_registry", _active_registry)
    db = str(build(tmp_path / "saas.db"))
    r = answer_question(db, ACTIVE_Q, model=Fake(), semantic_layer=True)
    assert r.clarification is None              # governed metric resolves it -> proceeds
    # and it actually generated (reached generate_sql)
    assert any(e.get("node") == "generate_sql" for e in r.trace)
