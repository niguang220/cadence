"""Tests for the QueryEnhance pre-step node (LLM rewrite + governed-metric guardrail).

The unit tests pin ``enhance_query`` directly with a tiny fake model; the graph tests
prove the original/enhanced boundary (enhanced feeds retrieval/planner/generation, the
original is preserved), that the governed metric block still reaches generation after
enhancement, and that a HITL clarification resume flows THROUGH query_enhance without a
model lookup KeyError. No API key, no tokens.
"""
from agent.query_enhance import enhance_query
from agent.semantic_layer import MetricDef


class _Fake:
    def __init__(self, reply):
        self._r = reply

    def invoke(self, prompt):
        self.prompt = prompt
        return type("R", (), {"content": self._r})()

    def bind(self, **_):
        return self


# The real MetricDef has 7 required fields; enhance_query only reads ``name``, so the
# extra fields are valid placeholders. Keep the assertions as specified in the brief.
def _metric(name):
    return MetricDef(name, [], "logged in >=1", "count", "user", [], "")


# --- unit: enhance_query -------------------------------------------------------

def test_enhance_returns_structured_result():
    m = _Fake('{"enhanced_question": "count active users this month", '
              '"rewrite_diff": "added time window", "warnings": []}')
    r = enhance_query("count active users", [_metric("active users")], m)
    assert r.enhanced_question and "active users" in r.governed_terms


def test_enhance_falls_back_to_original_when_governed_term_dropped():
    # teeth: the rewrite silently removed the governed term -> guardrail must KEEP the
    # governed term (fall back to original), not merely warn, or downstream loses it.
    m = _Fake('{"enhanced_question": "count people this month", "warnings": []}')
    r = enhance_query("count active users", [_metric("active users")], m)
    assert "active users" in r.enhanced_question.lower() and r.warnings


def test_enhance_unparseable_falls_back_to_original():
    m = _Fake("not json")
    r = enhance_query("count users", [], m)
    assert r.enhanced_question == "count users" and r.rewrite_diff == ""


# --- graph boundary: the original/enhanced split -------------------------------

def test_enhanced_noop_keeps_sql_task_and_retrieval_byte_identical():
    # baseline guard: when the enhancement is a no-op (empty or identical to the
    # original) the SQL-generation task and the retrieval input are byte-identical to
    # today's, so the non-enhanced path is preserved exactly.
    from agent.graph import _retrieval_question, _sql_task
    base = {"question": "how many accounts?", "schema": "S",
            "plan": [{"kind": "sql", "instruction": "count accounts"}], "step_index": 0}
    baseline_task, baseline_q = _sql_task(base), _retrieval_question(base)
    for noop in ("", base["question"]):
        s = {**base, "enhanced_question": noop}
        assert _sql_task(s) == baseline_task
        assert _retrieval_question(s) == baseline_q
    # a real rewrite feeds BOTH the original and the enhanced question into generation,
    # and drives retrieval/planning off the enhanced form -- but never mutates the original.
    changed = {**base, "enhanced_question": "how many active accounts this quarter?"}
    assert _sql_task(changed) != baseline_task
    assert "how many accounts?" in _sql_task(changed)                      # original kept
    assert "how many active accounts this quarter?" in _sql_task(changed)  # enhanced added
    assert _retrieval_question(changed) == "how many active accounts this quarter?"


def test_governed_metric_block_survives_enhancement(tmp_path, monkeypatch):
    # a governed-metric question still injects the metric block into SQL generation
    # AFTER the enhancement node runs (enhancement must not strip the governed defs).
    import agent.graph as graphmod
    from agent.db.build_saas_db import build
    from agent.pipeline import answer_question
    from agent.semantic_layer import format_metrics, load_metrics
    from conftest import PlanningFakeModel

    class _StubReg:
        def retrieve(self, *a, **k):
            return [m for m in load_metrics() if m.name == "mrr"]
        def format(self, metrics):
            return format_metrics(metrics)

    monkeypatch.setattr(graphmod, "_metric_registry", lambda: _StubReg())
    db = str(build(tmp_path / "saas.db"))
    model = PlanningFakeModel("SELECT 1")
    res = answer_question(db, "what is our MRR by region?", model=model, semantic_layer=True)
    assert model.saw_enhance                                       # the enhance node ran
    assert any(t.get("node") == "query_enhance" for t in res.trace)
    # the governed metric block reached the (last) generation prompt after enhancement
    assert "measure:" in model.last_prompt and "mrr" in model.last_prompt.lower()


def test_hitl_resume_flows_through_query_enhance(tmp_path):
    # Critical: a HITL checkpoint omits the model, so query_enhance must recover it via
    # the thread's model registry. A clarification resume routes THROUGH query_enhance;
    # it must not raise a KeyError/RuntimeError looking up the model.
    from agent.db.build_demo_db import build
    from agent.pipeline import resume_question_session, start_question_session
    from conftest import PlanningFakeModel

    db = build(tmp_path / "t.db")
    model = PlanningFakeModel(
        "SELECT customer_id, SUM(total) AS total_spend "
        "FROM invoice GROUP BY customer_id ORDER BY total_spend DESC LIMIT 5")
    thread_id, first = start_question_session(db, "who are the best customers?", model=model)
    assert isinstance(first, dict)                     # paused for clarification; enhance not yet run
    assert not model.saw_enhance
    result = resume_question_session(thread_id, "sales")   # resumes THROUGH query_enhance
    assert result.execution.ok                         # no KeyError getting the model on resume
    assert model.saw_enhance                           # enhance ran on the HITL resume path
    assert any(t.get("node") == "query_enhance" for t in result.trace)
