import agent.graph as graphmod
from agent.pipeline import answer_question, resume_question_session, start_question_session
from agent.db.build_saas_db import build
from agent.semantic_layer import format_metrics, load_metrics


class StubMetricRegistry:
    def __init__(self, metrics):
        self._metrics = metrics
        self.retrieve_calls = 0

    def retrieve(self, *a, **k):              # deterministic stub -> no fastembed in CI
        self.retrieve_calls += 1
        return self._metrics

    def format(self, metrics):
        return format_metrics(metrics)


def _only_mrr_registry():
    return StubMetricRegistry([m for m in load_metrics() if m.name == "mrr"])


def _registry_must_not_be_used():
    raise AssertionError("semantic_layer=False must not access metric registry")


class CapturingModel:
    """Fake model: records prompts, returns trivial SQL. No bind_tools -> plain path.
    Plan-aware: a planner prompt yields one SQL step so generation still runs."""
    def __init__(self):
        self.prompts = []
        self.saw_consistency = False
    def invoke(self, prompt):
        text = prompt if isinstance(prompt, str) else str(prompt)
        # semantic_consistency is the LAST model call on a validated SQL step; a pure
        # SIDE-CHANNEL that must NOT append to prompts (else prompts[-1] would be the
        # consistency prompt, not the generation prompt those tests assert on).
        if "semantic-consistency judge" in text:
            self.saw_consistency = True
            return type("R", (), {"content": '{"ok": true}'})()
        self.prompts.append(text)
        # query_enhance runs first on the proceed path; a passthrough keeps generation
        # byte-identical (the metric block is not in the enhance prompt).
        if "governed metric terms" in text:
            return type("R", (), {"content": '{"enhanced_question": ""}'})()
        if text.rstrip().endswith("JSON:") and "Output a JSON array of steps" in text:
            return type("R", (), {"content": '[{"kind": "sql", "instruction": "answer the question"}]'})()
        return type("R", (), {"content": "SELECT 1"})()

def test_semantic_block_injected_when_on(tmp_path, monkeypatch):
    registry = _only_mrr_registry()
    monkeypatch.setattr(graphmod, "_metric_registry", lambda: registry)
    db = str(build(tmp_path / "saas.db"))
    m = CapturingModel()
    result = answer_question(db, "what is our MRR by region?", model=m, semantic_layer=True)
    joined = "\n".join(m.prompts)
    assert "mrr" in joined and "measure:" in joined
    assert registry.retrieve_calls == 1
    preflight = next(t for t in result.trace if t["node"] == "preflight_context")
    assert preflight["semantic_metrics"] == ["mrr"]

def test_no_block_when_off(tmp_path, monkeypatch):
    monkeypatch.setattr(graphmod, "_metric_registry", _registry_must_not_be_used)
    db = str(build(tmp_path / "saas.db"))
    m = CapturingModel()
    answer_question(db, "what is our MRR by region?", model=m, semantic_layer=False)
    joined = "\n".join(m.prompts)
    assert "measure:" not in joined

def test_semantic_block_persists_on_repair(tmp_path, monkeypatch):
    """Governed defs MUST stay injected on repair, else ON is inconsistent."""
    registry = _only_mrr_registry()
    monkeypatch.setattr(graphmod, "_metric_registry", lambda: registry)
    db = str(build(tmp_path / "saas.db"))
    class FailThenOK:                          # 1st SQL fails -> forces a repair turn
        # Plan-aware: the planner is the first invoke (counted in n) but yields a plan,
        # not SQL; _gen indexes generations so the FIRST generation still fails.
        def __init__(self): self.prompts=[]; self.n=0; self._gen=0; self.saw_consistency=False
        def invoke(self, p):
            text = p if isinstance(p,str) else str(p)
            # semantic_consistency is the LAST model call on a validated SQL step; a pure
            # SIDE-CHANNEL that must NOT append to prompts or advance n/_gen.
            if "semantic-consistency judge" in text:
                self.saw_consistency = True
                return type("R",(),{"content": '{"ok": true}'})()
            self.prompts.append(text); self.n+=1
            # query_enhance runs first on the proceed path; a passthrough must NOT
            # advance _gen, else the enhance call would consume the failing draft.
            if "governed metric terms" in text:
                return type("R",(),{"content": '{"enhanced_question": ""}'})()
            if text.rstrip().endswith("JSON:") and "Output a JSON array of steps" in text:
                return type("R",(),{"content": '[{"kind": "sql", "instruction": "x"}]'})()
            self._gen += 1
            return type("R",(),{"content": "SELECT * FROM nope" if self._gen==1 else "SELECT 1"})()
        def bind(self, **k): return self
    m = FailThenOK()
    answer_question(db, "what is our MRR by region?", model=m, semantic_layer=True)
    assert m.n >= 2 and "measure:" in m.prompts[-1]
    assert registry.retrieve_calls == 1


def test_hitl_checkpoint_keeps_semantic_metrics_serializable(tmp_path, monkeypatch):
    registry = _only_mrr_registry()
    monkeypatch.setattr(graphmod, "_metric_registry", lambda: registry)
    db = str(build(tmp_path / "saas.db"))
    thread_id, first = start_question_session(
        db,
        "who are the best accounts?",
        model=CapturingModel(),
        semantic_layer=True,
    )
    assert thread_id
    # The semantic layer suppresses clarification, so the FIRST HITL pause is plan
    # approval; approve to run the plan to completion. This also exercises the metric
    # serialization across the checkpoint boundary, which is the point of this test.
    assert isinstance(first, dict) and first.get("plan")
    _, value = resume_question_session(thread_id, {"decision": "approve"})
    assert not isinstance(value, dict)
    assert registry.retrieve_calls == 1
    preflight = next(t for t in value.trace if t["node"] == "preflight_context")
    assert preflight["semantic_metrics"] == ["mrr"]
    generated = next(t for t in value.trace if t["node"] == "generate_sql")
    assert generated["semantic_metrics"] == ["mrr"]
