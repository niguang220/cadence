import agent.graph as graphmod
from agent.pipeline import answer_question, start_question_session
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
    """Fake model: records prompts, returns trivial SQL. No bind_tools -> plain path."""
    def __init__(self): self.prompts = []
    def invoke(self, prompt):
        self.prompts.append(prompt if isinstance(prompt, str) else str(prompt))
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
        def __init__(self): self.prompts=[]; self.n=0
        def invoke(self, p):
            self.prompts.append(p if isinstance(p,str) else str(p)); self.n+=1
            return type("R",(),{"content": "SELECT * FROM nope" if self.n==1 else "SELECT 1"})()
        def bind(self, **k): return self
    m = FailThenOK()
    answer_question(db, "what is our MRR by region?", model=m, semantic_layer=True)
    assert m.n >= 2 and "measure:" in m.prompts[-1]
    assert registry.retrieve_calls == 1


def test_hitl_checkpoint_keeps_semantic_metrics_serializable(tmp_path, monkeypatch):
    registry = _only_mrr_registry()
    monkeypatch.setattr(graphmod, "_metric_registry", lambda: registry)
    db = str(build(tmp_path / "saas.db"))
    thread_id, value = start_question_session(
        db,
        "who are the best accounts?",
        model=CapturingModel(),
        semantic_layer=True,
    )
    assert thread_id
    assert not isinstance(value, dict)
    assert registry.retrieve_calls == 1
    preflight = next(t for t in value.trace if t["node"] == "preflight_context")
    assert preflight["semantic_metrics"] == ["mrr"]
    generated = next(t for t in value.trace if t["node"] == "generate_sql")
    assert generated["semantic_metrics"] == ["mrr"]
