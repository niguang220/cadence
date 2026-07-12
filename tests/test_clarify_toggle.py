"""Tests for the global clarify toggle added in F1.

The toggle lets the ablation eval hold clarification constant across OFF/ON so
the only difference between conditions is the semantic block (clean single-factor).
Default behaviour (clarify=True) must be unchanged.
"""
from agent.pipeline import answer_question
from agent.db.build_saas_db import build

# "active" is in detect_ambiguity's _VAGUE list and has no metric hint, so this
# question reliably triggers a clarification request when clarify=True.
ACTIVE_Q = "How many active users did we have as of 2025-06-30?"


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


def test_clarify_true_default_still_clarifies(tmp_path):
    db = str(build(tmp_path / "saas.db"))
    r = answer_question(db, ACTIVE_Q, model=Fake(), semantic_layer=False)  # default clarify=True
    assert r.clarification is not None


def test_clarify_false_proceeds_to_generation(tmp_path):
    db = str(build(tmp_path / "saas.db"))
    r = answer_question(db, ACTIVE_Q, model=Fake(), semantic_layer=False, clarify=False)
    assert r.clarification is None
    assert any(e.get("node") == "generate_sql" for e in r.trace)
