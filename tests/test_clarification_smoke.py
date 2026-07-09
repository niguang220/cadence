from agent.db.build_demo_db import build
from evals.clarification_smoke import run_smoke


def test_clarification_smoke_passes_with_fake_models(tmp_path):
    db = build(tmp_path / "t.db")
    assert run_smoke(db) == []
