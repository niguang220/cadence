"""Load-smoke for the Streamlit demo: the page renders without any agent/API call.

Skipped when streamlit isn't installed -- CI runs the service-free suite without the
``[demo]`` extra. Run locally after ``pip install -e ".[demo]"``.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "demo" / "app.py")


def test_demo_app_loads_without_agent_run():
    # AppTest.run() renders the script but clicks nothing, so no model is created and no
    # DEEPSEEK_API_KEY is needed -- this is a pure UI load-smoke, service-free.
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Cadence" in t.value for t in at.title)
    assert len(at.text_input) >= 1
    assert any(b.label == "Ask" for b in at.button)


def test_demo_reliability_scorecard_renders():
    # the deterministic scorecard is service-free (no API / no Docker), so clicking its
    # button is a CI-safe check that the harness panel renders real output.
    at = AppTest.from_file(APP, default_timeout=60).run()
    [b for b in at.button if b.label == "Run the reliability checks"][0].click()
    at.run()
    assert not at.exception
    assert any(m.label == "Routing cases" for m in at.metric)


def test_demo_opens_with_the_governance_compare_hook():
    # the page leads with the "same question, two answers, one is wrong" governance hook,
    # not with a plain Ask box -- highest-differentiation screen first. Rendering only (the
    # button needs the API to click), so this is service-free.
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any(b.label == "Run both" for b in at.button)
    assert any("Two answers" in s.value for s in at.subheader)


def test_compare_concurrent_preserves_off_then_on_order(monkeypatch):
    # the opener runs semantic-layer OFF and ON concurrently; the result order must stay
    # (OFF, ON) regardless of which thread finishes first.
    import demo.app as app
    monkeypatch.setattr(app, "_run", lambda q, *, semantic_layer: f"sl={semantic_layer}")
    off, on = app._compare_concurrent("Q")
    assert off == "sl=False" and on == "sl=True"
