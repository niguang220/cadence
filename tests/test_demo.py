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
