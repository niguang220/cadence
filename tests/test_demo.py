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
