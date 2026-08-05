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


def test_governance_block_detected_from_execution_error():
    # a PII-blocked run carries a "governance violation: ..." execution error; the demo
    # detects it to render a governance callout instead of a generic refusal.
    import demo.app as app
    ex = type("E", (), {"error": "governance violation: query references blocked PII "
                        "columns: user.email"})()
    res = type("R", (), {"execution": ex, "sql": "SELECT email FROM user"})()
    assert app._governance_block(res) == "query references blocked PII columns: user.email"


def test_governance_block_none_for_clean_or_missing_execution():
    import demo.app as app
    clean = type("R", (), {"execution": type("E", (), {"error": ""})(), "sql": "SELECT 1"})()
    missing = type("R", (), {"execution": None, "sql": ""})()
    assert app._governance_block(clean) is None
    assert app._governance_block(missing) is None


def test_chart_png_decoded_from_python_analysis():
    # a Python step's base64 PNG (from the sandbox) is decoded to bytes for st.image.
    import base64
    import demo.app as app
    res = type("R", (), {"python_analysis": {"analysis": {"chart": _REAL_1PX_PNG}}})()
    assert app._chart_png(res) == base64.b64decode(_REAL_1PX_PNG)


def test_chart_png_none_when_absent():
    import demo.app as app
    assert app._chart_png(type("R", (), {"python_analysis": None})()) is None
    assert app._chart_png(type("R", (), {"python_analysis": {"analysis": {}}})()) is None


_REAL_1PX_PNG = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
                 "C0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def _res_with_chart(b64):
    import demo.app as app  # noqa: F401
    return type("R", (), {"python_analysis": {"analysis": {"chart": b64}}})()


def test_chart_png_accepts_a_real_png():
    import demo.app as app
    out = app._chart_png(_res_with_chart(_REAL_1PX_PNG))
    assert out is not None and out.startswith(b"\x89PNG\r\n\x1a\n")


def test_chart_png_rejects_untrusted_bytes_from_the_sandbox():
    import base64
    import demo.app as app
    # not base64 at all
    assert app._chart_png(_res_with_chart("!!!not base64!!!")) is None
    # valid base64 but not a PNG (no magic) -- don't hand arbitrary bytes to the host parser
    assert app._chart_png(_res_with_chart(base64.b64encode(b"GIF89a not a png").decode())) is None
    # a PNG header declaring 10000x1 -- a decompression-bomb shape; reject on dimensions
    bomb = (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR"
            + (10000).to_bytes(4, "big") + (1).to_bytes(4, "big") + b"\x00" * 8)
    assert app._chart_png(_res_with_chart(base64.b64encode(bomb).decode())) is None


def test_governance_block_only_for_a_pii_block_not_a_parse_failure():
    import demo.app as app
    pii = type("R", (), {"execution": type("E", (), {"error": "governance violation: "
              "query references blocked PII columns: user.email"})()})()
    parse = type("R", (), {"execution": type("E", (), {"error": "governance violation: "
                "could not parse SQL for governance: boom"})()})()
    assert app._governance_block(pii) == "query references blocked PII columns: user.email"
    assert app._governance_block(parse) is None


def test_ran_ok_distinguishes_a_real_answer_from_a_refusal():
    import demo.app as app
    ok = type("R", (), {"sql": "SELECT 1",
                        "execution": type("E", (), {"ok": True, "error": ""})()})()
    refusal = type("R", (), {"sql": "",
                             "execution": type("E", (), {"ok": False, "error": ""})()})()
    pii = type("R", (), {"sql": "SELECT email FROM user", "execution": type("E", (), {
        "ok": False, "error": "governance violation: query references blocked PII "
        "columns: user.email"})()})()
    assert app._ran_ok(ok) is True
    assert app._ran_ok(refusal) is False
    assert app._ran_ok(pii) is False
