"""Tests for the Phoenix tracing toggle (no server, no observability extra needed)."""
from agent import observability


def test_setup_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(observability, "_initialized", False)
    monkeypatch.delenv("PHOENIX_ENABLED", raising=False)
    assert observability.setup_phoenix() is False   # disabled -> does nothing, no import


def test_setup_fails_soft_without_the_extra(monkeypatch):
    # enabled but the import fails (extra not installed / collector down) -> warn, not crash
    monkeypatch.setattr(observability, "_initialized", False)
    monkeypatch.setenv("PHOENIX_ENABLED", "1")
    monkeypatch.setitem(__import__("sys").modules, "phoenix.otel", None)  # force ImportError
    assert observability.setup_phoenix() is False
