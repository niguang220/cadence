"""Optional Phoenix / OpenTelemetry tracing.

When ``PHOENIX_ENABLED`` is set, ``setup_phoenix`` wires OpenInference's LangChain
instrumentation to a Phoenix collector, so every agent run shows up in the Phoenix UI
as a span tree (retrieve -> generate -> execute -> ...), with the prompts, responses,
token counts and latency the console tracing already captures -- just in a dashboard.

It is a no-op (and a no-dependency) when disabled, so the core agent never depends on
the observability stack. Enable it with the optional extra: ``pip install ".[observability]"``
plus a running Phoenix (see docker-compose.yml).
"""
from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)
_initialized = False


def setup_phoenix() -> bool:
    """Register OpenInference tracing to a Phoenix collector once, if enabled.

    Returns True when registration succeeded. Idempotent (safe to call per run) and
    fail-soft: a missing extra or a setup error logs a warning and leaves the agent
    running untraced rather than crashing. Note True means *registered*, not that the
    collector is reachable -- OTLP export is async, so an unreachable Phoenix just means
    spans quietly never arrive, no error here.
    """
    global _initialized
    if _initialized or not os.getenv("PHOENIX_ENABLED"):
        return _initialized
    try:
        from phoenix.otel import register
        register(
            project_name=os.getenv("PHOENIX_PROJECT", "datapilot"),
            endpoint=os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces"),
            auto_instrument=True,   # picks up the installed openinference LangChain instrumentor
        )
        _initialized = True
        _log.info("Phoenix tracing enabled")
    except Exception:
        _log.warning("PHOENIX_ENABLED set but tracing setup failed; running untraced",
                     exc_info=True)
    return _initialized
