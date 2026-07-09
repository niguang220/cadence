"""Deterministic smoke check for HITL clarification behavior.

This is intentionally separate from the semantic-layer ablation. It uses fake
models, makes no API calls, and answers a narrower question: does the
clarification path ask with the right options, resolve valid short replies into
typed intent, and refuse invalid metrics instead of guessing?

Run:
    .venv/bin/python evals/clarification_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from agent.clarify import build_clarification_options
from agent.db.build_demo_db import DB_PATH, build
from agent.db.introspect import introspect
from agent.pipeline import resume_question_session, start_question_session
from agent.semantic_layer import load_metrics


class FakeModel:
    def __init__(self, sql: str):
        self.sql = sql
        self.calls = 0
        self.last_prompt = ""

    def invoke(self, prompt):
        self.calls += 1
        self.last_prompt = prompt
        return type("R", (), {"content": self.sql})()


def _check(failures: list[str], condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def run_smoke(db_path: str | Path) -> list[str]:
    failures: list[str] = []
    db = Path(db_path)

    model = FakeModel(
        "SELECT customer_id, SUM(total) AS total_spend "
        "FROM invoice GROUP BY customer_id ORDER BY total_spend DESC LIMIT 5"
    )
    thread_id, first = start_question_session(db, "who are the best customers?", model=model)
    _check(failures, isinstance(first, dict), "ambiguous question should interrupt")
    if isinstance(first, dict):
        labels = {o["label"] for o in first.get("options", [])}
        confidences = {o.get("confidence") for o in first.get("options", [])}
        _check(failures, "Total" in labels and "Count" in labels,
               "customer options should include a sum and count option")
        _check(failures, confidences == {"fallback"}, "schema-derived options should be fallback")
    _check(failures, model.calls == 0, "model should not be called before clarification")

    result = resume_question_session(thread_id, "sales")
    _check(failures, result.execution.ok, "valid clarification should resume to execution")
    _check(failures, model.calls == 1, "valid clarification should call model once")
    _check(failures, "metric: total" in model.last_prompt, "typed intent should reach prompt")
    clarify_trace = next((t for t in result.trace if t.get("node") == "clarify_check" and t.get("resumed")), {})
    _check(failures, clarify_trace.get("intent_verdict") == "ok", "valid clarification intent should be ok")

    invalid_model = FakeModel("SELECT 1")
    thread_id, first = start_question_session(db, "who are the best customers?", model=invalid_model)
    _check(failures, isinstance(first, dict), "invalid path should start with interrupt")
    invalid = resume_question_session(thread_id, "profit")
    _check(failures, not invalid.execution.ok, "invalid clarification should refuse")
    _check(failures, invalid_model.calls == 0, "invalid clarification should not call model")
    invalid_trace = next((t for t in invalid.trace if t.get("node") == "clarify_check" and t.get("resumed")), {})
    _check(failures, invalid_trace.get("intent_verdict") == "refuse", "invalid intent should be refused")

    metrics = [m for m in load_metrics() if m.name == "mrr"]
    options = build_clarification_options("best accounts", tables=introspect(db), metrics=metrics)
    _check(failures, [o["label"] for o in options] == ["MRR"], "governed metric should suppress fallback options")
    _check(failures, {o.get("confidence") for o in options} == {"governed"}, "semantic options should be governed")
    return failures


def main() -> int:
    db = DB_PATH if DB_PATH.exists() else build()
    failures = run_smoke(db)
    if failures:
        print("CLARIFICATION SMOKE FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("CLARIFICATION SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
