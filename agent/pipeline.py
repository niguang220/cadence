"""The public ``answer_question`` entry point.

The generation/formatting steps live in ``agent/generation.py``; the wiring that
runs them is the LangGraph state machine in ``agent/graph.py``. ``answer_question``
just builds the model and delegates to that graph, so callers and tests have one
stable entry while the graph grows.
"""
from __future__ import annotations

from pathlib import Path

from agent.db.introspect import Table
from agent.generation import AnswerResult
from agent.llm import create_sql_model


def answer_question(
    db_path: str | Path,
    question: str,
    *,
    model=None,
    k: int = 5,
    tables: list[Table] | None = None,
    semantic_layer: bool = False,
    threshold: float = 0.5,
    clarify: bool = True,
) -> AnswerResult:
    """Answer a question by running the agent graph. Pass ``model`` to inject a
    fake in tests (real use defaults to DeepSeek, needs DEEPSEEK_API_KEY); pass
    ``tables`` to reuse a cached introspection instead of re-reading the schema.
    Pass ``clarify=False`` to bypass clarify_check entirely (used in the ablation
    eval to hold clarification constant across OFF and ON conditions)."""
    from agent.graph import run_agent  # local import avoids a circular import

    model = model or create_sql_model()
    return run_agent(db_path, question, model=model, k=k, tables=tables,
                     semantic_layer=semantic_layer, threshold=threshold,
                     clarify=clarify)


def start_question_session(
    db_path: str | Path,
    question: str,
    *,
    model=None,
    k: int = 5,
    tables: list[Table] | None = None,
    semantic_layer: bool = False,
    threshold: float = 0.5,
    thread_id: str | None = None,
) -> tuple[str, AnswerResult | dict]:
    """Start a LangGraph HITL-capable run.

    Returns ``(thread_id, value)``. ``value`` is either an ``AnswerResult`` or an
    interrupt payload containing ``question`` and ``clarification``.
    """
    from agent.graph import start_agent_session

    model = model or create_sql_model()
    return start_agent_session(
        db_path, question, model=model, k=k, tables=tables,
        semantic_layer=semantic_layer, threshold=threshold, thread_id=thread_id)


def resume_question_session(thread_id: str, response) -> tuple[str, AnswerResult | dict]:
    """Resume a LangGraph HITL run that paused for a clarification or plan approval.

    ``response`` is the human's reply -- a clarification string, or a plan-approval dict
    ``{"decision": "approve"|"reject"|"edit", "plan": [...]?}``. Returns ``(thread_id,
    value)`` like ``start_question_session``: an ``AnswerResult`` when the run completes,
    or the next interrupt payload if it pauses again (e.g. a clarification resume flowing
    into plan approval, or an invalid plan edit re-interrupting).
    """
    from agent.graph import resume_agent_session

    return resume_agent_session(thread_id, response)
