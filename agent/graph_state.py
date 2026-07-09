"""Shared state for the agent graph.

One TypedDict flows through every node. ``total=False`` so nodes only return the
keys they touch; LangGraph merges them. ``trace`` uses an ``operator.add`` reducer
so each node *appends* its step instead of overwriting the list.

Phase 2 adds the reliability loop; the fields it needs (``error``, ``retry_count``,
``clarification``) live here from the start so the loop is a wiring change, not a
state-schema change.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

from agent.execution import ExecutionResult


class AgentState(TypedDict, total=False):
    # inputs (set once at invoke)
    question: str
    db_path: str
    model: Any                       # chat model (real or a fake in tests); omitted in HITL checkpoints
    k: int
    tables: Any                      # cached introspection (list[Table]) or None
    semantic_layer: bool             # inject governed metric definitions into prompts
    threshold: float                 # cosine similarity threshold for metric retrieval (default 0.5)
    clarify: bool                    # when False, skip clarify_check entirely (held constant in ablation)
    hitl: bool                       # when True, clarify_check pauses with LangGraph interrupt
    thread_id: str                   # HITL session id; used to recover non-serializable runtime objects

    # working set
    retrieved_tables: list[str]
    semantic_metrics: list[dict[str, Any]]  # serializable governed metrics bound in preflight
    clarification_options: list[dict[str, Any]]
    clarification_response: str
    normalized_clarification: str
    clarification_intent: dict[str, str]
    schema: str
    sql: str
    error: Optional[str]             # set by validate when the result is bad/suspicious
    repair_kind: str                 # why we're repairing (exec_error / missing_order / ...)
    result: Optional[ExecutionResult]
    attempts: int                    # generations done so far (0 = none yet; >0 = repairing)

    # outputs
    answer: str
    assumptions: list[str]
    clarification: Optional[str]
    trace: Annotated[list[dict], operator.add]
