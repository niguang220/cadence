"""The agent as a LangGraph state machine.

    preflight_context
    -> clarify_check (Phase 2/#8)
    -> retrieve_schema --(no tables)--> END (refuse)
    -> generate_sql ---(CANNOT_ANSWER)-> END (decline)
    -> execute -> validate --(ok)--------> respond -> END
                          \\--(repair & attempts<MAX)--> generate_sql

Clarification is deterministic and traceable: ``clarify_check`` asks only when
the question is ambiguous, maps the human reply into a typed metric intent, and
refuses invalid intent instead of guessing.

``generate_sql`` is the SINGLE place SQL is written, for both the first draft and
repairs: on a retry it gets the failing query + the problem fed back, and (for a
real model) a higher temperature so it doesn't regenerate the same broken SQL.
This keeps the full context (question + schema + retry hint) on every attempt --
a separate, context-starved "repair" node fails on schema-linking errors.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agent.clarify import (
    build_clarification_options,
    detect_ambiguity,
    format_clarification_intent,
    format_clarification_prompt,
    normalize_clarification_response,
    parse_clarification_intent,
)
from agent.db.introspect import introspect, render_schema
from agent.execution import ExecutionResult, run_query
from agent.graph_state import AgentState
from agent.generation import (AnswerResult, _extract_sql, _format_answer, _NO_QUERY,
                             generate_sql)
from agent.governance import check_result_governance, check_sql_governance
from agent.hybrid_retriever import retrieve
from agent.observability import setup_phoenix
from agent.prompts import (CANNOT_ANSWER, REPAIR_INSTRUCTION, REPAIR_PROMPT,
                           TOOL_SYSTEM_PROMPT)
from agent.semantic_layer import MetricDef, MetricRegistry
from agent.tools import build_get_schema_tool
from agent.usage import UsageCallback
from agent.validate import validate_result

MAX_ATTEMPTS = 3            # total generations (1 draft + up to 2 repairs)
MAX_TOOL_ROUNDS = 2         # times the model may call get_schema before it must answer
_RETRY_TEMPERATURE = 0.3    # temp 0 would regenerate the identical broken SQL
# Worst case = MAX_ATTEMPTS * (MAX_TOOL_ROUNDS + 1) = 9 LLM calls; the typical path
# is 1 (no tool, no repair). The two budgets are independent and both bounded.

_METRIC_REGISTRY = None
_HITL_CHECKPOINTER = InMemorySaver()
# Checkpoints must be serializable, so the runtime model is kept out of graph state.
# This in-memory registry is enough for the local demo; a service deployment would
# rebuild the model from provider config per thread/process.
_HITL_MODELS = {}


def _metric_registry() -> MetricRegistry:
    global _METRIC_REGISTRY
    if _METRIC_REGISTRY is None:
        _METRIC_REGISTRY = MetricRegistry.load()
    return _METRIC_REGISTRY


def _serialize_metrics(metrics: list[MetricDef]) -> list[dict]:
    return [asdict(m) for m in metrics]


def _deserialize_metrics(metrics: list[dict]) -> list[MetricDef]:
    return [MetricDef(**m) for m in metrics]


def _semantic_metrics(state: AgentState) -> list[MetricDef]:
    """Return metrics bound for this request.

    Preflight owns metric retrieval and stores the result in graph state so later
    nodes do not independently re-run retrieval and drift from the options shown
    to the user. The fallback keeps direct unit tests of downstream nodes usable.
    """
    if "semantic_metrics" in state:
        return _deserialize_metrics(state.get("semantic_metrics", []))
    if not state.get("semantic_layer"):
        return []
    thresh = state.get("threshold", 0.5)
    return _metric_registry().retrieve(state["question"], threshold=thresh)


def _preflight_context(state: AgentState) -> dict:
    tables = state.get("tables") or introspect(state["db_path"])
    thresh = state.get("threshold", 0.5)
    metrics = (
        _metric_registry().retrieve(state["question"], threshold=thresh)
        if state.get("semantic_layer") else []
    )
    options = build_clarification_options(state["question"], tables=tables, metrics=metrics)
    out = {
        "semantic_metrics": _serialize_metrics(metrics),
        "clarification_options": options,
        "trace": [{
            "node": "preflight_context",
            "tables": len(tables),
            "semantic_metrics": [m.name for m in metrics],
            "clarification_options": [o["label"] for o in options],
        }],
    }
    if not state.get("hitl"):
        out["tables"] = tables
    return out


def _clarify_check(state: AgentState) -> dict:
    if not state.get("clarify", True):
        # Global toggle: bypass clarification entirely so the ablation eval can hold
        # this factor constant across OFF and ON conditions (clean single-factor design).
        return {"trace": [{"node": "clarify_check", "skipped": True}]}
    clarification = detect_ambiguity(state["question"])
    options = state.get("clarification_options", [])
    if clarification and state.get("semantic_layer"):
        # The semantic layer governs this term -- check if a metric resolves it.
        # If so, suppress the clarification: the governed definition will be injected
        # into generation, making an upfront question unnecessary.
        if _semantic_metrics(state):
            return {"trace": [{"node": "clarify_check", "ambiguous": True,
                               "resolved_by_semantic_layer": True}]}
    if clarification:
        prompt = format_clarification_prompt(clarification, options)
        if state.get("hitl"):
            payload = {"question": state["question"], "clarification": prompt}
            if options:
                payload["options"] = options
            response = interrupt(payload)
            return _resolve_clarification_intent(state, str(response), options)
        # ask instead of guessing -- the question has no single right answer
        return {
            "clarification": prompt,
            "answer": prompt,
            "result": ExecutionResult(False, error="(clarification requested)"),
            "trace": [{
                "node": "clarify_check",
                "ambiguous": True,
                "clarification_options": [o["label"] for o in options],
            }],
        }
    return {"trace": [{"node": "clarify_check", "ambiguous": False}]}


def _resolve_clarification_intent(
    state: AgentState,
    response: str,
    options: list[dict],
) -> dict:
    normalized = normalize_clarification_response(response)
    intent = parse_clarification_intent(response, options)
    if intent:
        return {
            "question": f"{state['question']}\nClarification: {normalized}.",
            "clarification": None,
            "clarification_response": response,
            "normalized_clarification": normalized,
            "clarification_intent": intent,
            "trace": [{
                "node": "clarify_check",
                "ambiguous": True,
                "resumed": True,
                "clarification_response": response,
                "normalized_clarification": normalized,
                "clarification_intent": intent,
                "intent_verdict": "ok",
                "clarification_options": [o["label"] for o in options],
            }],
        }

    answer = (
        "I couldn't map that clarification to one of the available metrics. "
        "I can't answer this question reliably without a defined metric."
    )
    return {
        "clarification": None,
        "answer": answer,
        "result": ExecutionResult(False, error="(invalid clarification)"),
        "trace": [{
            "node": "clarify_check",
            "ambiguous": True,
            "resumed": True,
            "clarification_response": response,
            "normalized_clarification": normalized,
            "clarification_intent": None,
            "intent_verdict": "refuse",
            "clarification_options": [o["label"] for o in options],
        }],
    }


def _retrieve_schema(state: AgentState) -> dict:
    tables = state.get("tables")
    if tables is None:
        tables = introspect(state["db_path"])
    top_k = retrieve(state["question"], tables, k=state.get("k", 5))
    if not top_k:
        # honest refusal: don't dump the whole schema and let the model
        # hallucinate a query for an off-topic question.
        return {
            "tables": tables,
            "retrieved_tables": [],
            "result": ExecutionResult(False, error=_NO_QUERY),
            "answer": "I couldn't identify any tables relevant to this question.",
            "trace": [{"node": "retrieve_schema", "tables": [], "retrieval_failed": True}],
        }
    schema = render_schema(tables, only=top_k, include_fk_neighbors=True)
    return {
        "tables": tables,
        "retrieved_tables": top_k,
        "schema": schema,
        "trace": [{"node": "retrieve_schema", "tables": top_k}],
    }


def _generate_plain(state: AgentState, model, attempts: int, block: str = "") -> str:
    """Single-shot generation for models without tool support (e.g. test fakes)."""
    if attempts == 0:
        return generate_sql(state["question"], state["schema"], model,
                           semantic_block=block)
    prompt = REPAIR_PROMPT.format(
        schema=state["schema"], question=state["question"],
        failed_sql=state.get("sql", ""), problem=state.get("error", ""),
        semantic_block=block,
    )
    if hasattr(model, "bind"):                          # real model: raise temperature
        model = model.bind(temperature=_RETRY_TEMPERATURE)
    response = model.invoke(prompt)
    return _extract_sql(getattr(response, "content", response))


def _generate_with_tools(state: AgentState, model, attempts: int, requested: list[str],
                         block: str = "") -> str:
    """Generation with a get_schema tool the model can call to pull in a table the
    retriever missed -- so schema-linking misses are recoverable. Bounded tool rounds."""
    tables = state["tables"]
    tool = build_get_schema_tool(tables, requested)
    bound = model.bind_tools([tool])
    if attempts > 0 and hasattr(bound, "bind"):
        bound = bound.bind(temperature=_RETRY_TEMPERATURE)

    system = TOOL_SYSTEM_PROMPT.format(
        catalog=", ".join(sorted(t.name for t in tables)), schema=state["schema"],
        semantic_block=block)
    if attempts == 0:
        human = state["question"]
    else:
        human = REPAIR_INSTRUCTION.format(
            question=state["question"], failed_sql=state.get("sql", ""),
            problem=state.get("error", ""), semantic_block=block)
    messages = [SystemMessage(content=system), HumanMessage(content=human)]

    for _ in range(MAX_TOOL_ROUNDS):
        response = bound.invoke(messages)
        calls = getattr(response, "tool_calls", None)
        if not calls:
            return _extract_sql(getattr(response, "content", response))
        messages.append(response)
        for call in calls:
            result = tool.invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call.get("id", "")))
    # tool budget spent: one final turn to force a query
    response = bound.invoke(messages)
    return _extract_sql(getattr(response, "content", response))


def _model_for_state(state: AgentState, config):
    thread_id = state.get("thread_id") or (config or {}).get("configurable", {}).get("thread_id")
    return _HITL_MODELS.get(thread_id)


def _generate_sql(state: AgentState, config=None) -> dict:
    attempts = state.get("attempts", 0)
    model = state.get("model") or _model_for_state(state, config)
    if model is None:
        raise RuntimeError("No model available for SQL generation")
    requested: list[str] = []

    metric_block = ""
    mets = []
    if state.get("semantic_layer"):
        registry = _metric_registry()
        mets = _semantic_metrics(state)
        metric_block = registry.format(mets)
    block = format_clarification_intent(state.get("clarification_intent")) + metric_block

    if hasattr(model, "bind_tools"):
        sql = _generate_with_tools(state, model, attempts, requested, block)
    else:
        sql = _generate_plain(state, model, attempts, block)

    entry = {"node": "generate_sql", "sql": sql, "attempt": attempts + 1}
    if requested:
        entry["requested_tables"] = requested
    if mets:
        entry["semantic_metrics"] = [m.name for m in mets]
    if state.get("clarification_intent"):
        entry["clarification_intent"] = state["clarification_intent"]
    out = {"attempts": attempts + 1, "sql": sql, "error": None}
    if not sql or sql.strip().upper().startswith(CANNOT_ANSWER):
        out.update({
            "result": ExecutionResult(False, error=_NO_QUERY),
            "answer": "I couldn't write a reliable query for this question.",
            "trace": [{**entry, "declined": True}],
        })
    else:
        out["trace"] = [entry]
    return out


def _execute(state: AgentState) -> dict:
    governance = check_sql_governance(state["sql"], state["tables"])
    if governance.ok:
        execution = run_query(state["db_path"], state["sql"], tables=state["tables"])
    else:
        execution = ExecutionResult(False, error=f"governance violation: {governance.reason}")
    trace = {"node": "execute", "ok": execution.ok,
             "rows": len(execution.rows), "error": execution.error}
    trace["governance"] = (
        "blocked"
        if not governance.ok or execution.error.startswith("governance violation:")
        else "ok"
    )
    if governance.columns:
        trace["blocked_columns"] = governance.columns
    return {
        "result": execution,
        "trace": [trace],
    }


def _validate(state: AgentState) -> dict:
    v = validate_result(state["question"], state["sql"], state["result"])
    if v.ok:
        return {"error": None, "trace": [{"node": "validate", "verdict": "ok"}]}
    return {
        "error": v.reason,
        "repair_kind": v.repair_kind,
        "trace": [{"node": "validate", "verdict": "repair", "kind": v.repair_kind}],
    }


def _respond(state: AgentState) -> dict:
    if state.get("error"):                              # exhausted the repair budget
        return {"answer": f"I couldn't answer that reliably: {state['error']}.",
                "trace": [{"node": "respond", "refused": True, "error": state["error"]}]}
    governance = check_result_governance(state["result"].columns, state["tables"])
    if not governance.ok:
        return {"answer": f"I couldn't answer that reliably: {governance.reason}.",
                "trace": [{"node": "respond", "refused": True,
                           "governance": "blocked",
                           "blocked_columns": governance.columns}]}
    return {"answer": _format_answer(state["result"]), "trace": [{"node": "respond"}]}


def _route_after_clarify(state: AgentState) -> str:
    if state.get("answer"):
        return END
    return END if state.get("clarification") else "retrieve_schema"


def _route_after_retrieve(state: AgentState) -> str:
    return "generate_sql" if state.get("retrieved_tables") else END


def _route_after_generate(state: AgentState) -> str:
    # _generate_sql sets `answer` only when it declines (CANNOT_ANSWER / empty).
    return END if state.get("answer") else "execute"


def _route_after_validate(state: AgentState) -> str:
    if not state.get("error"):
        return "respond"
    if state.get("repair_kind") == "governance_block":
        return "respond"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:       # bounded: give up and refuse
        return "respond"
    return "generate_sql"                              # repair: re-enter with the hint


def _build(*, checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("preflight_context", _preflight_context)
    g.add_node("clarify_check", _clarify_check)
    g.add_node("retrieve_schema", _retrieve_schema)
    g.add_node("generate_sql", _generate_sql)
    g.add_node("execute", _execute)
    g.add_node("validate", _validate)
    g.add_node("respond", _respond)
    g.add_edge(START, "preflight_context")
    g.add_edge("preflight_context", "clarify_check")
    g.add_conditional_edges("clarify_check", _route_after_clarify,
                            {"retrieve_schema": "retrieve_schema", END: END})
    g.add_conditional_edges("retrieve_schema", _route_after_retrieve,
                            {"generate_sql": "generate_sql", END: END})
    g.add_conditional_edges("generate_sql", _route_after_generate,
                            {"execute": "execute", END: END})
    g.add_edge("execute", "validate")
    g.add_conditional_edges("validate", _route_after_validate,
                            {"respond": "respond", "generate_sql": "generate_sql"})
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer)


_AGENT = _build()
_HITL_AGENT = _build(checkpointer=_HITL_CHECKPOINTER)


def _config(callbacks: list | None = None, thread_id: str | None = None) -> dict:
    config: dict = {}
    if callbacks is not None:
        config["callbacks"] = callbacks
    if thread_id is not None:
        config["configurable"] = {"thread_id": thread_id}
    return config


def _to_answer(final: AgentState, usage: UsageCallback) -> AnswerResult:
    return AnswerResult(
        question=final["question"],
        retrieved_tables=final.get("retrieved_tables", []),
        sql=final.get("sql", ""),
        execution=final.get("result") or ExecutionResult(False, error=_NO_QUERY),
        answer=final.get("answer", ""),
        assumptions=final.get("assumptions", []),
        clarification=final.get("clarification"),
        trace=final.get("trace", []),
        usage=usage.summary(),
    )


def run_agent(db_path: str | Path, question: str, *, model, k: int = 5,
              tables=None, semantic_layer: bool = False,
              threshold: float = 0.5, clarify: bool = True) -> AnswerResult:
    """Invoke the compiled graph and map the final state to an AnswerResult."""
    setup_phoenix()  # no-op unless PHOENIX_ENABLED; sends span trees to the Phoenix UI
    # a callback captures token + latency for every model call nested in the nodes,
    # so the generate code carries no tracing plumbing. This relies on LangChain
    # propagating the config callbacks down to the raw model.invoke()s inside nodes
    # (verified working with DeepSeek; a version coupling worth knowing).
    usage = UsageCallback()
    final = _AGENT.invoke({
        "question": question,
        "db_path": str(db_path),
        "model": model,
        "k": k,
        "tables": tables,
        "semantic_layer": semantic_layer,
        "threshold": threshold,
        "clarify": clarify,
        "trace": [],
    }, config=_config(callbacks=[usage]))
    return _to_answer(final, usage)


def start_agent_session(db_path: str | Path, question: str, *, model, k: int = 5,
                        tables=None, semantic_layer: bool = False,
                        threshold: float = 0.5,
                        thread_id: str | None = None) -> tuple[str, AnswerResult | dict]:
    """Start a HITL-capable run. Returns ``(thread_id, result_or_interrupt)``.

    If the graph pauses for clarification, the second value is the interrupt payload:
    ``{"question": ..., "clarification": ...}``. Otherwise it is an ``AnswerResult``.
    Resume an interrupted run with ``resume_agent_session`` and the same thread id.
    """
    setup_phoenix()
    thread_id = thread_id or str(uuid4())
    _HITL_MODELS[thread_id] = model
    usage = UsageCallback()
    out = _HITL_AGENT.invoke({
        "question": question,
        "db_path": str(db_path),
        "k": k,
        "tables": tables,
        "semantic_layer": semantic_layer,
        "threshold": threshold,
        "clarify": True,
        "hitl": True,
        "thread_id": thread_id,
        "trace": [],
    }, config=_config(callbacks=[usage], thread_id=thread_id))
    interrupts = out.get("__interrupt__") if isinstance(out, dict) else None
    if interrupts:
        return thread_id, interrupts[0].value
    _HITL_MODELS.pop(thread_id, None)
    return thread_id, _to_answer(out, usage)


def resume_agent_session(thread_id: str, clarification_response: str) -> AnswerResult:
    """Resume a HITL run that paused in ``clarify_check``."""
    usage = UsageCallback()
    final = _HITL_AGENT.invoke(Command(resume=clarification_response),
                               config=_config(callbacks=[usage], thread_id=thread_id))
    _HITL_MODELS.pop(thread_id, None)
    return _to_answer(final, usage)
