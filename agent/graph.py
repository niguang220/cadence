"""The agent as a LangGraph state machine (planner-driven step loop).

    preflight_context
    -> intent_recognition --(out_of_scope)--> END (refuse)
    -> clarify_check
    -> schema_recall (pure retrieval; never refuses)
    -> table_relation
    -> feasibility_assessment --(no_recalled_tables)--> END (refuse)
    -> planner -> plan_validate --(invalid & attempts<MAX)--> planner
                              \\--(invalid & exhausted)------> respond (refuse)
    -> dispatch (reads plan[step_index].kind):
         "sql"    -> generate_sql --(CANNOT_ANSWER)--> END (decline)
                     -> execute -> validate --(ok)-----------> step_advance
                                          \\--(repair & attempts<MAX)--> generate_sql
         "python" -> python_generate -> python_execute -> python_analyze --(ok)--> step_advance
                                          \\--(fail & attempts<MAX)----> python_generate
    -> step_advance --(more steps)--> dispatch ;  --(done)--> respond -> END

The planner decomposes the question into a 1-2 step plan (one SQL step, optionally
followed by one Python analysis step); ``validate_plan`` rejects malformed plans and
the graph replans (bounded). SQL steps reuse the same generate/execute/validate/repair
body as before; Python steps run generated code in an isolated sandbox.

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
from agent.feasibility import assess_feasibility
from agent.graph_state import AgentState
from agent.generation import (AnswerResult, _extract_sql, _format_answer, _NO_QUERY,
                             generate_sql)
from agent.governance import check_result_governance, check_sql_governance
from agent.hybrid_retriever import retrieve
from agent.intent import classify_intent
from agent.observability import setup_phoenix
from agent.plan import deserialize_plan, serialize_plan, validate_plan
from agent.planner import plan_query
from agent.prompts import (CANNOT_ANSWER, REPAIR_INSTRUCTION, REPAIR_PROMPT,
                           TOOL_SYSTEM_PROMPT)
from agent.python_step import analyze_python_output, generate_python
from agent.query_enhance import enhance_query
from agent.sandbox import SandboxResult, run_in_sandbox   # re-exported here so tests can fake the sandbox
from agent.schema_relations import join_paths
from agent.semantic_layer import MetricDef, MetricRegistry
from agent.tools import build_get_schema_tool
from agent.usage import UsageCallback
from agent.validate import validate_result

MAX_ATTEMPTS = 3            # total generations (1 draft + up to 2 repairs)
MAX_PLAN_ATTEMPTS = 2       # planner retries before the graph gives up and refuses
MAX_PYTHON_ATTEMPTS = 3     # python-step retries before the graph gives up and refuses
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


def _intent_recognition(state: AgentState) -> dict:
    v = classify_intent(state["question"])
    if v.kind == "out_of_scope":
        return {"answer": "I can only answer questions about this database's data.",
                "result": ExecutionResult(False, error=_NO_QUERY),
                "trace": [{"node": "intent_recognition", "intent_kind": v.kind,
                           "reason": v.reason, "refused": True}]}
    return {"trace": [{"node": "intent_recognition", "intent_kind": v.kind}]}


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


def _retrieval_question(state: AgentState) -> str:
    """The question that drives schema recall, planning, and SQL generation.

    query_enhance rewrites the question to add time/entity context; that enhanced form
    is the retrieval/generation input. The ORIGINAL ``state["question"]`` is untouched
    and stays the anchor for the answer, the trace, and semantic consistency. When the
    enhancement is a no-op (empty or identical) this returns the original verbatim, so
    the baseline stays byte-identical.
    """
    return state.get("enhanced_question") or state["question"]


def _query_enhance(state: AgentState, config=None) -> dict:
    """Pre-step LLM rewrite: add context for better retrieval/generation under a
    governed-metric guardrail. Sets ``enhanced_question`` only -- never overwrites the
    original ``question``. Gets the model like the other LLM nodes so a HITL resume
    (checkpoint omits the model) still works."""
    model = state.get("model") or _model_for_state(state, config)
    if model is None:
        raise RuntimeError("No model available for query enhancement")
    result = enhance_query(state["question"], _semantic_metrics(state), model)
    entry = {"node": "query_enhance",
             "enhanced": result.enhanced_question != state["question"],
             "rewrite_diff": result.rewrite_diff}
    if result.governed_terms:
        entry["governed_terms"] = result.governed_terms
    if result.warnings:
        entry["warnings"] = result.warnings
    return {"enhanced_question": result.enhanced_question, "trace": [entry]}


def _schema_recall(state: AgentState) -> dict:
    """Pure retrieval: top-k table recall. Does NOT refuse on empty recall --
    that decision belongs to feasibility_assessment (the single deterministic
    refusal owner), which sees this node's empty ``retrieved_tables`` and issues
    the ``no_recalled_tables`` refusal."""
    tables = state.get("tables")
    if tables is None:
        tables = introspect(state["db_path"])
    top_k = retrieve(_retrieval_question(state), tables, k=state.get("k", 5))
    if not top_k:
        return {
            "tables": tables,
            "retrieved_tables": [],
            "schema": "",
            "trace": [{"node": "schema_recall", "tables": [], "retrieval_failed": True}],
        }
    schema = render_schema(tables, only=top_k, include_fk_neighbors=True)
    return {
        "tables": tables,
        "retrieved_tables": top_k,
        "schema": schema,
        "trace": [{"node": "schema_recall", "tables": top_k}],
    }


def _table_relation(state: AgentState) -> dict:
    """Deterministic table-relation node: zero LLM. Computes the direct FK-edge
    join hints among the recalled (top-k) tables and appends a short "Join paths:"
    hint to the rendered schema -- but only when there is something to hint at, so
    a recalled set with no direct FK edges leaves the schema byte-identical."""
    paths = join_paths(state["tables"], state["retrieved_tables"])
    out = {
        "join_paths": paths,
        "trace": [{"node": "table_relation", "paths": len(paths), "join_paths": paths}],
    }
    if paths:
        hint = "\n".join(f"{p['from']} -> {p['to']} (on {p['on']})" for p in paths)
        out["schema"] = f"{state['schema']}\n\nJoin paths:\n{hint}"
    return out


def _feasibility_assessment(state: AgentState) -> dict:
    """Deterministic feasibility gate: zero LLM. Owns the ONE hard refusal
    (empty recall, moved here from schema_recall) and traces other signals
    (e.g. a missing direct join edge) as risk, never as a refusal."""
    v = assess_feasibility(state["question"], state["tables"], state["retrieved_tables"],
                           _semantic_metrics(state), state.get("join_paths", []))
    if not v.feasible:
        return {
            "answer": v.message,
            "result": ExecutionResult(False, error=_NO_QUERY),
            "feasibility_reason": v.reason_code,
            "trace": [{"node": "feasibility_assessment", "refused": True,
                       "reason_code": v.reason_code, "message": v.message}],
        }
    return {"trace": [{"node": "feasibility_assessment", "feasible": True, "risks": v.risks}]}


def _generate_plain(state: AgentState, model, attempts: int, block: str = "") -> str:
    """Single-shot generation for models without tool support (e.g. test fakes)."""
    if attempts == 0:
        return generate_sql(_sql_task(state), state["schema"], model,
                           semantic_block=block)
    prompt = REPAIR_PROMPT.format(
        schema=state["schema"], question=_sql_task(state),
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
        human = _sql_task(state)
    else:
        human = REPAIR_INSTRUCTION.format(
            question=_sql_task(state), failed_sql=state.get("sql", ""),
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


def _planner(state: AgentState, config=None) -> dict:
    model = state.get("model") or _model_for_state(state, config)
    block = ""
    if state.get("semantic_layer"):
        block = _metric_registry().format(_semantic_metrics(state))
    plan = plan_query(_retrieval_question(state), state["schema"], model, semantic_block=block,
                      feedback=state.get("error") or "")   # replan with the reject reason
    attempts = state.get("plan_attempts", 0) + 1
    return {"plan": serialize_plan(plan), "plan_attempts": attempts, "step_index": 0,
            "trace": [{"node": "planner", "steps": [s.kind for s in plan.steps],
                      "attempt": attempts}]}


def _plan_validate(state: AgentState) -> dict:
    v = validate_plan(deserialize_plan(state["plan"]))
    return {"error": None if v.ok else v.reason,
            "trace": [{"node": "plan_validate", "ok": v.ok, "reason": v.reason}]}


def _current_step(state: AgentState) -> dict:
    return state["plan"][state["step_index"]]


def _sql_task(state: AgentState) -> str:
    """What the SQL step should generate for. In a multi-step plan the planner may
    decompose the SQL step (e.g. "pull raw monthly rows"); surface that instruction
    alongside the original question so the SQL is driven by the step, not only the
    overall question. Falls back to the question when there is no plan (direct callers).
    Keeping the question as the anchor and adding the instruction as context is the
    conservative fix; fully instruction-driven generation is a later refinement (it
    needs real-model checking that the planner emits faithful SQL instructions).

    SQL generation gets BOTH the original question and the enhanced rewrite (the metric
    block is threaded separately) so the governed intent and the added time/entity
    context both reach generation. When the enhancement is a no-op (empty or identical)
    the base is the original question verbatim, so the non-enhanced baseline stays
    byte-identical."""
    question = state["question"]
    enhanced = state.get("enhanced_question")
    base = (f"{question}\n\nEnhanced for retrieval: {enhanced}"
            if enhanced and enhanced != question else question)
    plan = state.get("plan")
    if not plan:
        return base
    instruction = (plan[state.get("step_index", 0)] or {}).get("instruction", "")
    if instruction.strip():
        return f"{base}\n\nFor this step: {instruction}"
    return base


def _dispatch_step(state: AgentState) -> dict:
    step = _current_step(state)
    # reset per-step SQL/python retry counters at the start of each step
    reset = {"attempts": 0, "python_attempts": 0, "error": None}
    reset["trace"] = [{"node": "dispatch", "step_index": state["step_index"], "kind": step["kind"]}]
    return reset


def _python_generate(state: AgentState, config=None) -> dict:
    model = state.get("model") or _model_for_state(state, config)
    prior = state["result"]                       # the single SQL step's ExecutionResult
    prev_err, prev_code = "", ""
    if state.get("python_attempts", 0) > 0:       # a retry: feed the failure back so it
        prev_err = (state.get("python_analysis") or {}).get("error", "")  # fixes, not repeats
        prev_code = state.get("python_code", "")
    code = generate_python(_current_step(state)["instruction"], prior, model,
                           previous_error=prev_err, previous_code=prev_code)
    attempts = state.get("python_attempts", 0) + 1
    return {"python_code": code, "python_attempts": attempts,
            "trace": [{"node": "python_generate", "attempt": attempts}]}


def _python_execute(state: AgentState) -> dict:
    prior = state["result"]
    # `truncated` tells the analysis code (and, via _respond, the user) that these rows
    # are only the first max_rows of a larger result -- so the analysis isn't silently
    # computed on a partial sample.
    payload = {"columns": prior.columns, "rows": [list(r) for r in prior.rows],
               "truncated": prior.truncated}
    sandbox = run_in_sandbox(state["python_code"], payload)
    return {"python_analysis": {"_sandbox_ok": sandbox.ok, "stdout": sandbox.stdout,
                                "stderr": sandbox.stderr, "error": sandbox.error},
            "trace": [{"node": "python_execute", "ok": sandbox.ok}]}


def _python_analyze(state: AgentState) -> dict:
    raw = state["python_analysis"]
    parsed = analyze_python_output(SandboxResult(
        raw["_sandbox_ok"], stdout=raw.get("stdout", ""), stderr=raw.get("stderr", ""),
        error=raw.get("error", "")))
    return {"python_analysis": parsed,
            "error": None if parsed["ok"] else parsed["error"],
            "trace": [{"node": "python_analyze", "ok": parsed["ok"]}]}


def _step_advance(state: AgentState) -> dict:
    step = _current_step(state)
    return {"step_index": state["step_index"] + 1,
            "attempts": 0, "python_attempts": 0, "error": None,
            "trace": [{"node": "step_advance", "completed": step["kind"]}]}


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
    answer = _format_answer(state["result"])
    if "python_analysis" not in state:                  # SQL-only: answer AND trace unchanged
        return {"answer": answer, "trace": [{"node": "respond"}]}
    analysis = (state.get("python_analysis") or {}).get("analysis")   # a Python step ran
    answer = f"{answer}\nAnalysis: {analysis}"
    truncated = state["result"].truncated
    if truncated:                                       # be honest: analysis on a sample
        answer += (f"\n(Note: this analysis is based on the first {len(state['result'].rows)} "
                   "rows; the query result was truncated.)")
    return {"answer": answer,
            "trace": [{"node": "respond", "python_analysis": True, "truncated": truncated}]}


def _route_after_intent(state: AgentState) -> str:
    return END if state.get("answer") else "clarify_check"


def _route_after_clarify(state: AgentState) -> str:
    if state.get("answer"):
        return END
    return END if state.get("clarification") else "query_enhance"


def _route_after_feasibility(state: AgentState) -> str:
    return END if state.get("answer") else "planner"


def _route_after_generate(state: AgentState) -> str:
    # _generate_sql sets `answer` only when it declines (CANNOT_ANSWER / empty).
    return END if state.get("answer") else "execute"


def _route_after_plan_validate(state: AgentState) -> str:
    if not state.get("error"):
        return "dispatch"
    if state.get("plan_attempts", 0) >= MAX_PLAN_ATTEMPTS:   # bounded: give up and refuse
        return "respond"
    return "planner"                                         # replan with a fresh attempt


def _route_dispatch(state: AgentState) -> str:
    return "python_generate" if _current_step(state)["kind"] == "python" else "generate_sql"


def _route_after_sql_validate(state: AgentState) -> str:
    if not state.get("error"):
        return "step_advance"
    if state.get("repair_kind") == "governance_block":
        return "respond"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:       # bounded: give up and refuse
        return "respond"
    return "generate_sql"                              # repair: re-enter with the hint


def _route_after_python_analyze(state: AgentState) -> str:
    if not state.get("error"):
        return "step_advance"
    if state.get("python_attempts", 0) >= MAX_PYTHON_ATTEMPTS:   # bounded: give up and refuse
        return "respond"
    return "python_generate"                              # repair: re-generate with the failure


def _route_after_step_advance(state: AgentState) -> str:
    return "dispatch" if state["step_index"] < len(state["plan"]) else "respond"


def _build(*, checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("preflight_context", _preflight_context)
    g.add_node("intent_recognition", _intent_recognition)
    g.add_node("clarify_check", _clarify_check)
    g.add_node("query_enhance", _query_enhance)
    g.add_node("schema_recall", _schema_recall)
    g.add_node("table_relation", _table_relation)
    g.add_node("feasibility_assessment", _feasibility_assessment)
    g.add_node("planner", _planner)
    g.add_node("plan_validate", _plan_validate)
    g.add_node("dispatch", _dispatch_step)
    g.add_node("generate_sql", _generate_sql)
    g.add_node("execute", _execute)
    g.add_node("validate", _validate)
    g.add_node("python_generate", _python_generate)
    g.add_node("python_execute", _python_execute)
    g.add_node("python_analyze", _python_analyze)
    g.add_node("step_advance", _step_advance)
    g.add_node("respond", _respond)
    g.add_edge(START, "preflight_context")
    g.add_edge("preflight_context", "intent_recognition")
    g.add_conditional_edges("intent_recognition", _route_after_intent,
                            {"clarify_check": "clarify_check", END: END})
    # query_enhance runs ONLY on the proceed path (a clarification/intent refusal never
    # reaches it): rewrite the question for retrieval/generation, then recall schema.
    g.add_conditional_edges("clarify_check", _route_after_clarify,
                            {"query_enhance": "query_enhance", END: END})
    g.add_edge("query_enhance", "schema_recall")
    # schema_recall (top-k retrieval, pure -- never refuses) -> table_relation
    # (deterministic FK-edge join hints, zero LLM) -> feasibility_assessment (the
    # single deterministic refusal owner, including empty recall) -> planner:
    g.add_edge("schema_recall", "table_relation")
    g.add_edge("table_relation", "feasibility_assessment")
    g.add_conditional_edges("feasibility_assessment", _route_after_feasibility,
                            {"planner": "planner", END: END})
    g.add_edge("planner", "plan_validate")
    g.add_conditional_edges("plan_validate", _route_after_plan_validate,
                            {"dispatch": "dispatch", "planner": "planner", "respond": "respond"})
    g.add_conditional_edges("dispatch", _route_dispatch,
                            {"generate_sql": "generate_sql", "python_generate": "python_generate"})
    # generate_sql keeps its decline short-circuit EXPLICITLY: on CANNOT_ANSWER it sets
    # `answer` and _route_after_generate routes to END; otherwise it proceeds to execute.
    g.add_conditional_edges("generate_sql", _route_after_generate,
                            {"execute": "execute", END: END})
    g.add_edge("execute", "validate")
    # SQL step tail: validate now advances the step instead of responding
    g.add_conditional_edges("validate", _route_after_sql_validate,
                            {"step_advance": "step_advance", "generate_sql": "generate_sql",
                             "respond": "respond"})
    g.add_edge("python_generate", "python_execute")
    g.add_edge("python_execute", "python_analyze")
    g.add_conditional_edges("python_analyze", _route_after_python_analyze,
                            {"step_advance": "step_advance", "python_generate": "python_generate",
                             "respond": "respond"})
    g.add_conditional_edges("step_advance", _route_after_step_advance,
                            {"dispatch": "dispatch", "respond": "respond"})
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
