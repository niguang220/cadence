"""Manual driver for the sandbox surface (needs DEEPSEEK_API_KEY AND local Docker).

Isolated Python-step eval, pass@1 single-shot (no repair loop) for clean attribution:
build an ExecutionResult from the fixed fixture input (truncated=False), ask the real
model to generate the program, run it in the PRODUCTION Docker sandbox, parse, and
compare against the gold with the case tolerance.
"""
from __future__ import annotations

from agent.execution import ExecutionResult
from agent.python_step import analyze_python_output, generate_python
from agent.sandbox import run_in_sandbox
from evalharness.computation_oracle import computation_match
from evalharness.sandbox_eval import SandboxOutcome


def run_sandbox(cases, model) -> list[SandboxOutcome]:
    outcomes = []
    for case in cases:
        sql_result = ExecutionResult(True, columns=case.input["columns"],
                                     rows=[tuple(r) for r in case.input["rows"]], truncated=False)
        program = generate_python(case.instruction, sql_result, model)
        sandbox = run_in_sandbox(program, {"columns": sql_result.columns, "rows": sql_result.rows})
        parsed = analyze_python_output(sandbox)
        if not parsed["ok"]:
            raise RuntimeError(f"{case.id}: sandbox run failed: {parsed['error']}")
        matched = computation_match(parsed["analysis"], case.expected_output, tolerance=case.tolerance)
        outcomes.append(SandboxOutcome(case.id, matched))
    return outcomes
