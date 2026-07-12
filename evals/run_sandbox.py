"""Manual driver for the sandbox surface (needs DEEPSEEK_API_KEY AND local Docker).

Isolated Python-step eval, pass@1 single-shot (no repair loop) for clean attribution:
build an ExecutionResult from the fixed fixture input (truncated=False), ask the real
model to generate the program, run it in the PRODUCTION Docker sandbox, parse, and
compare against the gold with the case tolerance.

Only the computation cases are measured -- the ``adv_*`` fixtures exist solely for the CI
teeth (validate_wrong_program) and would double-weight the real task shapes. A MODEL
failure (bad program: non-zero exit / timeout / non-JSON / unexpected shape) counts as
``matched=False`` and stays in the denominator (pass@1, no survivorship bias); only an
INFRASTRUCTURE failure (Docker unavailable, or a model/API error raised by generate_python)
aborts the whole run.
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
        if case.wrong_program:                 # adversarial fixture: CI teeth only, never measured
            continue
        sql_result = ExecutionResult(True, columns=case.input["columns"],
                                     rows=[tuple(r) for r in case.input["rows"]], truncated=False)
        program = generate_python(case.instruction, sql_result, model)   # API error -> raises (infra)
        sandbox = run_in_sandbox(program, {"columns": sql_result.columns, "rows": sql_result.rows})
        if not sandbox.ok:
            if sandbox.error == "docker not available":
                raise RuntimeError("Docker is not available -- cannot run the sandbox surface")
            # a non-zero exit / timeout is a MODEL failure: count it as unmatched, do not abort.
            outcomes.append(SandboxOutcome(case.id, matched=False, error=sandbox.error or "sandbox failed"))
            continue
        parsed = analyze_python_output(sandbox)
        if not parsed["ok"]:
            outcomes.append(SandboxOutcome(case.id, matched=False, error=parsed["error"]))
            continue
        try:
            matched = computation_match(parsed["analysis"], case.expected_output, tolerance=case.tolerance)
        except ValueError as exc:              # unexpected output shape (e.g. a chart) -> model failure
            outcomes.append(SandboxOutcome(case.id, matched=False, error=str(exc)))
            continue
        outcomes.append(SandboxOutcome(case.id, matched=matched))
    return outcomes
