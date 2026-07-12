"""Manual driver for the sandbox surface (needs DEEPSEEK_API_KEY AND local Docker).

Isolated Python-step eval, pass@1 single-shot (no repair loop) for clean attribution:
build an ExecutionResult from the fixed fixture input (truncated=False), ask the real
model to generate the program, run it in the PRODUCTION Docker sandbox, parse, and
compare against the gold with the case tolerance.

Only the computation cases are measured -- the ``adv_*`` fixtures exist solely for the CI
teeth (validate_wrong_program) and would double-weight the real task shapes.

Infrastructure vs. model failure is decided by a **preflight**, not by parsing error
strings (a down daemon and a missing image both surface as "sandbox exited non-zero", so
string-matching would miscount them as model misses and emit a fake match_rate). A trivial
container runs once before measuring: if it fails (daemon down / image missing / cannot
start) the whole run ABORTS with no score. After preflight succeeds, a per-case sandbox
failure (non-zero exit / timeout / non-JSON / unexpected shape) is a MODEL failure ->
matched=False, kept in the denominator.
"""
from __future__ import annotations

from agent.execution import ExecutionResult
from agent.python_step import analyze_python_output, generate_python
from agent.sandbox import run_in_sandbox
from evalharness.computation_oracle import computation_match
from evalharness.sandbox_eval import SandboxOutcome


def _preflight_docker() -> None:
    """Prove the daemon + image are up by actually starting a trivial container.
    A preflight failure is INFRASTRUCTURE, not a model miss, so it aborts the run."""
    probe = run_in_sandbox("import sys; sys.stdout.write('{}')", {"columns": [], "rows": []})
    if not probe.ok:
        raise RuntimeError(
            "sandbox preflight failed -- Docker daemon or image unavailable "
            f"({probe.error}: {probe.stderr.strip()[:200]}); refusing to emit a measured score")


def run_sandbox(cases, model) -> list[SandboxOutcome]:
    _preflight_docker()                        # infra up? else abort with no score
    outcomes = []
    for case in cases:
        if case.wrong_program:                 # adversarial fixture: CI teeth only, never measured
            continue
        sql_result = ExecutionResult(True, columns=case.input["columns"],
                                     rows=[tuple(r) for r in case.input["rows"]], truncated=False)
        program = generate_python(case.instruction, sql_result, model)   # API error -> raises (infra)
        sandbox = run_in_sandbox(program, {"columns": sql_result.columns, "rows": sql_result.rows})
        if not sandbox.ok:
            # preflight already proved infra is up, so a per-case failure is the MODEL's
            # program (non-zero exit / timeout): count it as unmatched, do not abort.
            outcomes.append(SandboxOutcome(case.id, matched=False, error=sandbox.error or "sandbox failed"))
            continue
        parsed = analyze_python_output(sandbox)
        if not parsed["ok"]:
            outcomes.append(SandboxOutcome(case.id, matched=False, error=parsed["error"]))
            continue
        try:
            matched = computation_match(parsed["analysis"], case.expected_output, tolerance=case.tolerance)
        except ValueError as exc:              # unexpected PREDICTED shape (e.g. a chart) -> model failure
            outcomes.append(SandboxOutcome(case.id, matched=False, error=str(exc)))
            continue
        outcomes.append(SandboxOutcome(case.id, matched=matched))
    return outcomes
