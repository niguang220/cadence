"""The sandbox computation surface: scorer + fixture/contract teeth.

Measured match-rate (real model + real Docker) lives in evals/run_sandbox.py. Here we
keep the pure scorer and the deterministic teeth: ``validate_wrong_program`` runs a
stdlib-only wrong_program through the sandbox's LOCAL runner seam (host process, no
Docker) and asserts, in order, that it exits 0, emits parseable JSON, and diverges from
the gold. This proves the fixture is a *runnable-but-miscomputing* plausible-wrong
program; it does not test container isolation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from agent.python_step import analyze_python_output
from agent.sandbox import SandboxResult, _subprocess_runner
from evalharness.computation_oracle import computation_match


@dataclass
class SandboxOutcome:
    case_id: str
    matched: bool


def score_sandbox(outcomes: list[SandboxOutcome]) -> dict:
    return {
        "match_rate": sum(o.matched for o in outcomes) / len(outcomes) if outcomes else 0.0,
        "support": len(outcomes),
    }


def validate_wrong_program(case, *, timeout: float = 5.0) -> None:
    """Assert the fixture's wrong_program is runnable-but-miscomputing (no Docker)."""
    stdin_data = {"columns": case.input["columns"], "rows": case.input["rows"]}
    proc = _subprocess_runner(["python", "-c", case.wrong_program],
                              json.dumps(stdin_data), timeout)
    assert proc.returncode == 0, f"{case.id}: wrong_program did not exit 0"
    parsed = analyze_python_output(SandboxResult(True, stdout=proc.stdout))
    assert parsed["ok"], f"{case.id}: wrong_program did not emit parseable JSON"
    assert not computation_match(parsed["analysis"], case.expected_output,
                                 tolerance=case.tolerance), \
        f"{case.id}: wrong_program matched gold -- it has no teeth"
