"""The sandbox computation surface: scorer + fixture/contract teeth.

Measured match-rate (real model + real Docker) lives in evals/run_sandbox.py. Here we
keep the pure scorer and the deterministic teeth: ``validate_wrong_program`` runs a
stdlib-only wrong_program through the sandbox's LOCAL runner seam (host process, no
Docker) and checks, in order, that it exits 0, emits parseable JSON, and diverges from
the gold. This proves the fixture is a *runnable-but-miscomputing* plausible-wrong
program; it does not test container isolation.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from agent.python_step import analyze_python_output
from agent.sandbox import SandboxResult, _subprocess_runner
from evalharness.computation_oracle import computation_match


@dataclass
class SandboxOutcome:
    case_id: str
    matched: bool
    error: str = ""       # a MODEL failure (bad program / timeout / non-JSON); counts as unmatched


def score_sandbox(outcomes: list[SandboxOutcome]) -> dict:
    return {
        "match_rate": sum(o.matched for o in outcomes) / len(outcomes) if outcomes else 0.0,
        "support": len(outcomes),
    }


def validate_wrong_program(case, *, timeout: float = 5.0) -> None:
    """Check the fixture's wrong_program is runnable-but-miscomputing (no Docker).

    Uses ``sys.executable`` (a bare "python" is absent on python3-only hosts) and RAISES
    explicitly rather than ``assert`` -- a bare assert is stripped under ``python -O``,
    which would let a broken fixture pass this teeth check silently.
    """
    stdin_data = {"columns": case.input["columns"], "rows": case.input["rows"]}
    proc = _subprocess_runner([sys.executable, "-c", case.wrong_program],
                              json.dumps(stdin_data), timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"{case.id}: wrong_program did not exit 0")
    parsed = analyze_python_output(SandboxResult(True, stdout=proc.stdout))
    if not parsed["ok"]:
        raise RuntimeError(f"{case.id}: wrong_program did not emit parseable JSON")
    if computation_match(parsed["analysis"], case.expected_output, tolerance=case.tolerance):
        raise RuntimeError(f"{case.id}: wrong_program matched gold -- it has no teeth")
