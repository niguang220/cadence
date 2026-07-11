"""The Python analysis step: generate a stdin→stdout JSON program from the step
instruction + the prior SQL rows, then parse the sandbox's stdout back into state.
Execution itself is agent.sandbox.run_in_sandbox (kept separate so this module stays
service-free and unit-testable)."""
from __future__ import annotations

import json

from agent.execution import ExecutionResult
from agent.prompts import PYTHON_GEN_PROMPT, PYTHON_REPAIR_BLOCK
from agent.sandbox import SandboxResult


def _strip_code_fence(text: str) -> str:
    """Return the program inside a ```lang ... ``` fence, or the text unchanged when
    unfenced. Line-based on purpose (no lazy-regex traps): drop the opening ```lang
    line, then strip a trailing closing fence. A ```python language tag can never leak
    into the returned program (that would crash `python -c` with a NameError)."""
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    after_open = text.split("\n", 1)
    body = after_open[1] if len(after_open) > 1 else ""
    body = body.rstrip()
    if body.endswith("```"):
        body = body[:-3]
    return body.strip()


def generate_python(instruction: str, sql_result: ExecutionResult, model,
                    *, previous_error: str = "", previous_code: str = "") -> str:
    repair = ""
    if previous_error:
        repair = PYTHON_REPAIR_BLOCK.format(previous_code=previous_code,
                                            previous_error=previous_error)
    prompt = PYTHON_GEN_PROMPT.format(
        repair_block=repair,
        instruction=instruction,
        columns=sql_result.columns,
        sample_rows=sql_result.rows[:5],
    )
    response = model.invoke(prompt)
    return _strip_code_fence(getattr(response, "content", response))


def analyze_python_output(result: SandboxResult) -> dict:
    if not result.ok:
        return {"ok": False, "analysis": None, "error": result.error}
    try:
        return {"ok": True, "analysis": json.loads(result.stdout), "error": ""}
    except (json.JSONDecodeError, ValueError):
        return {"ok": False, "analysis": None,
                "error": f"could not parse sandbox output as JSON: {result.stdout[:200]}"}
