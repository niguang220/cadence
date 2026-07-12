"""The sandbox surface: scorer units (fake outcomes) + fixture/contract teeth.

The teeth are honestly fixture/oracle validation, NOT container-isolation testing: each
wrong_program is run via the sandbox's local runner seam on the host (stdlib-only, no
pandas, no Docker) and must (a) exit 0, (b) emit parseable JSON, (c) diverge from
expected_output. All three are required -- a crashing/empty/non-JSON program would pass a
bare "diverges" check and give fake teeth. The measured match-rate itself comes only from
the manual Docker driver.
"""
from evalharness.golden import load_sandbox
from evalharness.sandbox_eval import SandboxOutcome, score_sandbox, validate_wrong_program


def test_score_sandbox_match_rate():
    s = score_sandbox([SandboxOutcome("a", True), SandboxOutcome("b", False), SandboxOutcome("c", True)])
    assert s["match_rate"] == 2 / 3 and s["support"] == 3


def test_wrong_programs_are_runnable_but_miscompute():
    adversarial = [c for c in load_sandbox() if c.wrong_program]
    assert adversarial, "expected at least one adversarial wrong_program fixture"
    for case in adversarial:
        validate_wrong_program(case)   # raises AssertionError if not runnable-but-wrong
