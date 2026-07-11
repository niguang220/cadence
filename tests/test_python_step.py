from agent.python_step import generate_python, analyze_python_output, _strip_code_fence
from agent.execution import ExecutionResult
from agent.sandbox import SandboxResult

class _FakeModel:
    def __init__(self, content): self._content = content
    def invoke(self, _p):
        class R: pass
        r = R(); r.content = self._content; return r

def test_generate_returns_program_text():
    m = _FakeModel("```python\nimport sys, json\nprint('{}')\n```")
    prog = generate_python("plot trend", ExecutionResult(True, ["m"], [(1,)]), m)
    assert "import sys" in prog and "```" not in prog
    assert prog.splitlines()[0] != "python"     # language tag must not leak as line 1
    compile(prog, "<test>", "exec")              # returned text is a runnable program

def test_analyze_parses_stdout_json():
    out = analyze_python_output(SandboxResult(True, stdout='{"growth": 0.12}'))
    assert out["ok"] and out["analysis"] == {"growth": 0.12}

def test_analyze_flags_sandbox_failure():
    out = analyze_python_output(SandboxResult(False, error="sandbox timed out"))
    assert not out["ok"] and "timed out" in out["error"]

def test_analyze_flags_non_json_stdout():
    out = analyze_python_output(SandboxResult(True, stdout="not json"))
    assert not out["ok"] and "parse" in out["error"].lower()

def test_generate_repair_feeds_back_previous_error_and_code():
    class _Capture:
        def invoke(self, p):
            self.prompt = p
            class R: pass
            r = R(); r.content = "print(1)"; return r
    m = _Capture()
    generate_python("x", ExecutionResult(True, ["a"], [(1,)]), m,
                    previous_error="KeyError: 'mrr'", previous_code="rows['mrr']")
    assert "KeyError: 'mrr'" in m.prompt and "rows['mrr']" in m.prompt

def test_strip_fence_python_tag():
    assert _strip_code_fence("```python\nimport x\nprint(1)\n```") == "import x\nprint(1)"

def test_strip_fence_py_tag():
    assert _strip_code_fence("```py\nprint(1)\n```") == "print(1)"

def test_strip_fence_bare_no_language():
    assert _strip_code_fence("```\nprint(1)\n```") == "print(1)"

def test_strip_fence_no_fence_returns_unchanged():
    assert _strip_code_fence("import sys\nprint(1)") == "import sys\nprint(1)"

def test_strip_fence_multiline_blank_line_survives():
    # litmus: a blank line inside the program must not truncate the body
    assert _strip_code_fence("```python\nimport x\n\nprint(1)\n```") == "import x\n\nprint(1)"

def test_strip_fence_closing_removed_on_code_line():
    assert _strip_code_fence("```python\nprint(1)```") == "print(1)"
