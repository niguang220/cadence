"""Shared pytest fixtures and fakes for the agent tests.

The SaaS DB is the product/eval domain, so `saas_db` builds it in a tmp path for the
plan-driven graph tests. `PlanningFakeModel` is a plan-aware fake chat model: under
the planner-driven graph the FIRST model call is the planner, so a fake that always
returned SQL would feed that SQL to the planner (unparseable -> empty plan -> refuse).
A plan-aware fake keeps "first call returns SELECT ..." tests meaningful.
"""
import pytest

from agent.db.build_saas_db import build


@pytest.fixture
def saas_db(tmp_path):
    return str(build(tmp_path / "saas.db"))


class PlanningFakeModel:
    """Plan-aware fake chat model: a planner prompt yields one SQL step; anything else
    yields the configured SQL. Keeps 'first call returns SELECT ...' tests meaningful.
    NOTE: PYTHON_GEN_PROMPT also contains 'JSON:', so match on PLANNER_PROMPT's unique
    'Output a JSON array of steps' + trailing 'JSON:', not on 'JSON:' alone."""

    def __init__(self, sql: str):
        self._sql = sql
        self.calls = 0
        self.last_prompt = None

    def invoke(self, prompt):
        self.calls += 1
        self.last_prompt = prompt
        text = prompt if isinstance(prompt, str) else str(prompt)
        is_planner = text.rstrip().endswith("JSON:") and "Output a JSON array of steps" in text
        content = ('[{"kind": "sql", "instruction": "answer the question"}]'
                   if is_planner else self._sql)
        return type("R", (), {"content": content})()

    def bind(self, **_):
        return self
