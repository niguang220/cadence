"""Strict loaders for the three Plan 4 golden sets.

Each set has its own dataclass. Loading is strict on purpose so a hand-authored JSON
cannot drift silently: an empty dataset, a duplicate id, or an unknown field (the
error names the key) all raise, as do invalid enum values. Per-field emptiness is
field-specific -- a gate question="" is a legal adversarial case.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

_GOLDEN_DIR = Path(__file__).resolve().parent.parent / "evals" / "golden"
GATE_PATH = _GOLDEN_DIR / "gate.json"
CONSISTENCY_PATH = _GOLDEN_DIR / "consistency.json"
SANDBOX_PATH = _GOLDEN_DIR / "sandbox.json"

_ROUTES = {"out_of_scope", "feasibility_refuse", "proceed"}
_CATEGORIES = {"measure", "grain", "entity", "dropped_filter"}


@dataclass
class GateCase:
    id: str
    question: str
    expected_route: str
    recalled_tables: list[str] = field(default_factory=list)
    paths: list[dict] = field(default_factory=list)
    note: str = ""


@dataclass
class ConsistencyCase:
    id: str
    question: str
    candidate_sql: str
    gold_sql: str
    category: str
    expected_caught: bool
    note: str = ""


@dataclass
class SandboxCase:
    id: str
    instruction: str
    input: dict
    expected_output: object
    tolerance: float = 1e-6
    wrong_program: str = ""


def _rows(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"golden set {path} is empty or not a list")
    return data


def _build(cls, path: Path):
    allowed = {f.name for f in fields(cls)}
    seen: set[str] = set()
    out = []
    for row in _rows(path):
        unknown = set(row) - allowed
        if unknown:
            raise ValueError(f"{path}: unknown field(s) {sorted(unknown)} in case {row.get('id')!r}")
        if not row.get("id"):
            raise ValueError(f"{path}: a case is missing a non-empty id")
        if row["id"] in seen:
            raise ValueError(f"{path}: duplicate id {row['id']!r}")
        seen.add(row["id"])
        out.append(cls(**row))
    return out


def load_gate(path: Path = GATE_PATH) -> list[GateCase]:
    cases = _build(GateCase, path)
    for c in cases:
        if c.expected_route not in _ROUTES:
            raise ValueError(f"{path}: case {c.id!r} has bad expected_route {c.expected_route!r}")
    return cases


def load_consistency(path: Path = CONSISTENCY_PATH) -> list[ConsistencyCase]:
    cases = _build(ConsistencyCase, path)
    for c in cases:
        if not isinstance(c.expected_caught, bool):
            raise ValueError(f"{path}: case {c.id!r} expected_caught must be a bool")
        if c.expected_caught and c.category not in _CATEGORIES:
            raise ValueError(f"{path}: adversarial case {c.id!r} needs category in {_CATEGORIES}")
        if not c.expected_caught and c.category != "":
            raise ValueError(f"{path}: clean case {c.id!r} must have empty category")
    return cases


def load_sandbox(path: Path = SANDBOX_PATH) -> list[SandboxCase]:
    cases = _build(SandboxCase, path)
    for c in cases:
        if not isinstance(c.input, dict) or "columns" not in c.input or "rows" not in c.input:
            raise ValueError(f"{path}: case {c.id!r} input needs 'columns' and 'rows'")
        if c.input.get("truncated"):
            raise ValueError(f"{path}: case {c.id!r} input.truncated must be false (full results only)")
        if _contains_chart(c.expected_output):
            raise ValueError(f"{path}: case {c.id!r} expected_output must not contain a chart (at any depth)")
    return cases


def _contains_chart(obj) -> bool:
    """True if a "chart" key appears anywhere in ``obj`` (charts are not a supported oracle).
    Recursive so a nested chart can't slip past into the comparator, where its ValueError
    would be misattributed to the model's output rather than the (invalid) fixture."""
    if isinstance(obj, dict):
        return "chart" in obj or any(_contains_chart(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_chart(v) for v in obj)
    return False
