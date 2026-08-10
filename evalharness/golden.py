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
SAAS_METRICS_PATH = _GOLDEN_DIR / "saas_metrics.json"
VALUE_LINKING_PATH = _GOLDEN_DIR / "value_linking.json"

_ROUTES = {"out_of_scope", "feasibility_refuse", "proceed"}
_CATEGORIES = {"measure", "grain", "entity", "dropped_filter"}
_VALUE_LINKING_CATEGORIES = {"en", "zh", "code", "homonym", "fuzzy",           # positive families
                             "no_hit", "pii", "public_col", "off_topic",       # negatives
                             "cross_table", "injection"}                        # safety
_VALUE_LINKING_ROLES = {"primary", "diagnostic", "negative", "safety"}
# roles that must resolve a value to a table (scored for candidate recall / Fusion@5)
_VALUE_LINKING_LINKING = {"primary", "diagnostic"}


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


@dataclass
class SaasMetricsCase:
    id: str
    category: str            # one of the 7 metric families, or "control"
    metric: str
    question: str
    gold_sql: str
    wrong_sql: str = ""
    required_tables: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class ValueLinkingCase:
    id: str
    category: str            # see _VALUE_LINKING_CATEGORIES
    question: str
    role: str = "primary"    # primary | diagnostic | negative | safety
    required_tables: list[str] = field(default_factory=list)
    expect_value_hit: bool = True
    gold_sql: str = ""       # required for primary/diagnostic (the full-agent E2E oracle)
    expected_table: str = ""     # the table the value must link to (primary/diagnostic)
    expected_column: str = ""    # the searchable column the value lives in
    discriminative_reason: str = ""   # why lexical can't naturally hit it (rank-sensitivity)
    note: str = ""


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


def load_saas_metrics(path: Path = SAAS_METRICS_PATH) -> list[SaasMetricsCase]:
    cases = _build(SaasMetricsCase, path)
    for c in cases:
        if not c.required_tables:
            raise ValueError(f"{path}: case {c.id!r} has empty required_tables")
        if not c.gold_sql.strip():
            raise ValueError(f"{path}: case {c.id!r} has empty gold_sql")
    return cases


def load_value_linking(path: Path = VALUE_LINKING_PATH) -> list[ValueLinkingCase]:
    cases = _build(ValueLinkingCase, path)
    for c in cases:
        if c.category not in _VALUE_LINKING_CATEGORIES:
            raise ValueError(f"{path}: case {c.id!r} bad category {c.category!r}")
        if c.role not in _VALUE_LINKING_ROLES:
            raise ValueError(f"{path}: case {c.id!r} bad role {c.role!r}")
        if c.role in _VALUE_LINKING_LINKING:        # primary / diagnostic must resolve a value
            if not c.expect_value_hit:
                raise ValueError(f"{path}: {c.role} case {c.id!r} must expect a value hit")
            if not c.required_tables:
                raise ValueError(f"{path}: {c.role} case {c.id!r} needs required_tables")
            if not c.gold_sql.strip():
                raise ValueError(f"{path}: {c.role} case {c.id!r} needs a gold_sql (E2E oracle)")
            if not (c.expected_table and c.expected_column):
                raise ValueError(f"{path}: {c.role} case {c.id!r} needs expected_table/expected_column")
            if not c.discriminative_reason.strip():
                raise ValueError(f"{path}: {c.role} case {c.id!r} needs a discriminative_reason")
        elif c.role == "negative":
            if c.expect_value_hit:
                raise ValueError(f"{path}: negative case {c.id!r} must not expect a value hit")
            if c.gold_sql.strip():
                raise ValueError(f"{path}: negative case {c.id!r} must not carry a gold_sql")
        elif c.role == "safety":                    # injection / cross_table DO hit, but must be safe
            if c.gold_sql.strip():
                raise ValueError(f"{path}: safety case {c.id!r} must not carry a gold_sql")
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
