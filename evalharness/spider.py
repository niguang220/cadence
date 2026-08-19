"""Frozen-slice Spider evaluation helpers.

This module keeps dataset validation, record derivation, and aggregation separate from
the paid runner in :mod:`evals.spider_external`.  Loading and scoring are therefore
service-free and unit-testable.  Raw records intentionally omit benchmark questions and
gold SQL; the frozen manifest index is the audit key.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sqlglot import exp, parse_one

from agent.execution import run_query
from agent.graph import run_agent
from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.serde import serialize_config
from evalharness.oracle import execution_match

SPIDER_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent / "evals" / "golden" / "spider_dev_slice.json"
)

_MANIFEST_KEYS = {
    "schema_version", "benchmark", "dataset_sha256", "dataset_size", "selection", "cases"
}
_SELECTION_KEYS = {"method", "seed", "n"}
_CASE_KEYS = {"index", "db_id"}
_SELECTION_METHOD = "sorted(random.Random(seed).sample(range(dataset_size), n))"


@dataclass(frozen=True)
class SpiderCase:
    index: int
    db_id: str
    question: str
    gold_sql: str
    db_path: Path


@dataclass
class SpiderRecord:
    index: int
    db_id: str
    retrieval_config: dict
    retrieval_k: int
    repeat_index: int
    predicted_sql: str
    exec_match: bool | None
    skipped_gold: bool
    failure_stage: str | None
    ordered: bool
    gold_tables: list[str]
    retrieved_tables: list[str] = field(default_factory=list)
    candidate_recall: float | None = None
    selection_recall: float | None = None
    context_recall: float | None = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieval_stage_events: list[dict] = field(default_factory=list)


def _require_exact_keys(value: dict, expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def load_spider_slice(
    spider_dir: str | Path, manifest_path: str | Path = SPIDER_MANIFEST_PATH
) -> list[SpiderCase]:
    """Load a frozen Spider slice and reject any source or manifest drift."""
    spider_dir = Path(spider_dir)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Spider manifest must be a JSON object")
    _require_exact_keys(manifest, _MANIFEST_KEYS, "manifest")
    if manifest["schema_version"] != 1 or manifest["benchmark"] != "spider-dev":
        raise ValueError("unsupported Spider manifest schema or benchmark")
    if not isinstance(manifest["selection"], dict):
        raise ValueError("manifest selection must be an object")
    _require_exact_keys(manifest["selection"], _SELECTION_KEYS, "manifest selection")
    selection = manifest["selection"]
    if selection["method"] != _SELECTION_METHOD:
        raise ValueError(f"unsupported Spider selection method {selection['method']!r}")
    if (
        not isinstance(selection["seed"], int)
        or isinstance(selection["seed"], bool)
        or not isinstance(selection["n"], int)
        or isinstance(selection["n"], bool)
        or selection["n"] < 1
    ):
        raise ValueError("manifest selection seed/n must be valid integers")

    dev_path = spider_dir / "dev.json"
    payload = dev_path.read_bytes()
    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_sha != manifest["dataset_sha256"]:
        raise ValueError(
            f"Spider dev.json SHA-256 mismatch: expected {manifest['dataset_sha256']}, "
            f"got {actual_sha}"
        )
    dev = json.loads(payload)
    if not isinstance(dev, list) or len(dev) != manifest["dataset_size"]:
        actual_size = len(dev) if isinstance(dev, list) else "non-list"
        raise ValueError(
            f"Spider dataset size mismatch: expected {manifest['dataset_size']}, got {actual_size}"
        )
    cases_manifest = manifest["cases"]
    if not isinstance(cases_manifest, list):
        raise ValueError("manifest cases must be a list")
    if manifest["selection"]["n"] != len(cases_manifest):
        raise ValueError("manifest selection.n does not match the number of cases")
    expected_indices = sorted(
        random.Random(selection["seed"]).sample(range(len(dev)), selection["n"])
    )
    frozen_indices = [case.get("index") if isinstance(case, dict) else None for case in cases_manifest]
    if frozen_indices != expected_indices:
        raise ValueError(
            "manifest case indices do not match the declared deterministic selection"
        )

    cases: list[SpiderCase] = []
    seen: set[int] = set()
    for position, frozen in enumerate(cases_manifest):
        if not isinstance(frozen, dict):
            raise ValueError(f"manifest case {position} must be an object")
        _require_exact_keys(frozen, _CASE_KEYS, f"manifest case {position}")
        index = frozen["index"]
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(dev):
            raise ValueError(f"manifest case {position} has invalid index {index!r}")
        if index in seen:
            raise ValueError(f"manifest contains duplicate index {index}")
        seen.add(index)
        source = dev[index]
        db_id = frozen["db_id"]
        if source.get("db_id") != db_id:
            raise ValueError(
                f"Spider index {index} database mismatch: manifest={db_id!r}, "
                f"source={source.get('db_id')!r}"
            )
        db_path = spider_dir / "database" / db_id / f"{db_id}.sqlite"
        if not db_path.is_file():
            raise FileNotFoundError(f"Spider database missing for index {index}: {db_path}")
        question, gold_sql = source.get("question"), source.get("query")
        if not isinstance(question, str) or not isinstance(gold_sql, str):
            raise ValueError(f"Spider index {index} lacks a string question/query")
        cases.append(SpiderCase(index, db_id, question, gold_sql, db_path))
    return cases


def sql_characteristics(sql: str) -> tuple[list[str], bool]:
    """Return physical source tables and whether result order is semantically scored."""
    tree = parse_one(sql, read="sqlite")
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    tables = sorted({
        table.name.lower()
        for table in tree.find_all(exp.Table)
        if table.name and table.name.lower() not in cte_names
    })
    return tables, tree.find(exp.Order) is not None


def preflight_spider_cases(cases: list[SpiderCase]) -> dict:
    """Parse and execute every trusted gold query without making model calls."""
    failures = []
    by_db: dict[str, int] = defaultdict(int)
    for case in cases:
        try:
            gold_tables, _ = sql_characteristics(case.gold_sql)
        except Exception as exc:  # sqlglot errors vary by release
            failures.append({"index": case.index, "stage": "gold_parse", "error": str(exc)})
            continue
        if not gold_tables:
            failures.append({"index": case.index, "stage": "gold_parse", "error": "no tables"})
            continue
        gold = run_query(case.db_path, case.gold_sql, assume_safe=True)
        if not gold.ok:
            failures.append({"index": case.index, "stage": "gold_execute", "error": gold.error})
            continue
        by_db[case.db_id] += 1
    return {
        "n_cases": len(cases),
        "runnable": len(cases) - len(failures),
        "failed": len(failures),
        "databases_covered": len(by_db),
        "failures": failures,
    }


def _recall(gold_tables: list[str], observed) -> float:
    gold = {name.lower() for name in gold_tables}
    found = {name.lower() for name in observed}
    return len(gold & found) / len(gold)


def _failure_stage(result, matched: bool) -> str | None:
    if matched:
        return None
    if result.clarification is not None:
        return "clarified"
    if not result.sql:
        return "no_sql"
    if not result.execution.ok:
        return "execution_error"
    return "answer_mismatch"


def record_spider_run(
    case: SpiderCase,
    result,
    gold_rows: list[tuple],
    *,
    config: RetrievalConfig,
    k: int,
    repeat_index: int,
) -> SpiderRecord:
    """Derive one compact, redacted Spider record from an agent result."""
    gold_tables, ordered = sql_characteristics(case.gold_sql)
    retrieval = result.retrieval_result
    if retrieval is None:
        raise ValueError(f"Spider index {case.index}: retrieval_result is missing")
    if retrieval.config_name != config.name:
        raise ValueError(
            f"Spider index {case.index}: result config {retrieval.config_name!r} "
            f"!= requested {config.name!r}"
        )
    matched = bool(
        result.execution.ok
        and execution_match(result.execution.rows, gold_rows, ordered=ordered)
    )
    usage = result.usage or {}
    return SpiderRecord(
        index=case.index,
        db_id=case.db_id,
        retrieval_config=serialize_config(config),
        retrieval_k=k,
        repeat_index=repeat_index,
        predicted_sql=result.sql,
        exec_match=matched,
        skipped_gold=False,
        failure_stage=_failure_stage(result, matched),
        ordered=ordered,
        gold_tables=gold_tables,
        retrieved_tables=list(result.retrieved_tables),
        candidate_recall=_recall(gold_tables, (c.table for c in retrieval.candidates)),
        selection_recall=_recall(gold_tables, retrieval.selection.anchor_tables),
        context_recall=_recall(gold_tables, retrieval.relation_plan.context_tables),
        latency_ms=float(usage.get("latency_ms", 0.0)),
        prompt_tokens=int(usage.get("input_tokens", 0)),
        completion_tokens=int(usage.get("output_tokens", 0)),
        retrieval_stage_events=[asdict(event) for event in retrieval.stage_events],
    )


def run_spider_case(
    case: SpiderCase,
    tables,
    model,
    *,
    config: RetrievalConfig,
    k: int,
    repeat_index: int,
) -> SpiderRecord:
    gold_tables, ordered = sql_characteristics(case.gold_sql)
    gold = run_query(case.db_path, case.gold_sql, assume_safe=True)
    if not gold.ok:
        return SpiderRecord(
            index=case.index,
            db_id=case.db_id,
            retrieval_config=serialize_config(config),
            retrieval_k=k,
            repeat_index=repeat_index,
            predicted_sql="",
            exec_match=None,
            skipped_gold=True,
            failure_stage="gold_unrunnable",
            ordered=ordered,
            gold_tables=gold_tables,
        )
    result = run_agent(
        case.db_path,
        case.question,
        model=model,
        tables=tables,
        semantic_layer=False,
        clarify=False,
        k=k,
        retrieval_config=config,
    )
    return record_spider_run(
        case, result, gold.rows, config=config, k=k, repeat_index=repeat_index
    )


def _mean(records: list[SpiderRecord], field_name: str) -> float:
    values = [getattr(record, field_name) for record in records]
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else 0.0


def summarize_spider(
    records: list[SpiderRecord], *, baseline: str = "lexical_baseline",
    candidate: str = "governed_rrf"
) -> dict:
    """Aggregate two paired configurations and evaluate the preregistered gate."""
    grouped: dict[str, list[SpiderRecord]] = defaultdict(list)
    keys_by_config: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for record in records:
        name = record.retrieval_config["name"]
        key = (record.index, record.repeat_index)
        if key in keys_by_config[name]:
            raise ValueError(f"duplicate Spider record for {name} index/repeat {key}")
        keys_by_config[name].add(key)
        grouped[name].append(record)
    if not keys_by_config[baseline] or keys_by_config[baseline] != keys_by_config[candidate]:
        raise ValueError("Spider comparison is empty or has unpaired config/index/repeat records")

    by_config = {}
    for name in (baseline, candidate):
        rows = grouped.get(name, [])
        scored = [row for row in rows if not row.skipped_gold]
        matched = sum(row.exec_match is True for row in scored)
        by_config[name] = {
            "n_records": len(rows),
            "scored": len(scored),
            "skipped_gold": len(rows) - len(scored),
            "matched": matched,
            "execution_match_rate": matched / len(scored) if scored else 0.0,
            "no_sql": sum(row.failure_stage == "no_sql" for row in scored),
            "candidate_recall": _mean(scored, "candidate_recall"),
            "selection_recall": _mean(scored, "selection_recall"),
            "context_recall": _mean(scored, "context_recall"),
            "avg_latency_ms": _mean(scored, "latency_ms"),
            "avg_prompt_tokens": _mean(scored, "prompt_tokens"),
            "avg_completion_tokens": _mean(scored, "completion_tokens"),
        }

    per_case: dict[tuple[str, int], int] = defaultdict(int)
    for record in records:
        if not record.skipped_gold:
            per_case[(record.retrieval_config["name"], record.index)] += int(
                record.exec_match is True
            )
    paired_indices = sorted(
        set(index for name, index in per_case if name == baseline)
        & set(index for name, index in per_case if name == candidate)
    )
    wins = [i for i in paired_indices if per_case[(candidate, i)] > per_case[(baseline, i)]]
    losses = [i for i in paired_indices if per_case[(candidate, i)] < per_case[(baseline, i)]]
    ties = [i for i in paired_indices if per_case[(candidate, i)] == per_case[(baseline, i)]]
    base, cand = by_config[baseline], by_config[candidate]
    checks = {
        "match_floor": cand["matched"] >= base["matched"] - 2,
        "paired_wins_gte_losses": len(wins) >= len(losses),
        "candidate_recall_gte_baseline": cand["candidate_recall"] >= base["candidate_recall"],
        "context_recall_gte_baseline": cand["context_recall"] >= base["context_recall"],
        "no_sql_ceiling": cand["no_sql"] <= base["no_sql"] + 2,
    }
    return {
        "n_records": len(records),
        "by_config": by_config,
        "paired": {
            "n_cases": len(paired_indices),
            "wins": len(wins),
            "losses": len(losses),
            "ties": len(ties),
            "win_indices": wins,
            "loss_indices": losses,
        },
        "candidate_gate": {"passed": all(checks.values()), "checks": checks},
    }
