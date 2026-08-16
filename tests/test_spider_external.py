"""Service-free tests for the frozen Spider external-validity harness."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict

import pytest

from agent.execution import ExecutionResult
from agent.generation import AnswerResult
from agent.retrieval.contracts import (
    RelationPlan,
    RetrievalConfig,
    RetrievalResult,
    SelectionDecision,
    TableCandidate,
)
from evalharness.spider import (
    SpiderCase,
    SpiderRecord,
    load_spider_slice,
    preflight_spider_cases,
    record_spider_run,
    sql_characteristics,
    summarize_spider,
)
from evals.spider_external import parse_args, run_comparison
from conftest import PlanningFakeModel
from agent.db.introspect import introspect


def _spider_fixture(tmp_path):
    spider = tmp_path / "spider"
    db_dir = spider / "database" / "tiny"
    db_dir.mkdir(parents=True)
    db = db_dir / "tiny.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript("CREATE TABLE pet(id INTEGER, name TEXT); INSERT INTO pet VALUES(1, 'Milo');")
    dev = [{"db_id": "tiny", "question": "How many pets?", "query": "SELECT count(*) FROM pet"}]
    payload = json.dumps(dev).encode()
    (spider / "dev.json").write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "benchmark": "spider-dev",
        "dataset_sha256": hashlib.sha256(payload).hexdigest(),
        "dataset_size": 1,
        "selection": {
            "method": "sorted(random.Random(seed).sample(range(dataset_size), n))",
            "seed": 7,
            "n": 1,
        },
        "cases": [{"index": 0, "db_id": "tiny"}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return spider, manifest_path, manifest


def test_loader_validates_frozen_source_and_gold_preflight(tmp_path):
    spider, manifest_path, _ = _spider_fixture(tmp_path)
    cases = load_spider_slice(spider, manifest_path)
    assert [(case.index, case.db_id) for case in cases] == [(0, "tiny")]
    assert preflight_spider_cases(cases) == {
        "n_cases": 1,
        "runnable": 1,
        "failed": 0,
        "databases_covered": 1,
        "failures": [],
    }


def test_loader_rejects_dataset_hash_drift(tmp_path):
    spider, manifest_path, _ = _spider_fixture(tmp_path)
    (spider / "dev.json").write_text("[]")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_spider_slice(spider, manifest_path)


def test_loader_rejects_database_id_drift(tmp_path):
    spider, manifest_path, manifest = _spider_fixture(tmp_path)
    manifest["cases"][0]["db_id"] = "other"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="database mismatch"):
        load_spider_slice(spider, manifest_path)


def test_loader_rejects_indices_that_do_not_match_declared_selection(tmp_path):
    spider, manifest_path, manifest = _spider_fixture(tmp_path)
    manifest["cases"][0]["index"] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="deterministic selection"):
        load_spider_slice(spider, manifest_path)


def test_sql_characteristics_excludes_cte_alias_and_detects_order():
    tables, ordered = sql_characteristics(
        "WITH chosen AS (SELECT id FROM pet) "
        "SELECT owner.name FROM owner JOIN chosen ON owner.pet_id = chosen.id ORDER BY owner.name"
    )
    assert tables == ["owner", "pet"]
    assert ordered is True


def _retrieval(config_name="rrf_hybrid"):
    return RetrievalResult(
        config_name=config_name,
        signals=[],
        candidates=[TableCandidate("PET", {}, 1.0, 1)],
        metric_matches=[],
        selection=SelectionDecision(["pet"], [], "topk", {}),
        relation_plan=RelationPlan("shortest_path", ["pet"], [], ["pet"], [], [], []),
        stage_events=[],
    )


def test_record_is_redacted_and_scores_ordered_execution(tmp_path):
    case = SpiderCase(
        4, "tiny", "private benchmark question", "SELECT name FROM pet ORDER BY name", tmp_path / "x.db"
    )
    result = AnswerResult(
        question=case.question,
        retrieved_tables=["pet"],
        sql="SELECT name FROM pet ORDER BY name",
        execution=ExecutionResult(True, ["name"], [("Milo",)]),
        answer="Milo",
        usage={"latency_ms": 12, "input_tokens": 7, "output_tokens": 3},
        retrieval_result=_retrieval(),
    )
    record = record_spider_run(
        case,
        result,
        [("Milo",)],
        config=RetrievalConfig.rrf_hybrid(),
        k=5,
        repeat_index=2,
    )
    payload = asdict(record)
    assert record.exec_match is True and record.ordered is True
    assert record.candidate_recall == record.selection_recall == record.context_recall == 1.0
    assert "question" not in payload and "gold_sql" not in payload
    assert "private benchmark question" not in json.dumps(payload)


def _record(name, index, matched, *, candidate_recall=1.0, context_recall=1.0, no_sql=False):
    return SpiderRecord(
        index=index,
        db_id="tiny",
        retrieval_config={"name": name},
        retrieval_k=5,
        repeat_index=0,
        predicted_sql="",
        exec_match=matched,
        skipped_gold=False,
        failure_stage="no_sql" if no_sql else (None if matched else "answer_mismatch"),
        ordered=False,
        gold_tables=["pet"],
        candidate_recall=candidate_recall,
        selection_recall=1.0,
        context_recall=context_recall,
    )


def test_summary_applies_paired_preregistered_gate():
    records = [
        _record("current_hybrid", 1, True),
        _record("current_hybrid", 2, False, candidate_recall=0.5, context_recall=0.5),
        _record("rrf_hybrid", 1, True),
        _record("rrf_hybrid", 2, True),
    ]
    summary = summarize_spider(records)
    assert summary["paired"] == {
        "n_cases": 2,
        "wins": 1,
        "losses": 0,
        "ties": 1,
        "win_indices": [2],
        "loss_indices": [],
    }
    assert summary["candidate_gate"]["passed"] is True


def test_cli_defaults_are_the_preregistered_run(tmp_path):
    args = parse_args(["--spider-dir", str(tmp_path)])
    assert (args.repeats, args.k, args.concurrency) == (3, 5, 4)
    assert args.preflight_only is False


def test_comparison_runs_both_configs_and_redacts_raw_records(tmp_path):
    spider, manifest_path, _ = _spider_fixture(tmp_path)
    case = load_spider_slice(spider, manifest_path)[0]
    tables = {"tiny": introspect(case.db_path)}
    output = run_comparison(
        [case],
        tables,
        PlanningFakeModel("SELECT count(*) FROM pet"),
        repeats=1,
        k=5,
        concurrency=1,
    )
    assert len(output["records"]) == 2
    assert [row["retrieval_config"]["name"] for row in output["records"]] == [
        "current_hybrid",
        "rrf_hybrid",
    ]
    # The tiny plural question deliberately exposes a real config difference: the legacy
    # retriever may reject it while RRF retrieves ``pet``. The runner must preserve that result.
    assert output["records"][1]["exec_match"] is True
    serialized = json.dumps(output["records"])
    assert "How many pets?" not in serialized
    assert "gold_sql" not in serialized and "question" not in serialized
