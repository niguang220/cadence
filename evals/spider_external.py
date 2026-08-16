"""Run the preregistered paired Cadence comparison on a frozen Spider slice.

The default invocation performs 180 paid agent runs (30 cases x two retrieval
configurations x three repeats).  Use ``--preflight-only`` to validate all source data
and gold queries without an API key or model calls.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from agent.db.introspect import introspect
from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.serde import serialize_config
from evalharness.spider import (
    SPIDER_MANIFEST_PATH,
    load_spider_slice,
    preflight_spider_cases,
    run_spider_case,
    summarize_spider,
)

_REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "reliability"
_MAX_CONCURRENCY = 16
_CONFIGS = (RetrievalConfig.current_hybrid(), RetrievalConfig.rrf_hybrid())


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen Spider external-validity comparison.")
    parser.add_argument("--spider-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=SPIDER_MANIFEST_PATH)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", type=Path, help="override the ignored raw JSON output path")
    return parser.parse_args(argv)


def run_comparison(cases, tables_by_db, model, *, repeats: int = 3, k: int = 5,
                   concurrency: int = 4) -> dict:
    if repeats < 1 or k < 1 or concurrency < 1:
        raise ValueError("repeats, k, and concurrency must all be >= 1")
    plan = [
        (repeat_index, config, case)
        for repeat_index in range(repeats)
        for config in _CONFIGS
        for case in cases
    ]

    def work(item):
        repeat_index, config, case = item
        return run_spider_case(
            case,
            tables_by_db[case.db_id],
            model,
            config=config,
            k=k,
            repeat_index=repeat_index,
        )

    if concurrency == 1:
        records = [work(item) for item in plan]
    else:
        records = [None] * len(plan)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(concurrency, _MAX_CONCURRENCY)
        ) as executor:
            futures = {executor.submit(work, item): i for i, item in enumerate(plan)}
            for future, index in futures.items():
                records[index] = future.result()
    return {
        "records": [asdict(record) for record in records],
        "summary": summarize_spider(records),
        "repeats": repeats,
        "retrieval_k": k,
        "retrieval_configs": [serialize_config(config) for config in _CONFIGS],
        "semantic_layer": False,
        "clarification": False,
    }


def build_report(cases, tables_by_db, model, *, model_name: str, manifest_path: Path,
                 dataset_sha256: str, repeats: int = 3, k: int = 5,
                 concurrency: int = 4) -> dict:
    result = run_comparison(
        cases, tables_by_db, model, repeats=repeats, k=k, concurrency=concurrency
    )
    return {
        "measured": True,
        "benchmark": "spider-dev",
        "oracle": "Cadence custom execution-match (not official Spider evaluator)",
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_sha256": dataset_sha256,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "n_cases": len(cases),
        "databases_covered": len({case.db_id for case in cases}),
        **result,
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    cases = load_spider_slice(args.spider_dir, args.manifest)
    preflight = preflight_spider_cases(cases)
    print(json.dumps(preflight, indent=2))
    if args.preflight_only:
        return
    if preflight["failed"]:
        print("gold-query preflight failed; refusing to start paid model calls", file=sys.stderr)
        raise SystemExit(2)
    if args.repeats < 1 or args.k < 1 or args.concurrency < 1:
        print("repeats, k, and concurrency must all be >= 1", file=sys.stderr)
        raise SystemExit(2)

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("Spider run requires DEEPSEEK_API_KEY; refusing to start", file=sys.stderr)
        raise SystemExit(1)
    from agent.llm import create_sql_model

    model = create_sql_model()
    model_name = getattr(model, "model_name", getattr(model, "model", "unknown"))
    db_paths = {case.db_id: case.db_path for case in cases}
    tables_by_db = {db_id: introspect(path) for db_id, path in db_paths.items()}
    dataset_sha = hashlib.sha256((args.spider_dir / "dev.json").read_bytes()).hexdigest()
    report = build_report(
        cases,
        tables_by_db,
        model,
        model_name=model_name,
        manifest_path=args.manifest,
        dataset_sha256=dataset_sha,
        repeats=args.repeats,
        k=args.k,
        concurrency=args.concurrency,
    )
    output = args.output
    if output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = _REPORT_DIR / f"spider_external_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
