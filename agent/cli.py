"""Command-line interface for running and inspecting Cadence.

The command surface keeps backend construction and value-index ingestion explicit. Query
execution never creates or refreshes an Elasticsearch index as a side effect.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.value_backend import ValueBackendError

_COMMANDS = {"ask", "retrieve", "index-values", "build-demo-db"}
# The canonical default is named once, here, and resolved through RetrievalConfig.default().
_DEFAULT_CONFIG_NAME = "governed_rrf"
_CONFIG_FACTORIES = {
    "governed_rrf": RetrievalConfig.default,
    "governed_rrf_value": RetrievalConfig.governed_rrf_value,
}


class CLIError(RuntimeError):
    """A user-actionable CLI configuration or dependency error."""


def _version() -> str:
    try:
        return version("cadence")
    except PackageNotFoundError:  # running directly from an uninstalled checkout
        return "0.1.0"


def _add_query_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("question", help="natural-language question")
    parser.add_argument("--db", type=Path, help="SQLite database path (default: bundled SaaS DB)")
    parser.add_argument("-k", type=int, default=5, help="number of schema tables to retrieve")
    parser.add_argument("--config", choices=tuple(_CONFIG_FACTORIES),
                        default=_DEFAULT_CONFIG_NAME,
                        help=f"retrieval preset (default: {_DEFAULT_CONFIG_NAME})")
    parser.add_argument("--semantic-layer", action=argparse.BooleanOptionalAction, default=True,
                        help="bind governed metric definitions for this request "
                             "(default: on; use --no-semantic-layer to opt out)")
    parser.add_argument("--es-url", help="Elasticsearch URL (or set CADENCE_ES_URL)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cadence", description="Reliability-first natural-language-to-SQL data agent.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    commands = parser.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="run the full agent (requires DEEPSEEK_API_KEY)")
    _add_query_options(ask)

    retrieve = commands.add_parser(
        "retrieve", help="inspect schema retrieval without an LLM or API key")
    _add_query_options(retrieve)

    index = commands.add_parser(
        "index-values", help="build or refresh the Elasticsearch value index")
    index.add_argument("--db", type=Path, help="SQLite database path (default: bundled SaaS DB)")
    index.add_argument("--es-url", help="Elasticsearch URL (or set CADENCE_ES_URL)")
    index.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    demo = commands.add_parser("build-demo-db", help="write a reproducible demo SQLite database")
    demo.add_argument("output", type=Path, help="output .db path")
    demo.add_argument("--kind", choices=("saas", "value"), default="saas")
    demo.add_argument("--force", action="store_true", help="replace an existing output file")
    demo.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def _normalize_argv(argv: list[str]) -> list[str]:
    """Map the original ``python -m agent QUESTION`` interface onto subcommands."""
    if not argv or argv[0] in _COMMANDS or argv[0] in {"-h", "--help", "--version"}:
        return argv
    if "--retrieval-only" in argv:
        return ["retrieve", *[arg for arg in argv if arg != "--retrieval-only"]]
    return ["ask", *argv]


def _database_path(requested: Path | None) -> Path:
    if requested is not None:
        path = requested.expanduser().resolve()
        if not path.is_file():
            raise CLIError(f"database does not exist: {path}")
        return path
    from agent.db.build_saas_db import DB_PATH, build

    if DB_PATH.exists():
        return DB_PATH
    with contextlib.redirect_stdout(io.StringIO()):
        return build()


def _config(name: str) -> RetrievalConfig:
    return _CONFIG_FACTORIES[name]()


def _value_backend(config: RetrievalConfig, tables, es_url: str | None, *, require_index: bool,
                   db_path: Path):
    if config.value_backend is None:
        return None
    url = es_url or os.environ.get("CADENCE_ES_URL")
    if not url:
        raise CLIError(
            f"retrieval preset {config.name!r} needs --es-url or CADENCE_ES_URL; "
            "run 'cadence index-values --help' for the ingestion step")
    try:
        from agent.retrieval.value_backend import ElasticsearchValueBackend
        from agent.retrieval.value_index import index_name

        backend = ElasticsearchValueBackend.from_url(url, index_name(tables))
    except ImportError as exc:
        raise CLIError("Elasticsearch support is not installed; run: pip install -e '.[es]'") from exc
    try:
        exists = backend.index_exists() if require_index else True
    except ValueBackendError as exc:
        raise CLIError(str(exc)) from exc
    if not exists:
        raise CLIError(
            "value index does not exist for this schema; run: "
            f"cadence index-values --db {str(db_path)!r} --es-url {url!r}")
    return backend


def _metric_hits(question: str, tables, enabled: bool):
    if not enabled:
        return []
    from agent.retrieval.metric_match import registry_governs, validate_all_metrics
    from agent.semantic_layer import MetricRegistry

    registry = MetricRegistry.load()
    if not registry_governs(registry, tables):
        return []                       # the registry does not govern this database
    validate_all_metrics(registry, tables)
    return [h for h in registry.retrieve_matches(question) if h.match_type == "alias"]


def _run_retrieve(args: argparse.Namespace) -> int:
    from agent.db.introspect import introspect
    from agent.retrieval.pipeline import run_retrieval
    from agent.retrieval.serde import serialize_result

    db = _database_path(args.db)
    tables = introspect(db)
    config = _config(args.config)
    backend = _value_backend(config, tables, args.es_url, require_index=True, db_path=db)
    result = run_retrieval(
        args.question, tables, config, k=args.k,
        metric_hits=_metric_hits(args.question, tables, args.semantic_layer),
        value_backend=backend, value_query=args.question,
    )
    if args.json:
        print(json.dumps(serialize_result(result), ensure_ascii=False, indent=2))
    else:
        print(f"config:     {result.config_name}")
        print(f"candidates: {[c.table for c in result.candidates]}")
        print(f"tables:     {result.relation_plan.context_tables}")
        if result.stage_events:
            print(f"events:     {[f'{e.stage}:{e.event}' for e in result.stage_events]}")
    return 0


def _answer_payload(result) -> dict:
    execution = result.execution
    payload = {
        "question": result.question,
        "retrieved_tables": result.retrieved_tables,
        "sql": result.sql,
        "answer": result.answer,
        "clarification": result.clarification,
        "execution": {
            "ok": execution.ok,
            "columns": execution.columns,
            "rows": execution.rows,
            "error": execution.error,
            "truncated": execution.truncated,
        },
        "usage": result.usage,
    }
    if result.retrieval_result is not None:
        from agent.retrieval.serde import serialize_result

        payload["retrieval"] = serialize_result(result.retrieval_result)
    return payload


def _run_ask(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv
    load_dotenv()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise CLIError("DEEPSEEK_API_KEY is required for 'cadence ask'")

    from agent.db.introspect import introspect
    from agent.pipeline import answer_question

    db = _database_path(args.db)
    tables = introspect(db)
    config = _config(args.config)
    backend = _value_backend(config, tables, args.es_url, require_index=True, db_path=db)
    result = answer_question(
        db, args.question, k=args.k, tables=tables, semantic_layer=args.semantic_layer,
        retrieval_config=config, value_backend=backend,
    )
    if args.json:
        print(json.dumps(_answer_payload(result), ensure_ascii=False, indent=2, default=str))
        return 0
    if result.clarification:
        print(result.clarification)
        return 0
    print(f"tables: {result.retrieved_tables}")
    print(f"SQL:    {result.sql}")
    print(f"answer: {result.answer}")
    events = result.retrieval_result.stage_events if result.retrieval_result else []
    if events:
        print(f"retrieval events: {[f'{e.stage}:{e.event}' for e in events]}")
    return 0


def _run_index(args: argparse.Namespace) -> int:
    from agent.db.introspect import introspect
    from agent.retrieval.value_index import build_value_index

    db = _database_path(args.db)
    tables = introspect(db)
    # Use a value-enabled config solely to construct the concrete backend. The index name is
    # schema-derived and shared by every value-enabled retrieval preset.
    backend = _value_backend(RetrievalConfig.governed_rrf_value(), tables, args.es_url,
                             require_index=False, db_path=db)
    try:
        result = build_value_index(tables, db, backend)
    except ValueBackendError as exc:
        raise CLIError(str(exc)) from exc
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(f"index:       {result.index}")
        print(f"documents:   {result.doc_count}")
        print(f"fingerprint: {result.fingerprint}")
    return 0


def _run_build_demo(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise CLIError(f"output already exists: {output} (pass --force to replace it)")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "value":
        from agent.db.build_value_db import build
    else:
        from agent.db.build_saas_db import build
    with contextlib.redirect_stdout(io.StringIO()):
        path = Path(build(output))
    payload = {"kind": args.kind, "path": str(path)}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"built {args.kind} demo database: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(_normalize_argv(raw))
    try:
        if args.command == "ask":
            return _run_ask(args)
        if args.command == "retrieve":
            return _run_retrieve(args)
        if args.command == "index-values":
            return _run_index(args)
        return _run_build_demo(args)
    except CLIError as exc:
        print(f"cadence: error: {exc}", file=sys.stderr)
        return 2
