"""Service-free tests for the public Cadence CLI."""
from __future__ import annotations

import json

import agent.cli as cli
from agent.db.build_saas_db import build as build_saas
from agent.db.build_value_db import build as build_value
from agent.retrieval.value_backend import FakeValueBackend


def test_shorthand_invocations_map_to_subcommands():
    assert cli._normalize_argv(["revenue by plan"]) == ["ask", "revenue by plan"]
    assert cli._normalize_argv(
        ["--retrieval-only", "revenue by plan", "-k", "3"]
    ) == ["retrieve", "revenue by plan", "-k", "3"]
    assert cli._normalize_argv(["retrieve", "q"]) == ["retrieve", "q"]
    assert cli._normalize_argv(["--help"]) == ["--help"]


def test_parser_exposes_retrieval_and_semantic_options():
    args = cli.build_parser().parse_args([
        "retrieve", "MRR by region", "--config", "governed_rrf", "--semantic-layer", "-k", "8",
    ])
    assert args.command == "retrieve"
    assert args.config == "governed_rrf"
    assert args.semantic_layer is True
    assert args.k == 8


def test_build_demo_db_refuses_overwrite_without_force(tmp_path, capsys):
    output = tmp_path / "value.db"
    assert cli.main(["build-demo-db", str(output), "--kind", "value"]) == 0
    assert output.is_file()
    assert cli.main(["build-demo-db", str(output), "--kind", "value"]) == 2
    assert "--force" in capsys.readouterr().err
    assert cli.main(["build-demo-db", str(output), "--kind", "value", "--force"]) == 0


def test_retrieve_json_is_real_typed_pipeline(tmp_path, capsys):
    db = build_saas(tmp_path / "saas.db")
    capsys.readouterr()
    rc = cli.main([
        "retrieve", "revenue by plan", "--db", str(db),
        "--config", "governed_rrf", "--json",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config_name"] == "governed_rrf"
    assert payload["candidates"]
    assert "plan" in payload["relation_plan"]["context_tables"]


def test_ask_fails_before_model_creation_without_api_key(monkeypatch, capsys):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert cli.main(["ask", "MRR by region"]) == 2
    assert "DEEPSEEK_API_KEY" in capsys.readouterr().err


def test_value_config_requires_es_url(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CADENCE_ES_URL", raising=False)
    db = build_value(tmp_path / "value.db")
    rc = cli.main([
        "retrieve", "tickets for Globex Corporation", "--db", str(db),
        "--config", "governed_rrf_value",
    ])
    assert rc == 2
    assert "--es-url or CADENCE_ES_URL" in capsys.readouterr().err


def test_value_config_reports_missing_index_before_query(tmp_path, monkeypatch, capsys):
    class MissingIndex:
        def index_exists(self):
            return False

    from agent.retrieval.value_backend import ElasticsearchValueBackend
    monkeypatch.setattr(ElasticsearchValueBackend, "from_url",
                        classmethod(lambda cls, url, index: MissingIndex()))
    db = build_value(tmp_path / "value.db")
    rc = cli.main([
        "retrieve", "tickets for Globex Corporation", "--db", str(db),
        "--config", "governed_rrf_value", "--es-url", "http://es.test:9200",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "value index does not exist" in err
    assert "cadence index-values" in err


def test_index_values_runs_explicit_ingestion_step(tmp_path, monkeypatch, capsys):
    backend = FakeValueBackend()
    monkeypatch.setattr(cli, "_value_backend", lambda *a, **k: backend)
    db = build_value(tmp_path / "value.db")
    rc = cli.main([
        "index-values", "--db", str(db), "--es-url", "http://es.test:9200", "--json",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["doc_count"] > 0
    assert backend._docs
