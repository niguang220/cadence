"""The canonical public retrieval default and its construction sites.

Every public entry point must construct its default from ONE source. These tests pin that
single source and assert the public construction sites agree with each other -- so a future default
change cannot land in five places and be forgotten in the sixth.
"""
from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import pytest

from agent.retrieval.contracts import RetrievalConfig


def test_default_is_a_single_canonical_source():
    default = RetrievalConfig.default()
    assert isinstance(default, RetrievalConfig)
    # the default must be the typed RRF architecture, not the legacy scaffold
    assert default.fusion == "rrf"
    assert default.relation_strategy == "shortest_path"
    assert default.lexical is True
    assert default.dense_backend == "memory"
    assert default.selector is None
    assert default.value_backend is None      # Elasticsearch value retrieval stays opt-in


@pytest.mark.parametrize("field,value", [
    ("fusion", "legacy_minmax"),
    ("relation_strategy", "legacy_one_hop"),
])
def test_removed_strategies_fail_loudly(field, value):
    values = {field: value}
    with pytest.raises(ValueError, match="unsupported"):
        RetrievalConfig(name="stale", **values)


@pytest.mark.parametrize(
    "module, func",
    [
        ("agent.graph", "run_agent"),
        ("agent.graph", "start_agent_session"),
        ("agent.pipeline", "answer_question"),
        ("agent.pipeline", "start_question_session"),
    ],
)
def test_public_entry_points_default_to_the_canonical_config(module, func):
    import importlib

    fn = getattr(importlib.import_module(module), func)
    param = inspect.signature(fn).parameters["retrieval_config"]
    assert param.default == RetrievalConfig.default(), (
        f"{module}.{func} does not use the canonical default"
    )


def test_graph_fallback_uses_the_canonical_config():
    from agent.graph import _retrieval_config

    assert _retrieval_config({}) == RetrievalConfig.default()


def test_cli_default_config_is_the_canonical_one():
    import agent.cli as cli

    assert cli._config(cli._DEFAULT_CONFIG_NAME) == RetrievalConfig.default()
    parser = cli.build_parser()
    for command in ("ask", "retrieve"):
        action = next(
            a for a in parser._subparsers._group_actions[0].choices[command]._actions
            if a.dest == "config"
        )
        assert action.default == cli._DEFAULT_CONFIG_NAME


def test_cli_exposes_only_product_configs():
    import agent.cli as cli

    assert set(cli._CONFIG_FACTORIES) == {"governed_rrf", "governed_rrf_value"}


def test_readme_exposes_only_product_retrieval_presets():
    text = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    assert "`governed_rrf`" in text and "`governed_rrf_value`" in text
    for internal in ("current_hybrid", "legacy_minmax", "rrf_hybrid",
                     "value_ablation", "dense_value"):
        assert internal not in text, f"evaluation or retired preset leaked into README: {internal}"


def test_config_is_frozen_so_the_shared_default_cannot_be_mutated():
    default = RetrievalConfig.default()
    with pytest.raises(dataclasses.FrozenInstanceError):
        default.fusion = "rrf"     # type: ignore[misc]
