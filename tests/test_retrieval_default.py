"""The canonical public retrieval default and the legacy compatibility alias.

Every public entry point must construct its default from ONE source. These tests pin that
single source, pin the deprecated ``current_hybrid`` alias to the preserved legacy config,
and assert the six public construction sites agree with each other -- so a future default
change cannot land in five places and be forgotten in the sixth.
"""
from __future__ import annotations

import dataclasses
import inspect

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


def test_legacy_minmax_preserves_the_old_retrieval_implementation():
    legacy = RetrievalConfig.legacy_minmax()
    assert legacy.name == "legacy_minmax"
    assert legacy.fusion == "legacy_minmax"
    assert legacy.relation_strategy == "legacy_one_hop"
    assert legacy.lexical is True
    assert legacy.dense_backend == "memory"
    assert legacy.value_backend is None


def test_current_hybrid_is_a_deprecated_alias_for_legacy_minmax():
    # one release cycle of source compatibility: same config, same behaviour
    assert RetrievalConfig.current_hybrid() == RetrievalConfig.legacy_minmax()


def test_default_is_not_the_legacy_config():
    assert RetrievalConfig.default() != RetrievalConfig.legacy_minmax()


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


def test_legacy_alias_is_still_selectable_in_the_cli():
    import agent.cli as cli

    # deprecated but not removed: existing scripts keep working for one cycle
    assert cli._config("current_hybrid") == RetrievalConfig.legacy_minmax()
    assert cli._config("legacy_minmax") == RetrievalConfig.legacy_minmax()


def test_config_is_frozen_so_the_shared_default_cannot_be_mutated():
    default = RetrievalConfig.default()
    with pytest.raises(dataclasses.FrozenInstanceError):
        default.fusion = "legacy_minmax"     # type: ignore[misc]
