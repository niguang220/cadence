"""CLI surface of the e2e baseline driver: reproducible one-config-per-run invocations.

Pure/service-free -- exercises arg parsing, the config-name mapping, and the config-stamped
report path. The real-API run itself (main) is not exercised here."""
import pytest

from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.serde import serialize_config
from evals.run_e2e import config_by_name, parse_args, report_path


def test_config_by_name_maps_the_three_presets():
    assert serialize_config(config_by_name("lexical_baseline")) == \
        serialize_config(RetrievalConfig.lexical_baseline())
    assert serialize_config(config_by_name("current_hybrid")) == \
        serialize_config(RetrievalConfig.current_hybrid())
    assert serialize_config(config_by_name("rrf_hybrid")) == \
        serialize_config(RetrievalConfig.rrf_hybrid())


def test_config_by_name_rejects_unknown():
    with pytest.raises(ValueError):
        config_by_name("es_full")


def test_parse_args_defaults_to_lexical_baseline_5_5():
    a = parse_args([])
    assert a.retrieval_config == "lexical_baseline"
    assert a.repeats == 5 and a.k == 5


def test_parse_args_overrides():
    a = parse_args(["--retrieval-config", "rrf_hybrid", "--repeats", "3", "--k", "8"])
    assert a.retrieval_config == "rrf_hybrid" and a.repeats == 3 and a.k == 8


def test_parse_args_rejects_unsupported_config():
    with pytest.raises(SystemExit):        # argparse choices reject qdrant/es/full_rag here
        parse_args(["--retrieval-config", "qdrant_full"])


def test_report_path_includes_config_name(tmp_path):
    p = report_path("rrf_hybrid", "20260810_161500", report_dir=tmp_path)
    assert p.name == "e2e_baseline_rrf_hybrid_20260810_161500.json"
    # different configs at the same second get distinct, unambiguous filenames
    other = report_path("current_hybrid", "20260810_161500", report_dir=tmp_path)
    assert other.name != p.name
    # still matches the .git/info/exclude glob docs/reliability/e2e_baseline_*.json
    assert p.name.startswith("e2e_baseline_") and p.name.endswith(".json")
