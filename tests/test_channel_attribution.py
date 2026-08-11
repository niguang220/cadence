"""Channel attribution — pure logic (per-channel Recall/Precision + per-gold-table classification),
then a deterministic real run confirming value is excluded (inert) and semantic-ON protected anchors
come from the real MetricRegistry, never from gold."""
from evalharness.channel_attribution import channel_recall_precision, classify_gold_table


def test_channel_recall_precision():
    ordered = ["a", "b", "c", "d", "e", "gold2", "x", "y", "z", "w", "u", "v", "s", "t", "gold1"]
    m = channel_recall_precision(ordered, ["gold1", "gold2"])
    assert m["recall"][5] == 0.0 and m["recall"][10] == 0.5 and m["recall"][15] == 1.0
    assert m["precision"][5] == 0.0 and m["precision"][10] == 1 / 10


def test_channel_recall_precision_absent_gold():
    m = channel_recall_precision(["a", "b"], ["gold"])
    assert m["recall"][15] == 0.0 and m["precision"][5] == 0.0


def test_classify_lexical_top5_but_rrf_demoted():
    tags = classify_gold_table(lex_rank=2, dense_rank=None, fused_rank=9, selector_kept=False,
                               context_recovered=False, governance_protected=False)
    assert "lexical_top5_but_rrf_demoted_out_of_top5" in tags


def test_classify_dense_top5_but_rrf_demoted():
    tags = classify_gold_table(lex_rank=None, dense_rank=3, fused_rank=8, selector_kept=False,
                               context_recovered=False, governance_protected=False)
    assert "dense_top5_but_rrf_demoted_out_of_top5" in tags


def test_classify_both_recalled_but_selector_dropped():
    tags = classify_gold_table(lex_rank=7, dense_rank=8, fused_rank=6, selector_kept=False,
                               context_recovered=False, governance_protected=False)
    assert "both_recalled_but_selector_dropped_at_top5" in tags


def test_classify_absent_from_both_channels_top15():
    tags = classify_gold_table(lex_rank=None, dense_rank=None, fused_rank=None, selector_kept=False,
                               context_recovered=False, governance_protected=False)
    assert tags == ["absent_from_both_channels_top15"]


def test_classify_governance_only_recovered_by_protected_anchor():
    # plan for arpu: absent lexically, dense demoted, but the metric governance protects it -> kept
    tags = classify_gold_table(lex_rank=None, dense_rank=8, fused_rank=None, selector_kept=True,
                               context_recovered=True, governance_protected=True)
    assert "governance_protected_recovery" in tags


# --- real deterministic run on the expanded schema (fastembed + real MetricRegistry; no LLM) --------

def _setup():
    import tempfile

    from agent.db.build_saas_db import build
    from agent.db.introspect import introspect
    from agent.retrieval.value_backend import FakeValueBackend
    from agent.semantic_layer import MetricRegistry
    d = tempfile.mkdtemp()
    db = str(build(f"{d}/e.db", confounders=True))
    return db, introspect(db), FakeValueBackend(), MetricRegistry.load()


def test_real_run_is_deterministic_and_value_is_inert():
    from agent.retrieval.contracts import RetrievalConfig
    from evals.channel_attribution import per_case
    from evalharness.golden import load_saas_metrics
    _, tables, be, reg = _setup()
    case = next(c for c in load_saas_metrics() if c.id == "arpu_asof")
    a = per_case(tables, case, be, RetrievalConfig.dense_value(), reg, semantic_layer=True)
    b = per_case(tables, case, be, RetrievalConfig.dense_value(), reg, semantic_layer=True)
    assert a["gold_tables"] == b["gold_tables"] and a["fused_top5"] == b["fused_top5"]
    assert a["value_inert"] is True                          # value channel excluded (no searchable cols)


def test_semantic_on_protected_anchors_come_from_registry_not_gold():
    from agent.retrieval.contracts import RetrievalConfig
    from evals.channel_attribution import per_case
    from evalharness.golden import load_saas_metrics
    _, tables, be, reg = _setup()
    cfg = RetrievalConfig.dense_value()
    arpu = next(c for c in load_saas_metrics() if c.id == "arpu_asof")
    on = per_case(tables, arpu, be, cfg, reg, semantic_layer=True)
    off = per_case(tables, arpu, be, cfg, reg, semantic_layer=False)
    assert off["protected_anchors"] == []                    # OFF: nothing protected
    assert "plan" in on["protected_anchors"]                 # ON: arpu governance protects plan (not gold-injected)
    # a plain control with no governed-metric alias must have NO protected anchors even though it has gold
    ctrl = next(c for c in load_saas_metrics() if c.id == "ctrl_account_count")
    assert per_case(tables, ctrl, be, cfg, reg, semantic_layer=True)["protected_anchors"] == []


def test_arpu_plan_is_governance_recovered_on_semantic_on():
    from agent.retrieval.contracts import RetrievalConfig
    from evals.channel_attribution import per_case
    from evalharness.golden import load_saas_metrics
    _, tables, be, reg = _setup()
    arpu = next(c for c in load_saas_metrics() if c.id == "arpu_asof")
    on = per_case(tables, arpu, be, RetrievalConfig.dense_value(), reg, semantic_layer=True)
    plan = next(g for g in on["gold_tables"] if g["table"] == "plan")
    assert plan["governance_protected"] and plan["selector_kept"]   # protected anchor keeps it
    assert "governance_protected_recovery" in plan["tags"]
