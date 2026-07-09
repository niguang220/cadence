from agent.semantic_layer import MetricRegistry, load_metrics, format_metrics, MetricDef, retrieve_metrics

def test_load_returns_metricdefs():
    ms = load_metrics()
    assert ms and all(isinstance(m, MetricDef) for m in ms)
    names = {m.name for m in ms}
    assert {"active_user","mrr","mrr_weighted_churn"} <= names

def test_registry_loads_and_retrieves_metrics():
    registry = MetricRegistry.load(embed=lambda xs: [[0.0] for _ in xs])
    got = registry.retrieve("what is our MRR by region?")
    assert "mrr" in {m.name for m in got}
    assert "measure:" in registry.format(got)

def test_format_renders_blocks_without_full_select():
    ms = [m for m in load_metrics() if m.name == "active_user"]
    out = format_metrics(ms)
    assert "active_user" in out
    assert "measure:" in out and "COUNT(DISTINCT user_id)" in out
    assert "filters" in out.lower()
    assert "SELECT" not in out.upper()        # building blocks only, never a full query

def test_format_empty_is_empty_string():
    assert format_metrics([]) == ""

def test_alias_exact_hit():
    ms = load_metrics()
    got = retrieve_metrics("what is our MRR by region?", ms, embed=lambda xs: [[0.0] for _ in xs])
    assert "mrr" in {m.name for m in got}

def test_semantic_paraphrase_hit_via_fake_embed():
    ms = load_metrics()
    # fake embed: question and active_user share vector [1,0]; everything else [0,1]
    def fake(texts):
        out = []
        for t in texts:
            out.append([1.0, 0.0] if ("regularly" in t or "active_user" in t or "active user" in t)
                       else [0.0, 1.0])
        return out
    got = retrieve_metrics("how many people use the product regularly", ms, threshold=0.5, embed=fake)
    assert "active_user" in {m.name for m in got}

def test_below_threshold_miss():
    ms = load_metrics()
    # identical vectors give cosine 1, not 0. Make the question ORTHOGONAL to every metric:
    # first input (question) = [1,0], all metric texts = [0,1] -> cosine 0 < 0.99.
    fake = lambda xs: [[1.0, 0.0]] + [[0.0, 1.0] for _ in xs[1:]]
    got = retrieve_metrics("zzzz unrelated text", ms, threshold=0.99, embed=fake)
    assert got == []

def test_embed_failure_degrades_to_alias_only():
    ms = load_metrics()
    def boom(texts):
        raise RuntimeError("no embedding model")
    got = retrieve_metrics("what is our MRR this month?", ms, embed=boom)
    assert "mrr" in {m.name for m in got}   # alias still works; dense stage degraded, not crashed

def test_top_k_caps_dense_injection():
    ms = load_metrics()
    # no alias hit; fake embed makes every metric match -> without a cap all would be returned
    fake = lambda xs: [[1.0] for _ in xs]          # cosine 1 for all
    got = retrieve_metrics("give me the figures for last quarter", ms, threshold=0.1, top_k=3, embed=fake)
    assert len(got) <= 3

def test_alias_hit_survives_small_top_k():
    ms = load_metrics()
    got = retrieve_metrics("what is our MRR?", ms, top_k=1, embed=lambda xs: [[0.0] for _ in xs])
    assert "mrr" in {m.name for m in got}          # alias guaranteed even at top_k=1
