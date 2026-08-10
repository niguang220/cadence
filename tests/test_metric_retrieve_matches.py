from agent.semantic_layer import MetricRegistry, load_metrics


def test_alias_hit_is_typed_with_score():
    hits = MetricRegistry.load().retrieve_matches("what is mrr?")
    mrr = [h for h in hits if h.metric.name == "mrr"]
    assert mrr and mrr[0].match_type == "alias" and mrr[0].score == 1.0


def test_retrieve_is_compat_wrapper_same_metrics():
    reg = MetricRegistry.load()
    assert [m.name for m in reg.retrieve("mrr")] == [h.metric.name for h in reg.retrieve_matches("mrr")]


def test_dense_best_effort_swallows_embed_failure():
    def boom(texts):
        raise RuntimeError("model down")
    reg = MetricRegistry(load_metrics(), embed=boom)
    # alias exact match needs no embedding -> still works
    hits = reg.retrieve_matches("what is mrr")
    assert [h.metric.name for h in hits] == ["mrr"] and hits[0].match_type == "alias"
    # a no-alias question would trigger dense recall -> embed raises -> swallowed -> empty, no crash
    assert reg.retrieve_matches("tell me something unrelated") == []
