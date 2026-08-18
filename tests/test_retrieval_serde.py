from agent.retrieval.contracts import (ChannelTableResult, MetricMatch, RelationEdge, RelationPlan,
                                        RetrievalConfig, RetrievalResult, RetrievalSignal,
                                        RetrievalStageEvent, SelectionDecision, TableCandidate)
from agent.retrieval.serde import (deserialize_config, deserialize_result, serialize_config,
                                   serialize_result)


def test_config_roundtrip():
    for c in (RetrievalConfig.legacy_minmax(), RetrievalConfig.rrf_hybrid(),
              RetrievalConfig.full_rag()):
        assert deserialize_config(serialize_config(c)) == c


def _full_result():
    sig = RetrievalSignal("lexical", "table", "account", None, "acme", 5.0, "name")
    ctr = ChannelTableResult("lexical", "account", 5.0, 1, [sig])
    legacy = TableCandidate("account", {}, None, 1)                       # fusion_score=None
    rrf = TableCandidate("account", {"lexical": ctr}, 0.016, 1)
    return RetrievalResult(
        config_name="rrf_hybrid", signals=[sig], candidates=[legacy, rrf],
        metric_matches=[MetricMatch("mrr", "alias", 1.0, ["account"], ["account.account_id"], [])],
        selection=SelectionDecision(["account"], [], "topk", {}),
        relation_plan=RelationPlan("shortest_path", ["a", "b"], ["m"], ["a", "b", "m"],
                                   [RelationEdge("a", "a_id", "m", "id", "physical_fk")],
                                   ["c"], [["a", "m", "b"]]),
        stage_events=[RetrievalStageEvent("relation", "unconnected_anchor", {"unconnected": ["c"]})])


def test_result_roundtrip_including_none_and_events():
    r = _full_result()
    back = deserialize_result(serialize_result(r))
    assert back == r
    assert back.candidates[0].fusion_score is None
    assert back.stage_events[0].event == "unconnected_anchor"
