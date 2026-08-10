import random

from agent.retrieval.aggregate import aggregate
from agent.retrieval.contracts import RetrievalSignal


def _sig(ch, table, score, term="t", tt="table", col=None, mt="m"):
    return RetrievalSignal(channel=ch, target_type=tt, table=table, column=col,
                           query_term=term, raw_score=score, match_type=mt)


def test_lexical_same_token_many_columns_counts_once_then_sums_tokens():
    sigs = [_sig("lexical", "account", 5, term="acme"),
            _sig("lexical", "account", 3, term="acme"),   # same token -> keep max (5)
            _sig("lexical", "account", 2, term="corp")]    # different token -> +2
    out = aggregate("lexical", sigs)
    assert {r.table: r.raw_table_score for r in out}["account"] == 7.0   # max(5,3)+2


def test_dense_table_score_is_max_similarity_keeps_top3():
    sigs = [_sig("dense", "account", 0.9, tt="column", col="name"),
            _sig("dense", "account", 0.4, tt="column", col="id"),
            _sig("dense", "account", 0.8, tt="column", col="alias"),
            _sig("dense", "account", 0.1, tt="column", col="x")]
    r = aggregate("dense", sigs)[0]
    assert r.raw_table_score == 0.9 and len(r.signals) == 3


def test_channel_rank_is_deterministic_ties_break_by_name():
    sigs = [_sig("lexical", "b_table", 5), _sig("lexical", "a_table", 5)]
    out = aggregate("lexical", sigs)
    assert [r.table for r in out] == ["a_table", "b_table"]
    assert [r.channel_rank for r in out] == [1, 2]


def test_value_orders_by_quality_bucket_then_score():
    sigs = [_sig("value", "a", 0.99, mt="fuzzy"),         # low bucket, high score
            _sig("value", "b", 0.10, mt="exact_keyword")]  # top bucket, low score
    out = aggregate("value", sigs)
    assert [r.table for r in out] == ["b", "a"]           # exact_keyword bucket wins


def test_kept_signals_ordering_is_input_order_independent():
    base = [
        _sig("dense", "account", 0.5, tt="column", col="b"),
        _sig("dense", "account", 0.5, tt="column", col="a"),   # tie on score -> col tiebreak
        _sig("dense", "account", 0.9, tt="column", col="z"),
    ]
    shuffled = base[:]
    random.Random(0).shuffle(shuffled)
    a = aggregate("dense", base)[0].signals
    b = aggregate("dense", shuffled)[0].signals
    assert [(s.raw_score, s.column) for s in a] == [(s.raw_score, s.column) for s in b]
    assert [(s.raw_score, s.column) for s in a] == [(0.9, "z"), (0.5, "a"), (0.5, "b")]
