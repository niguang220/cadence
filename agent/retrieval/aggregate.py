from __future__ import annotations

from collections import defaultdict

from agent.retrieval.contracts import ChannelTableResult, RetrievalSignal

_VALUE_BUCKET = {"exact_keyword": 4, "exact_phrase": 3, "token_match": 2, "fuzzy": 1}


def aggregate(channel: str, signals: list[RetrievalSignal]) -> list[ChannelTableResult]:
    by_table: dict[str, list[RetrievalSignal]] = defaultdict(list)
    for s in signals:
        by_table[s.table].append(s)

    results: list[tuple[float, int, ChannelTableResult]] = []   # (rank_score, bucket, result)
    for table, sigs in by_table.items():
        if channel == "lexical":
            best_per_term: dict[str, RetrievalSignal] = {}
            for s in sigs:
                cur = best_per_term.get(s.query_term)
                key = (s.raw_score, s.match_type, s.column or "")
                if cur is None or key > (cur.raw_score, cur.match_type, cur.column or ""):
                    best_per_term[s.query_term] = s
            kept = sorted(best_per_term.values(),
                          key=lambda s: (-s.raw_score, s.query_term, s.match_type, s.column or ""))
            score = sum(s.raw_score for s in kept)
            bucket = 0
        elif channel == "dense":
            kept = sorted(sigs, key=lambda s: (-s.raw_score, s.table, s.column or ""))[:3]
            score = max(s.raw_score for s in sigs)
            bucket = 0
        elif channel == "value":
            best = sorted(sigs, key=lambda s: (-_VALUE_BUCKET.get(s.match_type, 0), -s.raw_score,
                                                s.target_type, s.column or "", s.document_id or ""))[0]
            kept = [best]
            score = best.raw_score
            bucket = _VALUE_BUCKET.get(best.match_type, 0)
        else:
            raise ValueError(f"unknown channel {channel!r}")
        results.append((score, bucket, ChannelTableResult(
            channel=channel, table=table, raw_table_score=score, channel_rank=0, signals=kept)))

    if channel == "value":
        results.sort(key=lambda r: (-r[1], -r[0], r[2].table))
    else:
        results.sort(key=lambda r: (-r[0], r[2].table))
    out = []
    for i, (_, _, ctr) in enumerate(results, start=1):
        ctr.channel_rank = i
        out.append(ctr)
    return out
