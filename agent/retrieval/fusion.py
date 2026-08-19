from __future__ import annotations

from agent.retrieval.contracts import ChannelTableResult, TableCandidate


def weighted_rrf(channel_results: dict[str, list[ChannelTableResult]], *,
                 rrf_constant: int = 60, weights: dict[str, float] | None = None,
                 candidate_k: int = 15) -> list[TableCandidate]:
    """Fuse per-channel ranked tables by Weighted Reciprocal Rank Fusion:
        score(table) = Σ_channel weight[channel] / (rrf_constant + channel_rank)
    over channels where the table was recalled. Ties: (-score, best channel_rank, table).
    Reads ONLY channel_rank (never raw scores)."""
    weights = weights or {}
    acc: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    per_table_channels: dict[str, dict[str, ChannelTableResult]] = {}
    for channel, results in channel_results.items():
        w = weights.get(channel, 1.0)
        for ctr in results:
            acc[ctr.table] = acc.get(ctr.table, 0.0) + w / (rrf_constant + ctr.channel_rank)
            best_rank[ctr.table] = min(best_rank.get(ctr.table, ctr.channel_rank), ctr.channel_rank)
            per_table_channels.setdefault(ctr.table, {})[channel] = ctr
    ranked = sorted(acc.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], kv[0]))
    out = []
    for i, (table, score) in enumerate(ranked[:candidate_k], start=1):
        out.append(TableCandidate(table=table, channel_results=per_table_channels[table],
                                  fusion_score=score, fusion_rank=i))
    return out
