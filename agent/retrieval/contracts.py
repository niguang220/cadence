from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Literal


class UnsupportedRetrievalCapability(Exception):
    """Raised when a config requests an unavailable retrieval capability.

    Carries the full list so the caller sees every missing capability in one failure.
    """
    def __init__(self, capabilities: list[str]):
        self.capabilities = list(capabilities)
        super().__init__(f"unsupported retrieval capabilities: {self.capabilities}")


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    lexical: bool = True
    # WHICH lexical scorer runs behind LexicalChannel. "hand_weighted" is the original
    # hand-curated scorer, kept as the comparison arm; "bm25" is the standard in-memory BM25.
    lexical_backend: Literal["hand_weighted", "bm25"] = "hand_weighted"
    dense_backend: Literal["memory", "qdrant"] | None = "memory"
    value_backend: Literal["es"] | None = None
    selector: Literal["llm"] | None = None
    fusion: Literal["legacy_minmax", "rrf"] = "rrf"
    relation_strategy: Literal["legacy_one_hop", "shortest_path"] = "shortest_path"
    candidate_k: int = 15
    context_anchor_k: int = 5
    rrf_constant: int = 60
    max_bridge_hops: int = 3
    # Per-channel Weighted-RRF weights. Weighted RRF always accepted these; making them
    # configuration is what lets a chosen weighting actually ship. Both must stay positive:
    # a zero lexical weight is dense-only ranking, not hybrid fusion.
    lexical_weight: float = 1.0
    dense_weight: float = 1.0

    def with_weights(self, *, lexical: float, dense: float) -> "RetrievalConfig":
        """Return a copy carrying explicit channel weights. Rejects a non-positive weight so a
        degenerate single-channel ranking cannot be produced by configuration."""
        if lexical <= 0 or dense <= 0:
            raise ValueError(
                f"channel weights must be positive (got lexical={lexical}, dense={dense}); "
                "a zero weight is single-channel ranking, not hybrid fusion")
        return replace(self, lexical_weight=float(lexical), dense_weight=float(dense))

    def channel_weights(self) -> dict[str, float]:
        """Weights as weighted_rrf consumes them. The value channel is opt-in and not part of
        the weighting decision, so it stays at the neutral 1.0."""
        return {"lexical": self.lexical_weight, "dense": self.dense_weight, "value": 1.0}

    @classmethod
    def default(cls) -> "RetrievalConfig":
        """THE canonical public retrieval default -- the single source every public entry
        point constructs from. Changing the shipping default means changing this one method.

        Governed typed RRF: lexical + in-memory dense channels, Weighted RRF fusion,
        deterministic Top-K selection, protected anchors when governance is enabled, and
        shortest-path relation planning. Elasticsearch value retrieval stays opt-in."""
        return cls.governed_rrf()

    @classmethod
    def governed_rrf(cls) -> "RetrievalConfig":
        """The shipping default, selected by the deterministic backend/weight matrix.

        BM25 at the neutral equal weighting: the frozen selection surfaces could not tell the six
        cells apart, so the rule fell back to the standard external implementation at the
        untuned weight rather than fitting a weight to a surface that could not measure it. The
        matrix provenance records surfaces_discriminated=False for exactly this reason."""
        return cls(name="governed_rrf", lexical=True, lexical_backend="bm25",
                   dense_backend="memory", value_backend=None, selector=None, fusion="rrf",
                   relation_strategy="shortest_path",
                   lexical_weight=1.0, dense_weight=1.0)

    @classmethod
    def lexical_baseline(cls) -> "RetrievalConfig":
        return cls(name="lexical_baseline", lexical=True, dense_backend=None,
                   value_backend=None, selector=None, fusion="rrf",
                   relation_strategy="shortest_path")

    @classmethod
    def legacy_minmax(cls) -> "RetrievalConfig":
        """The preserved pre-migration retrieval implementation: min-max fusion over the
        legacy hybrid retriever plus one-hop FK closure. Retained for one release cycle as
        a historical comparator for already-published results. NOT a supported product mode
        and never a default."""
        return cls(name="legacy_minmax", lexical=True, dense_backend="memory",
                   value_backend=None, selector=None, fusion="legacy_minmax",
                   relation_strategy="legacy_one_hop")

    @classmethod
    def current_hybrid(cls) -> "RetrievalConfig":
        """Deprecated alias for ``legacy_minmax`` kept for one release cycle so existing
        callers keep working. Use ``legacy_minmax()`` (comparator) or ``default()`` (product)."""
        return cls.legacy_minmax()

    @classmethod
    def rrf_hybrid(cls) -> "RetrievalConfig":
        return cls(name="rrf_hybrid", lexical=True, dense_backend="memory",
                   value_backend=None, selector=None, fusion="rrf",
                   relation_strategy="shortest_path")

    @classmethod
    def value_ablation(cls) -> "RetrievalConfig":
        return cls(name="value_ablation", lexical=True, dense_backend=None,
                   value_backend="es", selector=None, fusion="rrf",
                   relation_strategy="shortest_path")

    @classmethod
    def dense_value(cls) -> "RetrievalConfig":
        """Factorial cell: lexical + dense + value (no LLM selector). The dense+value corner of the
        Stage 2 ablation over {dense off/on} x {value off/on}, with lexical as the admission floor."""
        return cls(name="dense_value", lexical=True, dense_backend="memory",
                   value_backend="es", selector=None, fusion="rrf",
                   relation_strategy="shortest_path")

    @classmethod
    def full_rag(cls) -> "RetrievalConfig":
        return cls(name="full_rag", lexical=True, dense_backend="memory",
                   value_backend="es", selector="llm", fusion="rrf",
                   relation_strategy="shortest_path")


@dataclass
class RetrievalSignal:
    channel: Literal["lexical", "dense", "value"]
    target_type: Literal["table", "column", "value"]
    table: str                            # owner table this hit votes for (required)
    column: str | None
    query_term: str
    raw_score: float                      # only comparable WITHIN its own channel
    match_type: str
    document_id: str | None = None
    matched_value: str | None = None      # canonical DB value; ONLY set for a searchable value hit


@dataclass
class ChannelTableResult:
    channel: Literal["lexical", "dense", "value"]
    table: str
    raw_table_score: float
    channel_rank: int
    signals: list[RetrievalSignal]


@dataclass
class TableCandidate:
    table: str
    channel_results: dict[str, ChannelTableResult]
    # None for legacy compatibility candidates — the legacy min-max fusion is not this
    # pipeline's RRF; only fusion_rank is meaningful there. Real float for RRF candidates.
    fusion_score: float | None
    fusion_rank: int


@dataclass
class MetricMatch:
    metric: str
    match_type: str
    score: float
    required_tables: list[str]
    required_columns: list[str]
    required_filters: list[str]


@dataclass
class SelectionDecision:
    anchor_tables: list[str]
    dropped_tables: list[str]
    selector: Literal["topk", "noop", "llm"]
    model_reason: dict[str, str]           # explanation only, NOT trusted evidence


@dataclass
class RelationEdge:
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    source: Literal["physical_fk", "logical_fk"]


@dataclass
class RelationPlan:
    strategy: Literal["legacy_one_hop", "shortest_path"]
    anchors: list[str]
    bridges: list[str]
    context_tables: list[str]
    edges: list[RelationEdge]
    unconnected_anchors: list[str]
    ambiguous_paths: list[list[str]]


@dataclass
class RetrievalStageEvent:
    stage: Literal["channel", "admission", "fusion", "selection", "relation"]
    event: str                             # "admission_rejected" | "dense_degraded" | "selector_fallback" | "unconnected_anchor"
    detail: dict[str, object] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    config_name: str
    signals: list[RetrievalSignal]
    candidates: list[TableCandidate]
    metric_matches: list[MetricMatch]
    selection: SelectionDecision
    relation_plan: RelationPlan
    stage_events: list[RetrievalStageEvent] = field(default_factory=list)
