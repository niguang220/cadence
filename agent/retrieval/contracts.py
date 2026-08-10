from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


class UnsupportedRetrievalCapability(Exception):
    """Raised at pipeline execution when a config requests a capability not built in this
    stage (es / qdrant / llm). Carries the full list so the caller sees every missing one."""
    def __init__(self, capabilities: list[str]):
        self.capabilities = list(capabilities)
        super().__init__(f"unsupported retrieval capabilities: {self.capabilities}")


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    lexical: bool = True
    dense_backend: Literal["memory", "qdrant"] | None = "memory"
    value_backend: Literal["es"] | None = None
    selector: Literal["llm"] | None = None
    fusion: Literal["legacy_minmax", "rrf"] = "rrf"
    relation_strategy: Literal["legacy_one_hop", "shortest_path"] = "shortest_path"
    candidate_k: int = 15
    context_anchor_k: int = 5
    rrf_constant: int = 60
    max_bridge_hops: int = 3

    @classmethod
    def lexical_baseline(cls) -> "RetrievalConfig":
        return cls(name="lexical_baseline", lexical=True, dense_backend=None,
                   value_backend=None, selector=None, fusion="rrf",
                   relation_strategy="shortest_path")

    @classmethod
    def current_hybrid(cls) -> "RetrievalConfig":
        return cls(name="current_hybrid", lexical=True, dense_backend="memory",
                   value_backend=None, selector=None, fusion="legacy_minmax",
                   relation_strategy="legacy_one_hop")

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
