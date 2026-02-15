"""Core Pydantic domain models for Carapace."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EntityKind(str, Enum):
    PR = "pr"
    ISSUE = "issue"
    TICKET = "ticket"


class CIStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class FilterState(str, Enum):
    PASS = "pass"
    SUPPRESS = "suppress"
    SKIP = "skip"


class LowPassAction(str, Enum):
    IGNORE = "ignore"
    CLOSE = "close"


class EdgeTier(str, Enum):
    STRONG = "strong"
    WEAK = "weak"


class DecisionState(str, Enum):
    CANONICAL = "canonical"
    DUPLICATE = "duplicate_of"
    RELATED = "related_non_duplicate"
    TIE_BREAK = "needs_human_tie_break"


class DiffHunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str
    context: str = ""
    added_lines: list[str] = Field(default_factory=list)
    removed_lines: list[str] = Field(default_factory=list)


class ExternalReviewSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    overall_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    summary: str = ""
    risk: dict[str, float] = Field(default_factory=dict)


class SourceEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    provider: str = "github"
    repo: str
    kind: EntityKind
    state: str = "open"
    is_draft: bool = False
    number: int | None = None
    title: str
    body: str = ""
    labels: list[str] = Field(default_factory=list)
    author: str
    author_association: str | None = None
    is_bot: bool = False
    base_branch: str | None = None
    head_branch: str | None = None
    linked_issues: list[str] = Field(default_factory=list)
    soft_linked_issues: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    diff_hunks: list[DiffHunk] = Field(default_factory=list)
    commits: list[str] = Field(default_factory=list)
    patch_ids: list[str] = Field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    ci_status: CIStatus = CIStatus.UNKNOWN
    approvals: int = 0
    review_comments: int = 0
    external_reviews: list[ExternalReviewSignal] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def churn(self) -> int:
        return max(0, self.additions) + max(0, self.deletions)


class Fingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str
    tokens: list[str] = Field(default_factory=list)
    module_buckets: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    hunk_signatures: list[str] = Field(default_factory=list)
    linked_issues: list[str] = Field(default_factory=list)
    soft_linked_issues: list[str] = Field(default_factory=list)
    commits: list[str] = Field(default_factory=list)
    patch_ids: list[str] = Field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    ci_status: CIStatus = CIStatus.UNKNOWN
    approvals: int = 0
    reviewer_score: float = 0.0
    text_embedding: list[float] = Field(default_factory=list)
    diff_embedding: list[float] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)


class LowPassDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str
    state: FilterState
    action: LowPassAction = LowPassAction.IGNORE
    reason_codes: list[str] = Field(default_factory=list)
    priority_weight: float = 1.0


class SimilarityBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lineage: float
    structure: float
    file_overlap: float = 0.0
    hunk_overlap: float = 0.0
    hard_link_overlap: float = 0.0
    soft_link_overlap: float = 0.0
    semantic: float
    semantic_text: float = 0.0
    semantic_diff: float = 0.0
    minhash: float = 0.0
    simhash: float = 0.0
    winnow: float = 0.0
    size_penalty: float
    total: float


class SimilarityEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_a: str
    entity_b: str
    score: float
    tier: EdgeTier
    breakdown: SimilarityBreakdown


class Cluster(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    members: list[str]


class MemberDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str
    state: DecisionState
    score: float
    reason: str = ""
    duplicate_of: str | None = None


class CanonicalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    canonical_entity_id: str | None
    member_decisions: list[MemberDecision]


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str
    labels: list[str] = Field(default_factory=list)
    queue_key: str | None = None
    comment: str | None = None
    close: bool = False


class EngineReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    processed_entities: int
    active_entities: int
    suppressed_entities: int
    skipped_entities: int
    clusters: list[Cluster]
    edges: list[SimilarityEdge]
    canonical_decisions: list[CanonicalDecision]
    low_pass: list[LowPassDecision]
    routing: list[RoutingDecision]
    profile: dict[str, Any] = Field(default_factory=dict)
