"""Configuration models and loading for Carapace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class LowPassConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hard_skip_labels: list[str] = Field(default_factory=lambda: ["invalid", "wontfix", "duplicate", "stale"])
    soft_suppress_labels: list[str] = Field(default_factory=lambda: ["question", "discussion"])
    skip_closed: bool = True
    skip_drafts: bool = True
    ignore_recent_pr_hours: int | None = None
    boost_labels: dict[str, float] = Field(default_factory=lambda: {"security": 1.5, "regression": 1.5})
    stale_days: int | None = 90
    stale_action: str = "skip"
    suppress_docs_only_if_no_ci: bool = True
    docs_only_suffixes: list[str] = Field(default_factory=lambda: [".md", ".rst", ".txt"])
    docs_only_prefixes: list[str] = Field(default_factory=lambda: ["docs/", ".github/"])
    suppress_bot_authors: bool = False
    bot_author_patterns: list[str] = Field(default_factory=lambda: ["[bot]", "bot"])
    issue_template_match_threshold: float = 0.7
    issue_template_max_content_tokens: int = 12
    issue_template_action: str = "suppress"
    issue_one_liner_max_tokens: int = 14
    issue_one_liner_action: str = "suppress"
    pr_min_body_tokens: int = 0
    pr_missing_context_action: str = "suppress"
    pr_large_max_files: int = 40
    pr_large_max_churn: int = 5000
    pr_large_action: str = "skip"
    pr_unenriched_max_age_hours: int | None = None
    pr_unenriched_action: str = "suppress"


class SimilarityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_advanced_algorithms: bool = True
    advanced_for_issues: bool = False
    min_score: float = 0.72
    strong_score: float = 0.82
    top_k_candidates: int = 64
    cluster_tail_prune_score: float | None = 0.16
    max_file_bucket_size: int = 300
    max_module_bucket_size: int = 800
    min_candidate_votes: int = 1
    min_candidate_votes_large: int = 2
    large_run_threshold: int = 2000
    weight_lineage: float = 0.35
    weight_structure: float = 0.30
    weight_semantic: float = 0.12
    semantic_text_share: float = 0.65
    semantic_diff_share: float = 0.35
    weight_minhash: float = 0.10
    weight_simhash: float = 0.06
    weight_winnow: float = 0.07
    size_penalty_weight: float = 0.10
    lineage_strong_overlap: float = 0.50
    hard_link_strong_overlap: float = 0.50
    hard_link_weak_overlap: float = 0.20
    hard_link_pr_strong_semantic_min: float = 0.50
    hard_link_issue_pr_strong_semantic_min: float = 0.35
    hard_link_weak_semantic_min: float = 0.45
    pr_semantic_structure_min: float = 0.20
    pr_semantic_min: float = 0.70
    pr_semantic_simhash_min: float = 0.70
    pr_semantic_title_salient_overlap_min: float = 0.08
    weak_structure_min: float = 0.25
    weak_semantic_min: float = 0.30
    weak_minhash_min: float = 0.40
    weak_simhash_min: float = 0.65
    weak_winnow_min: float = 0.20
    strong_minhash_min: float = 0.85
    strong_winnow_min: float = 0.55
    unstructured_semantic_min: float = 0.96
    unstructured_minhash_min: float = 0.90
    unstructured_winnow_min: float = 0.85
    unstructured_pr_title_overlap_min: float = 0.95
    unstructured_pr_semantic_text_min: float = 0.58
    unstructured_pr_simhash_min: float = 0.64
    minhash_num_perm: int = 64
    minhash_bands: int = 8
    minhash_shingle_k: int = 3
    simhash_bits: int = 64
    simhash_chunk_bits: int = 16
    winnow_kgram: int = 5
    winnow_window: int = 4


class CanonicalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weight_coverage: float = 5.0
    weight_centrality: float = 4.0
    weight_ci: float = 3.0
    weight_reviewer: float = 2.0
    weight_approvals: float = 1.5
    weight_priority: float = 1.0
    weight_size_penalty: float = 1.0
    duplicate_threshold: float = 0.22
    duplicate_file_overlap_min: float = 0.80
    duplicate_hunk_overlap_min: float = 0.80
    duplicate_title_salient_overlap_min: float = 0.20
    duplicate_semantic_text_min: float = 0.85
    duplicate_file_title_overlap_min: float = 0.80
    duplicate_hard_link_overlap_min: float = 0.50
    duplicate_hard_link_file_overlap_min: float = 0.90
    duplicate_hard_link_hunk_overlap_min: float = 0.20
    duplicate_hard_link_title_overlap_min: float = 0.80
    duplicate_title_mismatch_overlap_max: float = 0.05
    duplicate_title_mismatch_semantic_text_max: float = 0.72
    duplicate_title_mismatch_hard_link_override_min: float = 1.0
    tie_margin: float = 0.10
    tie_break_min_similarity: float = 0.25
    tie_break_hard_link_min: float = 0.50
    tie_break_hunk_overlap_min: float = 0.20
    tie_break_file_overlap_min: float = 0.80
    tie_break_semantic_text_min: float = 0.75


class LabelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical: str = "triage/canonical"
    duplicate: str = "triage/duplicate"
    related: str = "triage/related"
    linked_pair: str = "triage/linked-pair"
    quarantine: str = "triage/quarantine"
    noise_suppressed: str = "triage/noise-suppressed"
    close_candidate: str = "triage/close-candidate"
    ready_human: str = "triage/ready-human"


class ActionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    safe_mode: bool = True
    add_comments: bool = True
    queue_on_suppress: bool = True
    close_comment: str = "Auto-closing due to low-signal triage policy. Reopen with more details if needed."


class IngestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_closed: bool = False
    include_drafts: bool = False
    include_issues: bool = True
    page_size: int = 100
    resume: bool = True
    enrich_pr_details: bool = False
    enrich_issue_comments: bool = False
    state_checkpoint_interval_pages: int = 5


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "local-hash"
    model: str = "hash-embed-v1"
    dimensions: int = 256
    endpoint: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 10.0


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "sqlite"
    sqlite_path: str = ".carapace/carapace.db"
    persist_runs: bool = False


class CarapaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low_pass: LowPassConfig = Field(default_factory=LowPassConfig)
    similarity: SimilarityConfig = Field(default_factory=SimilarityConfig)
    canonical: CanonicalConfig = Field(default_factory=CanonicalConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    action: ActionConfig = Field(default_factory=ActionConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    return data or {}


def load_effective_config(
    repo_path: str | Path,
    org_defaults: dict[str, Any] | None = None,
    system_defaults: dict[str, Any] | None = None,
    runtime_override: dict[str, Any] | None = None,
) -> CarapaceConfig:
    """Load config with precedence runtime > repo .carapace.yaml > org > system."""
    repo = Path(repo_path)
    repo_config = _load_yaml(repo / ".carapace.yaml")

    merged: dict[str, Any] = {}
    if system_defaults:
        merged = _deep_merge(merged, system_defaults)
    if org_defaults:
        merged = _deep_merge(merged, org_defaults)
    if repo_config:
        merged = _deep_merge(merged, repo_config)
    if runtime_override:
        merged = _deep_merge(merged, runtime_override)

    return CarapaceConfig.model_validate(merged)
