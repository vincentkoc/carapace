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
    boost_labels: dict[str, float] = Field(default_factory=lambda: {"security": 1.5, "regression": 1.5})
    stale_days: int | None = 90
    suppress_docs_only_if_no_ci: bool = True
    docs_only_suffixes: list[str] = Field(default_factory=lambda: [".md", ".rst", ".txt"])
    docs_only_prefixes: list[str] = Field(default_factory=lambda: ["docs/", ".github/"])
    suppress_bot_authors: bool = False
    bot_author_patterns: list[str] = Field(default_factory=lambda: ["[bot]", "bot"])


class SimilarityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_advanced_algorithms: bool = True
    min_score: float = 0.72
    strong_score: float = 0.82
    top_k_candidates: int = 64
    weight_lineage: float = 0.35
    weight_structure: float = 0.30
    weight_semantic: float = 0.12
    weight_minhash: float = 0.10
    weight_simhash: float = 0.06
    weight_winnow: float = 0.07
    size_penalty_weight: float = 0.10
    lineage_strong_overlap: float = 0.50
    weak_structure_min: float = 0.25
    weak_semantic_min: float = 0.30
    weak_minhash_min: float = 0.40
    weak_simhash_min: float = 0.65
    weak_winnow_min: float = 0.20
    strong_minhash_min: float = 0.85
    strong_winnow_min: float = 0.55
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
    duplicate_threshold: float = 0.78
    tie_margin: float = 0.10


class LabelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical: str = "triage/canonical"
    duplicate: str = "triage/duplicate"
    related: str = "triage/related"
    quarantine: str = "triage/quarantine"
    noise_suppressed: str = "triage/noise-suppressed"
    ready_human: str = "triage/ready-human"


class ActionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    safe_mode: bool = True
    add_comments: bool = True
    queue_on_suppress: bool = True


class IngestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_closed: bool = False
    include_drafts: bool = False
    include_issues: bool = True
    page_size: int = 100
    resume: bool = True
    enrich_pr_details: bool = False
    enrich_issue_comments: bool = False


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
