"""Service interfaces used by command/runtime orchestration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from carapace.config import CarapaceConfig
from carapace.models import SourceEntity


class SourceConnector(Protocol):
    def fetch_open_entities(
        self,
        max_prs: int,
        include_issues: bool,
        max_issues: int,
        include_drafts: bool,
        include_closed: bool,
        enrich_pr_details: bool,
        enrich_issue_comments: bool,
    ) -> list[SourceEntity]: ...

    def enrich_entity(
        self,
        entity: SourceEntity,
        include_comments: bool = False,
        mode: str = "minimal",
        include_simple_scores: bool = False,
    ) -> SourceEntity: ...


class SourceConnectorFactory(Protocol):
    def __call__(
        self,
        *,
        repo: str,
        gh_bin: str = "gh",
        rate_limit_retries: int = 2,
        secondary_backoff_base_seconds: float = 5.0,
        rate_limit_max_sleep_seconds: float = 90.0,
    ) -> SourceConnector: ...


class SinkConnectorFactory(Protocol):
    def __call__(
        self,
        *,
        repo: str,
        gh_bin: str = "gh",
        rate_limit_retries: int = 2,
        secondary_backoff_base_seconds: float = 5.0,
        rate_limit_max_sleep_seconds: float = 90.0,
        entity_number_resolver: Callable[[str], int],
        dry_run: bool,
    ) -> Any: ...


class Storage(Protocol):
    def get_ingest_state(self, repo: str) -> dict[str, Any]: ...

    def ingest_quality_stats(self, repo: str, kind: str | None = None) -> dict[str, int]: ...

    def load_ingested_entities(
        self,
        repo: str,
        include_closed: bool,
        include_drafts: bool,
        kind: str | None,
    ) -> list[SourceEntity]: ...

    def ingest_audit_summary(self, repo: str) -> dict[str, Any]: ...

    def upsert_ingest_entities(
        self,
        repo: str,
        entities: list[SourceEntity],
        *,
        source: str = "ingest",
        enrich_level: str | None = None,
    ) -> int: ...

    def get_enrichment_watermarks(self, repo: str, kind: str = "pr") -> dict[str, dict[str, Any]]: ...

    def load_ingested_entities_by_ids(self, repo: str, entity_ids: list[str]) -> dict[str, SourceEntity]: ...

    def get_latest_run_report(self, repo: str) -> Any | None: ...

    def list_ingested_repos(self) -> list[str]: ...

    def get_author_metrics_cache(self, repo: str, authors: list[str]) -> dict[str, dict[str, float | int | str]]: ...

    def load_latest_run_embeddings(self, repo: str, entity_ids: list[str]) -> tuple[int | None, dict[str, list[float]]]: ...

    def load_json_cache(self, repo: str, cache_key: str, *, source_signature: str) -> dict[str, Any] | None: ...

    def upsert_json_cache(self, repo: str, cache_key: str, *, source_signature: str, payload: dict[str, Any]) -> None: ...


class StorageFactory(Protocol):
    def __call__(self, db_path: str | Path) -> Storage: ...


class IngestLoader(Protocol):
    def __call__(
        self,
        connector: Any,
        storage: Any,
        *,
        repo: str,
        ingest_cfg: Any,
        max_prs: int,
        max_issues: int,
    ) -> Any: ...


class EnrichmentRunner(Protocol):
    def __call__(
        self,
        *,
        args: Any,
        config: CarapaceConfig,
        storage: Any,
        entities: list[SourceEntity],
        connector_factory: Callable[..., Any],
    ) -> tuple[list[SourceEntity], int]: ...
