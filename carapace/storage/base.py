"""Storage backend interfaces for Carapace persistence."""

from __future__ import annotations

from typing import Protocol

from carapace.models import EngineReport, Fingerprint, SourceEntity


class StorageBackend(Protocol):
    def init_schema(self) -> None: ...

    def save_run(
        self,
        entities: list[SourceEntity],
        fingerprints: dict[str, Fingerprint],
        report: EngineReport,
        embedding_model: str,
    ) -> int: ...

    def list_runs(self, limit: int = 20) -> list[dict]: ...

    def upsert_ingest_entities(
        self,
        repo: str,
        entities: list[SourceEntity],
        *,
        source: str = "ingest",
        enrich_level: str | None = None,
    ) -> int: ...

    def load_ingested_entities(
        self,
        repo: str,
        include_closed: bool = False,
        include_drafts: bool = False,
        kind: str | None = None,
    ) -> list[SourceEntity]: ...

    def get_ingest_state(self, repo: str) -> dict: ...

    def save_ingest_state(
        self,
        repo: str,
        *,
        pr_next_page: int,
        issue_next_page: int,
        phase: str,
        completed: bool,
    ) -> None: ...

    def ingest_quality_stats(self, repo: str, kind: str | None = None) -> dict[str, int]: ...

    def get_enrichment_watermarks(self, repo: str, kind: str | None = None) -> dict[str, dict[str, str | None]]: ...

    def mark_entities_closed_except(self, repo: str, *, kind: str, seen_entity_ids: set[str]) -> int: ...

    def ingest_audit_summary(self, repo: str) -> dict: ...

    def load_fingerprint_cache(
        self,
        repo: str,
        entities: list[SourceEntity],
        *,
        model_id: str,
    ) -> dict[str, Fingerprint]: ...

    def upsert_fingerprint_cache(
        self,
        repo: str,
        entities: list[SourceEntity],
        fingerprints: dict[str, Fingerprint],
        *,
        model_id: str,
    ) -> int: ...
