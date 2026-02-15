"""Storage backend interfaces for Carapace persistence."""

from __future__ import annotations

from typing import Protocol

from carapace.models import EngineReport, Fingerprint, SourceEntity


class StorageBackend(Protocol):
    def init_schema(self) -> None:
        ...

    def save_run(
        self,
        entities: list[SourceEntity],
        fingerprints: dict[str, Fingerprint],
        report: EngineReport,
        embedding_model: str,
    ) -> int:
        ...

    def list_runs(self, limit: int = 20) -> list[dict]:
        ...
