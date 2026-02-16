"""Typed command runtime dependency container."""

from __future__ import annotations

from dataclasses import dataclass

from carapace.services.interfaces import (
    EnrichmentRunner,
    IngestLoader,
    SinkConnectorFactory,
    SourceConnectorFactory,
    StorageFactory,
)


@dataclass(frozen=True)
class CommandRuntime:
    source_connector_cls: SourceConnectorFactory
    sink_connector_cls: SinkConnectorFactory
    storage_cls: StorageFactory
    ingest_loader: IngestLoader
    enrich_entities: EnrichmentRunner

