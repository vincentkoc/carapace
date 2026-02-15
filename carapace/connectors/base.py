"""Connector interfaces for source and sink providers."""

from __future__ import annotations

from typing import Protocol

from carapace.models import SourceEntity


class SourceConnector(Protocol):
    def list_open_entities(self) -> list[SourceEntity]: ...

    def get_entity(self, entity_id: str) -> SourceEntity: ...

    def get_diff_or_change_set(self, entity_id: str) -> dict: ...

    def get_reviews_and_checks(self, entity_id: str) -> dict: ...


class SinkConnector(Protocol):
    def apply_labels(self, entity_id: str, labels: list[str]) -> None: ...

    def post_comment(self, entity_id: str, body: str) -> None: ...

    def set_status(self, entity_id: str, state: str, context: str) -> None: ...

    def route_to_queue(self, entity_id: str, queue_key: str) -> None: ...
