"""Connector interfaces for source and sink providers."""

from __future__ import annotations

from typing import Protocol

from carapace.models import SourceEntity


class SourceConnector(Protocol):
    def fetch_open_entities(
        self,
        max_prs: int = 200,
        include_issues: bool = True,
        max_issues: int = 200,
        include_drafts: bool = False,
        include_closed: bool = False,
        enrich_pr_details: bool = True,
        enrich_issue_comments: bool = True,
    ) -> list[SourceEntity]: ...

    def fetch_pull_page(
        self,
        *,
        page: int,
        per_page: int = 100,
        state: str = "open",
        include_drafts: bool = False,
        enrich_details: bool = False,
        enrich_comments: bool = False,
    ) -> list[SourceEntity]: ...

    def fetch_issue_page(self, *, page: int, per_page: int = 100, state: str = "open") -> list[SourceEntity]: ...

    def list_open_entities(self) -> list[SourceEntity]: ...

    def get_entity(self, entity_id: str) -> SourceEntity: ...

    def enrich_entity(
        self,
        entity: SourceEntity,
        include_comments: bool = False,
        mode: str = "minimal",
        include_simple_scores: bool = False,
    ) -> SourceEntity: ...

    def get_diff_or_change_set(self, entity_id: str) -> dict: ...

    def get_reviews_and_checks(self, entity_id: str) -> dict: ...


class SinkConnector(Protocol):
    def apply_labels(self, entity_id: str, labels: list[str]) -> None: ...

    def post_comment(self, entity_id: str, body: str) -> None: ...

    def set_status(self, entity_id: str, state: str, context: str) -> None: ...

    def route_to_queue(self, entity_id: str, queue_key: str) -> None: ...

    def close_entity(self, entity_id: str) -> None: ...
