"""Routing action application helpers."""

from __future__ import annotations

from carapace.connectors.base import SinkConnector
from carapace.models import EngineReport


def apply_routing_decisions(report: EngineReport, sink: SinkConnector) -> None:
    for entry in report.routing:
        if entry.labels:
            sink.apply_labels(entry.entity_id, entry.labels)
        if entry.comment:
            sink.post_comment(entry.entity_id, entry.comment)
        if entry.queue_key:
            sink.route_to_queue(entry.entity_id, entry.queue_key)
