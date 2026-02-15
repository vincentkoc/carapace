import json
from pathlib import Path

from carapace.actioning import apply_routing_decisions
from carapace.models import (
    CanonicalDecision,
    Cluster,
    DecisionState,
    EngineReport,
    MemberDecision,
    RoutingDecision,
)
from carapace.reporting import render_markdown_report, write_report_bundle


class FakeSink:
    def __init__(self) -> None:
        self.labels = []
        self.comments = []
        self.routes = []
        self.closed = []

    def apply_labels(self, entity_id: str, labels: list[str]) -> None:
        self.labels.append((entity_id, labels))

    def post_comment(self, entity_id: str, body: str) -> None:
        self.comments.append((entity_id, body))

    def set_status(self, entity_id: str, state: str, context: str) -> None:
        _ = (entity_id, state, context)

    def route_to_queue(self, entity_id: str, queue_key: str) -> None:
        self.routes.append((entity_id, queue_key))

    def close_entity(self, entity_id: str) -> None:
        self.closed.append(entity_id)


def _report() -> EngineReport:
    return EngineReport(
        processed_entities=2,
        active_entities=2,
        suppressed_entities=0,
        skipped_entities=0,
        clusters=[Cluster(id="cluster-1", members=["pr:1", "pr:2"])],
        edges=[],
        canonical_decisions=[
            CanonicalDecision(
                cluster_id="cluster-1",
                canonical_entity_id="pr:1",
                member_decisions=[
                    MemberDecision(entity_id="pr:1", state=DecisionState.CANONICAL, score=5.0),
                    MemberDecision(entity_id="pr:2", state=DecisionState.DUPLICATE, score=4.0, duplicate_of="pr:1"),
                ],
            )
        ],
        low_pass=[],
        routing=[
            RoutingDecision(entity_id="pr:1", labels=["triage/canonical"], comment="ok"),
            RoutingDecision(entity_id="pr:2", labels=["triage/duplicate"], queue_key="quarantine", close=True),
        ],
        profile={"timing_seconds": {"total": 1.23}},
    )


def test_render_and_write_report_bundle(tmp_path: Path) -> None:
    report = _report()
    markdown = render_markdown_report(report)
    assert "# Carapace Triage Report" in markdown
    assert "Canonical: pr:1" in markdown

    write_report_bundle(report, tmp_path)
    assert (tmp_path / "triage_report.md").exists()
    assert (tmp_path / "clusters.json").exists()
    assert (tmp_path / "labels_to_apply.json").exists()
    assert (tmp_path / "scan_profile.json").exists()

    labels_payload = json.loads((tmp_path / "labels_to_apply.json").read_text())
    assert labels_payload["pr:2"]["queue_key"] == "quarantine"
    assert labels_payload["pr:2"]["close"] is True


def test_apply_routing_decisions_invokes_sink_methods() -> None:
    sink = FakeSink()
    apply_routing_decisions(_report(), sink)

    assert len(sink.labels) == 2
    assert len(sink.comments) == 1
    assert len(sink.routes) == 1
    assert sink.closed == ["pr:2"]
