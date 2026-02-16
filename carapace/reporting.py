"""Report generation utilities for offline workflows."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from carapace.models import EngineReport, SourceEntity


def _entity_titles(entities: list[SourceEntity] | None) -> dict[str, str]:
    if not entities:
        return {}
    return {entity.id: entity.title for entity in entities}


def _entity_index(entities: list[SourceEntity] | None) -> dict[str, SourceEntity]:
    if not entities:
        return {}
    return {entity.id: entity for entity in entities}


def _display_entity(entity_id: str | None, titles: dict[str, str]) -> str:
    if not entity_id:
        return "none"
    title = titles.get(entity_id, "").strip()
    if not title:
        return entity_id
    return f'{entity_id} — "{title}"'


def _provider_score(entity: SourceEntity, provider: str) -> float | None:
    values = [signal.overall_score for signal in entity.external_reviews if provider in signal.provider.lower()]
    if not values:
        return None
    return sum(values) / len(values)


def _member_signal_suffix(entity: SourceEntity | None) -> str:
    if entity is None:
        return ""
    if entity.kind.value != "pr":
        return ""

    mergeable = "unknown"
    if entity.mergeable is True:
        mergeable = "yes"
    elif entity.mergeable is False:
        mergeable = "no"

    greptile = _provider_score(entity, "greptile")
    coderabbit = _provider_score(entity, "coderabbit")

    parts = [f"mergeable={mergeable}"]
    if entity.mergeable_state:
        parts.append(f"merge_state={entity.mergeable_state}")
    if greptile is not None:
        parts.append(f"greptile={greptile:.2f}")
    if coderabbit is not None:
        parts.append(f"coderabbit={coderabbit:.2f}")
    parts.append(f"ci={entity.ci_status.value}")
    parts.append(f"approvals={entity.approvals}")
    return " " + " ".join(parts)


def render_markdown_report(
    report: EngineReport,
    entities: list[SourceEntity] | None = None,
    *,
    include_singleton_orphans: bool = False,
) -> str:
    titles = _entity_titles(entities)
    entity_by_id = _entity_index(entities)
    cluster_by_id = {cluster.id: cluster for cluster in report.clusters}
    cluster_type_counts = Counter(cluster.cluster_type for cluster in report.clusters)
    lines: list[str] = []
    lines.append("# Carapace Triage Report")
    lines.append("")
    lines.append(f"- Processed entities: {report.processed_entities}")
    lines.append(f"- Active: {report.active_entities}")
    lines.append(f"- Suppressed: {report.suppressed_entities}")
    lines.append(f"- Skipped: {report.skipped_entities}")
    lines.append(f"- Similarity edges: {len(report.edges)}")
    lines.append(f"- Clusters: {len(report.clusters)}")
    if cluster_type_counts:
        type_summary = ", ".join(f"{cluster_type}={count}" for cluster_type, count in sorted(cluster_type_counts.items(), key=lambda item: (-item[1], item[0])))
        lines.append(f"- Cluster types: {type_summary}")
    lines.append("")

    decisions = list(report.canonical_decisions)

    def _cluster_size(cluster_id: str) -> int:
        cluster = cluster_by_id.get(cluster_id)
        if cluster is None:
            return 0
        return len(cluster.members)

    decisions.sort(
        key=lambda decision: (
            -_cluster_size(decision.cluster_id),
            decision.cluster_id,
        )
    )

    omitted_singleton_orphans = 0
    for decision in decisions:
        cluster = cluster_by_id.get(decision.cluster_id)
        cluster_type = cluster.cluster_type if cluster else "unknown"
        if not include_singleton_orphans and cluster_type == "singleton_orphan":
            omitted_singleton_orphans += 1
            continue

        lines.append(f"## {decision.cluster_id}")
        lines.append("")
        lines.append(f"- Type: {cluster_type}")
        lines.append(f"- Canonical: {_display_entity(decision.canonical_entity_id, titles)}")
        if decision.canonical_pr_entity_id:
            lines.append(f"- Canonical PR: {_display_entity(decision.canonical_pr_entity_id, titles)}")
        if decision.canonical_issue_entity_id:
            lines.append(f"- Canonical Issue: {_display_entity(decision.canonical_issue_entity_id, titles)}")
        for member in sorted(decision.member_decisions, key=lambda item: item.score, reverse=True):
            state = member.state.value
            extra = f" duplicate_of={member.duplicate_of}" if member.duplicate_of else ""
            signal_suffix = _member_signal_suffix(entity_by_id.get(member.entity_id))
            lines.append(f"- {_display_entity(member.entity_id, titles)}: state={state} score={member.score:.3f}{extra}{signal_suffix}")
        lines.append("")

    if omitted_singleton_orphans:
        lines.append(f"_Omitted {omitted_singleton_orphans} singleton_orphan clusters. Set include_singleton_orphans=true to print them._")

    return "\n".join(lines)


def write_report_bundle(report: EngineReport, output_dir: str | Path, entities: list[SourceEntity] | None = None) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "clusters.json").write_text(report.model_dump_json(indent=2))

    labels_payload = {
        entry.entity_id: {
            "labels": entry.labels,
            "queue_key": entry.queue_key,
            "comment": entry.comment,
            "close": entry.close,
        }
        for entry in report.routing
    }
    (out / "labels_to_apply.json").write_text(json.dumps(labels_payload, indent=2))
    (out / "triage_report.md").write_text(render_markdown_report(report, entities=entities, include_singleton_orphans=False))
    (out / "scan_profile.json").write_text(json.dumps(report.profile, indent=2))
