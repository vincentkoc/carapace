"""Report generation utilities for offline workflows."""

from __future__ import annotations

import json
from pathlib import Path

from carapace.models import EngineReport, SourceEntity


def _entity_titles(entities: list[SourceEntity] | None) -> dict[str, str]:
    if not entities:
        return {}
    return {entity.id: entity.title for entity in entities}


def _display_entity(entity_id: str | None, titles: dict[str, str]) -> str:
    if not entity_id:
        return "none"
    title = titles.get(entity_id, "").strip()
    if not title:
        return entity_id
    return f'{entity_id} — "{title}"'


def render_markdown_report(report: EngineReport, entities: list[SourceEntity] | None = None) -> str:
    titles = _entity_titles(entities)
    lines: list[str] = []
    lines.append("# Carapace Triage Report")
    lines.append("")
    lines.append(f"- Processed entities: {report.processed_entities}")
    lines.append(f"- Active: {report.active_entities}")
    lines.append(f"- Suppressed: {report.suppressed_entities}")
    lines.append(f"- Skipped: {report.skipped_entities}")
    lines.append(f"- Similarity edges: {len(report.edges)}")
    lines.append(f"- Clusters: {len(report.clusters)}")
    lines.append("")

    for decision in report.canonical_decisions:
        lines.append(f"## {decision.cluster_id}")
        lines.append("")
        lines.append(f"- Canonical: {_display_entity(decision.canonical_entity_id, titles)}")
        for member in sorted(decision.member_decisions, key=lambda item: item.score, reverse=True):
            state = member.state.value
            extra = f" duplicate_of={member.duplicate_of}" if member.duplicate_of else ""
            lines.append(f"- {_display_entity(member.entity_id, titles)}: state={state} score={member.score:.3f}{extra}")
        lines.append("")

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
    (out / "triage_report.md").write_text(render_markdown_report(report, entities=entities))
    (out / "scan_profile.json").write_text(json.dumps(report.profile, indent=2))
