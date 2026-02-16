"""Canonical selection within similarity clusters."""

from __future__ import annotations

import math

from carapace.config import CanonicalConfig
from carapace.models import (
    CanonicalDecision,
    CIStatus,
    Cluster,
    DecisionState,
    Fingerprint,
    LowPassDecision,
    MemberDecision,
    SimilarityEdge,
)


def _edge_lookup(edges: list[SimilarityEdge]) -> dict[tuple[str, str], SimilarityEdge]:
    table: dict[tuple[str, str], SimilarityEdge] = {}
    for edge in edges:
        a, b = sorted((edge.entity_a, edge.entity_b))
        key = (a, b)
        table[key] = edge
    return table


def _similarity(edge_table: dict[tuple[str, str], SimilarityEdge], a: str, b: str) -> float:
    if a == b:
        return 1.0
    x, y = sorted((a, b))
    edge = edge_table.get((x, y))
    if edge is None:
        return 0.0
    return edge.score


def _lineage_overlap(edge_table: dict[tuple[str, str], SimilarityEdge], a: str, b: str) -> float:
    x, y = sorted((a, b))
    edge = edge_table.get((x, y))
    if edge is None:
        return 0.0
    return edge.breakdown.lineage


def _edge_metrics(edge_table: dict[tuple[str, str], SimilarityEdge], a: str, b: str) -> tuple[float, float, float, float, float]:
    x, y = sorted((a, b))
    edge = edge_table.get((x, y))
    if edge is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        edge.breakdown.file_overlap,
        edge.breakdown.hunk_overlap,
        edge.breakdown.hard_link_overlap,
        edge.breakdown.title_salient_overlap,
        edge.breakdown.semantic_text,
    )


def _title_mismatch_veto(
    *,
    lineage_overlap: float,
    title_salient_overlap: float,
    semantic_text: float,
    hard_link_overlap: float,
    cfg: CanonicalConfig,
) -> bool:
    return (
        lineage_overlap < 0.5
        and title_salient_overlap <= cfg.duplicate_title_mismatch_overlap_max
        and semantic_text <= cfg.duplicate_title_mismatch_semantic_text_max
        and hard_link_overlap < cfg.duplicate_title_mismatch_hard_link_override_min
    )


def _ci_score(ci: CIStatus) -> float:
    if ci == CIStatus.PASS:
        return 1.0
    if ci == CIStatus.FAIL:
        return -1.0
    return 0.0


def _entity_kind(entity_id: str) -> str:
    if ":" not in entity_id:
        return "default"
    return entity_id.split(":", maxsplit=1)[0]


def rank_canonicals(
    clusters: list[Cluster],
    fingerprints: dict[str, Fingerprint],
    edges: list[SimilarityEdge],
    low_pass: dict[str, LowPassDecision],
    cfg: CanonicalConfig,
) -> list[CanonicalDecision]:
    edge_table = _edge_lookup(edges)
    decisions: list[CanonicalDecision] = []

    for cluster in clusters:
        members = cluster.members
        if not members:
            decisions.append(CanonicalDecision(cluster_id=cluster.id, canonical_entity_id=None, member_decisions=[]))
            continue

        cluster_files: set[str] = set()
        for member in members:
            cluster_files.update(fingerprints[member].changed_files)

        scores: dict[str, float] = {}
        for member in members:
            fp = fingerprints[member]
            sims = [_similarity(edge_table, member, other) for other in members if other != member]
            centrality = sum(sims) / max(1, len(sims))

            coverage = 0.0
            if cluster_files:
                coverage = len(set(fp.changed_files) & cluster_files) / len(cluster_files)

            approvals_norm = min(1.0, fp.approvals / 3.0)
            churn = fp.additions + fp.deletions
            size_penalty = min(1.0, math.log1p(churn) / 8.0)
            lp = low_pass.get(member)
            priority = lp.priority_weight if lp else 1.0

            score = (
                cfg.weight_coverage * coverage
                + cfg.weight_centrality * centrality
                + cfg.weight_ci * _ci_score(fp.ci_status)
                + cfg.weight_reviewer * fp.reviewer_score
                + cfg.weight_approvals * approvals_norm
                + cfg.weight_priority * priority
                - cfg.weight_size_penalty * size_penalty
            )
            scores[member] = score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ranked_by_kind: dict[str, list[tuple[str, float]]] = {}
        for member, member_score in ranked:
            ranked_by_kind.setdefault(_entity_kind(member), []).append((member, member_score))

        canonical_by_kind = {kind: entries[0][0] for kind, entries in ranked_by_kind.items() if entries}
        canonical_pr = canonical_by_kind.get("pr")
        canonical_issue = canonical_by_kind.get("issue")
        canonical = canonical_pr or canonical_issue or ranked[0][0]

        member_decisions: list[MemberDecision] = []
        member_index: dict[str, int] = {}
        for member, canonical_score in ranked:
            member_kind = _entity_kind(member)
            canonical_for_kind = canonical_by_kind.get(member_kind, canonical)

            if member == canonical_for_kind:
                state = DecisionState.CANONICAL
                if member_kind in {"pr", "issue"} and len(ranked_by_kind) > 1:
                    reason = f"Top canonical score in {member_kind} lane"
                else:
                    reason = "Top canonical score in cluster"
                duplicate_of = None
                member_score = canonical_score
            else:
                sim_to_canonical = _similarity(edge_table, member, canonical_for_kind)
                lineage_to_canonical = _lineage_overlap(edge_table, member, canonical_for_kind)
                file_overlap, hunk_overlap, hard_link_overlap, title_salient_overlap, semantic_text = _edge_metrics(edge_table, member, canonical_for_kind)
                canonical_kind = _entity_kind(canonical_for_kind)
                same_kind = member_kind == canonical_kind or member_kind == "default" or canonical_kind == "default"
                meets_similarity = sim_to_canonical >= cfg.duplicate_threshold or lineage_to_canonical >= 0.5
                lineage_supported = lineage_to_canonical >= 0.5 and (
                    hard_link_overlap >= cfg.duplicate_hard_link_overlap_min
                    or hunk_overlap >= cfg.duplicate_hunk_overlap_min
                    or (title_salient_overlap >= cfg.duplicate_title_salient_overlap_min and semantic_text >= cfg.duplicate_semantic_text_min)
                )
                has_duplicate_evidence = (
                    lineage_supported
                    or (
                        hard_link_overlap >= cfg.duplicate_hard_link_overlap_min
                        and (file_overlap >= cfg.duplicate_hard_link_file_overlap_min or hunk_overlap >= cfg.duplicate_hard_link_hunk_overlap_min or title_salient_overlap >= cfg.duplicate_hard_link_title_overlap_min)
                    )
                    or hunk_overlap >= cfg.duplicate_hunk_overlap_min
                    or (file_overlap >= cfg.duplicate_file_overlap_min and title_salient_overlap >= cfg.duplicate_file_title_overlap_min and semantic_text >= cfg.duplicate_semantic_text_min)
                    or (title_salient_overlap >= cfg.duplicate_title_salient_overlap_min and semantic_text >= cfg.duplicate_semantic_text_min and hunk_overlap >= cfg.duplicate_hunk_overlap_min)
                )
                title_mismatch_veto = _title_mismatch_veto(
                    lineage_overlap=lineage_to_canonical,
                    title_salient_overlap=title_salient_overlap,
                    semantic_text=semantic_text,
                    hard_link_overlap=hard_link_overlap,
                    cfg=cfg,
                )
                if title_mismatch_veto:
                    has_duplicate_evidence = False
                if same_kind and meets_similarity and has_duplicate_evidence:
                    state = DecisionState.DUPLICATE
                    reason = f"Similarity {sim_to_canonical:.2f} / lineage {lineage_to_canonical:.2f} meets duplicate criteria"
                    duplicate_of = canonical_for_kind
                else:
                    state = DecisionState.RELATED
                    reason = f"Related cluster member with similarity {sim_to_canonical:.2f}"
                    duplicate_of = None
                member_score = sim_to_canonical

            member_decisions.append(
                MemberDecision(
                    entity_id=member,
                    state=state,
                    score=member_score,
                    reason=reason,
                    duplicate_of=duplicate_of,
                )
            )
            member_index[member] = len(member_decisions) - 1

        for member_kind, kind_ranked in ranked_by_kind.items():
            if len(kind_ranked) <= 1:
                continue
            canonical_for_kind = canonical_by_kind[member_kind]
            if kind_ranked[0][1] - kind_ranked[1][1] >= cfg.tie_margin:
                continue

            runner_up_id = kind_ranked[1][0]
            idx = member_index.get(runner_up_id)
            if idx is None:
                continue
            decision = member_decisions[idx]
            # Only escalate to tie-break for non-duplicate runner-up.
            if decision.state != DecisionState.RELATED:
                continue
            file_overlap, hunk_overlap, hard_link_overlap, title_salient_overlap, semantic_text = _edge_metrics(edge_table, decision.entity_id, canonical_for_kind)
            lineage_overlap = _lineage_overlap(edge_table, decision.entity_id, canonical_for_kind)
            if _title_mismatch_veto(
                lineage_overlap=lineage_overlap,
                title_salient_overlap=title_salient_overlap,
                semantic_text=semantic_text,
                hard_link_overlap=hard_link_overlap,
                cfg=cfg,
            ):
                continue
            tie_break_evidence = hard_link_overlap >= cfg.tie_break_hard_link_min or hunk_overlap >= cfg.tie_break_hunk_overlap_min or (file_overlap >= cfg.tie_break_file_overlap_min and semantic_text >= cfg.tie_break_semantic_text_min)
            if decision.score < cfg.tie_break_min_similarity or not tie_break_evidence:
                continue
            member_decisions[idx] = MemberDecision(
                entity_id=decision.entity_id,
                state=DecisionState.TIE_BREAK,
                score=decision.score,
                reason="Near tie with canonical candidate",
                duplicate_of=None,
            )

        decisions.append(
            CanonicalDecision(
                cluster_id=cluster.id,
                canonical_entity_id=canonical,
                canonical_pr_entity_id=canonical_pr,
                canonical_issue_entity_id=canonical_issue,
                member_decisions=member_decisions,
            )
        )

    return decisions
