"""Canonical selection within similarity clusters."""

from __future__ import annotations

import math
from carapace.config import CanonicalConfig
from carapace.models import (
    CIStatus,
    CanonicalDecision,
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
        key = tuple(sorted((edge.entity_a, edge.entity_b)))
        table[key] = edge
    return table


def _similarity(edge_table: dict[tuple[str, str], SimilarityEdge], a: str, b: str) -> float:
    if a == b:
        return 1.0
    edge = edge_table.get(tuple(sorted((a, b))))
    if edge is None:
        return 0.0
    return edge.score


def _lineage_overlap(edge_table: dict[tuple[str, str], SimilarityEdge], a: str, b: str) -> float:
    edge = edge_table.get(tuple(sorted((a, b))))
    if edge is None:
        return 0.0
    return edge.breakdown.lineage


def _ci_score(ci: CIStatus) -> float:
    if ci == CIStatus.PASS:
        return 1.0
    if ci == CIStatus.FAIL:
        return -1.0
    return 0.0


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
            priority = low_pass.get(member).priority_weight if member in low_pass else 1.0

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
        canonical = ranked[0][0]

        member_decisions: list[MemberDecision] = []
        for member, score in ranked:
            if member == canonical:
                state = DecisionState.CANONICAL
                reason = "Top canonical score in cluster"
                duplicate_of = None
            else:
                sim_to_canonical = _similarity(edge_table, member, canonical)
                lineage_to_canonical = _lineage_overlap(edge_table, member, canonical)
                if sim_to_canonical >= cfg.duplicate_threshold or lineage_to_canonical >= 0.5:
                    state = DecisionState.DUPLICATE
                    reason = (
                        f"Similarity {sim_to_canonical:.2f} / lineage {lineage_to_canonical:.2f} "
                        "meets duplicate criteria"
                    )
                    duplicate_of = canonical
                else:
                    state = DecisionState.RELATED
                    reason = f"Related cluster member with similarity {sim_to_canonical:.2f}"
                    duplicate_of = None

            member_decisions.append(
                MemberDecision(
                    entity_id=member,
                    state=state,
                    score=score,
                    reason=reason,
                    duplicate_of=duplicate_of,
                )
            )

        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < cfg.tie_margin:
            for idx, decision in enumerate(member_decisions):
                if decision.entity_id == ranked[1][0] and decision.state != DecisionState.CANONICAL:
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
                member_decisions=member_decisions,
            )
        )

    return decisions
