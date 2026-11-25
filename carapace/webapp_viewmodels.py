"""View-model builders for the Carapace web app."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from carapace.models import EngineReport, SourceEntity


@dataclass(frozen=True)
class ClusterSummary:
    cluster_id: str
    cluster_type: str
    members: list[str]
    canonical: str | None
    canonical_pr: str | None
    canonical_issue: str | None
    priority: float
    issue_attention: float
    duplicate_pressure: float


def _association_trust(association: str | None) -> float:
    key = (association or "").upper()
    mapping = {
        "OWNER": 1.0,
        "MEMBER": 0.9,
        "COLLABORATOR": 0.8,
        "CONTRIBUTOR": 0.6,
        "FIRST_TIME_CONTRIBUTOR": 0.25,
        "FIRST_TIMER": 0.2,
        "NONE": 0.3,
    }
    return mapping.get(key, 0.4)


def _issue_attention(entity: SourceEntity) -> float:
    comments = int(entity.metadata.get("comment_count", 0) or 0)
    reactions = int(entity.metadata.get("reaction_total", 0) or 0)
    return (comments * 0.6) + (reactions * 0.4)


def _compute_cluster_summaries(
    report: EngineReport,
    entities: dict[str, SourceEntity],
) -> list[ClusterSummary]:
    canonical_by_cluster = {decision.cluster_id: decision for decision in report.canonical_decisions}
    summaries: list[ClusterSummary] = []

    for cluster in report.clusters:
        decision = canonical_by_cluster.get(cluster.id)
        canonical = decision.canonical_entity_id if decision else None
        canonical_pr = decision.canonical_pr_entity_id if decision else None
        canonical_issue = decision.canonical_issue_entity_id if decision else None

        issue_attention = 0.0
        for entity_id in cluster.members:
            entity = entities.get(entity_id)
            if entity and entity.kind.value == "issue":
                issue_attention += _issue_attention(entity)

        duplicate_pressure = max(0, len(cluster.members) - 1)
        cluster_size = len(cluster.members)
        priority = (issue_attention * 0.45) + (cluster_size * 0.35) + (duplicate_pressure * 0.20)

        summaries.append(
            ClusterSummary(
                cluster_id=cluster.id,
                cluster_type=cluster.cluster_type,
                members=cluster.members,
                canonical=canonical,
                canonical_pr=canonical_pr,
                canonical_issue=canonical_issue,
                priority=priority,
                issue_attention=issue_attention,
                duplicate_pressure=float(duplicate_pressure),
            )
        )

    summaries.sort(key=lambda item: (item.priority, len(item.members)), reverse=True)
    return summaries


def _node_size_from_priority(priority: float) -> float:
    return max(12.0, min(80.0, 12.0 + (priority * 1.5)))


def _canonical_title(canonical_id: str | None, entities: dict[str, SourceEntity]) -> str | None:
    if not canonical_id:
        return None
    entity = entities.get(canonical_id)
    if entity is None:
        return None
    return entity.title


def _build_ingest_linkage_summaries(repo: str, entities: list[SourceEntity]) -> list[ClusterSummary]:
    if not entities:
        return []

    entity_by_id = {entity.id: entity for entity in entities}
    issue_by_number = {str(entity.number): entity.id for entity in entities if entity.kind.value == "issue" and entity.number is not None}

    parent = {entity.id: entity.id for entity in entities}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        parent[root_right] = root_left

    for entity in entities:
        refs = set(entity.linked_issues) | set(entity.soft_linked_issues)
        for ref in refs:
            target = issue_by_number.get(ref)
            if target is not None and target != entity.id:
                union(entity.id, target)

    components: dict[str, list[str]] = defaultdict(list)
    for entity_id in parent:
        components[find(entity_id)].append(entity_id)

    summaries: list[ClusterSummary] = []
    counter = 0
    for members in components.values():
        if len(members) < 2:
            continue
        counter += 1
        ordered_members = sorted(members)
        pr_members = [entity_by_id[mid] for mid in ordered_members if entity_by_id[mid].kind.value == "pr"]
        issue_members = [entity_by_id[mid] for mid in ordered_members if entity_by_id[mid].kind.value == "issue"]
        if pr_members and issue_members:
            cluster_type = "linked_pair" if len(ordered_members) == 2 else "duplicate_candidate"
        elif pr_members:
            cluster_type = "duplicate_candidate"
        elif issue_members:
            cluster_type = "duplicate_candidate"
        else:
            cluster_type = "mixed_pair"

        canonical_pr = max(pr_members, key=lambda item: item.updated_at).id if pr_members else None
        canonical_issue = max(issue_members, key=lambda item: item.updated_at).id if issue_members else None
        canonical = canonical_pr or canonical_issue

        issue_attention = sum(_issue_attention(item) for item in issue_members)
        duplicate_pressure = max(0, len(ordered_members) - 1)
        priority = (issue_attention * 0.45) + (len(ordered_members) * 0.35) + (duplicate_pressure * 0.20)

        summaries.append(
            ClusterSummary(
                cluster_id=f"ingest-cluster-{counter}",
                cluster_type=cluster_type,
                members=ordered_members,
                canonical=canonical,
                canonical_pr=canonical_pr,
                canonical_issue=canonical_issue,
                priority=priority,
                issue_attention=issue_attention,
                duplicate_pressure=float(duplicate_pressure),
            )
        )

    summaries.sort(key=lambda item: (item.priority, len(item.members)), reverse=True)
    return summaries


def _filter_cluster_summaries(
    summaries: list[ClusterSummary],
    *,
    kind: str,
    min_members: int,
    max_clusters: int,
) -> list[ClusterSummary]:
    filtered = [summary for summary in summaries if len(summary.members) >= min_members and (kind == "all" or summary.cluster_type == kind)]
    return filtered[:max_clusters]


def _build_cluster_map_payload(
    *,
    repo: str,
    report: EngineReport,
    entities: dict[str, SourceEntity],
    summaries: list[ClusterSummary],
    kind: str,
    min_members: int,
    max_clusters: int,
    max_bridges: int,
) -> dict[str, Any]:
    chosen = _filter_cluster_summaries(
        summaries,
        kind=kind,
        min_members=min_members,
        max_clusters=max_clusters,
    )
    if not chosen:
        return {
            "repo": repo,
            "mode": "cluster_map",
            "node_count": 0,
            "edge_count": 0,
            "elements": {"nodes": [], "edges": []},
        }

    chosen_ids = {summary.cluster_id for summary in chosen}
    cluster_lookup = {cluster.id: cluster for cluster in report.clusters}

    nodes: list[dict[str, Any]] = []
    for summary in chosen:
        canonical_title = _canonical_title(summary.canonical, entities)
        nodes.append(
            {
                "data": {
                    "id": summary.cluster_id,
                    "kind": "cluster",
                    "label": summary.cluster_id,
                    "short_label": summary.cluster_id,
                    "canonical_id": summary.canonical,
                    "canonical_title": canonical_title,
                    "cluster_type": summary.cluster_type,
                    "member_count": len(summary.members),
                    "priority": round(summary.priority, 3),
                    "canonical": summary.canonical,
                    "size": max(12.0, min(64.0, 12.0 + (len(summary.members) * 4.0))),
                }
            }
        )

    entity_to_cluster: dict[str, str] = {}
    cluster_authors: dict[str, set[str]] = {}
    cluster_refs: dict[str, set[str]] = {}
    for summary in chosen:
        cid = summary.cluster_id
        members = cluster_lookup[cid].members if cid in cluster_lookup else summary.members
        for entity_id in members:
            entity_to_cluster[entity_id] = cid

        authors: set[str] = set()
        refs: set[str] = set()
        for entity_id in members:
            entity = entities.get(entity_id)
            if entity is None:
                continue
            authors.add(entity.author)
            refs.update(entity.linked_issues)
            refs.update(entity.soft_linked_issues)
            if entity.kind.value == "issue" and entity.number is not None:
                refs.add(str(entity.number))
        cluster_authors[cid] = authors
        cluster_refs[cid] = refs

    bridge_weights: dict[tuple[str, str], float] = defaultdict(float)

    # Content-based cross-cluster signal from pairwise similarity edges.
    for edge in report.edges:
        left = entity_to_cluster.get(edge.entity_a)
        right = entity_to_cluster.get(edge.entity_b)
        if left is None or right is None or left == right:
            continue
        pair = (left, right) if left < right else (right, left)
        bridge_weights[pair] += edge.score

    # Metadata-based bridges (shared authors/refs).
    chosen_list = sorted(chosen_ids)
    for idx, left in enumerate(chosen_list):
        for right in chosen_list[idx + 1 :]:
            shared_authors = cluster_authors[left] & cluster_authors[right]
            shared_refs = cluster_refs[left] & cluster_refs[right]
            if shared_authors:
                bridge_weights[(left, right)] += 0.8 * len(shared_authors)
            if shared_refs:
                bridge_weights[(left, right)] += 1.2 * len(shared_refs)

    sorted_pairs = sorted(bridge_weights.items(), key=lambda item: item[1], reverse=True)[:max_bridges]
    edges: list[dict[str, Any]] = []
    for (left, right), weight in sorted_pairs:
        if weight <= 0:
            continue
        edges.append(
            {
                "data": {
                    "id": f"cluster-bridge:{left}:{right}",
                    "source": left,
                    "target": right,
                    "kind": "cluster_bridge",
                    "weight": round(weight, 3),
                }
            }
        )

    return {
        "repo": repo,
        "mode": "cluster_map",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "elements": {"nodes": nodes, "edges": edges},
    }


def _build_cluster_map_from_summaries(
    *,
    repo: str,
    summaries: list[ClusterSummary],
    entities: dict[str, SourceEntity],
    kind: str,
    min_members: int,
    max_clusters: int,
    max_bridges: int,
    mode: str,
) -> dict[str, Any]:
    chosen = _filter_cluster_summaries(
        summaries,
        kind=kind,
        min_members=min_members,
        max_clusters=max_clusters,
    )
    if not chosen:
        return {
            "repo": repo,
            "mode": mode,
            "node_count": 0,
            "edge_count": 0,
            "elements": {"nodes": [], "edges": []},
        }

    nodes = [
        {
            "data": {
                "id": summary.cluster_id,
                "kind": "cluster",
                "label": summary.cluster_id,
                "short_label": summary.cluster_id,
                "canonical_id": summary.canonical,
                "canonical_title": _canonical_title(summary.canonical, entities),
                "cluster_type": summary.cluster_type,
                "member_count": len(summary.members),
                "priority": round(summary.priority, 3),
                "canonical": summary.canonical,
                "size": max(12.0, min(64.0, 12.0 + (len(summary.members) * 4.0))),
            }
        }
        for summary in chosen
    ]

    chosen_ids = {summary.cluster_id for summary in chosen}
    cluster_authors: dict[str, set[str]] = {}
    cluster_refs: dict[str, set[str]] = {}
    for summary in chosen:
        authors: set[str] = set()
        refs: set[str] = set()
        for entity_id in summary.members:
            entity = entities.get(entity_id)
            if entity is None:
                continue
            authors.add(entity.author)
            refs.update(entity.linked_issues)
            refs.update(entity.soft_linked_issues)
            if entity.kind.value == "issue" and entity.number is not None:
                refs.add(str(entity.number))
        cluster_authors[summary.cluster_id] = authors
        cluster_refs[summary.cluster_id] = refs

    bridge_weights: dict[tuple[str, str], float] = defaultdict(float)
    chosen_list = sorted(chosen_ids)
    for idx, left in enumerate(chosen_list):
        for right in chosen_list[idx + 1 :]:
            shared_authors = cluster_authors[left] & cluster_authors[right]
            shared_refs = cluster_refs[left] & cluster_refs[right]
            if shared_authors:
                bridge_weights[(left, right)] += 0.8 * len(shared_authors)
            if shared_refs:
                bridge_weights[(left, right)] += 1.2 * len(shared_refs)

    sorted_pairs = sorted(bridge_weights.items(), key=lambda item: item[1], reverse=True)[:max_bridges]
    edges = [
        {
            "data": {
                "id": f"cluster-bridge:{left}:{right}",
                "source": left,
                "target": right,
                "kind": "cluster_bridge",
                "weight": round(weight, 3),
            }
        }
        for (left, right), weight in sorted_pairs
        if weight > 0
    ]

    return {
        "repo": repo,
        "mode": mode,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "elements": {"nodes": nodes, "edges": edges},
    }


def _build_cluster_detail_payload(
    *,
    repo: str,
    report: EngineReport,
    entities: dict[str, SourceEntity],
    summaries: list[ClusterSummary],
    author_cache: dict[str, dict[str, float | int | str]],
    cluster_id: str,
    min_edge_score: float,
    include_authors: bool,
) -> dict[str, Any]:
    cluster = next((item for item in report.clusters if item.id == cluster_id), None)
    if cluster is None:
        raise HTTPException(status_code=404, detail=f"Unknown cluster id: {cluster_id}")
    summary = next((item for item in summaries if item.cluster_id == cluster_id), None)
    canonical_id = summary.canonical if summary else None

    member_ids = set(cluster.members)
    shadow_ids = set(cluster.shadow_members)
    include_ids = sorted(member_ids | shadow_ids)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    issue_index = {entity.number: entity.id for entity in entities.values() if entity.kind.value == "issue" and entity.number is not None}
    author_nodes: set[str] = set()

    for entity_id in include_ids:
        entity = entities.get(entity_id)
        if entity is None:
            continue
        trust_cache = author_cache.get(entity.author, {})
        trust_score = max(_association_trust(entity.author_association), float(trust_cache.get("trust_score", 0.0) or 0.0))
        is_shadow = entity_id in shadow_ids
        nodes.append(
            {
                "data": {
                    "id": entity.id,
                    "kind": entity.kind.value,
                    "number": entity.number,
                    "label": entity.title,
                    "canonical": entity.id == canonical_id,
                    "author": entity.author,
                    "state": entity.state,
                    "shadow": is_shadow,
                    "trust_score": round(trust_score, 3),
                    "size": 17 if entity.kind.value == "pr" else 15,
                }
            }
        )

        if include_authors:
            author_id = f"author:{entity.author}"
            if author_id not in author_nodes:
                author_nodes.add(author_id)
                nodes.append(
                    {
                        "data": {
                            "id": author_id,
                            "kind": "author",
                            "label": entity.author,
                            "size": 10 + (trust_score * 8),
                            "trust_score": round(trust_score, 3),
                        }
                    }
                )
            edges.append(
                {
                    "data": {
                        "id": f"detail-authored:{entity.id}:{author_id}",
                        "source": entity.id,
                        "target": author_id,
                        "kind": "authored_by",
                        "weight": round(trust_score, 3),
                    }
                }
            )

        for ref in set(entity.linked_issues):
            if not ref.isdigit():
                continue
            issue_id = issue_index.get(int(ref))
            if issue_id and issue_id in include_ids and issue_id != entity.id:
                edges.append(
                    {
                        "data": {
                            "id": f"detail-ref-hard:{entity.id}:{issue_id}:{ref}",
                            "source": entity.id,
                            "target": issue_id,
                            "kind": "references_hard",
                            "weight": 1.2,
                        }
                    }
                )
        for ref in set(entity.soft_linked_issues):
            if not ref.isdigit():
                continue
            issue_id = issue_index.get(int(ref))
            if issue_id and issue_id in include_ids and issue_id != entity.id:
                edges.append(
                    {
                        "data": {
                            "id": f"detail-ref-soft:{entity.id}:{issue_id}:{ref}",
                            "source": entity.id,
                            "target": issue_id,
                            "kind": "references_soft",
                            "weight": 0.75,
                        }
                    }
                )

    for edge in report.edges:
        if edge.score < min_edge_score:
            continue
        if edge.entity_a not in include_ids or edge.entity_b not in include_ids:
            continue
        edges.append(
            {
                "data": {
                    "id": f"detail-sim:{edge.entity_a}:{edge.entity_b}",
                    "source": edge.entity_a,
                    "target": edge.entity_b,
                    "kind": "similarity",
                    "weight": round(edge.score, 4),
                    "tier": edge.tier.value,
                }
            }
        )

    return {
        "repo": repo,
        "mode": "cluster_detail",
        "cluster": {
            "id": cluster.id,
            "cluster_type": cluster.cluster_type,
            "member_count": len(cluster.members),
            "shadow_count": len(cluster.shadow_members),
            "canonical": summary.canonical if summary else None,
            "canonical_pr": summary.canonical_pr if summary else None,
            "canonical_issue": summary.canonical_issue if summary else None,
            "priority": round(summary.priority, 3) if summary else 0.0,
        },
        "node_count": len(nodes),
        "edge_count": len(edges),
        "elements": {"nodes": nodes, "edges": edges},
    }


def _build_ingest_cluster_detail_payload(
    *,
    repo: str,
    summary: ClusterSummary,
    entities: dict[str, SourceEntity],
    author_cache: dict[str, dict[str, float | int | str]],
    include_authors: bool,
) -> dict[str, Any]:
    include_ids = sorted(summary.members)
    issue_index = {entity.number: entity.id for entity in entities.values() if entity.kind.value == "issue" and entity.number is not None}

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    author_nodes: set[str] = set()
    entity_set = set(include_ids)

    for entity_id in include_ids:
        entity = entities.get(entity_id)
        if entity is None:
            continue
        trust_cache = author_cache.get(entity.author, {})
        trust_score = max(_association_trust(entity.author_association), float(trust_cache.get("trust_score", 0.0) or 0.0))
        nodes.append(
            {
                "data": {
                    "id": entity.id,
                    "kind": entity.kind.value,
                    "number": entity.number,
                    "label": entity.title,
                    "canonical": entity.id == summary.canonical,
                    "author": entity.author,
                    "state": entity.state,
                    "shadow": False,
                    "trust_score": round(trust_score, 3),
                    "size": 17 if entity.kind.value == "pr" else 15,
                }
            }
        )

        if include_authors:
            author_id = f"author:{entity.author}"
            if author_id not in author_nodes:
                author_nodes.add(author_id)
                nodes.append(
                    {
                        "data": {
                            "id": author_id,
                            "kind": "author",
                            "label": entity.author,
                            "size": 10 + (trust_score * 8),
                            "trust_score": round(trust_score, 3),
                        }
                    }
                )
            edges.append(
                {
                    "data": {
                        "id": f"ingest-detail-authored:{entity.id}:{author_id}",
                        "source": entity.id,
                        "target": author_id,
                        "kind": "authored_by",
                        "weight": round(trust_score, 3),
                    }
                }
            )

        for ref in set(entity.linked_issues):
            if not ref.isdigit():
                continue
            issue_id = issue_index.get(int(ref))
            if issue_id and issue_id in entity_set and issue_id != entity.id:
                edges.append(
                    {
                        "data": {
                            "id": f"ingest-detail-ref-hard:{entity.id}:{issue_id}:{ref}",
                            "source": entity.id,
                            "target": issue_id,
                            "kind": "references_hard",
                            "weight": 1.2,
                        }
                    }
                )
        for ref in set(entity.soft_linked_issues):
            if not ref.isdigit():
                continue
            issue_id = issue_index.get(int(ref))
            if issue_id and issue_id in entity_set and issue_id != entity.id:
                edges.append(
                    {
                        "data": {
                            "id": f"ingest-detail-ref-soft:{entity.id}:{issue_id}:{ref}",
                            "source": entity.id,
                            "target": issue_id,
                            "kind": "references_soft",
                            "weight": 0.75,
                        }
                    }
                )

    # Soft cohesion edge for same-author members in ingest fallback detail.
    for idx, left in enumerate(include_ids):
        left_entity = entities.get(left)
        if left_entity is None:
            continue
        for right in include_ids[idx + 1 :]:
            right_entity = entities.get(right)
            if right_entity is None:
                continue
            if left_entity.author == right_entity.author:
                edges.append(
                    {
                        "data": {
                            "id": f"ingest-detail-author-link:{left}:{right}",
                            "source": left,
                            "target": right,
                            "kind": "similarity",
                            "weight": 0.6,
                            "tier": "weak",
                        }
                    }
                )

    return {
        "repo": repo,
        "mode": "cluster_detail_ingest",
        "cluster": {
            "id": summary.cluster_id,
            "cluster_type": summary.cluster_type,
            "member_count": len(summary.members),
            "shadow_count": 0,
            "canonical": summary.canonical,
            "canonical_pr": summary.canonical_pr,
            "canonical_issue": summary.canonical_issue,
            "priority": round(summary.priority, 3),
        },
        "node_count": len(nodes),
        "edge_count": len(edges),
        "elements": {"nodes": nodes, "edges": edges},
    }


def _entity_signal_score(
    entity: SourceEntity,
    *,
    canonical: bool,
    cluster_size: int,
) -> float:
    comments = int(entity.metadata.get("comment_count", 0) or 0)
    reactions = int(entity.metadata.get("reaction_total", 0) or 0)
    approvals = int(entity.approvals or 0)
    review_comments = int(entity.review_comments or 0)

    age_days = max(0.0, (datetime.now(UTC) - entity.updated_at).total_seconds() / 86400.0)
    recency = 1.0 / (1.0 + (age_days / 10.0))
    churn_bonus = min(2.0, math.log1p(max(0, entity.churn)))

    score = 1.0 + (comments * 0.35) + (reactions * 0.20) + (approvals * 0.80) + (review_comments * 0.12) + (cluster_size * 0.42) + recency + churn_bonus
    if canonical:
        score *= 1.35
    return float(score)


def _project_vectors_to_plane(vectors: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
    if not vectors:
        return {}
    dims = max(len(vec) for vec in vectors.values() if vec)
    if dims <= 0:
        return {entity_id: (0.0, 0.0) for entity_id in vectors}

    def _weight(axis: str, idx: int) -> float:
        digest = hashlib.blake2b(f"{axis}:{idx}".encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big") / ((1 << 64) - 1)
        return (value * 2.0) - 1.0

    wx = [_weight("x", idx) for idx in range(dims)]
    wy = [_weight("y", idx) for idx in range(dims)]

    raw_points: dict[str, tuple[float, float]] = {}
    xs: list[float] = []
    ys: list[float] = []
    for entity_id, vec in vectors.items():
        length = min(len(vec), dims)
        x = sum(float(vec[idx]) * wx[idx] for idx in range(length))
        y = sum(float(vec[idx]) * wy[idx] for idx in range(length))
        raw_points[entity_id] = (x, y)
        xs.append(x)
        ys.append(y)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(1e-9, max_x - min_x)
    span_y = max(1e-9, max_y - min_y)

    projected: dict[str, tuple[float, float]] = {}
    for entity_id, (x, y) in raw_points.items():
        nx = ((x - min_x) / span_x) * 2.0 - 1.0
        ny = ((y - min_y) / span_y) * 2.0 - 1.0
        projected[entity_id] = (nx, ny)
    return projected


def _build_embedding_neighbor_edges(
    points: dict[str, tuple[float, float]],
    *,
    k: int,
    max_distance: float,
) -> list[dict[str, Any]]:
    if not points or k <= 0 or max_distance <= 0.0:
        return []

    cell = max_distance
    grid: dict[tuple[int, int], list[str]] = defaultdict(list)
    for entity_id, (x, y) in points.items():
        cell_id = (int(math.floor(x / cell)), int(math.floor(y / cell)))
        grid[cell_id].append(entity_id)

    edge_set: set[tuple[str, str]] = set()
    edges: list[dict[str, Any]] = []
    max_distance_sq = max_distance * max_distance

    for entity_id, (x, y) in points.items():
        gx, gy = int(math.floor(x / cell)), int(math.floor(y / cell))
        candidates: list[tuple[float, str]] = []
        for cx in range(gx - 1, gx + 2):
            for cy in range(gy - 1, gy + 2):
                for other_id in grid.get((cx, cy), []):
                    if other_id == entity_id:
                        continue
                    ox, oy = points[other_id]
                    distance_sq = ((x - ox) ** 2) + ((y - oy) ** 2)
                    if distance_sq <= max_distance_sq:
                        candidates.append((distance_sq, other_id))
        if not candidates:
            continue

        candidates.sort(key=lambda item: item[0])
        for distance_sq, other_id in candidates[:k]:
            left, right = (entity_id, other_id) if entity_id < other_id else (other_id, entity_id)
            key = (left, right)
            if key in edge_set:
                continue
            edge_set.add(key)
            distance = math.sqrt(distance_sq)
            weight = max(0.02, 1.0 - (distance / max_distance))
            edges.append(
                {
                    "data": {
                        "id": f"atlas-nn:{left}:{right}",
                        "source": left,
                        "target": right,
                        "kind": "embedding_neighbor",
                        "weight": round(weight, 5),
                        "distance": round(distance, 5),
                    }
                }
            )
    return edges


def _build_embedding_atlas_payload(
    *,
    repo: str,
    entities: list[SourceEntity],
    vectors: dict[str, list[float]],
    cluster_by_entity: dict[str, str],
    canonical_entities: set[str],
    cluster_sizes: dict[str, int],
    include_edges: bool,
    edge_k: int,
    edge_max_distance: float,
    min_signal: float,
    max_nodes: int,
) -> dict[str, Any]:
    points = _project_vectors_to_plane(vectors)
    nodes_raw: list[tuple[float, dict[str, Any]]] = []
    for entity in entities:
        if entity.id not in points:
            continue
        cluster_id = cluster_by_entity.get(entity.id)
        cluster_size = cluster_sizes.get(cluster_id, 1) if cluster_id else 1
        canonical = entity.id in canonical_entities
        signal = _entity_signal_score(entity, canonical=canonical, cluster_size=cluster_size)
        if signal < min_signal:
            continue
        x, y = points[entity.id]
        size = 2.8 + min(12.0, math.log1p(signal) * (2.4 if canonical else 2.0))
        nodes_raw.append(
            (
                signal,
                {
                    "data": {
                        "id": entity.id,
                        "kind": entity.kind.value,
                        "number": entity.number,
                        "label": entity.title,
                        "short_label": entity.id,
                        "cluster_id": cluster_id,
                        "canonical": canonical,
                        "signal": round(signal, 4),
                        "x": round(x, 6),
                        "y": round(y, 6),
                        "size": round(size, 4),
                        "author": entity.author,
                        "state": entity.state,
                    }
                },
            )
        )

    if max_nodes > 0 and len(nodes_raw) > max_nodes:
        canonical_nodes = [item for item in nodes_raw if bool(item[1]["data"]["canonical"])]
        noncanonical = [item for item in nodes_raw if not bool(item[1]["data"]["canonical"])]
        canonical_nodes.sort(key=lambda item: item[0], reverse=True)
        noncanonical.sort(key=lambda item: item[0], reverse=True)
        keep = canonical_nodes[:max_nodes]
        remaining = max(0, max_nodes - len(keep))
        keep.extend(noncanonical[:remaining])
        nodes_raw = keep

    nodes = [item[1] for item in sorted(nodes_raw, key=lambda row: row[0], reverse=True)]
    node_ids = {node["data"]["id"] for node in nodes}
    points_selected = {entity_id: point for entity_id, point in points.items() if entity_id in node_ids}
    edges = _build_embedding_neighbor_edges(points_selected, k=edge_k, max_distance=edge_max_distance) if include_edges else []

    return {
        "repo": repo,
        "mode": "embedding_atlas",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "elements": {"nodes": nodes, "edges": edges},
    }


def _build_graph_payload(
    *,
    repo: str,
    report: EngineReport,
    entities: dict[str, SourceEntity],
    summaries: dict[str, ClusterSummary],
    author_cache: dict[str, dict[str, float | int | str]],
    cluster_id: str | None,
    min_edge_score: float,
    max_clusters: int,
    include_authors: bool,
    max_similarity_edges: int,
) -> dict[str, Any]:
    include_members: set[str] = set()
    include_clusters: set[str] = set()
    cluster_lookup = {cluster.id: cluster for cluster in report.clusters}
    summary_values = sorted(summaries.values(), key=lambda item: (item.priority, len(item.members)), reverse=True)
    default_cluster_ids = [item.cluster_id for item in summary_values if len(item.members) > 1][:max_clusters]

    if cluster_id:
        target = cluster_lookup.get(cluster_id)
        if target is None:
            raise HTTPException(status_code=404, detail=f"Unknown cluster id: {cluster_id}")
        include_clusters.add(target.id)
        include_members.update(target.members)
        include_members.update(target.shadow_members)
    else:
        for cid in default_cluster_ids:
            include_clusters.add(cid)
            cluster = cluster_lookup[cid]
            include_members.update(cluster.members)
            include_members.update(cluster.shadow_members)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for cid in sorted(include_clusters):
        summary = summaries.get(cid)
        if summary is None:
            continue
        nodes.append(
            {
                "data": {
                    "id": cid,
                    "kind": "cluster",
                    "label": cid,
                    "cluster_type": summary.cluster_type,
                    "priority": round(summary.priority, 2),
                    "size": _node_size_from_priority(summary.priority),
                }
            }
        )

    author_nodes: set[str] = set()
    issue_index = {entity.number: entity.id for entity in entities.values() if entity.kind.value == "issue" and entity.number is not None}

    for entity_id in sorted(include_members):
        entity = entities.get(entity_id)
        if entity is None:
            continue

        trust_cache = author_cache.get(entity.author, {})
        merged_pr_count = int(trust_cache.get("merged_pr_count", 0) or 0)
        cached_trust = float(trust_cache.get("trust_score", 0.0) or 0.0)
        trust_score = max(_association_trust(entity.author_association), cached_trust)

        nodes.append(
            {
                "data": {
                    "id": entity.id,
                    "kind": entity.kind.value,
                    "label": entity.title,
                    "author": entity.author,
                    "state": entity.state,
                    "trust_score": round(trust_score, 3),
                    "merged_pr_count": merged_pr_count,
                    "comment_count": int(entity.metadata.get("comment_count", 0) or 0),
                    "reaction_total": int(entity.metadata.get("reaction_total", 0) or 0),
                    "size": 20 if entity.kind.value == "pr" else 18,
                }
            }
        )

        if include_authors:
            author_id = f"author:{entity.author}"
            if author_id not in author_nodes:
                author_nodes.add(author_id)
                nodes.append(
                    {
                        "data": {
                            "id": author_id,
                            "kind": "author",
                            "label": entity.author,
                            "trust_score": round(trust_score, 3),
                            "merged_pr_count": merged_pr_count,
                            "size": 16 + (trust_score * 10.0),
                        }
                    }
                )

            edges.append(
                {
                    "data": {
                        "id": f"authored_by:{entity.id}:{author_id}",
                        "source": entity.id,
                        "target": author_id,
                        "kind": "authored_by",
                        "weight": round(trust_score, 3),
                    }
                }
            )

        linked_issue_ids = set(entity.linked_issues) | set(entity.soft_linked_issues)
        for raw in linked_issue_ids:
            if not raw.isdigit():
                continue
            issue_num = int(raw)
            issue_id = issue_index.get(issue_num)
            if issue_id and issue_id in include_members and issue_id != entity.id:
                edges.append(
                    {
                        "data": {
                            "id": f"links:{entity.id}:{issue_id}:{raw}",
                            "source": entity.id,
                            "target": issue_id,
                            "kind": "references",
                            "weight": 1.0,
                        }
                    }
                )

    cluster_members = {cluster.id: set(cluster.members) for cluster in report.clusters if cluster.id in include_clusters}
    for cid, members in cluster_members.items():
        for entity_id in members:
            if entity_id not in include_members:
                continue
            edges.append(
                {
                    "data": {
                        "id": f"cluster:{cid}:{entity_id}",
                        "source": cid,
                        "target": entity_id,
                        "kind": "in_cluster",
                        "weight": 0.8,
                    }
                }
            )

    # Build lightweight lineage bridges between clusters to avoid isolated islands.
    cluster_authors: dict[str, set[str]] = {}
    cluster_issue_refs: dict[str, set[str]] = {}
    for cid in include_clusters:
        authors: set[str] = set()
        issue_refs: set[str] = set()
        for entity_id in cluster_members.get(cid, set()):
            entity = entities.get(entity_id)
            if entity is None:
                continue
            authors.add(entity.author)
            issue_refs.update(entity.linked_issues)
            issue_refs.update(entity.soft_linked_issues)
            if entity.kind.value == "issue" and entity.number is not None:
                issue_refs.add(str(entity.number))
        cluster_authors[cid] = authors
        cluster_issue_refs[cid] = issue_refs

    bridge_count = 0
    cluster_ids = sorted(include_clusters)
    for idx, left in enumerate(cluster_ids):
        for right in cluster_ids[idx + 1 :]:
            shared_authors = cluster_authors[left] & cluster_authors[right]
            shared_refs = cluster_issue_refs[left] & cluster_issue_refs[right]
            if not shared_authors and not shared_refs:
                continue
            bridge_weight = float(len(shared_authors) * 0.6 + len(shared_refs) * 1.0)
            edges.append(
                {
                    "data": {
                        "id": f"bridge:{left}:{right}",
                        "source": left,
                        "target": right,
                        "kind": "cluster_bridge",
                        "weight": round(bridge_weight, 3),
                    }
                }
            )
            bridge_count += 1
            if bridge_count >= 2000:
                break
        if bridge_count >= 2000:
            break

    similarity_edges_added = 0
    for edge in report.edges:
        if similarity_edges_added >= max_similarity_edges:
            break
        if edge.score < min_edge_score:
            continue
        if edge.entity_a not in include_members or edge.entity_b not in include_members:
            continue
        edges.append(
            {
                "data": {
                    "id": f"sim:{edge.entity_a}:{edge.entity_b}",
                    "source": edge.entity_a,
                    "target": edge.entity_b,
                    "kind": "similarity",
                    "tier": edge.tier.value,
                    "weight": round(edge.score, 4),
                }
            }
        )
        similarity_edges_added += 1

    return {
        "repo": repo,
        "mode": "run",
        "cluster_id": cluster_id,
        "max_clusters": max_clusters,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "elements": {"nodes": nodes, "edges": edges},
    }


def _build_ingest_fallback_graph_payload(
    *,
    repo: str,
    entities: list[SourceEntity],
    author_cache: dict[str, dict[str, float | int | str]],
    max_entities: int,
    include_authors: bool,
) -> dict[str, Any]:
    selected = sorted(entities, key=lambda item: item.updated_at, reverse=True)[:max_entities]
    if not selected:
        return {
            "repo": repo,
            "mode": "ingest_fallback",
            "cluster_id": None,
            "node_count": 0,
            "edge_count": 0,
            "elements": {"nodes": [], "edges": []},
        }

    issue_index = {entity.number: entity.id for entity in selected if entity.kind.value == "issue" and entity.number is not None}
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    author_nodes: set[str] = set()

    for entity in selected:
        trust_cache = author_cache.get(entity.author, {})
        merged_pr_count = int(trust_cache.get("merged_pr_count", 0) or 0)
        cached_trust = float(trust_cache.get("trust_score", 0.0) or 0.0)
        trust_score = max(_association_trust(entity.author_association), cached_trust)

        nodes.append(
            {
                "data": {
                    "id": entity.id,
                    "kind": entity.kind.value,
                    "label": entity.title,
                    "author": entity.author,
                    "state": entity.state,
                    "trust_score": round(trust_score, 3),
                    "merged_pr_count": merged_pr_count,
                    "comment_count": int(entity.metadata.get("comment_count", 0) or 0),
                    "reaction_total": int(entity.metadata.get("reaction_total", 0) or 0),
                    "size": 20 if entity.kind.value == "pr" else 18,
                }
            }
        )

        if include_authors:
            author_id = f"author:{entity.author}"
            if author_id not in author_nodes:
                author_nodes.add(author_id)
                nodes.append(
                    {
                        "data": {
                            "id": author_id,
                            "kind": "author",
                            "label": entity.author,
                            "trust_score": round(trust_score, 3),
                            "merged_pr_count": merged_pr_count,
                            "size": 16 + (trust_score * 10.0),
                        }
                    }
                )

            edges.append(
                {
                    "data": {
                        "id": f"authored_by:{entity.id}:{author_id}",
                        "source": entity.id,
                        "target": author_id,
                        "kind": "authored_by",
                        "weight": round(trust_score, 3),
                    }
                }
            )

        linked_issue_ids = set(entity.linked_issues) | set(entity.soft_linked_issues)
        for raw in linked_issue_ids:
            if not raw.isdigit():
                continue
            issue_num = int(raw)
            issue_id = issue_index.get(issue_num)
            if issue_id and issue_id != entity.id:
                edges.append(
                    {
                        "data": {
                            "id": f"links:{entity.id}:{issue_id}:{raw}",
                            "source": entity.id,
                            "target": issue_id,
                            "kind": "references",
                            "weight": 1.0,
                        }
                    }
                )

    return {
        "repo": repo,
        "mode": "ingest_fallback",
        "cluster_id": None,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "elements": {"nodes": nodes, "edges": edges},
    }

