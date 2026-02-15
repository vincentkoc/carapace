"""Incremental clustering over similarity edges."""

from __future__ import annotations

from collections import defaultdict

from carapace.models import Cluster, EdgeTier, SimilarityEdge


class UnionFind:
    def __init__(self, members: list[str]) -> None:
        self.parent = {m: m for m in members}
        self.rank = {m: 0 for m in members}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _adjacency(edges: list[SimilarityEdge]) -> dict[str, dict[str, SimilarityEdge]]:
    graph: dict[str, dict[str, SimilarityEdge]] = defaultdict(dict)
    for edge in edges:
        graph[edge.entity_a][edge.entity_b] = edge
        graph[edge.entity_b][edge.entity_a] = edge
    return graph


def _prune_weak_tails(
    members: list[str],
    adjacency: dict[str, dict[str, SimilarityEdge]],
    min_tail_score: float | None,
) -> list[list[str]]:
    if min_tail_score is None or len(members) < 3:
        return [sorted(members)]

    remaining = set(members)
    changed = True
    while changed and len(remaining) > 1:
        changed = False
        for node in list(remaining):
            neighbors = [(other, edge) for other, edge in adjacency.get(node, {}).items() if other in remaining]
            if len(neighbors) > 1:
                continue
            has_hard_link = any(edge.breakdown.hard_link_overlap >= 0.5 for _, edge in neighbors)
            has_lineage = any(edge.breakdown.lineage >= 0.5 for _, edge in neighbors)
            if has_hard_link or has_lineage:
                continue
            max_score = max((edge.score for _, edge in neighbors), default=0.0)
            if max_score < min_tail_score:
                remaining.remove(node)
                changed = True

    if remaining == set(members):
        return [sorted(members)]

    groups: list[list[str]] = []
    seen: set[str] = set()
    for node in sorted(remaining):
        if node in seen:
            continue
        stack = [node]
        component: list[str] = []
        seen.add(node)
        while stack:
            current = stack.pop()
            component.append(current)
            for other in adjacency.get(current, {}):
                if other in remaining and other not in seen:
                    seen.add(other)
                    stack.append(other)
        groups.append(sorted(component))

    for node in sorted(set(members) - remaining):
        groups.append([node])
    return groups


def build_clusters(
    entity_ids: list[str],
    edges: list[SimilarityEdge],
    *,
    tail_prune_score: float | None = None,
) -> list[Cluster]:
    uf = UnionFind(entity_ids)

    strong_neighbors: dict[str, set[str]] = defaultdict(set)
    weak_neighbors: dict[str, set[str]] = defaultdict(set)
    strong_edges = [edge for edge in edges if edge.tier == EdgeTier.STRONG]
    weak_edges = [edge for edge in edges if edge.tier == EdgeTier.WEAK]

    for edge in strong_edges:
        uf.union(edge.entity_a, edge.entity_b)
        strong_neighbors[edge.entity_a].add(edge.entity_b)
        strong_neighbors[edge.entity_b].add(edge.entity_a)

    for edge in weak_edges:
        weak_neighbors[edge.entity_a].add(edge.entity_b)
        weak_neighbors[edge.entity_b].add(edge.entity_a)

    for edge in weak_edges:
        common = strong_neighbors[edge.entity_a] & strong_neighbors[edge.entity_b]
        if common:
            uf.union(edge.entity_a, edge.entity_b)

    # Allow isolated weak pairs to merge when both sides have exactly one weak neighbor
    # and no strong-neighbor participation (avoids long weak-bridge chains).
    for edge in weak_edges:
        a = edge.entity_a
        b = edge.entity_b
        if uf.find(a) == uf.find(b):
            continue
        if strong_neighbors[a] or strong_neighbors[b]:
            continue
        if len(weak_neighbors[a]) == 1 and len(weak_neighbors[b]) == 1:
            uf.union(a, b)

    grouped: dict[str, list[str]] = defaultdict(list)
    for entity_id in entity_ids:
        grouped[uf.find(entity_id)].append(entity_id)

    edge_lookup = _adjacency(edges)
    expanded_groups: list[list[str]] = []
    for members in grouped.values():
        expanded_groups.extend(_prune_weak_tails(members, edge_lookup, tail_prune_score))

    clusters: list[Cluster] = []
    for idx, members in enumerate(sorted(expanded_groups, key=lambda items: (-len(items), items))):
        clusters.append(Cluster(id=f"cluster-{idx + 1}", members=members))

    return clusters
