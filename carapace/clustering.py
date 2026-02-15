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


def build_clusters(entity_ids: list[str], edges: list[SimilarityEdge]) -> list[Cluster]:
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

    clusters: list[Cluster] = []
    for idx, members in enumerate(sorted(grouped.values(), key=lambda items: (-len(items), items))):
        clusters.append(Cluster(id=f"cluster-{idx + 1}", members=sorted(members)))

    return clusters
