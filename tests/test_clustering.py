from carapace.clustering import build_clusters
from carapace.models import EdgeTier, SimilarityBreakdown, SimilarityEdge


def _edge(a: str, b: str, tier: EdgeTier, score: float = 0.8) -> SimilarityEdge:
    return SimilarityEdge(
        entity_a=a,
        entity_b=b,
        score=score,
        tier=tier,
        breakdown=SimilarityBreakdown(lineage=0.0, structure=0.0, semantic=0.0, size_penalty=0.0, total=score),
    )


def test_strong_edges_union_directly() -> None:
    clusters = build_clusters(
        ["a", "b", "c"],
        [_edge("a", "b", EdgeTier.STRONG)],
    )
    members = [set(c.members) for c in clusters]
    assert {"a", "b"} in members
    assert {"c"} in members


def test_weak_edge_requires_shared_strong_neighbor() -> None:
    clusters = build_clusters(
        ["a", "b", "c"],
        [
            _edge("a", "b", EdgeTier.STRONG),
            _edge("b", "c", EdgeTier.STRONG),
            _edge("a", "c", EdgeTier.WEAK),
        ],
    )
    assert len(clusters) == 1
    assert set(clusters[0].members) == {"a", "b", "c"}


def test_weak_without_shared_strong_neighbor_does_not_merge() -> None:
    clusters = build_clusters(
        ["a", "b", "c"],
        [
            _edge("a", "b", EdgeTier.WEAK),
        ],
    )
    members = [set(c.members) for c in clusters]
    assert {"a", "b"} in members
    assert {"c"} in members


def test_weak_chain_without_strong_neighbors_does_not_bridge_all_nodes() -> None:
    clusters = build_clusters(
        ["a", "b", "c"],
        [
            _edge("a", "b", EdgeTier.WEAK),
            _edge("b", "c", EdgeTier.WEAK),
        ],
    )
    # Avoid transitive weak-chain collapse; only isolated weak pairs can merge.
    members = [set(c.members) for c in clusters]
    assert {"a"} in members
    assert {"b"} in members
    assert {"c"} in members
