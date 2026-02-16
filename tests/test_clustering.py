from carapace.clustering import build_clusters
from carapace.models import EdgeTier, SimilarityBreakdown, SimilarityEdge


def _edge(a: str, b: str, tier: EdgeTier, score: float = 0.8, *, hard_link: float = 0.0, lineage: float = 0.0) -> SimilarityEdge:
    return SimilarityEdge(
        entity_a=a,
        entity_b=b,
        score=score,
        tier=tier,
        breakdown=SimilarityBreakdown(
            lineage=lineage,
            structure=0.0,
            semantic=0.0,
            hard_link_overlap=hard_link,
            size_penalty=0.0,
            total=score,
        ),
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


def test_tail_pruning_splits_low_score_leaf_nodes() -> None:
    clusters = build_clusters(
        ["a", "b", "c"],
        [
            _edge("a", "b", EdgeTier.STRONG, score=0.20),
            _edge("b", "c", EdgeTier.STRONG, score=0.10),
        ],
        tail_prune_score=0.15,
    )
    members = [set(c.members) for c in clusters]
    assert {"a", "b"} in members
    assert {"c"} in members


def test_tail_pruning_keeps_leaf_with_hard_link_signal() -> None:
    clusters = build_clusters(
        ["a", "b", "c"],
        [
            _edge("a", "b", EdgeTier.STRONG, score=0.20),
            _edge("b", "c", EdgeTier.STRONG, score=0.10, hard_link=0.5),
        ],
        tail_prune_score=0.15,
    )
    assert len(clusters) == 1
    assert set(clusters[0].members) == {"a", "b", "c"}


def test_weak_edge_with_hard_link_merges_directly() -> None:
    clusters = build_clusters(
        ["a", "b"],
        [
            _edge("a", "b", EdgeTier.WEAK, score=0.1, hard_link=1.0),
        ],
    )
    assert len(clusters) == 1
    assert set(clusters[0].members) == {"a", "b"}


def test_cluster_type_singleton_and_duplicate_candidate() -> None:
    clusters = build_clusters(
        ["pr:1", "pr:2", "issue:1"],
        [_edge("pr:1", "pr:2", EdgeTier.STRONG, score=0.9)],
    )
    by_id = {cluster.id: cluster for cluster in clusters}
    duplicate_cluster = next(cluster for cluster in by_id.values() if {"pr:1", "pr:2"} == set(cluster.members))
    singleton_cluster = next(cluster for cluster in by_id.values() if set(cluster.members) == {"issue:1"})
    assert duplicate_cluster.cluster_type == "duplicate_candidate"
    assert singleton_cluster.cluster_type == "singleton_orphan"


def test_cluster_type_linked_pair_for_hard_link_issue_pr() -> None:
    clusters = build_clusters(
        ["pr:1", "issue:1"],
        [_edge("pr:1", "issue:1", EdgeTier.WEAK, score=0.1, hard_link=1.0)],
    )
    assert len(clusters) == 1
    assert clusters[0].cluster_type == "linked_pair"
