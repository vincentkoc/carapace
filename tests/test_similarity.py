from carapace.config import SimilarityConfig
from carapace.models import Fingerprint
from carapace.similarity import (
    build_candidate_index,
    compute_similarity_edges,
    compute_similarity_edges_with_stats,
    retrieve_candidates,
    score_pair,
)


def _fp(
    entity_id: str,
    *,
    files: list[str],
    modules: list[str],
    issues: list[str],
    patch_ids: list[str],
    hunks: list[str],
    embedding: list[float],
    tokens: list[str] | None = None,
    additions: int = 10,
    deletions: int = 2,
) -> Fingerprint:
    return Fingerprint(
        entity_id=entity_id,
        tokens=tokens or [],
        changed_files=files,
        module_buckets=modules,
        linked_issues=issues,
        patch_ids=patch_ids,
        hunk_signatures=hunks,
        embedding=embedding,
        additions=additions,
        deletions=deletions,
    )


def test_candidate_retrieval_uses_module_issue_and_file_indices() -> None:
    cfg = SimilarityConfig()
    fps = {
        "a": _fp("a", files=["src/a.py"], modules=["src/*"], issues=["12"], patch_ids=[], hunks=[], embedding=[1, 0]),
        "b": _fp("b", files=["src/b.py"], modules=["src/*"], issues=[], patch_ids=[], hunks=[], embedding=[1, 0]),
        "c": _fp("c", files=["docs/a.md"], modules=["docs/*"], issues=["12"], patch_ids=[], hunks=[], embedding=[0, 1]),
    }
    idx = build_candidate_index(fps, cfg)

    candidates = retrieve_candidates("a", fps["a"], idx, cfg)
    assert set(candidates) == {"b", "c"}


def test_score_pair_prefers_lineage_and_penalizes_size() -> None:
    cfg = SimilarityConfig()
    a = _fp(
        "a",
        files=["src/cache.py"],
        modules=["src/*"],
        issues=[],
        patch_ids=["p1"],
        hunks=["h1"],
        embedding=[1.0, 0.0],
        additions=10,
        deletions=10,
    )
    b = _fp(
        "b",
        files=["src/cache.py"],
        modules=["src/*"],
        issues=[],
        patch_ids=["p1"],
        hunks=["h1"],
        embedding=[1.0, 0.0],
        additions=100,
        deletions=100,
    )

    idx = build_candidate_index({"a": a, "b": b}, cfg)
    total, breakdown = score_pair(a, b, cfg, idx=idx)
    assert breakdown.lineage == 1.0
    assert breakdown.structure > 0.0
    assert breakdown.minhash >= 0.0
    assert breakdown.simhash >= 0.0
    assert breakdown.winnow >= 0.0
    assert breakdown.size_penalty > 0.0
    assert total > 0.5


def test_compute_similarity_edges_creates_single_strong_edge() -> None:
    cfg = SimilarityConfig()
    fps = {
        "a": _fp(
            "a",
            files=["src/cache.py"],
            modules=["src/*"],
            issues=["1"],
            patch_ids=["p1"],
            hunks=["h1"],
            embedding=[1.0, 0.0, 0.0],
        ),
        "b": _fp(
            "b",
            files=["src/cache.py"],
            modules=["src/*"],
            issues=["1"],
            patch_ids=["p1"],
            hunks=["h1"],
            embedding=[1.0, 0.0, 0.0],
        ),
        "c": _fp(
            "c",
            files=["docs/guide.md"],
            modules=["docs/*"],
            issues=[],
            patch_ids=["p2"],
            hunks=["h2"],
            embedding=[0.0, 1.0, 0.0],
        ),
    }

    edges = compute_similarity_edges(fps, cfg)
    assert len(edges) == 1
    edge = edges[0]
    assert {edge.entity_a, edge.entity_b} == {"a", "b"}
    assert edge.tier.value == "strong"


def test_advanced_algorithms_retrieve_without_file_or_issue_overlap() -> None:
    cfg = SimilarityConfig(use_advanced_algorithms=True)
    a = _fp(
        "a",
        files=[],
        modules=[],
        issues=[],
        patch_ids=[],
        hunks=[],
        embedding=[1.0, 0.0],
        tokens=["fix", "cache", "key", "collision"],
    )
    b = _fp(
        "b",
        files=[],
        modules=[],
        issues=[],
        patch_ids=[],
        hunks=[],
        embedding=[1.0, 0.0],
        tokens=["fix", "cache", "key", "collision"],
    )
    idx = build_candidate_index({"a": a, "b": b}, cfg)
    candidates = retrieve_candidates("a", a, idx, cfg)
    assert "b" in candidates


def test_compute_similarity_edges_with_stats() -> None:
    cfg = SimilarityConfig()
    fps = {
        "a": _fp("a", files=["src/x.py"], modules=["src/*"], issues=[], patch_ids=["p1"], hunks=["h1"], embedding=[1, 0]),
        "b": _fp("b", files=["src/x.py"], modules=["src/*"], issues=[], patch_ids=["p1"], hunks=["h1"], embedding=[1, 0]),
    }
    edges, stats = compute_similarity_edges_with_stats(fps, cfg)
    assert len(edges) == 1
    assert stats.entities_total == 2
    assert stats.unique_pairs_scored >= 1
    assert stats.edges_emitted == 1


def test_unstructured_pairs_require_very_high_semantic_and_lexical_match() -> None:
    cfg = SimilarityConfig(use_advanced_algorithms=True)
    a = _fp(
        "issue:1",
        files=[],
        modules=[],
        issues=[],
        patch_ids=[],
        hunks=[],
        embedding=[1.0, 0.0],
        tokens=["summary", "steps", "reproduce", "expected", "behavior"],
    )
    b = _fp(
        "issue:2",
        files=[],
        modules=[],
        issues=[],
        patch_ids=[],
        hunks=[],
        embedding=[0.0, 1.0],
        tokens=["summary", "steps", "reproduce", "expected", "behavior"],
    )
    edges = compute_similarity_edges({"issue:1": a, "issue:2": b}, cfg)
    assert edges == []
