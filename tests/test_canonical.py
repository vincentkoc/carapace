from carapace.canonical import rank_canonicals
from carapace.config import CanonicalConfig
from carapace.models import (
    CIStatus,
    Cluster,
    DecisionState,
    EdgeTier,
    Fingerprint,
    LowPassDecision,
    SimilarityBreakdown,
    SimilarityEdge,
)


def _fp(entity_id: str, ci: CIStatus, reviewer: float, approvals: int, files: list[str], adds: int = 5, dels: int = 2) -> Fingerprint:
    return Fingerprint(
        entity_id=entity_id,
        changed_files=files,
        module_buckets=["src/*"],
        linked_issues=[],
        patch_ids=[],
        hunk_signatures=[],
        embedding=[1.0, 0.0],
        ci_status=ci,
        reviewer_score=reviewer,
        approvals=approvals,
        additions=adds,
        deletions=dels,
    )


def _edge(a: str, b: str, score: float, lineage: float = 0.0) -> SimilarityEdge:
    return SimilarityEdge(
        entity_a=a,
        entity_b=b,
        score=score,
        tier=EdgeTier.STRONG if score > 0.82 else EdgeTier.WEAK,
        breakdown=SimilarityBreakdown(
            lineage=lineage,
            structure=score,
            file_overlap=score,
            hunk_overlap=score,
            semantic=score,
            size_penalty=0.0,
            total=score,
        ),
    )


def test_rank_canonical_prefers_stronger_quality_signals() -> None:
    clusters = [Cluster(id="cluster-1", members=["a", "b"])]
    fingerprints = {
        "a": _fp("a", CIStatus.PASS, reviewer=0.9, approvals=2, files=["src/core.py"]),
        "b": _fp("b", CIStatus.FAIL, reviewer=0.2, approvals=0, files=["src/core.py"]),
    }
    low_pass = {
        "a": LowPassDecision(entity_id="a", state="pass", priority_weight=1.0),
        "b": LowPassDecision(entity_id="b", state="pass", priority_weight=1.0),
    }
    edges = [_edge("a", "b", score=0.9)]

    decisions = rank_canonicals(clusters, fingerprints, edges, low_pass, CanonicalConfig())
    d = decisions[0]
    assert d.canonical_entity_id == "a"
    by_id = {m.entity_id: m for m in d.member_decisions}
    assert by_id["a"].state == DecisionState.CANONICAL
    assert by_id["b"].state == DecisionState.DUPLICATE


def test_lineage_overlap_marks_duplicate_even_below_duplicate_threshold() -> None:
    cfg = CanonicalConfig(duplicate_threshold=0.95)
    clusters = [Cluster(id="cluster-1", members=["a", "b"])]
    fingerprints = {
        "a": _fp("a", CIStatus.PASS, reviewer=0.6, approvals=1, files=["src/cache.py"]),
        "b": _fp("b", CIStatus.PASS, reviewer=0.5, approvals=1, files=["src/cache.py"]),
    }
    low_pass = {
        "a": LowPassDecision(entity_id="a", state="pass", priority_weight=1.0),
        "b": LowPassDecision(entity_id="b", state="pass", priority_weight=1.0),
    }
    edges = [_edge("a", "b", score=0.6, lineage=0.8)]

    d = rank_canonicals(clusters, fingerprints, edges, low_pass, cfg)[0]
    non_canonical = next(m for m in d.member_decisions if m.entity_id != d.canonical_entity_id)
    assert non_canonical.state == DecisionState.DUPLICATE
    assert non_canonical.score == 0.6


def test_tie_margin_marks_runner_up_for_human_tie_break() -> None:
    cfg = CanonicalConfig(tie_margin=10.0, duplicate_threshold=0.95)
    clusters = [Cluster(id="cluster-1", members=["a", "b"])]
    fingerprints = {
        "a": _fp("a", CIStatus.PASS, reviewer=0.5, approvals=1, files=["src/x.py"]),
        "b": _fp("b", CIStatus.PASS, reviewer=0.49, approvals=1, files=["src/x.py"]),
    }
    low_pass = {
        "a": LowPassDecision(entity_id="a", state="pass", priority_weight=1.0),
        "b": LowPassDecision(entity_id="b", state="pass", priority_weight=1.0),
    }
    edges = [_edge("a", "b", score=0.8)]

    d = rank_canonicals(clusters, fingerprints, edges, low_pass, cfg)[0]
    states = {m.state for m in d.member_decisions}
    assert DecisionState.TIE_BREAK in states


def test_mixed_kind_cluster_marks_related_not_duplicate() -> None:
    clusters = [Cluster(id="cluster-1", members=["issue:1", "pr:1"])]
    fingerprints = {
        "issue:1": _fp("issue:1", CIStatus.UNKNOWN, reviewer=0.0, approvals=0, files=[]),
        "pr:1": _fp("pr:1", CIStatus.PASS, reviewer=0.5, approvals=1, files=["src/x.py"]),
    }
    low_pass = {
        "issue:1": LowPassDecision(entity_id="issue:1", state="pass", priority_weight=1.0),
        "pr:1": LowPassDecision(entity_id="pr:1", state="pass", priority_weight=1.0),
    }
    edges = [_edge("issue:1", "pr:1", score=0.9)]
    decision = rank_canonicals(clusters, fingerprints, edges, low_pass, CanonicalConfig(duplicate_threshold=0.1))[0]
    non_canonical = next(item for item in decision.member_decisions if item.entity_id != decision.canonical_entity_id)
    assert non_canonical.state == DecisionState.RELATED
