from carapace.config import CarapaceConfig
from carapace.models import CIStatus, EntityKind, ExternalReviewSignal, SourceEntity
from carapace.pipeline import CarapaceEngine


def _entity(entity_id: str, **kwargs) -> SourceEntity:
    payload = {
        "id": entity_id,
        "repo": "acme/repo",
        "kind": EntityKind.PR,
        "title": "Fix cache key handling",
        "body": "Improves cache key generation",
        "author": "dev",
        "changed_files": ["src/cache.py"],
        "additions": 10,
        "deletions": 2,
        "ci_status": CIStatus.PASS,
        "approvals": 1,
    }
    payload.update(kwargs)
    return SourceEntity.model_validate(payload)


def test_end_to_end_similarity_and_canonical() -> None:
    pr1 = _entity(
        "1",
        patch_ids=["patch-1"],
        external_reviews=[ExternalReviewSignal(provider="coderabbit", overall_score=0.9, confidence=0.8)],
    )
    pr2 = _entity(
        "2",
        patch_ids=["patch-1"],
        ci_status=CIStatus.FAIL,
        approvals=0,
        external_reviews=[ExternalReviewSignal(provider="coderabbit", overall_score=0.5, confidence=0.8)],
    )
    pr3 = _entity(
        "3",
        title="Refactor parser",
        body="Refactors parser internals",
        changed_files=["src/parser/core.py"],
        patch_ids=["patch-2"],
    )
    pr4 = _entity(
        "4",
        title="Docs tweak",
        body="Fix typo",
        changed_files=["docs/guide.md"],
        ci_status=CIStatus.UNKNOWN,
    )

    engine = CarapaceEngine(config=CarapaceConfig())
    report = engine.scan_entities([pr1, pr2, pr3, pr4])

    assert report.processed_entities == 4
    assert report.suppressed_entities == 1
    assert any(edge.entity_a == "1" and edge.entity_b == "2" for edge in report.edges)

    cluster_for_1 = next(cluster for cluster in report.clusters if "1" in cluster.members)
    canonical = next(item for item in report.canonical_decisions if item.cluster_id == cluster_for_1.id)
    assert canonical.canonical_entity_id == "1"

    routing = {item.entity_id: item.labels for item in report.routing}
    assert "triage/canonical" in routing["1"]
    assert "triage/duplicate" in routing["2"]
    assert "triage/noise-suppressed" in routing["4"]
