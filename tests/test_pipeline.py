from carapace.config import CarapaceConfig
from carapace.models import CIStatus, EntityKind, ExternalReviewSignal, SourceEntity
from carapace.pipeline import CarapaceEngine
from carapace.storage import SQLiteStorage


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


def test_pipeline_uses_fingerprint_cache_across_runs(tmp_path) -> None:
    class CountingEmbeddingProvider:
        def __init__(self) -> None:
            self.calls = 0

        def model_id(self) -> str:
            return "counting-model"

        def dimensions(self) -> int:
            return 4

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    storage = SQLiteStorage(tmp_path / "carapace.db")
    provider = CountingEmbeddingProvider()
    engine = CarapaceEngine(config=CarapaceConfig(), embedding_provider=provider, storage=storage)

    entities = [
        _entity("pr:1", repo="acme/repo"),
        _entity("pr:2", repo="acme/repo", title="Another"),
    ]
    engine.scan_entities(entities)
    assert provider.calls == 2
    engine.scan_entities(entities)
    assert provider.calls == 2


def test_singleton_cluster_not_labeled_canonical_or_related() -> None:
    entity = _entity("solo", repo="acme/repo", changed_files=["src/solo.py"])
    engine = CarapaceEngine(config=CarapaceConfig())
    report = engine.scan_entities([entity])
    routing = {item.entity_id: item.labels for item in report.routing}
    assert "triage/ready-human" in routing["solo"]
    assert "triage/canonical" not in routing["solo"]
    assert "triage/related" not in routing["solo"]


def test_close_candidate_is_labeled_but_not_closed_in_safe_mode() -> None:
    issue = SourceEntity.model_validate(
        {
            "id": "issue:1",
            "repo": "acme/repo",
            "kind": EntityKind.ISSUE,
            "title": "Template-only issue",
            "body": "## Summary\n## Steps to reproduce\n## Expected behavior\nOne line",
            "author": "dev",
        }
    )
    cfg = CarapaceConfig.model_validate(
        {
            "low_pass": {"issue_template_action": "close"},
            "action": {"safe_mode": True},
        }
    )
    report = CarapaceEngine(config=cfg).scan_entities([issue])
    route = report.routing[0]
    assert "triage/close-candidate" in route.labels
    assert route.close is False


def test_close_candidate_sets_close_when_safe_mode_off() -> None:
    issue = SourceEntity.model_validate(
        {
            "id": "issue:2",
            "repo": "acme/repo",
            "kind": EntityKind.ISSUE,
            "title": "Template-only issue",
            "body": "## Summary\n## Steps to reproduce\n## Expected behavior\nOne line",
            "author": "dev",
        }
    )
    cfg = CarapaceConfig.model_validate(
        {
            "low_pass": {"issue_template_action": "close"},
            "action": {"safe_mode": False},
        }
    )
    report = CarapaceEngine(config=cfg).scan_entities([issue])
    assert report.routing[0].close is True


def test_linked_pair_cluster_gets_linked_pair_label() -> None:
    issue = SourceEntity.model_validate(
        {
            "id": "issue:101",
            "repo": "acme/repo",
            "kind": EntityKind.ISSUE,
            "number": 101,
            "title": "Fix startup crash",
            "body": "The app crashes on startup after loading configuration.\nThis reproduces when plugin hooks initialize and session state is restored.",
            "author": "alice",
        }
    )
    pr = SourceEntity.model_validate(
        {
            "id": "pr:101",
            "repo": "acme/repo",
            "kind": EntityKind.PR,
            "number": 101,
            "title": "Fix startup crash",
                "body": "Fixes #101\nThe app crashes on startup after loading configuration.\nThis reproduces when plugin hooks initialize and session state is restored.",
                "author": "bob",
                "changed_files": ["src/app.py"],
                "linked_issues": ["101"],
            }
        )

    cfg = CarapaceConfig.model_validate(
        {
            "similarity": {
                "hard_link_issue_pr_strong_semantic_min": 0.0,
                "hard_link_weak_semantic_min": 0.0,
                "weak_semantic_min": 0.0,
            }
        }
    )
    report = CarapaceEngine(config=cfg).scan_entities([issue, pr])
    routing = {entry.entity_id: entry.labels for entry in report.routing}
    assert "triage/linked-pair" in routing["issue:101"]
    assert "triage/linked-pair" in routing["pr:101"]
    assert "triage/canonical" not in routing["issue:101"]
    assert "triage/canonical" not in routing["pr:101"]
