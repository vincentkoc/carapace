from carapace.connectors.github_gh import GithubGhSourceConnector, _extract_score_from_text
from carapace.models import CIStatus, EntityKind


class FakeGhClient:
    def __init__(self) -> None:
        self.repo = "openclaw/openclaw"

    def get_paginated(self, endpoint: str, per_page: int = 100, max_items: int | None = None):
        if endpoint.startswith("pulls?state=open"):
            return [
                {
                    "id": 1001,
                    "number": 1,
                    "title": "Fix cache key generation",
                    "body": "Fixes #77\nImproves behavior",
                    "user": {"login": "dev1", "type": "User"},
                    "labels": [{"name": "bug"}],
                    "base": {"ref": "main"},
                    "head": {"ref": "fix/cache", "sha": "abc123"},
                    "additions": 10,
                    "deletions": 2,
                    "created_at": "2026-02-14T10:00:00Z",
                    "updated_at": "2026-02-15T10:00:00Z",
                    "author_association": "CONTRIBUTOR",
                }
            ]

        if endpoint == "pulls/1/files":
            return [
                {
                    "filename": "src/cache.py",
                    "patch": "@@ -1,2 +1,2 @@\n-old\n+new",
                }
            ]

        if endpoint == "pulls/1/reviews":
            return [{"state": "APPROVED"}]

        if endpoint == "issues/1/comments":
            return [
                {
                    "body": "CodeRabbit score: 82%\nGood improvement overall",
                    "user": {"login": "coderabbitai[bot]", "type": "Bot"},
                }
            ]

        if endpoint.startswith("issues?state=open"):
            return []

        return []

    def _api_json(self, endpoint: str, method: str = "GET", body=None):
        if endpoint == "commits/abc123/status":
            return {"state": "success"}
        return {}


def test_github_connector_normalizes_pr() -> None:
    connector = GithubGhSourceConnector(repo="openclaw/openclaw")
    connector.client = FakeGhClient()  # type: ignore[assignment]

    entities = connector.fetch_open_entities(max_prs=1, include_issues=False, max_issues=0)

    assert len(entities) == 1
    pr = entities[0]
    assert pr.kind == EntityKind.PR
    assert pr.id == "pr:1"
    assert pr.linked_issues == ["77"]
    assert pr.ci_status == CIStatus.PASS
    assert pr.approvals == 1
    assert pr.changed_files == ["src/cache.py"]
    assert len(pr.diff_hunks) == 1
    assert pr.external_reviews[0].provider == "coderabbit"
    assert pr.external_reviews[0].overall_score == 0.82


def test_extract_score_from_text() -> None:
    assert _extract_score_from_text("score: 0.91") == 0.91
    assert _extract_score_from_text("quality 72%") == 0.72
    assert _extract_score_from_text("result 85/100") == 0.85
    assert _extract_score_from_text("no score") == 0.5
