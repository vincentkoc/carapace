from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from carapace.connectors.github_gh import (
    GithubGhClient,
    GithubGhSinkConnector,
    GithubGhSourceConnector,
    GithubRateLimitError,
    _extract_lineage,
    _extract_score_from_text,
)
from carapace.models import CIStatus, EntityKind, SourceEntity


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
                    "mergeable": True,
                    "mergeable_state": "clean",
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

        if endpoint == "pulls/1/commits":
            return [
                {"sha": "deadbeef1", "commit": {"message": "feat: cache key normalization"}},
                {"sha": "deadbeef2", "commit": {"message": "test: add coverage"}},
            ]

        if endpoint == "issues/1/comments":
            return [
                {
                    "body": "CodeRabbit score: 82%\nGood improvement overall\nRelated #88",
                    "user": {"login": "coderabbitai[bot]", "type": "Bot"},
                }
            ]

        if endpoint.startswith("issues?state=open"):
            return [
                {
                    "id": 2001,
                    "number": 77,
                    "title": "Parser bug",
                    "body": "Fix parser edge case related #99",
                    "user": {"login": "reporter", "type": "User"},
                    "labels": [{"name": "bug"}],
                    "created_at": "2026-02-13T10:00:00Z",
                    "updated_at": "2026-02-14T10:00:00Z",
                    "author_association": "NONE",
                },
                {
                    "id": 2002,
                    "number": 88,
                    "title": "PR mirror",
                    "body": "",
                    "user": {"login": "dev2", "type": "User"},
                    "labels": [],
                    "created_at": "2026-02-13T10:00:00Z",
                    "updated_at": "2026-02-14T10:00:00Z",
                    "author_association": "NONE",
                    "pull_request": {"url": "https://api.github.com/repos/x/y/pulls/88"},
                },
            ]

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
    assert pr.soft_linked_issues == ["88"]
    assert pr.ci_status == CIStatus.PASS
    assert pr.approvals == 1
    assert pr.changed_files == ["src/cache.py"]
    assert len(pr.diff_hunks) == 1
    assert pr.commits == ["deadbeef1", "deadbeef2"]
    assert len(pr.patch_ids) == 2
    assert pr.mergeable is True
    assert pr.mergeable_state == "clean"
    assert pr.external_reviews[0].provider == "coderabbit"
    assert pr.external_reviews[0].overall_score == 0.82


def test_extract_score_from_text() -> None:
    assert _extract_score_from_text("score: 0.91") == 0.91
    assert _extract_score_from_text("quality 72%") == 0.72
    assert _extract_score_from_text("result 85/100") == 0.85
    assert _extract_score_from_text("no score") == 0.5


def test_extract_lineage_uses_sha_not_commit_message() -> None:
    commits, patch_ids = _extract_lineage(
        [
            {"sha": "abc123", "commit": {"message": "fix: update docs"}},
            {"sha": "def456", "commit": {"message": "fix: update docs"}},
        ]
    )
    assert commits == ["abc123", "def456"]
    assert len(patch_ids) == 2
    assert patch_ids[0] != patch_ids[1]


def test_github_connector_includes_issues_and_skips_issue_pr_mirror() -> None:
    connector = GithubGhSourceConnector(repo="openclaw/openclaw")
    connector.client = FakeGhClient()  # type: ignore[assignment]

    entities = connector.fetch_open_entities(max_prs=1, include_issues=True, max_issues=20)
    ids = {e.id for e in entities}
    assert "pr:1" in ids
    assert "issue:77" in ids
    assert "issue:88" not in ids
    issue = [e for e in entities if e.id == "issue:77"][0]
    assert issue.soft_linked_issues == ["99"]


def test_enrich_entity_minimal_uses_files_fast_path() -> None:
    class FastPathClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def get_page(self, endpoint: str, *, page: int, per_page: int = 100):
            _ = (page, per_page)
            self.calls.append(("get_page", endpoint))
            if endpoint == "pulls/1/files":
                return [{"filename": "src/a.py", "patch": "@@ -1 +1 @@\n-old\n+new"}]
            if endpoint == "pulls/1/commits":
                return [{"sha": "abc123", "commit": {"message": "fix: sample"}}]
            return []

        def _api_json(self, endpoint: str, method: str = "GET", body=None):
            _ = (method, body)
            self.calls.append(("_api_json", endpoint))
            return {}

    connector = GithubGhSourceConnector(repo="openclaw/openclaw")
    fast_client = FastPathClient()
    connector.client = fast_client  # type: ignore[assignment]

    pr_entity = SourceEntity.model_validate(
        {
            "id": "pr:1",
            "repo": "openclaw/openclaw",
            "kind": "pr",
            "number": 1,
            "state": "open",
            "title": "t",
            "author": "alice",
            "changed_files": [],
        }
    )
    enriched = connector.enrich_entity(pr_entity, mode="minimal")
    assert enriched.changed_files == ["src/a.py"]
    assert len(enriched.diff_hunks) == 1
    assert enriched.commits == ["abc123"]
    assert len(enriched.patch_ids) == 1
    assert ("get_page", "pulls/1/files") in fast_client.calls
    assert ("get_page", "pulls/1/commits") in fast_client.calls
    assert ("_api_json", "pulls/1") not in fast_client.calls


def test_enrich_entity_minimal_can_include_simple_scores() -> None:
    class SimpleScoreClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def get_page(self, endpoint: str, *, page: int, per_page: int = 100):
            _ = (page, per_page)
            self.calls.append(("get_page", endpoint))
            if endpoint == "pulls/1/files":
                return [{"filename": "src/a.py", "patch": "@@ -1 +1 @@\n-old\n+new"}]
            if endpoint == "pulls/1/commits":
                return [{"sha": "abc123", "commit": {"message": "fix: sample"}}]
            return []

        def _api_json(self, endpoint: str, method: str = "GET", body=None):
            _ = (method, body)
            self.calls.append(("_api_json", endpoint))
            if endpoint == "pulls/1":
                return {"mergeable": True, "mergeable_state": "clean", "head": {"sha": "abc123"}}
            if endpoint == "commits/abc123/status":
                return {
                    "state": "success",
                    "statuses": [
                        {
                            "context": "greptile review",
                            "description": "score: 0.81",
                        }
                    ],
                }
            return {}

    connector = GithubGhSourceConnector(repo="openclaw/openclaw")
    score_client = SimpleScoreClient()
    connector.client = score_client  # type: ignore[assignment]

    pr_entity = SourceEntity.model_validate(
        {
            "id": "pr:1",
            "repo": "openclaw/openclaw",
            "kind": "pr",
            "number": 1,
            "state": "open",
            "title": "t",
            "author": "alice",
            "changed_files": [],
        }
    )
    enriched = connector.enrich_entity(pr_entity, mode="minimal", include_simple_scores=True)
    assert enriched.mergeable is True
    assert enriched.mergeable_state == "clean"
    assert enriched.ci_status == CIStatus.PASS
    assert enriched.external_reviews
    assert enriched.external_reviews[0].provider == "greptile"
    assert enriched.external_reviews[0].overall_score == 0.81
    assert ("_api_json", "pulls/1") in score_client.calls
    assert ("_api_json", "commits/abc123/status") in score_client.calls


class FakeSinkClient:
    def __init__(self) -> None:
        self.calls = []

    def _api_json(self, endpoint: str, method: str = "GET", body=None):
        self.calls.append((endpoint, method, body))
        return {}


def test_github_sink_dry_run_no_api_calls() -> None:
    sink = GithubGhSinkConnector(repo="openclaw/openclaw", entity_number_resolver=lambda _: 12, dry_run=True)
    fake_client = FakeSinkClient()
    sink.client = fake_client  # type: ignore[assignment]

    sink.apply_labels("pr:12", ["triage/duplicate"])
    sink.post_comment("pr:12", "hello")
    assert fake_client.calls == []


def test_github_sink_live_calls_api() -> None:
    sink = GithubGhSinkConnector(repo="openclaw/openclaw", entity_number_resolver=lambda _: 12, dry_run=False)
    fake_client = FakeSinkClient()
    sink.client = fake_client  # type: ignore[assignment]

    sink.apply_labels("pr:12", ["triage/duplicate"])
    sink.post_comment("pr:12", "hello")
    sink.close_entity("pr:12")

    assert ("issues/12/labels", "POST", {"labels": ["triage/duplicate"]}) in fake_client.calls
    assert ("issues/12/comments", "POST", {"body": "hello"}) in fake_client.calls
    assert ("issues/12", "PATCH", {"state": "closed"}) in fake_client.calls


def test_github_client_raises_rate_limit_error_with_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"api": 0, "rate_limit": 0}

    def _fake_run(cmd, text=True, capture_output=True, check=False, input=None):  # noqa: ANN001,ARG001
        endpoint = cmd[2]
        if endpoint.endswith("/pulls?state=open"):
            calls["api"] += 1
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="gh: API rate limit exceeded for user ID 1 (HTTP 403)",
            )
        calls["rate_limit"] += 1
        return SimpleNamespace(
            returncode=0,
            stdout='{"resources":{"core":{"remaining":0,"reset":1700000000}}}',
            stderr="",
        )

    monkeypatch.setattr("subprocess.run", _fake_run)
    client = GithubGhClient(repo="openclaw/openclaw", rate_limit_retries=0, rate_limit_max_sleep_seconds=10.0)
    with pytest.raises(GithubRateLimitError) as exc:
        client._api_json("pulls?state=open")
    assert isinstance(exc.value.reset_at, datetime)
    assert exc.value.reset_at == datetime.fromtimestamp(1700000000, UTC)
    assert calls["api"] == 1
    assert calls["rate_limit"] == 1


def test_github_client_rate_limit_reset_fallback_none(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"api": 0, "rate_limit": 0}

    def _fake_run(cmd, text=True, capture_output=True, check=False, input=None):  # noqa: ANN001,ARG001
        endpoint = cmd[2]
        if endpoint.endswith("/pulls?state=open"):
            calls["api"] += 1
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="gh: secondary rate limit. please wait",
            )
        calls["rate_limit"] += 1
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="failed",
        )

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    client = GithubGhClient(repo="openclaw/openclaw", rate_limit_retries=1, secondary_backoff_base_seconds=1.0)
    with pytest.raises(GithubRateLimitError) as exc:
        client._api_json("pulls?state=open")
    assert exc.value.reset_at is None
    assert calls["api"] == 2
    assert calls["rate_limit"] == 2


def test_github_client_retries_and_succeeds_after_secondary_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"api": 0, "rate_limit": 0}

    def _fake_run(cmd, text=True, capture_output=True, check=False, input=None):  # noqa: ANN001,ARG001
        endpoint = cmd[2]
        if endpoint.endswith("/pulls?state=open"):
            calls["api"] += 1
            if calls["api"] == 1:
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="gh: secondary rate limit. please wait",
                )
            return SimpleNamespace(returncode=0, stdout='[{"id":1}]', stderr="")
        calls["rate_limit"] += 1
        return SimpleNamespace(returncode=1, stdout="", stderr="gh: unavailable")

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    client = GithubGhClient(repo="openclaw/openclaw", rate_limit_retries=2, secondary_backoff_base_seconds=1.0)
    payload = client._api_json("pulls?state=open")
    assert payload == [{"id": 1}]
    assert calls["api"] == 2
    assert calls["rate_limit"] == 1
