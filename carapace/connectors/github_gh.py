"""GitHub connectors backed by gh CLI for source ingestion and sink actioning."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from carapace.connectors.base import SinkConnector, SourceConnector
from carapace.models import CIStatus, DiffHunk, EntityKind, ExternalReviewSignal, SourceEntity

_ISSUE_RE = re.compile(r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*#(\d+)", re.IGNORECASE)


class GithubUser(BaseModel):
    model_config = ConfigDict(extra="ignore")

    login: str
    type: str | None = None


class GithubLabel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str


class GithubPull(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    number: int
    title: str
    body: str | None = None
    user: GithubUser
    labels: list[GithubLabel] = Field(default_factory=list)
    base: dict[str, Any] = Field(default_factory=dict)
    head: dict[str, Any] = Field(default_factory=dict)
    additions: int = 0
    deletions: int = 0
    created_at: datetime
    updated_at: datetime
    author_association: str | None = None


class GithubIssue(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    number: int
    title: str
    body: str | None = None
    user: GithubUser
    labels: list[GithubLabel] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    author_association: str | None = None
    pull_request: dict[str, Any] | None = None


class GithubFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    filename: str
    patch: str | None = None


class GithubReview(BaseModel):
    model_config = ConfigDict(extra="ignore")

    state: str | None = None


class GithubComment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    body: str | None = None
    user: GithubUser | None = None


class GithubGhClient:
    def __init__(self, repo: str, gh_bin: str = "gh") -> None:
        self.repo = repo
        self.gh_bin = gh_bin

    def get_paginated(self, endpoint: str, per_page: int = 100, max_items: int | None = None) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page = 1
        while True:
            query = f"{endpoint}{'&' if '?' in endpoint else '?'}per_page={per_page}&page={page}"
            payload = self._api_json(query)
            if not isinstance(payload, list) or not payload:
                break
            items.extend(payload)
            if max_items is not None and len(items) >= max_items:
                return items[:max_items]
            page += 1
        return items

    def _api_json(self, endpoint: str, method: str = "GET", body: dict[str, Any] | None = None) -> Any:
        cmd = [self.gh_bin, "api", f"repos/{self.repo}/{endpoint.lstrip('/')}" if not endpoint.startswith("repos/") else endpoint]
        cmd.extend(["-X", method])
        cmd.extend(["-H", "Accept: application/vnd.github+json"])
        if body is not None:
            cmd.extend(["--input", "-"])
            proc = subprocess.run(
                cmd,
                input=json.dumps(body),
                text=True,
                capture_output=True,
                check=False,
            )
        else:
            proc = subprocess.run(cmd, text=True, capture_output=True, check=False)

        if proc.returncode != 0:
            raise RuntimeError(f"gh api failed: {' '.join(cmd)}\n{proc.stderr.strip()}")

        output = proc.stdout.strip()
        if not output:
            return None
        return json.loads(output)


class GithubGhSourceConnector(SourceConnector):
    def __init__(self, repo: str, gh_bin: str = "gh") -> None:
        self.repo = repo
        self.client = GithubGhClient(repo=repo, gh_bin=gh_bin)

    def fetch_open_entities(
        self,
        max_prs: int = 200,
        include_issues: bool = True,
        max_issues: int = 200,
    ) -> list[SourceEntity]:
        entities: list[SourceEntity] = []

        pulls = self.client.get_paginated("pulls?state=open", max_items=max_prs)
        for item in pulls:
            entities.append(self._normalize_pull(GithubPull.model_validate(item)))

        if include_issues:
            issues = self.client.get_paginated("issues?state=open", max_items=max_issues)
            for item in issues:
                issue = GithubIssue.model_validate(item)
                # GitHub issue API includes PRs; ignore them here.
                if issue.pull_request:
                    continue
                entities.append(self._normalize_issue(issue))

        return entities

    def list_open_entities(self) -> list[SourceEntity]:
        return self.fetch_open_entities(max_prs=200, include_issues=True, max_issues=200)

    def get_entity(self, entity_id: str) -> SourceEntity:
        kind, number = self._parse_entity_id(entity_id)
        if kind == EntityKind.PR:
            payload = self.client._api_json(f"pulls/{number}")
            return self._normalize_pull(GithubPull.model_validate(payload))
        payload = self.client._api_json(f"issues/{number}")
        issue = GithubIssue.model_validate(payload)
        return self._normalize_issue(issue)

    def get_diff_or_change_set(self, entity_id: str) -> dict:
        kind, number = self._parse_entity_id(entity_id)
        if kind == EntityKind.PR:
            files = self.client.get_paginated(f"pulls/{number}/files")
            return {"files": files}
        return {"files": []}

    def get_reviews_and_checks(self, entity_id: str) -> dict:
        kind, number = self._parse_entity_id(entity_id)
        if kind != EntityKind.PR:
            return {"reviews": [], "ci_status": CIStatus.UNKNOWN.value}

        reviews = self.client.get_paginated(f"pulls/{number}/reviews")
        pull = self.client._api_json(f"pulls/{number}")
        head_sha = pull.get("head", {}).get("sha")
        ci_state = CIStatus.UNKNOWN
        if head_sha:
            status = self.client._api_json(f"commits/{head_sha}/status")
            ci_state = _normalize_ci_status(status.get("state"))
        return {"reviews": reviews, "ci_status": ci_state.value}

    def _normalize_pull(self, pull: GithubPull) -> SourceEntity:
        number = pull.number

        files_payload = self.client.get_paginated(f"pulls/{number}/files")
        files = [GithubFile.model_validate(item) for item in files_payload]
        changed_files = [item.filename for item in files]
        diff_hunks = _parse_diff_hunks(files)

        reviews_payload = self.client.get_paginated(f"pulls/{number}/reviews")
        reviews = [GithubReview.model_validate(item) for item in reviews_payload]
        approvals = sum(1 for review in reviews if (review.state or "").upper() == "APPROVED")

        comments_payload = self.client.get_paginated(f"issues/{number}/comments")
        comments = [GithubComment.model_validate(item) for item in comments_payload]

        head_sha = pull.head.get("sha")
        ci_status = CIStatus.UNKNOWN
        if head_sha:
            status_payload = self.client._api_json(f"commits/{head_sha}/status") or {}
            ci_status = _normalize_ci_status(status_payload.get("state"))

        labels = [label.name for label in pull.labels]
        linked_issues = _extract_linked_issues(pull.body or "")
        external_reviews = _extract_external_review_signals(comments)

        return SourceEntity(
            id=f"pr:{number}",
            provider="github",
            repo=self.repo,
            kind=EntityKind.PR,
            number=number,
            title=pull.title,
            body=pull.body or "",
            labels=labels,
            author=pull.user.login,
            author_association=pull.author_association,
            is_bot=(pull.user.type or "").lower() == "bot" or pull.user.login.endswith("[bot]"),
            base_branch=pull.base.get("ref"),
            head_branch=pull.head.get("ref"),
            linked_issues=linked_issues,
            changed_files=changed_files,
            diff_hunks=diff_hunks,
            commits=[],
            patch_ids=[],
            additions=pull.additions,
            deletions=pull.deletions,
            ci_status=ci_status,
            approvals=approvals,
            review_comments=len(reviews_payload),
            external_reviews=external_reviews,
            created_at=pull.created_at.astimezone(UTC),
            updated_at=pull.updated_at.astimezone(UTC),
            metadata={"source": "gh", "head_sha": head_sha},
        )

    def _normalize_issue(self, issue: GithubIssue) -> SourceEntity:
        labels = [label.name for label in issue.labels]
        return SourceEntity(
            id=f"issue:{issue.number}",
            provider="github",
            repo=self.repo,
            kind=EntityKind.ISSUE,
            number=issue.number,
            title=issue.title,
            body=issue.body or "",
            labels=labels,
            author=issue.user.login,
            author_association=issue.author_association,
            is_bot=(issue.user.type or "").lower() == "bot" or issue.user.login.endswith("[bot]"),
            linked_issues=_extract_linked_issues(issue.body or ""),
            created_at=issue.created_at.astimezone(UTC),
            updated_at=issue.updated_at.astimezone(UTC),
            metadata={"source": "gh"},
        )

    @staticmethod
    def _parse_entity_id(entity_id: str) -> tuple[EntityKind, int]:
        prefix, raw_num = entity_id.split(":", maxsplit=1)
        if prefix == "pr":
            return EntityKind.PR, int(raw_num)
        if prefix == "issue":
            return EntityKind.ISSUE, int(raw_num)
        raise ValueError(f"Unsupported entity id: {entity_id}")


class GithubGhSinkConnector(SinkConnector):
    def __init__(
        self,
        repo: str,
        entity_number_resolver: Callable[[str], int],
        gh_bin: str = "gh",
        dry_run: bool = True,
    ) -> None:
        self.repo = repo
        self.client = GithubGhClient(repo=repo, gh_bin=gh_bin)
        self.entity_number_resolver = entity_number_resolver
        self.dry_run = dry_run

    def apply_labels(self, entity_id: str, labels: list[str]) -> None:
        if not labels:
            return
        number = self.entity_number_resolver(entity_id)
        if self.dry_run:
            return
        self.client._api_json(f"issues/{number}/labels", method="POST", body={"labels": labels})

    def post_comment(self, entity_id: str, body: str) -> None:
        if not body:
            return
        number = self.entity_number_resolver(entity_id)
        if self.dry_run:
            return
        self.client._api_json(f"issues/{number}/comments", method="POST", body={"body": body})

    def set_status(self, entity_id: str, state: str, context: str) -> None:
        # Optional future implementation; no-op for v1.
        _ = (entity_id, state, context)

    def route_to_queue(self, entity_id: str, queue_key: str) -> None:
        # GitHub has no native queue primitive; represented by labels/filtered views.
        _ = (entity_id, queue_key)


def _normalize_ci_status(state: str | None) -> CIStatus:
    normalized = (state or "").lower()
    if normalized in {"success", "successful", "completed"}:
        return CIStatus.PASS
    if normalized in {"failure", "failed", "error"}:
        return CIStatus.FAIL
    return CIStatus.UNKNOWN


def _extract_linked_issues(text: str) -> list[str]:
    return sorted({match.group(1) for match in _ISSUE_RE.finditer(text)})


def _parse_diff_hunks(files: list[GithubFile]) -> list[DiffHunk]:
    hunks: list[DiffHunk] = []
    for file in files:
        patch = file.patch or ""
        if not patch:
            continue

        current_context = ""
        added: list[str] = []
        removed: list[str] = []

        def flush() -> None:
            nonlocal added, removed
            if current_context or added or removed:
                hunks.append(
                    DiffHunk(
                        file_path=file.filename,
                        context=current_context,
                        added_lines=added,
                        removed_lines=removed,
                    )
                )
            added = []
            removed = []

        for line in patch.splitlines():
            if line.startswith("@@"):
                flush()
                current_context = line
                continue
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+"):
                added.append(line[1:])
            elif line.startswith("-"):
                removed.append(line[1:])

        flush()

    return hunks


def _extract_external_review_signals(comments: list[GithubComment]) -> list[ExternalReviewSignal]:
    signals: list[ExternalReviewSignal] = []
    for comment in comments:
        body = (comment.body or "").strip()
        user_login = (comment.user.login if comment.user else "").lower()
        if not body:
            continue

        provider = None
        if "coderabbit" in user_login or "coderabbit" in body.lower():
            provider = "coderabbit"
        elif "greptile" in user_login or "greptile" in body.lower():
            provider = "greptile"
        if provider is None:
            continue

        score = _extract_score_from_text(body)
        signals.append(
            ExternalReviewSignal(
                provider=provider,
                overall_score=score,
                confidence=0.6,
                summary=body[:600],
            )
        )

    return signals


def _extract_score_from_text(body: str) -> float:
    # Tries to parse percentage scores (e.g., "85/100", "score: 0.82", "82%") from review comments.
    patterns = [
        re.compile(r"(\d{1,3})\s*/\s*100"),
        re.compile(r"score\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)", re.IGNORECASE),
        re.compile(r"(\d{1,3})\s*%"),
    ]

    for pattern in patterns:
        match = pattern.search(body)
        if not match:
            continue
        raw = match.group(1)
        value = float(raw)
        if value > 1.0:
            value = value / 100.0
        return min(1.0, max(0.0, value))

    return 0.5
