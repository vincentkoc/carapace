"""GitHub connectors backed by gh CLI for source ingestion and sink actioning."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from carapace.connectors.base import SinkConnector, SourceConnector
from carapace.models import CIStatus, DiffHunk, EntityKind, ExternalReviewSignal, SourceEntity

_HARD_ISSUE_RE = re.compile(r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*#(\d+)", re.IGNORECASE)
_SOFT_ISSUE_RE = re.compile(r"#(\d+)")
_GH_ENTITY_URL_RE = re.compile(r"github\\.com/[^\\s/]+/[^\\s/]+/(?:issues|pull)/(\\d+)", re.IGNORECASE)
_RATE_LIMIT_RE = re.compile(r"(?:api|secondary) rate limit", re.IGNORECASE)
logger = logging.getLogger(__name__)


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
    state: str = "open"
    draft: bool = False
    mergeable: bool | None = None
    mergeable_state: str | None = None


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
    state: str = "open"
    comments: int = 0
    reactions: dict[str, Any] = Field(default_factory=dict)


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


class GithubPullCommit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sha: str
    commit: dict[str, Any] = Field(default_factory=dict)


class GithubRateLimitError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        reset_at: datetime | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.reset_at = reset_at
        self.retry_after_seconds = retry_after_seconds


class GithubGhClient:
    def __init__(
        self,
        repo: str,
        gh_bin: str = "gh",
        *,
        rate_limit_retries: int = 2,
        secondary_backoff_base_seconds: float = 5.0,
        rate_limit_max_sleep_seconds: float = 90.0,
    ) -> None:
        self.repo = repo
        self.gh_bin = gh_bin
        self.rate_limit_retries = max(0, rate_limit_retries)
        self.secondary_backoff_base_seconds = max(1.0, secondary_backoff_base_seconds)
        self.rate_limit_max_sleep_seconds = max(1.0, rate_limit_max_sleep_seconds)
        self._rate_limit_lock = threading.Lock()
        self._global_backoff_until = 0.0

    def get_paginated(self, endpoint: str, per_page: int = 100, max_items: int | None = None) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page = 1
        while True:
            payload = self.get_page(endpoint, page=page, per_page=per_page)
            if not isinstance(payload, list) or not payload:
                break
            items.extend(payload)
            if max_items is not None and len(items) >= max_items:
                return items[:max_items]
            page += 1
        return items

    def get_page(self, endpoint: str, *, page: int, per_page: int = 100) -> list[dict[str, Any]]:
        # TODO: Add explicit GitHub rate-limit backoff/retry strategy for long-running service mode.
        query = f"{endpoint}{'&' if '?' in endpoint else '?'}per_page={per_page}&page={page}"
        payload = self._api_json(query)
        if not isinstance(payload, list):
            return []
        return payload

    def _api_json(self, endpoint: str, method: str = "GET", body: dict[str, Any] | None = None) -> Any:
        cmd = [self.gh_bin, "api", f"repos/{self.repo}/{endpoint.lstrip('/')}" if not endpoint.startswith("repos/") else endpoint]
        cmd.extend(["-X", method])
        cmd.extend(["-H", "Accept: application/vnd.github+json"])
        for attempt in range(self.rate_limit_retries + 1):
            self._wait_for_global_backoff()
            if body is not None:
                cmd_with_body = [*cmd, "--input", "-"]
                proc = subprocess.run(
                    cmd_with_body,
                    input=json.dumps(body),
                    text=True,
                    capture_output=True,
                    check=False,
                )
            else:
                proc = subprocess.run(cmd, text=True, capture_output=True, check=False)

            if proc.returncode == 0:
                output = proc.stdout.strip()
                if not output:
                    return None
                return json.loads(output)

            stderr = proc.stderr.strip()
            if not _RATE_LIMIT_RE.search(stderr):
                raise RuntimeError(f"gh api failed: {' '.join(cmd)}\n{stderr}")

            reset_at = self._get_rate_limit_reset_at()
            retry_after_seconds = self._compute_rate_limit_wait_seconds(reset_at=reset_at, attempt=attempt)
            self._set_global_backoff(retry_after_seconds)
            has_retry = attempt < self.rate_limit_retries

            logger.warning(
                "GitHub rate limit hit for %s (attempt %s/%s). backoff=%.1fs reset_at=%s",
                endpoint,
                attempt + 1,
                self.rate_limit_retries + 1,
                retry_after_seconds,
                reset_at.isoformat() if reset_at else "unknown",
            )

            if has_retry and retry_after_seconds <= self.rate_limit_max_sleep_seconds:
                continue

            raise GithubRateLimitError(
                f"gh api failed: {' '.join(cmd)}\n{stderr}",
                reset_at=reset_at,
                retry_after_seconds=retry_after_seconds,
            )
        raise RuntimeError(f"gh api failed unexpectedly after retries for endpoint={endpoint}")

    def _wait_for_global_backoff(self) -> None:
        while True:
            with self._rate_limit_lock:
                wait_seconds = self._global_backoff_until - time.monotonic()
            if wait_seconds <= 0:
                return
            time.sleep(wait_seconds)

    def _set_global_backoff(self, wait_seconds: float) -> None:
        target = time.monotonic() + max(0.0, wait_seconds)
        with self._rate_limit_lock:
            self._global_backoff_until = max(self._global_backoff_until, target)

    def _compute_rate_limit_wait_seconds(self, *, reset_at: datetime | None, attempt: int) -> float:
        if reset_at is not None:
            until_reset = (reset_at - datetime.now(UTC)).total_seconds()
            if until_reset > self.rate_limit_max_sleep_seconds:
                return until_reset
            return max(1.0, until_reset + 1.0)
        backoff = self.secondary_backoff_base_seconds * (2**attempt)
        return float(min(self.rate_limit_max_sleep_seconds, max(1.0, backoff)))

    def _get_rate_limit_reset_at(self) -> datetime | None:
        cmd = [self.gh_bin, "api", "rate_limit", "-X", "GET", "-H", "Accept: application/vnd.github+json"]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            return None
        payload = proc.stdout.strip()
        if not payload:
            return None
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None

        reset_epochs: list[int] = []
        resources = data.get("resources")
        if isinstance(resources, dict):
            for resource in resources.values():
                if not isinstance(resource, dict):
                    continue
                remaining = resource.get("remaining")
                reset = resource.get("reset")
                if isinstance(remaining, int) and remaining <= 0 and isinstance(reset, int):
                    reset_epochs.append(reset)
        rate = data.get("rate")
        if isinstance(rate, dict):
            remaining = rate.get("remaining")
            reset = rate.get("reset")
            if isinstance(remaining, int) and remaining <= 0 and isinstance(reset, int):
                reset_epochs.append(reset)
        if not reset_epochs:
            return None
        return datetime.fromtimestamp(max(reset_epochs), UTC)


class GithubGhSourceConnector(SourceConnector):
    def __init__(
        self,
        repo: str,
        gh_bin: str = "gh",
        *,
        rate_limit_retries: int = 2,
        secondary_backoff_base_seconds: float = 5.0,
        rate_limit_max_sleep_seconds: float = 90.0,
    ) -> None:
        self.repo = repo
        self.client = GithubGhClient(
            repo=repo,
            gh_bin=gh_bin,
            rate_limit_retries=rate_limit_retries,
            secondary_backoff_base_seconds=secondary_backoff_base_seconds,
            rate_limit_max_sleep_seconds=rate_limit_max_sleep_seconds,
        )

    def fetch_open_entities(
        self,
        max_prs: int = 200,
        include_issues: bool = True,
        max_issues: int = 200,
        include_drafts: bool = False,
        include_closed: bool = False,
        enrich_pr_details: bool = True,
        enrich_issue_comments: bool = True,
    ) -> list[SourceEntity]:
        entities: list[SourceEntity] = []
        pr_state = "all" if include_closed else "open"
        issue_state = "all" if include_closed else "open"

        pulls = self.client.get_paginated(f"pulls?state={pr_state}", max_items=max_prs)
        for item in pulls:
            pull = GithubPull.model_validate(item)
            if not include_drafts and pull.draft:
                continue
            entities.append(
                self._normalize_pull(
                    pull,
                    enrich_files=enrich_pr_details,
                    enrich_reviews=enrich_pr_details,
                    enrich_ci=enrich_pr_details,
                    enrich_comments=enrich_issue_comments,
                    enrich_lineage=enrich_pr_details,
                )
            )

        if include_issues:
            issues = self.client.get_paginated(f"issues?state={issue_state}", max_items=max_issues)
            for item in issues:
                issue = GithubIssue.model_validate(item)
                # GitHub issue API includes PRs; ignore them here.
                if issue.pull_request:
                    continue
                entities.append(self._normalize_issue(issue))

        return entities

    def list_open_entities(self) -> list[SourceEntity]:
        return self.fetch_open_entities(max_prs=200, include_issues=True, max_issues=200)

    def fetch_pull_page(
        self,
        *,
        page: int,
        per_page: int = 100,
        state: str = "open",
        include_drafts: bool = False,
        enrich_details: bool = False,
        enrich_comments: bool = False,
    ) -> list[SourceEntity]:
        payload = self.client.get_page(f"pulls?state={state}", page=page, per_page=per_page)
        logger.debug("Fetched PR page %s (%s items)", page, len(payload))
        entities: list[SourceEntity] = []
        for item in payload:
            pull = GithubPull.model_validate(item)
            if not include_drafts and pull.draft:
                continue
            entities.append(
                self._normalize_pull(
                    pull,
                    enrich_files=enrich_details,
                    enrich_reviews=enrich_details,
                    enrich_ci=enrich_details,
                    enrich_comments=enrich_comments,
                    enrich_lineage=enrich_details,
                )
            )
        return entities

    def fetch_issue_page(self, *, page: int, per_page: int = 100, state: str = "open") -> list[SourceEntity]:
        payload = self.client.get_page(f"issues?state={state}", page=page, per_page=per_page)
        logger.debug("Fetched issue page %s (%s items)", page, len(payload))
        entities: list[SourceEntity] = []
        for item in payload:
            issue = GithubIssue.model_validate(item)
            if issue.pull_request:
                continue
            entities.append(self._normalize_issue(issue))
        return entities

    def get_entity(self, entity_id: str) -> SourceEntity:
        kind, number = self._parse_entity_id(entity_id)
        if kind == EntityKind.PR:
            payload = self.client._api_json(f"pulls/{number}")
            return self._normalize_pull(
                GithubPull.model_validate(payload),
                enrich_files=True,
                enrich_reviews=True,
                enrich_ci=True,
                enrich_comments=False,
                enrich_lineage=True,
            )
        payload = self.client._api_json(f"issues/{number}")
        issue = GithubIssue.model_validate(payload)
        return self._normalize_issue(issue)

    def enrich_entity(
        self,
        entity: SourceEntity,
        include_comments: bool = False,
        mode: str = "minimal",
        include_simple_scores: bool = False,
    ) -> SourceEntity:
        if entity.kind != EntityKind.PR or entity.number is None:
            return entity
        if mode == "minimal":
            # Fast path: avoid fetching pull metadata and only pull first page of changed files.
            files_payload = self.client.get_page(f"pulls/{entity.number}/files", page=1, per_page=100)
            files = [GithubFile.model_validate(item) for item in files_payload]
            changed_files = [item.filename for item in files]
            diff_hunks = _parse_diff_hunks(files)
            commits_payload = self.client.get_page(f"pulls/{entity.number}/commits", page=1, per_page=100)
            commits, patch_ids = _extract_lineage(commits_payload)
            update = {
                "changed_files": changed_files,
                "diff_hunks": diff_hunks,
                "commits": commits,
                "patch_ids": patch_ids,
            }
            if include_simple_scores:
                pull_payload = self.client._api_json(f"pulls/{entity.number}") or {}
                mergeable = pull_payload.get("mergeable")
                mergeable_state = pull_payload.get("mergeable_state")
                reviews_payload = self.client.get_page(f"pulls/{entity.number}/reviews", page=1, per_page=100)
                approvals = 0
                for review_item in reviews_payload:
                    state = (review_item.get("state") or "").upper()
                    if state == "APPROVED":
                        approvals += 1
                head_sha = (pull_payload.get("head") or {}).get("sha")
                ci_status = entity.ci_status
                status_payload: dict[str, Any] = {}
                if head_sha:
                    status_payload = self.client._api_json(f"commits/{head_sha}/status") or {}
                    ci_status = _normalize_ci_status(status_payload.get("state"))
                status_signals = _extract_external_review_signals_from_status(status_payload)
                external_reviews = _merge_external_review_signals(entity.external_reviews, status_signals)
                update.update(
                    {
                        "mergeable": mergeable,
                        "mergeable_state": mergeable_state,
                        "ci_status": ci_status,
                        "approvals": approvals,
                        "external_reviews": external_reviews,
                    }
                )
            return entity.model_copy(update=update)

        payload = self.client._api_json(f"pulls/{entity.number}")
        pull = GithubPull.model_validate(payload)
        return self._normalize_pull(
            pull,
            enrich_files=True,
            enrich_reviews=True,
            enrich_ci=True,
            enrich_comments=include_comments,
            enrich_lineage=True,
        )

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

    def _normalize_pull(
        self,
        pull: GithubPull,
        *,
        enrich_files: bool = True,
        enrich_reviews: bool = True,
        enrich_ci: bool = True,
        enrich_comments: bool = True,
        enrich_lineage: bool = True,
        file_pages_limit: int | None = None,
        commits_pages_limit: int | None = None,
    ) -> SourceEntity:
        number = pull.number

        changed_files: list[str] = []
        diff_hunks: list[DiffHunk] = []
        commits: list[str] = []
        patch_ids: list[str] = []
        approvals = 0
        reviews_payload: list[dict[str, Any]] = []
        comments: list[GithubComment] = []
        head_sha = pull.head.get("sha")
        ci_status = CIStatus.UNKNOWN
        status_payload: dict[str, Any] = {}

        if enrich_files:
            if file_pages_limit is not None and file_pages_limit <= 1:
                files_payload = self.client.get_page(f"pulls/{number}/files", page=1, per_page=100)
            else:
                files_payload = self.client.get_paginated(f"pulls/{number}/files")
            files = [GithubFile.model_validate(item) for item in files_payload]
            changed_files = [item.filename for item in files]
            diff_hunks = _parse_diff_hunks(files)

        if enrich_reviews:
            reviews_payload = self.client.get_paginated(f"pulls/{number}/reviews")
            reviews = [GithubReview.model_validate(item) for item in reviews_payload]
            approvals = sum(1 for review in reviews if (review.state or "").upper() == "APPROVED")

        if enrich_ci:
            if head_sha:
                status_payload = self.client._api_json(f"commits/{head_sha}/status") or {}
                ci_status = _normalize_ci_status(status_payload.get("state"))

        if enrich_lineage:
            if commits_pages_limit is not None and commits_pages_limit <= 1:
                commits_payload = self.client.get_page(f"pulls/{number}/commits", page=1, per_page=100)
            else:
                commits_payload = self.client.get_paginated(f"pulls/{number}/commits")
            commits, patch_ids = _extract_lineage(commits_payload)

        if enrich_comments:
            comments_payload = self.client.get_paginated(f"issues/{number}/comments")
            comments = [GithubComment.model_validate(item) for item in comments_payload]

        labels = [label.name for label in pull.labels]
        linked_issues = _extract_hard_linked_issues(pull.body or "")
        soft_linked_issues = _extract_soft_linked_issues(pull.body or "", linked_issues)
        if comments:
            soft_linked_issues = _merge_issue_refs(
                soft_linked_issues,
                _extract_soft_links_from_comments(comments, linked_issues),
            )
        external_reviews = _merge_external_review_signals(
            _extract_external_review_signals(comments),
            _extract_external_review_signals_from_status(status_payload),
        )

        return SourceEntity(
            id=f"pr:{number}",
            provider="github",
            repo=self.repo,
            kind=EntityKind.PR,
            state=pull.state,
            is_draft=bool(pull.draft),
            number=number,
            title=pull.title,
            body=pull.body or "",
            labels=labels,
            author=pull.user.login,
            author_association=pull.author_association,
            is_bot=(pull.user.type or "").lower() == "bot" or pull.user.login.endswith("[bot]"),
            base_branch=pull.base.get("ref"),
            head_branch=pull.head.get("ref"),
            mergeable=pull.mergeable,
            mergeable_state=pull.mergeable_state,
            linked_issues=linked_issues,
            soft_linked_issues=soft_linked_issues,
            changed_files=changed_files,
            diff_hunks=diff_hunks,
            commits=commits,
            patch_ids=patch_ids,
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
        linked_issues = _extract_hard_linked_issues(issue.body or "")
        soft_linked_issues = _extract_soft_linked_issues(issue.body or "", linked_issues)
        reaction_total = _sum_issue_reactions(issue.reactions)
        return SourceEntity(
            id=f"issue:{issue.number}",
            provider="github",
            repo=self.repo,
            kind=EntityKind.ISSUE,
            state=issue.state,
            number=issue.number,
            title=issue.title,
            body=issue.body or "",
            labels=labels,
            author=issue.user.login,
            author_association=issue.author_association,
            is_bot=(issue.user.type or "").lower() == "bot" or issue.user.login.endswith("[bot]"),
            linked_issues=linked_issues,
            soft_linked_issues=soft_linked_issues,
            created_at=issue.created_at.astimezone(UTC),
            updated_at=issue.updated_at.astimezone(UTC),
            metadata={
                "source": "gh",
                "comment_count": issue.comments,
                "reactions": issue.reactions,
                "reaction_total": reaction_total,
            },
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
        *,
        rate_limit_retries: int = 2,
        secondary_backoff_base_seconds: float = 5.0,
        rate_limit_max_sleep_seconds: float = 90.0,
    ) -> None:
        self.repo = repo
        self.client = GithubGhClient(
            repo=repo,
            gh_bin=gh_bin,
            rate_limit_retries=rate_limit_retries,
            secondary_backoff_base_seconds=secondary_backoff_base_seconds,
            rate_limit_max_sleep_seconds=rate_limit_max_sleep_seconds,
        )
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
        # TODO: Implement GitHub checks/status publishing so Carapace appears directly in PR check-runs.
        _ = (entity_id, state, context)

    def route_to_queue(self, entity_id: str, queue_key: str) -> None:
        # GitHub has no native queue primitive; represented by labels/filtered views.
        _ = (entity_id, queue_key)

    def close_entity(self, entity_id: str) -> None:
        number = self.entity_number_resolver(entity_id)
        if self.dry_run:
            return
        self.client._api_json(f"issues/{number}", method="PATCH", body={"state": "closed"})


def _normalize_ci_status(state: str | None) -> CIStatus:
    normalized = (state or "").lower()
    if normalized in {"success", "successful", "completed"}:
        return CIStatus.PASS
    if normalized in {"failure", "failed", "error"}:
        return CIStatus.FAIL
    return CIStatus.UNKNOWN


def _merge_issue_refs(first: list[str], second: list[str]) -> list[str]:
    return sorted(set(first) | set(second))


def _extract_hard_linked_issues(text: str) -> list[str]:
    return sorted({match.group(1) for match in _HARD_ISSUE_RE.finditer(text)})


def _extract_soft_linked_issues(text: str, hard_links: list[str] | None = None) -> list[str]:
    hard_set = set(hard_links or [])
    refs = {match.group(1) for match in _SOFT_ISSUE_RE.finditer(text)}
    refs.update(match.group(1) for match in _GH_ENTITY_URL_RE.finditer(text))
    return sorted(ref for ref in refs if ref not in hard_set)


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


def _stable_patch_like_id(text: str) -> str:
    material = text.encode("utf-8")
    return hashlib.blake2b(material, digest_size=20).hexdigest()


def _extract_lineage(commits_payload: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    commits: list[str] = []
    patch_ids: list[str] = []
    seen_commits: set[str] = set()
    seen_patch_ids: set[str] = set()

    for item in commits_payload:
        commit_obj = GithubPullCommit.model_validate(item)
        sha = commit_obj.sha
        if sha and sha not in seen_commits:
            commits.append(sha)
            seen_commits.add(sha)

        # Use commit SHA as the lineage token to avoid collisions from generic commit messages.
        if not sha:
            continue
        patch_like_id = _stable_patch_like_id(sha)
        if patch_like_id in seen_patch_ids:
            continue
        patch_ids.append(patch_like_id)
        seen_patch_ids.add(patch_like_id)

    return commits, patch_ids


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


def _extract_external_review_signals_from_status(status_payload: dict[str, Any]) -> list[ExternalReviewSignal]:
    signals: list[ExternalReviewSignal] = []
    statuses = status_payload.get("statuses") if isinstance(status_payload, dict) else None
    if not isinstance(statuses, list):
        return signals

    for status in statuses:
        if not isinstance(status, dict):
            continue
        context = (status.get("context") or "").strip()
        description = (status.get("description") or "").strip()
        haystack = f"{context} {description}".lower()
        provider = None
        if "coderabbit" in haystack:
            provider = "coderabbit"
        elif "greptile" in haystack:
            provider = "greptile"
        if provider is None:
            continue

        signal_text = f"{context}\n{description}".strip()
        score = _extract_score_from_text(signal_text)
        signals.append(
            ExternalReviewSignal(
                provider=provider,
                overall_score=score,
                confidence=0.55,
                summary=signal_text[:600],
            )
        )
    return signals


def _merge_external_review_signals(first: list[ExternalReviewSignal], second: list[ExternalReviewSignal]) -> list[ExternalReviewSignal]:
    merged: dict[str, ExternalReviewSignal] = {}
    for signal in [*first, *second]:
        key = signal.provider.lower()
        existing = merged.get(key)
        if existing is None:
            merged[key] = signal
            continue
        # Keep highest score and confidence while preserving latest summary context.
        merged[key] = ExternalReviewSignal(
            provider=existing.provider,
            overall_score=max(existing.overall_score, signal.overall_score),
            confidence=max(existing.confidence, signal.confidence),
            summary=signal.summary or existing.summary,
            risk={**existing.risk, **signal.risk},
        )
    return list(merged.values())


def _extract_soft_links_from_comments(comments: list[GithubComment], hard_links: list[str] | None = None) -> list[str]:
    merged: list[str] = []
    for comment in comments:
        body = (comment.body or "").strip()
        if not body:
            continue
        merged = _merge_issue_refs(merged, _extract_soft_linked_issues(body, hard_links))
    return merged


def _sum_issue_reactions(reactions: dict[str, Any] | None) -> int:
    if not reactions:
        return 0
    total = 0
    for key, value in reactions.items():
        if key == "url":
            continue
        if isinstance(value, int):
            total += value
    return total


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
