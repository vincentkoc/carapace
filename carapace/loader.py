"""Stateful ingestion loader utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from carapace.config import IngestConfig
from carapace.connectors.github_gh import GithubGhSourceConnector
from carapace.storage import SQLiteStorage

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    repo: str
    prs_ingested: int
    issues_ingested: int
    pr_pages: int
    issue_pages: int


def ingest_github_to_sqlite(
    connector: GithubGhSourceConnector,
    storage: SQLiteStorage,
    *,
    repo: str,
    ingest_cfg: IngestConfig,
    max_prs: int,
    max_issues: int,
) -> IngestResult:
    state = storage.get_ingest_state(repo)

    # Resume only unfinished runs. Completed runs restart from page 1 for fresh reconciliation.
    should_resume = ingest_cfg.resume and not bool(state["completed"])
    pr_page = state["pr_next_page"] if should_resume else 1
    issue_page = state["issue_next_page"] if should_resume else 1

    pr_state = "all" if ingest_cfg.include_closed else "open"
    issue_state = "all" if ingest_cfg.include_closed else "open"

    total_prs = 0
    total_issues = 0
    pr_pages = 0
    issue_pages = 0
    seen_pr_ids: set[str] = set()
    seen_issue_ids: set[str] = set()
    pr_exhausted = False
    issue_exhausted = False

    logger.info("Starting ingest for %s (resume=%s, from_pr_page=%s, from_issue_page=%s)", repo, should_resume, pr_page, issue_page)

    while True:
        if max_prs and total_prs >= max_prs:
            break
        page_entities = connector.fetch_pull_page(
            page=pr_page,
            per_page=ingest_cfg.page_size,
            state=pr_state,
            include_drafts=ingest_cfg.include_drafts,
            enrich_details=ingest_cfg.enrich_pr_details,
            enrich_comments=ingest_cfg.enrich_issue_comments,
        )
        if not page_entities:
            pr_exhausted = True
            break

        if max_prs:
            page_entities = page_entities[: max(0, max_prs - total_prs)]
        seen_pr_ids.update(entity.id for entity in page_entities)
        written = storage.upsert_ingest_entities(repo, page_entities)
        total_prs += written
        pr_pages += 1
        pr_page += 1

        storage.save_ingest_state(
            repo,
            pr_next_page=pr_page,
            issue_next_page=issue_page,
            phase="prs",
            completed=False,
        )
        logger.debug("Ingested PR page %s (%s entities, total=%s)", pr_page - 1, written, total_prs)

    if ingest_cfg.include_issues:
        while True:
            if max_issues and total_issues >= max_issues:
                break
            page_entities = connector.fetch_issue_page(
                page=issue_page,
                per_page=ingest_cfg.page_size,
                state=issue_state,
            )
            if not page_entities:
                issue_exhausted = True
                break

            if max_issues:
                page_entities = page_entities[: max(0, max_issues - total_issues)]
            seen_issue_ids.update(entity.id for entity in page_entities)
            written = storage.upsert_ingest_entities(repo, page_entities)
            total_issues += written
            issue_pages += 1
            issue_page += 1

            storage.save_ingest_state(
                repo,
                pr_next_page=pr_page,
                issue_next_page=issue_page,
                phase="issues",
                completed=False,
            )
            logger.debug("Ingested issue page %s (%s entities, total=%s)", issue_page - 1, written, total_issues)

    # Reconcile entity closures only on full refresh from page 1 with open-only fetch.
    if not should_resume and not ingest_cfg.include_closed:
        if pr_exhausted and (max_prs == 0):
            closed_prs = storage.mark_entities_closed_except(repo, kind="pr", seen_entity_ids=seen_pr_ids)
            logger.info("Reconciled PR states: marked %s stale open PRs as closed", closed_prs)
        if ingest_cfg.include_issues and issue_exhausted and (max_issues == 0):
            closed_issues = storage.mark_entities_closed_except(repo, kind="issue", seen_entity_ids=seen_issue_ids)
            logger.info("Reconciled issue states: marked %s stale open issues as closed", closed_issues)

    prs_complete = pr_exhausted and (max_prs == 0)
    issues_complete = (not ingest_cfg.include_issues) or (issue_exhausted and (max_issues == 0))
    completed = prs_complete and issues_complete
    final_phase = "done" if completed else ("issues" if ingest_cfg.include_issues else "prs")

    storage.save_ingest_state(
        repo,
        pr_next_page=pr_page,
        issue_next_page=issue_page,
        phase=final_phase,
        completed=completed,
    )
    logger.info(
        "Ingest completed for %s: prs=%s issues=%s pages(pr=%s,issues=%s) complete=%s",
        repo,
        total_prs,
        total_issues,
        pr_pages,
        issue_pages,
        completed,
    )

    return IngestResult(
        repo=repo,
        prs_ingested=total_prs,
        issues_ingested=total_issues,
        pr_pages=pr_pages,
        issue_pages=issue_pages,
    )
