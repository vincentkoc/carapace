"""PR enrichment workflow for stored entities."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime, timedelta
from typing import Any

from carapace.config import CarapaceConfig
from carapace.connectors.github_gh import GithubGhSourceConnector, GithubRateLimitError
from carapace.models import SourceEntity
from carapace.storage import SQLiteStorage

logger = logging.getLogger(__name__)


def enrich_entities_if_needed(
    *,
    args: argparse.Namespace,
    config: CarapaceConfig,
    storage: SQLiteStorage,
    entities: list[SourceEntity],
    connector_factory: Callable[..., Any] = GithubGhSourceConnector,
) -> tuple[list[SourceEntity], int]:
    connector = connector_factory(
        repo=args.repo,
        gh_bin=args.gh_bin,
        rate_limit_retries=args.gh_rate_limit_retries,
        secondary_backoff_base_seconds=args.gh_secondary_backoff_seconds,
        rate_limit_max_sleep_seconds=args.gh_rate_limit_max_sleep_seconds,
    )
    watermarks = storage.get_enrichment_watermarks(args.repo, kind="pr")
    target_enrich_level = args.enrich_mode
    if args.enrich_mode == "minimal" and args.enrich_simple_scores:
        target_enrich_level = "minimal+scores"
    targets: list[tuple[int, SourceEntity]] = []
    skipped_recent = 0
    recent_cutoff: datetime | None = None
    if config.low_pass.ignore_recent_pr_hours is not None:
        recent_cutoff = datetime.now(UTC) - timedelta(hours=config.low_pass.ignore_recent_pr_hours)
    for idx, entity in enumerate(entities):
        if entity.kind.value != "pr":
            continue
        if recent_cutoff is not None and entity.updated_at >= recent_cutoff:
            skipped_recent += 1
            continue
        wm = watermarks.get(entity.id, {})
        enriched_for_updated_at = wm.get("enriched_for_updated_at")
        level = wm.get("enrich_level")
        same_version = enriched_for_updated_at == entity.updated_at.isoformat()
        if args.enrich_mode == "minimal":
            needs = (not same_version) or (len(entity.changed_files) == 0) or (level != target_enrich_level)
            if args.enrich_simple_scores:
                needs = needs or (entity.ci_status.value == "unknown") or (entity.mergeable is None)
        else:
            needs = (not same_version) or (level != "full") or (len(entity.changed_files) == 0) or (entity.ci_status.value == "unknown")
        if needs:
            targets.append((idx, entity))

    if not targets:
        logger.info("No PR entities need enrichment for repo=%s", args.repo)
        return entities, 0

    if skipped_recent:
        logger.info(
            "Skipped %s recent PRs from enrichment using ignore_recent_pr_hours=%s",
            skipped_recent,
            config.low_pass.ignore_recent_pr_hours,
        )
    logger.info(
        "Enriching %s PR entities with missing details (mode=%s workers=%s)",
        len(targets),
        args.enrich_mode,
        args.enrich_workers,
    )

    enriched_updates: list[SourceEntity] = []
    state_updates: list[SourceEntity] = []
    rate_limited = False
    rate_limit_reset_at: datetime | None = None
    rate_limit_error_count = 0

    def _enrich_one(target: tuple[int, SourceEntity]) -> tuple[int, SourceEntity, bool, datetime | None]:
        entity_index, entity_obj = target
        try:
            enriched_entity = connector.enrich_entity(
                entity_obj,
                include_comments=args.enrich_comments,
                mode=args.enrich_mode,
                include_simple_scores=args.enrich_simple_scores,
            )
            return entity_index, enriched_entity, True, None
        except GithubRateLimitError as exc:
            return entity_index, entity_obj, False, exc.reset_at
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            message = str(exc)
            if "404" in message and entity_obj.kind.value == "pr":
                logger.debug("PR %s resolved as closed during enrichment", entity_obj.id)
                return entity_index, entity_obj.model_copy(update={"state": "closed"}), False, None
            logger.warning("Failed to enrich %s: %s", entity_obj.id, exc)
            return entity_index, entity_obj, False, None

    completed = 0
    total_targets = len(targets)
    target_by_index = {index: entity for index, entity in targets}
    progress_every = max(1, args.enrich_progress_every)
    flush_every = max(1, args.enrich_flush_every)
    heartbeat_seconds = max(1.0, args.enrich_heartbeat_seconds)
    started_at = time.monotonic()
    last_progress_log = started_at

    def _log_progress(force: bool = False) -> None:
        nonlocal last_progress_log
        now = time.monotonic()
        if not force and (now - last_progress_log) < heartbeat_seconds and completed % progress_every != 0:
            return
        elapsed = max(0.001, now - started_at)
        rate = completed / elapsed
        remaining = total_targets - completed
        eta_seconds = int(remaining / rate) if rate > 0 else -1
        logger.info(
            "Enrichment progress: %s/%s (%.1f%%) rate=%.2f/s eta=%ss",
            completed,
            total_targets,
            (completed * 100.0) / max(1, total_targets),
            rate,
            eta_seconds if eta_seconds >= 0 else "?",
        )
        last_progress_log = now

    def _flush_updates() -> None:
        if enriched_updates:
            storage.upsert_ingest_entities(
                args.repo,
                list(enriched_updates),
                source="enrichment",
                enrich_level=target_enrich_level,
            )
            enriched_updates.clear()
        if state_updates:
            storage.upsert_ingest_entities(args.repo, list(state_updates), source="ingest")
            state_updates.clear()

    max_workers = max(1, args.enrich_workers)
    target_iter = iter(targets)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending = set()

        def _submit_next() -> bool:
            try:
                target = next(target_iter)
            except StopIteration:
                return False
            pending.add(pool.submit(_enrich_one, target))
            return True

        for _ in range(max_workers):
            if not _submit_next():
                break

        while pending:
            done, _ = wait(pending, timeout=heartbeat_seconds, return_when=FIRST_COMPLETED)
            if not done:
                _log_progress(force=True)
                continue
            pending.difference_update(done)
            for future in done:
                entity_index, enriched_entity, enrichment_success, limited_until = future.result()
                original_entity = target_by_index.get(entity_index, entities[entity_index])
                if enrichment_success:
                    enriched_updates.append(enriched_entity)
                elif enriched_entity.state != original_entity.state:
                    state_updates.append(enriched_entity)
                elif limited_until is not None:
                    rate_limited = True
                    rate_limit_error_count += 1
                    if rate_limit_reset_at is None or limited_until > rate_limit_reset_at:
                        rate_limit_reset_at = limited_until

                entities[entity_index] = enriched_entity
                completed += 1

                if completed % flush_every == 0:
                    _flush_updates()

                _log_progress(force=completed == total_targets)

            if rate_limited:
                # Drop queued futures that haven't started yet, then drain in-flight work.
                to_cancel = list(pending)
                cancelled = 0
                for future in to_cancel:
                    if future.cancel():
                        cancelled += 1
                pending = {future for future in pending if not future.cancelled()}
                if cancelled:
                    logger.warning("Cancelled %s queued enrichment tasks due to GitHub rate limiting", cancelled)
                continue

            while len(pending) < max_workers:
                if not _submit_next():
                    break

    _flush_updates()
    _log_progress(force=True)
    if rate_limited:
        remaining = max(0, total_targets - completed)
        if rate_limit_reset_at:
            now = datetime.now(UTC)
            wait_seconds = max(0, int((rate_limit_reset_at - now).total_seconds()))
            logger.warning(
                "GitHub rate limit hit during enrichment: completed=%s/%s remaining=%s reset_at=%s (~%ss). Re-run process/enrich later to continue.",
                completed,
                total_targets,
                remaining,
                rate_limit_reset_at.isoformat(),
                wait_seconds,
            )
        else:
            logger.warning(
                "GitHub rate limit hit during enrichment: completed=%s/%s remaining=%s. Re-run process/enrich later to continue.",
                completed,
                total_targets,
                remaining,
            )
        if rate_limit_error_count > 1:
            logger.warning("Encountered %s rate-limit enrichment errors in this run", rate_limit_error_count)
    logger.info("Persisted enriched entity data back to SQLite")
    return entities, completed
