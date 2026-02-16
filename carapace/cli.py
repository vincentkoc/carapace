"""CLI entrypoint for offline and GitHub-backed Carapace runs."""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from carapace.actioning import apply_routing_decisions
from carapace.config import CarapaceConfig, load_effective_config
from carapace.connectors.github_gh import GithubGhSinkConnector, GithubGhSourceConnector, GithubRateLimitError
from carapace.loader import ingest_github_to_sqlite
from carapace.logging_utils import configure_logging
from carapace.models import SourceEntity
from carapace.pipeline import CarapaceEngine
from carapace.repo_validation import validate_repo_path_matches
from carapace.reporting import write_report_bundle
from carapace.storage import SQLiteStorage

logger = logging.getLogger(__name__)


def _load_json_entities(path: Path) -> list[SourceEntity]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("Input must be a JSON array of entities")
    return [SourceEntity.model_validate(item) for item in raw]


def _load_yaml_dict(path: str | None) -> dict | None:
    if not path:
        return None
    data = yaml.safe_load(Path(path).read_text())
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must decode to a mapping")
    return data


def _load_config(args: argparse.Namespace) -> CarapaceConfig:
    return load_effective_config(
        repo_path=args.repo_path,
        org_defaults=_load_yaml_dict(args.org_config),
        system_defaults=_load_yaml_dict(args.system_config),
        runtime_override=_load_yaml_dict(args.runtime_override),
    )


def _build_engine(config: CarapaceConfig, storage: SQLiteStorage | None = None) -> CarapaceEngine:
    return CarapaceEngine(config=config, storage=storage)


def _add_common_config_flags(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument("--repo-path", default=".", help="Repository root path")
    cmd.add_argument("--org-config", help="Optional org defaults YAML")
    cmd.add_argument("--system-config", help="Optional system defaults YAML")
    cmd.add_argument("--runtime-override", help="Optional runtime override YAML")


def _add_repo_validation_flags(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument(
        "--skip-repo-path-check",
        action="store_true",
        help="Skip validation that repo-path origin matches --repo",
    )


def _add_report_flags(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument(
        "--report-include-singletons",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include singleton_orphan clusters in triage_report.md detail section",
    )


def _add_github_rate_limit_flags(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument(
        "--gh-rate-limit-retries",
        type=int,
        default=2,
        help="Retries per GitHub API call when rate-limited",
    )
    cmd.add_argument(
        "--gh-secondary-backoff-seconds",
        type=float,
        default=5.0,
        help="Base backoff for secondary limits (exponential per retry)",
    )
    cmd.add_argument(
        "--gh-rate-limit-max-sleep-seconds",
        type=float,
        default=90.0,
        help="Maximum automatic sleep before surfacing a rate-limit failure",
    )


def _validate_repo_path_if_needed(args: argparse.Namespace) -> None:
    if getattr(args, "skip_repo_path_check", False):
        return
    validate_repo_path_matches(args.repo_path, args.repo)


def _entity_number_resolver(entities: list[SourceEntity]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for entity in entities:
        if entity.number is not None:
            mapping[entity.id] = entity.number
    return mapping


def _maybe_apply_routing(args: argparse.Namespace, entities: list[SourceEntity], report) -> None:
    if not getattr(args, "apply_routing", False):
        return

    id_to_number = _entity_number_resolver(entities)
    sink = GithubGhSinkConnector(
        repo=args.repo,
        gh_bin=args.gh_bin,
        rate_limit_retries=args.gh_rate_limit_retries,
        secondary_backoff_base_seconds=args.gh_secondary_backoff_seconds,
        rate_limit_max_sleep_seconds=args.gh_rate_limit_max_sleep_seconds,
        entity_number_resolver=lambda entity_id: id_to_number[entity_id],
        dry_run=not args.live_actions,
    )
    apply_routing_decisions(report, sink)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Carapace triage engine")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Run offline scan from JSON entity snapshot")
    scan.add_argument("--input", required=True, help="Path to input JSON array of entities")
    scan.add_argument("--output-dir", default="./carapace-out", help="Output directory")
    _add_common_config_flags(scan)
    _add_report_flags(scan)

    ingest = sub.add_parser("ingest-github", help="Ingest GitHub entities into SQLite with resumable state")
    ingest.add_argument("--repo", required=True, help="GitHub repo slug, e.g. owner/repo")
    ingest.add_argument("--gh-bin", default="gh", help="Path/name of gh binary")
    ingest.add_argument("--max-prs", type=int, default=0, help="Max PRs to ingest (0 = no explicit cap)")
    ingest.add_argument("--max-issues", type=int, default=0, help="Max issues to ingest (0 = no explicit cap)")
    _add_common_config_flags(ingest)
    _add_repo_validation_flags(ingest)
    _add_github_rate_limit_flags(ingest)

    process = sub.add_parser("process-stored", help="Process previously ingested entities from SQLite")
    process.add_argument("--repo", required=True, help="GitHub repo slug, e.g. owner/repo")
    process.add_argument("--gh-bin", default="gh", help="Path/name of gh binary")
    process.add_argument("--output-dir", default="./carapace-out", help="Output directory")
    process.add_argument(
        "--entity-kind",
        choices=["all", "pr", "issue"],
        default="all",
        help="Entity kind scope for loading/processing",
    )
    process.add_argument("--limit", type=int, default=0, help="Optional processing cap (0 = all loaded entities)")
    process.add_argument(
        "--enrich-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enrich missing PR details from GitHub before processing",
    )
    process.add_argument(
        "--enrich-comments",
        action="store_true",
        help="When enriching, also fetch issue comments (slower; needed for external review signal extraction)",
    )
    process.add_argument(
        "--enrich-simple-scores",
        action="store_true",
        help="When enriching in minimal mode, also fetch mergeable/check-status review signals (extra API calls)",
    )
    process.add_argument(
        "--enrich-mode",
        choices=["minimal", "full"],
        default="minimal",
        help="Enrichment mode: minimal (files/hunks only) or full (files+reviews+ci+optional comments)",
    )
    process.add_argument(
        "--enrich-workers",
        type=int,
        default=6,
        help="Parallel workers used for enrichment API calls",
    )
    process.add_argument(
        "--enrich-progress-every",
        type=int,
        default=100,
        help="Log enrichment progress every N completed PRs",
    )
    process.add_argument(
        "--enrich-flush-every",
        type=int,
        default=50,
        help="Persist enrichment results every N completed PRs",
    )
    process.add_argument(
        "--enrich-heartbeat-seconds",
        type=float,
        default=10.0,
        help="Emit heartbeat progress logs at this interval while enrichment is running",
    )
    process.add_argument("--apply-routing", action="store_true", help="Apply labels/comments using routing decisions")
    process.add_argument(
        "--live-actions",
        action="store_true",
        help="When used with --apply-routing, perform real GitHub writes (default is dry-run)",
    )
    _add_common_config_flags(process)
    _add_repo_validation_flags(process)
    _add_github_rate_limit_flags(process)
    _add_report_flags(process)

    enrich = sub.add_parser("enrich-stored", help="Enrich stored PR details in SQLite without running scan")
    enrich.add_argument("--repo", required=True, help="GitHub repo slug, e.g. owner/repo")
    enrich.add_argument("--gh-bin", default="gh", help="Path/name of gh binary")
    enrich.add_argument(
        "--entity-kind",
        choices=["all", "pr"],
        default="pr",
        help="Entity kind scope for loading before enrichment (issues are ignored by enrichment)",
    )
    enrich.add_argument("--limit", type=int, default=0, help="Optional load cap (0 = all loaded entities)")
    enrich.add_argument(
        "--enrich-comments",
        action="store_true",
        help="When enriching in full mode, also fetch issue comments",
    )
    enrich.add_argument(
        "--enrich-simple-scores",
        action="store_true",
        help="When enriching in minimal mode, also fetch mergeable/check-status review signals (extra API calls)",
    )
    enrich.add_argument(
        "--enrich-mode",
        choices=["minimal", "full"],
        default="minimal",
        help="Enrichment mode: minimal (files/hunks only) or full",
    )
    enrich.add_argument(
        "--enrich-workers",
        type=int,
        default=6,
        help="Parallel workers used for enrichment API calls",
    )
    enrich.add_argument(
        "--enrich-progress-every",
        type=int,
        default=100,
        help="Log enrichment progress every N completed PRs",
    )
    enrich.add_argument(
        "--enrich-flush-every",
        type=int,
        default=50,
        help="Persist enrichment results every N completed PRs",
    )
    enrich.add_argument(
        "--enrich-heartbeat-seconds",
        type=float,
        default=10.0,
        help="Emit heartbeat progress logs at this interval while enrichment is running",
    )
    _add_common_config_flags(enrich)
    _add_repo_validation_flags(enrich)
    _add_github_rate_limit_flags(enrich)

    audit = sub.add_parser("db-audit", help="Show ingest DB audit and integrity summary for a repo")
    audit.add_argument("--repo", required=True, help="GitHub repo slug, e.g. owner/repo")
    audit.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary",
    )
    _add_common_config_flags(audit)
    _add_repo_validation_flags(audit)

    scan_github = sub.add_parser("scan-github", help="One-shot GitHub ingest+process")
    scan_github.add_argument("--repo", required=True, help="GitHub repo slug, e.g. owner/repo")
    scan_github.add_argument("--gh-bin", default="gh", help="Path/name of gh binary")
    scan_github.add_argument("--max-prs", type=int, default=200, help="Max open PRs to ingest")
    scan_github.add_argument(
        "--include-issues",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override ingest.include_issues for one-shot scan",
    )
    scan_github.add_argument("--max-issues", type=int, default=200, help="Max open issues to ingest")
    scan_github.add_argument("--output-dir", default="./carapace-out", help="Output directory")
    scan_github.add_argument("--save-input-json", action="store_true", help="Write fetched entities.json snapshot")
    scan_github.add_argument("--apply-routing", action="store_true", help="Apply labels/comments using routing decisions")
    scan_github.add_argument(
        "--live-actions",
        action="store_true",
        help="When used with --apply-routing, perform real GitHub writes (default is dry-run)",
    )
    _add_common_config_flags(scan_github)
    _add_repo_validation_flags(scan_github)
    _add_github_rate_limit_flags(scan_github)
    _add_report_flags(scan_github)

    return parser


def _run_scan(args: argparse.Namespace) -> int:
    entities = _load_json_entities(Path(args.input))
    config = _load_config(args)
    engine = _build_engine(config)
    report = engine.scan_entities(entities)
    write_report_bundle(
        report,
        args.output_dir,
        entities=entities,
        include_singleton_orphans=args.report_include_singletons,
    )
    logger.info("Scan complete: processed=%s clusters=%s", report.processed_entities, len(report.clusters))
    return 0


def _run_ingest_github(args: argparse.Namespace) -> int:
    _validate_repo_path_if_needed(args)
    config = _load_config(args)
    if config.storage.backend != "sqlite":
        raise ValueError("ingest-github currently requires storage.backend=sqlite")

    storage = SQLiteStorage(config.storage.sqlite_path)
    connector = GithubGhSourceConnector(
        repo=args.repo,
        gh_bin=args.gh_bin,
        rate_limit_retries=args.gh_rate_limit_retries,
        secondary_backoff_base_seconds=args.gh_secondary_backoff_seconds,
        rate_limit_max_sleep_seconds=args.gh_rate_limit_max_sleep_seconds,
    )

    result = ingest_github_to_sqlite(
        connector,
        storage,
        repo=args.repo,
        ingest_cfg=config.ingest,
        max_prs=args.max_prs,
        max_issues=args.max_issues,
    )
    logger.info(
        "Ingest done repo=%s prs=%s issues=%s pages(pr=%s issues=%s)",
        result.repo,
        result.prs_ingested,
        result.issues_ingested,
        result.pr_pages,
        result.issue_pages,
    )
    return 0


def _kind_filter_from_arg(entity_kind: str) -> str | None:
    if entity_kind == "all":
        return None
    return entity_kind


def _load_stored_entities(
    *,
    storage: SQLiteStorage,
    repo: str,
    config: CarapaceConfig,
    entity_kind: str,
    limit: int,
) -> list[SourceEntity]:
    entities = storage.load_ingested_entities(
        repo=repo,
        include_closed=config.ingest.include_closed,
        include_drafts=config.ingest.include_drafts,
        kind=_kind_filter_from_arg(entity_kind),
    )
    if limit > 0:
        entities = entities[:limit]
    return entities


def _enrich_entities_if_needed(
    *,
    args: argparse.Namespace,
    config: CarapaceConfig,
    storage: SQLiteStorage,
    entities: list[SourceEntity],
) -> tuple[list[SourceEntity], int]:
    connector = GithubGhSourceConnector(
        repo=args.repo,
        gh_bin=args.gh_bin,
        rate_limit_retries=args.gh_rate_limit_retries,
        secondary_backoff_base_seconds=args.gh_secondary_backoff_seconds,
        rate_limit_max_sleep_seconds=args.gh_rate_limit_max_sleep_seconds,
    )
    watermarks = storage.get_enrichment_watermarks(args.repo, kind="pr")
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
            needs = (not same_version) or (len(entity.changed_files) == 0)
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
                enrich_level=args.enrich_mode,
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
                "GitHub rate limit hit during enrichment: completed=%s/%s remaining=%s reset_at=%s (~%ss). Re-run process-stored/enrich-stored to continue.",
                completed,
                total_targets,
                remaining,
                rate_limit_reset_at.isoformat(),
                wait_seconds,
            )
        else:
            logger.warning(
                "GitHub rate limit hit during enrichment: completed=%s/%s remaining=%s. Re-run process-stored/enrich-stored later to continue.",
                completed,
                total_targets,
                remaining,
            )
        if rate_limit_error_count > 1:
            logger.warning("Encountered %s rate-limit enrichment errors in this run", rate_limit_error_count)
    logger.info("Persisted enriched entity data back to SQLite")
    return entities, completed


def _run_process_stored(args: argparse.Namespace) -> int:
    _validate_repo_path_if_needed(args)
    config = _load_config(args)
    if config.storage.backend != "sqlite":
        raise ValueError("process-stored currently requires storage.backend=sqlite")

    storage = SQLiteStorage(config.storage.sqlite_path)
    state = storage.get_ingest_state(args.repo)
    if not state["completed"]:
        logger.warning(
            "Ingest state for %s is incomplete (phase=%s, pr_next_page=%s, issue_next_page=%s). Processing partial snapshot.",
            args.repo,
            state["phase"],
            state["pr_next_page"],
            state["issue_next_page"],
        )

    quality = storage.ingest_quality_stats(args.repo)
    pr_quality = storage.ingest_quality_stats(args.repo, kind="pr")
    issue_quality = storage.ingest_quality_stats(args.repo, kind="issue")
    logger.info(
        "Ingest quality stats: total=%s missing_changed_files=%s missing_diff_hunks=%s ci_unknown=%s enriched_rows=%s",
        quality["total"],
        quality["missing_changed_files"],
        quality["missing_diff_hunks"],
        quality["ci_unknown"],
        quality["enriched_rows"],
    )
    logger.info(
        "PR quality stats: total=%s missing_changed_files=%s missing_diff_hunks=%s ci_unknown=%s enriched_rows=%s",
        pr_quality["total"],
        pr_quality["missing_changed_files"],
        pr_quality["missing_diff_hunks"],
        pr_quality["ci_unknown"],
        pr_quality["enriched_rows"],
    )
    logger.info(
        "Issue quality stats: total=%s missing_changed_files=%s missing_diff_hunks=%s ci_unknown=%s enriched_rows=%s",
        issue_quality["total"],
        issue_quality["missing_changed_files"],
        issue_quality["missing_diff_hunks"],
        issue_quality["ci_unknown"],
        issue_quality["enriched_rows"],
    )

    entities = _load_stored_entities(
        storage=storage,
        repo=args.repo,
        config=config,
        entity_kind=args.entity_kind,
        limit=args.limit,
    )

    pr_count = sum(1 for entity in entities if entity.kind.value == "pr")
    issue_count = sum(1 for entity in entities if entity.kind.value == "issue")
    logger.info("Loaded %s ingested entities for processing (prs=%s issues=%s)", len(entities), pr_count, issue_count)

    if args.enrich_missing:
        entities, enriched_count = _enrich_entities_if_needed(
            args=args,
            config=config,
            storage=storage,
            entities=entities,
        )
        logger.info("Enrichment finished for %s PRs before scan", enriched_count)

    # Re-apply state filters after enrichment may have changed PR state (e.g. closed during processing).
    entities = [entity for entity in entities if (config.ingest.include_closed or entity.state == "open") and (config.ingest.include_drafts or not entity.is_draft)]
    logger.info("Entities after post-enrichment state filtering: %s", len(entities))

    engine = _build_engine(config, storage=storage)
    report = engine.scan_entities(entities)
    write_report_bundle(
        report,
        args.output_dir,
        entities=entities,
        include_singleton_orphans=args.report_include_singletons,
    )
    _maybe_apply_routing(args, entities, report)

    logger.info("Process complete: processed=%s clusters=%s", report.processed_entities, len(report.clusters))
    return 0


def _run_enrich_stored(args: argparse.Namespace) -> int:
    _validate_repo_path_if_needed(args)
    config = _load_config(args)
    if config.storage.backend != "sqlite":
        raise ValueError("enrich-stored currently requires storage.backend=sqlite")

    storage = SQLiteStorage(config.storage.sqlite_path)
    entities = _load_stored_entities(
        storage=storage,
        repo=args.repo,
        config=config,
        entity_kind=args.entity_kind,
        limit=args.limit,
    )
    pr_count = sum(1 for entity in entities if entity.kind.value == "pr")
    issue_count = sum(1 for entity in entities if entity.kind.value == "issue")
    logger.info("Loaded %s entities for enrichment (prs=%s issues=%s)", len(entities), pr_count, issue_count)

    _, enriched_count = _enrich_entities_if_needed(
        args=args,
        config=config,
        storage=storage,
        entities=entities,
    )
    logger.info("Enrich-stored complete: enriched_prs=%s", enriched_count)
    return 0


def _run_scan_github(args: argparse.Namespace) -> int:
    _validate_repo_path_if_needed(args)
    config = _load_config(args)

    connector = GithubGhSourceConnector(
        repo=args.repo,
        gh_bin=args.gh_bin,
        rate_limit_retries=args.gh_rate_limit_retries,
        secondary_backoff_base_seconds=args.gh_secondary_backoff_seconds,
        rate_limit_max_sleep_seconds=args.gh_rate_limit_max_sleep_seconds,
    )
    include_issues = config.ingest.include_issues if args.include_issues is None else args.include_issues
    entities = connector.fetch_open_entities(
        max_prs=args.max_prs,
        include_issues=include_issues,
        max_issues=args.max_issues,
        include_drafts=config.ingest.include_drafts,
        include_closed=config.ingest.include_closed,
        enrich_pr_details=config.ingest.enrich_pr_details,
        enrich_issue_comments=config.ingest.enrich_issue_comments,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_input_json:
        (out_dir / "entities.json").write_text(json.dumps([entity.model_dump(mode="json") for entity in entities], indent=2))

    engine = _build_engine(config)
    report = engine.scan_entities(entities)
    write_report_bundle(
        report,
        out_dir,
        entities=entities,
        include_singleton_orphans=args.report_include_singletons,
    )
    _maybe_apply_routing(args, entities, report)
    logger.info("One-shot scan complete: processed=%s clusters=%s", report.processed_entities, len(report.clusters))
    return 0


def _run_db_audit(args: argparse.Namespace) -> int:
    _validate_repo_path_if_needed(args)
    config = _load_config(args)
    if config.storage.backend != "sqlite":
        raise ValueError("db-audit currently requires storage.backend=sqlite")

    storage = SQLiteStorage(config.storage.sqlite_path)
    state = storage.get_ingest_state(args.repo)
    summary = storage.ingest_audit_summary(args.repo)
    quality = storage.ingest_quality_stats(args.repo)
    pr_quality = storage.ingest_quality_stats(args.repo, kind="pr")
    issue_quality = storage.ingest_quality_stats(args.repo, kind="issue")

    payload = {
        "repo": args.repo,
        "db_path": str(config.storage.sqlite_path),
        "ingest_state": state,
        "summary": summary,
        "quality": {
            "all": quality,
            "pr": pr_quality,
            "issue": issue_quality,
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Repo: {args.repo}")
    print(f"DB: {config.storage.sqlite_path}")
    print("Ingest state: completed={completed} phase={phase} pr_next_page={pr_next_page} issue_next_page={issue_next_page}".format(**state))
    print(
        "Entities: total={total} prs={prs} issues={issues}".format(
            total=summary["total"],
            prs=summary["by_kind"].get("pr", 0),
            issues=summary["by_kind"].get("issue", 0),
        )
    )
    print(
        "PRs: open={open} closed={closed} drafts={drafts} enriched={enriched}".format(
            open=summary["by_kind_state"].get("pr", {}).get("open", 0),
            closed=summary["by_kind_state"].get("pr", {}).get("closed", 0),
            drafts=summary["draft_prs"],
            enriched=summary["enriched_prs"],
        )
    )
    print(
        "Issues: open={open} closed={closed}".format(
            open=summary["by_kind_state"].get("issue", {}).get("open", 0),
            closed=summary["by_kind_state"].get("issue", {}).get("closed", 0),
        )
    )
    print("Quality(all): missing_changed_files={missing_changed_files} missing_diff_hunks={missing_diff_hunks} ci_unknown={ci_unknown} enriched_rows={enriched_rows}".format(**quality))
    print("Quality(pr): missing_changed_files={missing_changed_files} missing_diff_hunks={missing_diff_hunks} ci_unknown={ci_unknown} enriched_rows={enriched_rows}".format(**pr_quality))
    print("Quality(issue): missing_changed_files={missing_changed_files} missing_diff_hunks={missing_diff_hunks} ci_unknown={ci_unknown} enriched_rows={enriched_rows}".format(**issue_quality))
    print(
        "Integrity: kind_id_prefix_mismatch={kind_id_prefix_mismatch} kind_payload_mismatch={kind_payload_mismatch} repo_payload_mismatch={repo_payload_mismatch} entity_number_mismatch={entity_number_mismatch}".format(
            **summary["integrity"]
        )
    )
    levels = summary["enrich_levels"]
    print("Enrich levels (PR): " + ", ".join(f"{level}={count}" for level, count in sorted(levels.items())))
    print(
        "Fingerprint cache rows: total={total} by_model={models}".format(
            total=summary["fingerprint_cache_rows"],
            models=", ".join(f"{model}={count}" for model, count in sorted(summary["fingerprint_cache_by_model"].items())),
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    if args.command == "scan":
        return _run_scan(args)
    if args.command == "ingest-github":
        return _run_ingest_github(args)
    if args.command == "process-stored":
        return _run_process_stored(args)
    if args.command == "enrich-stored":
        return _run_enrich_stored(args)
    if args.command == "scan-github":
        return _run_scan_github(args)
    if args.command == "db-audit":
        return _run_db_audit(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
