"""Process and enrich commands over stored entities."""

from __future__ import annotations

import argparse
import logging
from typing import cast

from carapace.commands.common import (
    CommandRuntime,
    build_engine,
    load_config,
    load_stored_entities,
    maybe_apply_routing,
    validate_repo_path_if_needed,
)
from carapace.reporting import write_report_bundle
from carapace.storage.base import StorageBackend

logger = logging.getLogger(__name__)


def run_process(args: argparse.Namespace, *, runtime: CommandRuntime) -> int:
    validate_repo_path_if_needed(args)
    config = load_config(args)
    if config.storage.backend != "sqlite":
        raise ValueError("process currently requires storage.backend=sqlite")

    storage = runtime.storage_cls(config.storage.sqlite_path)
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

    entities = load_stored_entities(
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
        entities, enriched_count = runtime.enrich_entities(
            args=args,
            config=config,
            storage=storage,
            entities=entities,
            connector_factory=runtime.source_connector_cls,
        )
        logger.info("Enrichment finished for %s PRs before scan", enriched_count)

    entities = [entity for entity in entities if (config.ingest.include_closed or entity.state == "open") and (config.ingest.include_drafts or not entity.is_draft)]
    logger.info("Entities after post-enrichment state filtering: %s", len(entities))

    engine = build_engine(config, storage=cast(StorageBackend, storage))
    report = engine.scan_entities(entities)
    write_report_bundle(
        report,
        args.output_dir,
        entities=entities,
        include_singleton_orphans=args.report_include_singletons,
    )
    maybe_apply_routing(args, entities, report, runtime=runtime)

    logger.info("Process complete: processed=%s clusters=%s", report.processed_entities, len(report.clusters))
    return 0


def run_enrich(args: argparse.Namespace, *, runtime: CommandRuntime) -> int:
    validate_repo_path_if_needed(args)
    config = load_config(args)
    if config.storage.backend != "sqlite":
        raise ValueError("enrich currently requires storage.backend=sqlite")

    storage = runtime.storage_cls(config.storage.sqlite_path)
    entities = load_stored_entities(
        storage=storage,
        repo=args.repo,
        config=config,
        entity_kind=args.entity_kind,
        limit=args.limit,
    )
    pr_count = sum(1 for entity in entities if entity.kind.value == "pr")
    issue_count = sum(1 for entity in entities if entity.kind.value == "issue")
    logger.info("Loaded %s entities for enrichment (prs=%s issues=%s)", len(entities), pr_count, issue_count)

    _, enriched_count = runtime.enrich_entities(
        args=args,
        config=config,
        storage=storage,
        entities=entities,
        connector_factory=runtime.source_connector_cls,
    )
    logger.info("Enrich-stored complete: enriched_prs=%s", enriched_count)
    return 0
