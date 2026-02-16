"""Ingest command."""

from __future__ import annotations

import argparse
import logging

from carapace.commands.common import CommandRuntime, load_config, validate_repo_path_if_needed

logger = logging.getLogger(__name__)


def run(args: argparse.Namespace, *, runtime: CommandRuntime) -> int:
    validate_repo_path_if_needed(args)
    config = load_config(args)
    if config.storage.backend != "sqlite":
        raise ValueError("ingest currently requires storage.backend=sqlite")

    storage = runtime.storage_cls(config.storage.sqlite_path)
    connector = runtime.source_connector_cls(
        repo=args.repo,
        gh_bin=args.gh_bin,
        rate_limit_retries=args.gh_rate_limit_retries,
        secondary_backoff_base_seconds=args.gh_secondary_backoff_seconds,
        rate_limit_max_sleep_seconds=args.gh_rate_limit_max_sleep_seconds,
    )
    result = runtime.ingest_loader(
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

