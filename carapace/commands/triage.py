"""One-shot triage command."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from carapace.commands.common import CommandRuntime, build_engine, load_config, maybe_apply_routing, validate_repo_path_if_needed
from carapace.reporting import write_report_bundle

logger = logging.getLogger(__name__)


def run(args: argparse.Namespace, *, runtime: CommandRuntime) -> int:
    validate_repo_path_if_needed(args)
    config = load_config(args)

    connector = runtime.source_connector_cls(
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

    engine = build_engine(config)
    report = engine.scan_entities(entities)
    write_report_bundle(
        report,
        out_dir,
        entities=entities,
        include_singleton_orphans=args.report_include_singletons,
    )
    maybe_apply_routing(args, entities, report, runtime=runtime)
    logger.info("One-shot scan complete: processed=%s clusters=%s", report.processed_entities, len(report.clusters))
    return 0

