"""CLI entrypoint for offline and GitHub-backed Carapace runs."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import yaml

from carapace.actioning import apply_routing_decisions
from carapace.config import CarapaceConfig, load_effective_config
from carapace.connectors.github_gh import GithubGhSinkConnector, GithubGhSourceConnector
from carapace.enrichment import enrich_entities_if_needed
from carapace.loader import ingest_github_to_sqlite
from carapace.logging_utils import configure_logging
from carapace.models import SourceEntity
from carapace.pipeline import CarapaceEngine
from carapace.repo_validation import validate_repo_path_matches
from carapace.reporting import write_report_bundle
from carapace.storage import SQLiteStorage

logger = logging.getLogger(__name__)


def _default_enrich_workers() -> int:
    raw = os.environ.get("CARAPACE_ENRICH_WORKERS")
    if raw:
        try:
            parsed = int(raw)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    cpu = os.cpu_count() or 8
    return max(4, min(16, cpu))


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

    scan = sub.add_parser("scan", aliases=["run"], help="Run offline scan from JSON entity snapshot")
    scan.add_argument("--input", required=True, help="Path to input JSON array of entities")
    scan.add_argument("--output-dir", default="./carapace-out", help="Output directory")
    _add_common_config_flags(scan)
    _add_report_flags(scan)

    ingest = sub.add_parser("ingest", aliases=["ingest-github"], help="Ingest GitHub entities into SQLite with resumable state")
    ingest.add_argument("--repo", required=True, help="GitHub repo slug, e.g. owner/repo")
    ingest.add_argument("--gh-bin", default="gh", help="Path/name of gh binary")
    ingest.add_argument("--max-prs", type=int, default=0, help="Max PRs to ingest (0 = no explicit cap)")
    ingest.add_argument("--max-issues", type=int, default=0, help="Max issues to ingest (0 = no explicit cap)")
    _add_common_config_flags(ingest)
    _add_repo_validation_flags(ingest)
    _add_github_rate_limit_flags(ingest)

    process = sub.add_parser("process", aliases=["process-stored"], help="Process previously ingested entities from SQLite")
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enriching in minimal mode, also fetch mergeable/approvals/check-status review signals (extra API calls)",
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
        default=_default_enrich_workers(),
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

    enrich = sub.add_parser("enrich", aliases=["enrich-stored"], help="Enrich stored PR details in SQLite without running scan")
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enriching in minimal mode, also fetch mergeable/approvals/check-status review signals (extra API calls)",
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
        default=_default_enrich_workers(),
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

    audit = sub.add_parser("audit", aliases=["db-audit"], help="Show ingest DB audit and integrity summary for a repo")
    audit.add_argument("--repo", required=True, help="GitHub repo slug, e.g. owner/repo")
    audit.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary",
    )
    _add_common_config_flags(audit)
    _add_repo_validation_flags(audit)

    serve_ui = sub.add_parser("serve", aliases=["serve-ui"], help="Run lightweight graph UI/API over stored SQLite data")
    serve_ui.add_argument("--repo", help="Default repo slug shown in UI, e.g. owner/repo")
    serve_ui.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve_ui.add_argument("--port", type=int, default=8765, help="Bind port")
    serve_ui.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload (development only)",
    )
    _add_common_config_flags(serve_ui)
    _add_repo_validation_flags(serve_ui)

    scan_github = sub.add_parser("triage", aliases=["scan-github"], help="One-shot GitHub ingest+process")
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
        raise ValueError("ingest currently requires storage.backend=sqlite")

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
    return enrich_entities_if_needed(
        args=args,
        config=config,
        storage=storage,
        entities=entities,
        connector_factory=GithubGhSourceConnector,
    )


def _run_process_stored(args: argparse.Namespace) -> int:
    _validate_repo_path_if_needed(args)
    config = _load_config(args)
    if config.storage.backend != "sqlite":
        raise ValueError("process currently requires storage.backend=sqlite")

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
        raise ValueError("enrich currently requires storage.backend=sqlite")

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
        raise ValueError("audit currently requires storage.backend=sqlite")

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


def _run_serve_ui(args: argparse.Namespace) -> int:
    _validate_repo_path_if_needed(args)
    config = _load_config(args)
    if config.storage.backend != "sqlite":
        raise ValueError("serve currently requires storage.backend=sqlite")

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError("Missing optional UI dependencies. Install with: pip install 'carapace[ui]'") from exc

    logger.info("Starting UI on http://%s:%s (repo=%s)", args.host, args.port, args.repo or "auto")
    if args.reload:
        os.environ["CARAPACE_REPO_PATH"] = str(args.repo_path)
        uvicorn.run(
            "carapace.webapp:create_app_from_env",
            host=args.host,
            port=args.port,
            reload=True,
            factory=True,
            log_level=args.log_level.lower(),
        )
    else:
        from carapace.webapp import create_app

        app = create_app(config)
        uvicorn.run(app, host=args.host, port=args.port, reload=False, log_level=args.log_level.lower())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    alias_to_canonical = {
        "run": "scan",
        "ingest-github": "ingest",
        "process-stored": "process",
        "enrich-stored": "enrich",
        "scan-github": "triage",
        "db-audit": "audit",
        "serve-ui": "serve",
    }
    args.command = alias_to_canonical.get(args.command, args.command)
    configure_logging(args.log_level)

    command_handlers = {
        "scan": _run_scan,
        "ingest": _run_ingest_github,
        "process": _run_process_stored,
        "enrich": _run_enrich_stored,
        "triage": _run_scan_github,
        "audit": _run_db_audit,
        "serve": _run_serve_ui,
    }
    handler = command_handlers.get(args.command)
    if handler is not None:
        return handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
