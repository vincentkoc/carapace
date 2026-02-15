"""CLI entrypoint for offline and GitHub-backed Carapace runs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from carapace.actioning import apply_routing_decisions
from carapace.config import CarapaceConfig, load_effective_config
from carapace.connectors.github_gh import GithubGhSinkConnector, GithubGhSourceConnector
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


def _build_engine(config: CarapaceConfig) -> CarapaceEngine:
    return CarapaceEngine(config=config)


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

    ingest = sub.add_parser("ingest-github", help="Ingest GitHub entities into SQLite with resumable state")
    ingest.add_argument("--repo", required=True, help="GitHub repo slug, e.g. owner/repo")
    ingest.add_argument("--gh-bin", default="gh", help="Path/name of gh binary")
    ingest.add_argument("--max-prs", type=int, default=0, help="Max PRs to ingest (0 = no explicit cap)")
    ingest.add_argument("--max-issues", type=int, default=0, help="Max issues to ingest (0 = no explicit cap)")
    _add_common_config_flags(ingest)
    _add_repo_validation_flags(ingest)

    process = sub.add_parser("process-stored", help="Process previously ingested entities from SQLite")
    process.add_argument("--repo", required=True, help="GitHub repo slug, e.g. owner/repo")
    process.add_argument("--gh-bin", default="gh", help="Path/name of gh binary")
    process.add_argument("--output-dir", default="./carapace-out", help="Output directory")
    process.add_argument("--limit", type=int, default=0, help="Optional processing cap (0 = all loaded entities)")
    process.add_argument("--apply-routing", action="store_true", help="Apply labels/comments using routing decisions")
    process.add_argument(
        "--live-actions",
        action="store_true",
        help="When used with --apply-routing, perform real GitHub writes (default is dry-run)",
    )
    _add_common_config_flags(process)
    _add_repo_validation_flags(process)

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

    return parser


def _run_scan(args: argparse.Namespace) -> int:
    entities = _load_json_entities(Path(args.input))
    config = _load_config(args)
    engine = _build_engine(config)
    report = engine.scan_entities(entities)
    write_report_bundle(report, args.output_dir)
    logger.info("Scan complete: processed=%s clusters=%s", report.processed_entities, len(report.clusters))
    return 0


def _run_ingest_github(args: argparse.Namespace) -> int:
    _validate_repo_path_if_needed(args)
    config = _load_config(args)
    if config.storage.backend != "sqlite":
        raise ValueError("ingest-github currently requires storage.backend=sqlite")

    storage = SQLiteStorage(config.storage.sqlite_path)
    connector = GithubGhSourceConnector(repo=args.repo, gh_bin=args.gh_bin)

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


def _run_process_stored(args: argparse.Namespace) -> int:
    _validate_repo_path_if_needed(args)
    config = _load_config(args)
    if config.storage.backend != "sqlite":
        raise ValueError("process-stored currently requires storage.backend=sqlite")

    storage = SQLiteStorage(config.storage.sqlite_path)
    entities = storage.load_ingested_entities(
        repo=args.repo,
        include_closed=config.ingest.include_closed,
        include_drafts=config.ingest.include_drafts,
    )
    if args.limit > 0:
        entities = entities[: args.limit]

    logger.info("Loaded %s ingested entities for processing", len(entities))
    engine = _build_engine(config)
    report = engine.scan_entities(entities)
    write_report_bundle(report, args.output_dir)
    _maybe_apply_routing(args, entities, report)

    logger.info("Process complete: processed=%s clusters=%s", report.processed_entities, len(report.clusters))
    return 0


def _run_scan_github(args: argparse.Namespace) -> int:
    _validate_repo_path_if_needed(args)
    config = _load_config(args)

    connector = GithubGhSourceConnector(repo=args.repo, gh_bin=args.gh_bin)
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
        (out_dir / "entities.json").write_text(
            json.dumps([entity.model_dump(mode="json") for entity in entities], indent=2)
        )

    engine = _build_engine(config)
    report = engine.scan_entities(entities)
    write_report_bundle(report, out_dir)
    _maybe_apply_routing(args, entities, report)
    logger.info("One-shot scan complete: processed=%s clusters=%s", report.processed_entities, len(report.clusters))
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
    if args.command == "scan-github":
        return _run_scan_github(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
