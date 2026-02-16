"""CLI parser construction."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from carapace.commands.common import (
    add_common_config_flags,
    add_github_rate_limit_flags,
    add_repo_validation_flags,
    add_report_flags,
)


def build_parser(default_enrich_workers: Callable[[], int]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Carapace triage engine")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", aliases=["run"], help="Run offline scan from JSON entity snapshot")
    scan.add_argument("--input", required=True, help="Path to input JSON array of entities")
    scan.add_argument("--output-dir", default="./carapace-out", help="Output directory")
    add_common_config_flags(scan)
    add_report_flags(scan)

    ingest = sub.add_parser("ingest", aliases=["ingest-github"], help="Ingest GitHub entities into SQLite with resumable state")
    ingest.add_argument("--repo", required=True, help="GitHub repo slug, e.g. owner/repo")
    ingest.add_argument("--gh-bin", default="gh", help="Path/name of gh binary")
    ingest.add_argument("--max-prs", type=int, default=0, help="Max PRs to ingest (0 = no explicit cap)")
    ingest.add_argument("--max-issues", type=int, default=0, help="Max issues to ingest (0 = no explicit cap)")
    add_common_config_flags(ingest)
    add_repo_validation_flags(ingest)
    add_github_rate_limit_flags(ingest)

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
        default=default_enrich_workers(),
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
    add_common_config_flags(process)
    add_repo_validation_flags(process)
    add_github_rate_limit_flags(process)
    add_report_flags(process)

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
        default=default_enrich_workers(),
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
    add_common_config_flags(enrich)
    add_repo_validation_flags(enrich)
    add_github_rate_limit_flags(enrich)

    audit = sub.add_parser("audit", aliases=["db-audit"], help="Show ingest DB audit and integrity summary for a repo")
    audit.add_argument("--repo", required=True, help="GitHub repo slug, e.g. owner/repo")
    audit.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary",
    )
    add_common_config_flags(audit)
    add_repo_validation_flags(audit)

    serve_ui = sub.add_parser("serve", aliases=["serve-ui"], help="Run lightweight graph UI/API over stored SQLite data")
    serve_ui.add_argument("--repo", help="Default repo slug shown in UI, e.g. owner/repo")
    serve_ui.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve_ui.add_argument("--port", type=int, default=8765, help="Bind port")
    serve_ui.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload (development only)",
    )
    add_common_config_flags(serve_ui)
    add_repo_validation_flags(serve_ui)

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
    add_common_config_flags(scan_github)
    add_repo_validation_flags(scan_github)
    add_github_rate_limit_flags(scan_github)
    add_report_flags(scan_github)

    return parser

