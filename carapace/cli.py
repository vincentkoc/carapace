"""CLI entrypoint for offline and GitHub-backed Carapace runs."""

from __future__ import annotations

import argparse

from carapace.commands import audit, ingest, process, scan, serve, triage
from carapace.commands.common import CommandRuntime, default_enrich_workers, normalize_command
from carapace.commands.parser import build_parser as build_cli_parser
from carapace.connectors.github_gh import GithubGhSinkConnector, GithubGhSourceConnector
from carapace.enrichment import enrich_entities_if_needed
from carapace.loader import ingest_github_to_sqlite
from carapace.logging_utils import configure_logging
from carapace.storage import SQLiteStorage


def _default_enrich_workers() -> int:
    return default_enrich_workers()


def build_parser() -> argparse.ArgumentParser:
    return build_cli_parser(default_enrich_workers=_default_enrich_workers)


def _runtime() -> CommandRuntime:
    return CommandRuntime(
        source_connector_cls=GithubGhSourceConnector,
        sink_connector_cls=GithubGhSinkConnector,
        storage_cls=SQLiteStorage,
        ingest_loader=ingest_github_to_sqlite,
        enrich_entities=enrich_entities_if_needed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.command = normalize_command(args.command)
    configure_logging(args.log_level)

    runtime = _runtime()
    command_handlers = {
        "scan": scan.run,
        "ingest": ingest.run,
        "process": process.run_process,
        "enrich": process.run_enrich,
        "triage": triage.run,
        "audit": audit.run,
        "serve": serve.run,
    }
    handler = command_handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args, runtime=runtime)


if __name__ == "__main__":
    raise SystemExit(main())
