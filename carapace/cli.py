"""CLI entrypoint for offline and GitHub-backed Carapace runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from carapace.actioning import apply_routing_decisions
from carapace.connectors.github_gh import GithubGhSinkConnector, GithubGhSourceConnector
from carapace.models import SourceEntity
from carapace.pipeline import CarapaceEngine
from carapace.reporting import write_report_bundle


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


def _build_engine(args: argparse.Namespace) -> CarapaceEngine:
    return CarapaceEngine.from_repo(
        repo_path=args.repo_path,
        org_defaults=_load_yaml_dict(args.org_config),
        system_defaults=_load_yaml_dict(args.system_config),
        runtime_override=_load_yaml_dict(args.runtime_override),
    )


def _add_common_config_flags(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument("--repo-path", default=".", help="Repository root path")
    cmd.add_argument("--org-config", help="Optional org defaults YAML")
    cmd.add_argument("--system-config", help="Optional system defaults YAML")
    cmd.add_argument("--runtime-override", help="Optional runtime override YAML")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Carapace triage engine")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Run offline scan from JSON entity snapshot")
    scan.add_argument("--input", required=True, help="Path to input JSON array of entities")
    scan.add_argument("--output-dir", default="./carapace-out", help="Output directory")
    _add_common_config_flags(scan)

    scan_github = sub.add_parser("scan-github", help="Ingest from GitHub via gh CLI and run triage")
    scan_github.add_argument("--repo", required=True, help="GitHub repo slug, e.g. owner/repo")
    scan_github.add_argument("--gh-bin", default="gh", help="Path/name of gh binary")
    scan_github.add_argument("--max-prs", type=int, default=200, help="Max open PRs to ingest")
    scan_github.add_argument(
        "--include-issues",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include open issues in scan",
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

    return parser


def _entity_number_resolver(entities: list[SourceEntity]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for entity in entities:
        if entity.number is not None:
            mapping[entity.id] = entity.number
    return mapping


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        entities = _load_json_entities(Path(args.input))
        engine = _build_engine(args)
        report = engine.scan_entities(entities)
        write_report_bundle(report, args.output_dir)
        return 0

    if args.command == "scan-github":
        connector = GithubGhSourceConnector(repo=args.repo, gh_bin=args.gh_bin)
        entities = connector.fetch_open_entities(
            max_prs=args.max_prs,
            include_issues=args.include_issues,
            max_issues=args.max_issues,
        )

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.save_input_json:
            (out_dir / "entities.json").write_text(
                json.dumps([entity.model_dump(mode="json") for entity in entities], indent=2)
            )

        engine = _build_engine(args)
        report = engine.scan_entities(entities)
        write_report_bundle(report, out_dir)

        if args.apply_routing:
            id_to_number = _entity_number_resolver(entities)
            sink = GithubGhSinkConnector(
                repo=args.repo,
                gh_bin=args.gh_bin,
                entity_number_resolver=lambda entity_id: id_to_number[entity_id],
                dry_run=not args.live_actions,
            )
            apply_routing_decisions(report, sink)

        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
