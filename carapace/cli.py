"""CLI entrypoint for offline Carapace runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Carapace triage engine")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Run offline scan from JSON entity snapshot")
    scan.add_argument("--input", required=True, help="Path to input JSON array of entities")
    scan.add_argument("--repo-path", default=".", help="Repository root path")
    scan.add_argument("--output-dir", default="./carapace-out", help="Output directory")
    scan.add_argument("--org-config", help="Optional org defaults YAML")
    scan.add_argument("--system-config", help="Optional system defaults YAML")
    scan.add_argument("--runtime-override", help="Optional runtime override YAML")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        entities = _load_json_entities(Path(args.input))
        engine = CarapaceEngine.from_repo(
            repo_path=args.repo_path,
            org_defaults=_load_yaml_dict(args.org_config),
            system_defaults=_load_yaml_dict(args.system_config),
            runtime_override=_load_yaml_dict(args.runtime_override),
        )
        report = engine.scan_entities(entities)
        write_report_bundle(report, args.output_dir)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
